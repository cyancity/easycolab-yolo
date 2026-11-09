"""远端内核里执行的启动脚本（挂载 Drive、装依赖、装回 colab_server 包、拉起 app.py 并提取隧道地址）。

字符串内容原样搬运，不要改动。
"""
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime


BOOT_CODE = r"""
import os, sys, time, re, shutil, subprocess

# 0. 配置无网页模式下的 Google Drive 云盘自动挂载/高速同步 (供模型库与 ComfyUI 节点持久化)
rc_stage = "/content/.easycolab_rclone.conf"
if os.path.exists(rc_stage):
    rc_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    rc_dst = os.path.join(rc_dir, "rclone.conf")
    shutil.move(rc_stage, rc_dst)
    os.chmod(rc_dst, 0o600)

if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("✅ 已通过 Google OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                # 容器未开放 FUSE 挂载权限时清理空目录，由 app.py 自动启用 rclone 云盘高速双向同步
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

# 1. 如果尚未安装 DiffSynth-Studio，自动快速安装 (含最新 transformers 以支持 Qwen3.8-27B)
if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    # 大包安装改为后台 + 心跳输出：内核长时间静默会触发桥接侧 iopub 超时（300s），
    # 超时后桥接会换新内核重跑 boot_code，而旧内核仍在装包，两个 pip 并发改坏 site-packages
    _pip = subprocess.Popen([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"])
    while _pip.poll() is None:
        time.sleep(30)
        print("⏳ 依赖安装中，请稍候...", flush=True)
    if _pip.returncode != 0:
        raise RuntimeError("pip install failed with exit code %s" % _pip.returncode)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

# 2. 检查是否已有运行中的 app.py 或 llama-server，先清理旧进程释放 GPU 显存
subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
subprocess.run(["pkill", "-f", "llama-server"], check=False)
# 等旧进程真正退出再拉起新实例：app.py 退出时有 Drive 同步等清理动作，固定 sleep(1) 可能不够，
# 新进程会因 7860 端口被占而启动失败
for _ in range(10):
    _probe = subprocess.run(["pgrep", "-f", "python.*/content/app.py"], capture_output=True)
    if _probe.returncode != 0:
        break
    time.sleep(1)

# 2.5 把平铺上传的包文件装回 /content/colab_server/（先清掉旧版本，避免残留已删除的模块）
_pkg_files = sorted(f for f in os.listdir("/content") if f.startswith("colab_server__") and f.endswith(".py"))
if _pkg_files:
    shutil.rmtree("/content/colab_server", ignore_errors=True)
    os.makedirs("/content/colab_server")
    for _f in _pkg_files:
        shutil.move("/content/" + _f, "/content/colab_server/" + _f[len("colab_server__"):])

# 3. 后台启动 /content/app.py 并等待 gradio.live 链接生成
log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", stdout=out_f, stderr=subprocess.STDOUT)

# 服务地址含 /k/<密钥> 路径段（API 全部挂载其下，裸路径 403），提取时必须保留
url_re = re.compile(r"https://[a-zA-Z0-9.-]+\.gradio\.live(?:/k/[A-Za-z0-9_-]+)?")
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        # Gradio 会先打印不带密钥段的 "Running on public URL"，只认带 /k/ 的完整地址
        keyed = [u for u in url_re.findall(txt) if "/k/" in u]
        if keyed:
            found_url = keyed[-1]
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
"""


def run_boot_script(target_session):
    """在远端内核执行 BOOT_CODE，旧内核不可用时换新内核重跑；返回内核输出列表"""
    def _on_kid(kid):
        target_session.kernel_id = kid
        state.store.add(target_session)

    def _on_sid(sid):
        target_session.session_id = sid
        state.store.add(target_session)

    def _run_boot_on_runtime(use_saved_kernel: bool):
        rt = ColabRuntime(
            target_session.url,
            target_session.token,
            session_name=target_session.name,
            history=state.history,
            kernel_id=target_session.kernel_id if use_saved_kernel else None,
            session_id=target_session.session_id if use_saved_kernel else None,
            on_kernel_started=_on_kid,
            on_session_started=_on_sid,
        )
        try:
            return rt.execute_code(BOOT_CODE, allow_stdin=False, timeout=300)
        finally:
            try:
                rt.stop()
            except Exception:
                pass

    try:
        outputs = _run_boot_on_runtime(use_saved_kernel=True)
    except Exception:
        target_session.kernel_id = None
        target_session.session_id = None
        outputs = _run_boot_on_runtime(use_saved_kernel=False)
