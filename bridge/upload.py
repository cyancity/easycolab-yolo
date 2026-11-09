"""把 app.py、runtime/、colab_server/ 包和 rclone 配置上传到 Colab 实例的 /content。"""
import os
import re
import time
import tempfile

from .credentials import _build_rclone_drive_conf


def upload_server_files(contents, app_py_path):
    for up_attempt in range(4):
        try:
            contents.upload(app_py_path, "content/app.py")
            break
        except Exception as up_err:
            if up_attempt == 3:
                raise up_err
            time.sleep(2.5)
    # runtime/ 下的附加模块（GPU 采集、Flash-Next 运行时）与 app.py 放在同一目录，app.py 按需 import
    rt_dir = os.path.join(os.path.dirname(os.path.abspath(app_py_path)), "runtime")
    for name in sorted(os.listdir(rt_dir)) if os.path.isdir(rt_dir) else []:
        if name.endswith(".py"):
            contents.upload(os.path.join(rt_dir, name), "content/" + name)

    # app.py 是薄入口，逻辑在同目录的 colab_server/ 包里。Jupyter contents 接口不会自动建目录，
    # 所以包文件先以平铺名 colab_server__<模块>.py 传到 /content，启动脚本再装回 /content/colab_server/
    pkg_local = os.path.join(os.path.dirname(os.path.abspath(app_py_path)), "colab_server")
    pkg_files = sorted(n for n in os.listdir(pkg_local) if n.endswith(".py")) if os.path.isdir(pkg_local) else []
    with open(app_py_path, "r", encoding="utf-8") as _af:
        uses_pkg = re.search(r"^from colab_server\b", _af.read(), re.M) is not None
    if uses_pkg and not pkg_files:
        raise RuntimeError("app.py 依赖 colab_server/ 包，但 %s 不存在或为空" % pkg_local)
    for name in pkg_files:
        contents.upload(os.path.join(pkg_local, name), "content/colab_server__" + name)

    rclone_conf_text = _build_rclone_drive_conf()
    if rclone_conf_text:
        tmp_rc_fd, tmp_rc_path = tempfile.mkstemp(prefix="easycolab_rc_", suffix=".conf")
        try:
            with os.fdopen(tmp_rc_fd, "w", encoding="utf-8") as rcf:
                rcf.write(rclone_conf_text)
            os.chmod(tmp_rc_path, 0o600)
            contents.upload(tmp_rc_path, "content/.easycolab_rclone.conf")
        except Exception:
            pass
        finally:
            if os.path.exists(tmp_rc_path):
                try:
                    os.remove(tmp_rc_path)
                except Exception:
                    pass
