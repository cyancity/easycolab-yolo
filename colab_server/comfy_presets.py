import os
import re
from typing import List, Dict, Any
import json
import urllib.parse
import urllib.request
from PIL import Image


# ==============================================================================
# ComfyUI 云端节点工作台 + Google Drive 自动软链与双向持久化引擎
# ==============================================================================
COMFY_STANDARD_MODEL_SUBDIRS = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "vae",
    "loras",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "audio_encoders",
    "embeddings",
    "style_models",
    "gguf",
]

_COMFY_PROTECTED_PIP_PKGS = re.compile(
    r"^\s*(torch|torchvision|torchaudio|xformers)($|[<>=!~;\s\[])",
    re.IGNORECASE,
)
_COMFY_STATE_FILE = "/tmp/.easycolab_comfy_state.json"
_COMFY_STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "counts": (0, 0, 0), "outputs": [], "packs": []}

COMFY_TEMPLATE_RAW_BASE = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/"

# 100% 免审核 (Gated=False) 官方原生节点预置图生视频 / 首尾帧运镜工作流包
COMFY_PRESET_WORKFLOW_PACKS: List[Dict[str, Any]] = [
    {
        "id": "wan22_i2v_flf2v",
        "title": "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        "badge": "新手首选 · 4步极速",
        "scenario": "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        "approx_size_gb": 35.4,
        "workflows": [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json",
        ],
        "models": [
            ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6.27),
            ("vae", "wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", 0.24),
            ("diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 13.31),
            ("diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 13.31),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.14),
            ("loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1.14),
        ],
    },
    {
        "id": "minimax_h3_i2v_flf2v",
        "title": "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        "badge": "自带立体声音效 · 8步加速",
        "scenario": "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        "approx_size_gb": 39.1,
        "workflows": [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
        ],
        "models": [
            ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.62),
            ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.56),
            ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 19.53),
            ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 14.61),
            ("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.82),
        ],
    },
    {
        "id": "ltx23_flf2v",
        "title": "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        "badge": "电影推拉摇移 · 仅 2 个权重",
        "scenario": "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        "approx_size_gb": 36.3,
        "workflows": [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json",
        ],
        "models": [
            ("checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors", "Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", 27.50),
            ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors", "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 8.80),
        ],
    },
    {
        "id": "hunyuan15_i2v_1080p",
        "title": "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        "badge": "720P + 1080P 双路超清",
        "scenario": "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        "approx_size_gb": 43.5,
        "workflows": [
            "06_Hunyuan1.5_单图直出1080P超清视频.json",
        ],
        "models": [
            ("clip_vision", "sigclip_vision_patch14_384.safetensors", "Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", 0.80),
            ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 8.74),
            ("text_encoders", "byt5_small_glyphxl_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", 0.41),
            ("latent_upscale_models", "hunyuanvideo15_latent_upsampler_1080p.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/latent_upscale_models/hunyuanvideo15_latent_upsampler_1080p.safetensors", 0.45),
            ("diffusion_models", "hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors", 15.20),
            ("vae", "hunyuanvideo15_vae_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", 2.35),
            ("diffusion_models", "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors", 15.51),
        ],
    },
]


def _ensure_comfyui_beginner_assets(comfy_root: str, persist_root: str):
    """
    为新手免折腾准备：
    1. 默认将 ComfyUI 界面语言设为简体中文 (Comfy.Locale = zh)；
    2. 生成 1280x720 预设示例首帧/尾帧图片至 input/ 目录并建立软链，即使未传图直接点「运行」也零报错。
    """
    try:
        settings_dir = os.path.join(persist_root, "user", "default")
        os.makedirs(settings_dir, exist_ok=True)
        settings_path = os.path.join(settings_dir, "comfy.settings.json")
        cfg_obj: Dict[str, Any] = {}
        if os.path.isfile(settings_path):
            try:
                cfg_obj = json.load(open(settings_path, "r", encoding="utf-8"))
            except Exception:
                cfg_obj = {}
        if not cfg_obj.get("Comfy.Locale"):
            cfg_obj["Comfy.Locale"] = "zh"
            with open(settings_path, "w", encoding="utf-8") as sf:
                json.dump(cfg_obj, sf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        from PIL import ImageDraw
        input_persist = os.path.join(persist_root, "input")
        os.makedirs(input_persist, exist_ok=True)
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(input_persist, os.path.join(comfy_root, "input"))

        start_img_p = os.path.join(input_persist, "easycolab_sample_start.png")
        end_img_p = os.path.join(input_persist, "easycolab_sample_end.png")

        if not os.path.isfile(start_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (16, 24, 48))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(18 * (1 - t) + 64 * t)
                g = int(28 * (1 - t) + 108 * t)
                b = int(68 * (1 - t) + 168 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 太阳光晕与海平面剪影（首帧：朝阳初升左侧）
            dr.ellipse([260, 210, 440, 390], fill=(255, 206, 126))
            dr.ellipse([285, 235, 415, 365], fill=(255, 240, 196))
            dr.polygon([(0, 520), (240, 430), (520, 495), (860, 410), (1140, 485), (1280, 450), (1280, 720), (0, 720)], fill=(18, 28, 46))
            dr.rectangle([0, 565, w, h], fill=(12, 22, 38))
            im.save(start_img_p, format="PNG")

        if not os.path.isfile(end_img_p):
            w, h = 1280, 720
            im = Image.new("RGB", (w, h), (32, 18, 52))
            dr = ImageDraw.Draw(im)
            for y in range(h):
                t = y / float(h - 1)
                r = int(42 * (1 - t) + 232 * t)
                g = int(24 * (1 - t) + 118 * t)
                b = int(76 * (1 - t) + 88 * t)
                dr.line([(0, y), (w, y)], fill=(r, g, b))
            # 尾帧：晚霞金日移至右侧高空，测试首尾帧运镜与光影平滑过渡
            dr.ellipse([800, 130, 1000, 330], fill=(255, 168, 92))
            dr.ellipse([830, 160, 970, 300], fill=(255, 232, 178))
            dr.polygon([(0, 540), (210, 460), (490, 510), (820, 395), (1090, 470), (1280, 430), (1280, 720), (0, 720)], fill=(24, 16, 36))
            dr.rectangle([0, 575, w, h], fill=(16, 12, 28))
            im.save(end_img_p, format="PNG")
    except Exception as e:
        print(f"[ComfyUI] 生成默认占位参考图提示: {e}")


def _fetch_official_comfy_template(template_filename: str) -> Dict[str, Any]:
    url = COMFY_TEMPLATE_RAW_BASE + template_filename
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab/2.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_zh_guide_note(node_id: int, pos: List[float], title: str, steps_md: str) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": pos,
        "size": [430, 340],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {},
        "widgets_values": [steps_md],
        "widgets_values_named": {"text": steps_md},
        "color": "#1e3a5f",
        "bgcolor": "#13263e",
    }


def _ensure_easycolab_comfy_preset_helper(comfy_root: str, persist_root: str):
    """
    预置轻量前端扩展：每次通过全新 *.trycloudflare.com 域名首次打开 ComfyUI 时，
    由于新域名 localStorage 为空，ComfyUI 默认会加载内置的 Z-Image Turbo 文生图模板，
    并把缺失的模型自动替换成目录里字母序第一个文件（例如把 minimax_h3_audio_vae 塞进图像 VAEDecode 导致维度 24!=32 报错）。
    本扩展会在检测到默认 Z-Image Turbo 占位图时，自动切换为用户已安装的 EasyColab 中文预置工作流。
    """
    if not os.path.isdir(comfy_root):
        return
    try:
        helper_dir = os.path.join(comfy_root, "custom_nodes", "easycolab_preset_helper")
        js_dir = os.path.join(helper_dir, "js")
        os.makedirs(js_dir, exist_ok=True)

        init_py = '''import os
import json
from aiohttp import web
import folder_paths
from server import PromptServer

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.get("/easycolab/default_workflow")
        async def _easycolab_get_default_workflow(request):
            try:
                get_ud = getattr(folder_paths, "get_user_directory", None)
                user_dir = get_ud() if callable(get_ud) else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "user",
                )
                default_dir = os.path.join(user_dir, "default")
                wf_dir = os.path.join(default_dir, "workflows")
                target_fn = ""
                ptr_path = os.path.join(default_dir, "easycolab_active_preset.json")
                if os.path.isfile(ptr_path):
                    try:
                        with open(ptr_path, "r", encoding="utf-8") as pf:
                            ptr = json.load(pf)
                        cand = str(ptr.get("workflow_filename") or "").strip()
                        if cand and os.path.isfile(os.path.join(wf_dir, cand)):
                            target_fn = cand
                    except Exception:
                        pass
                if not target_fn and os.path.isdir(wf_dir):
                    cands = sorted(
                        f for f in os.listdir(wf_dir)
                        if f.lower().endswith(".json") and not f.startswith(".")
                    )
                    if cands:
                        target_fn = cands[0]
                if target_fn:
                    wf_path = os.path.join(wf_dir, target_fn)
                    with open(wf_path, "r", encoding="utf-8") as wf:
                        wf_obj = json.load(wf)
                    return web.json_response({"ok": True, "filename": target_fn, "workflow": wf_obj})
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)})
            return web.json_response({"ok": False})
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
'''

        autoload_js = '''import { app } from "../../scripts/app.js";

function isBuiltInZImageStarterGraph(graph) {
    if (!graph || !Array.isArray(graph._nodes)) return false;
    if (graph._nodes.length === 0) return false;
    return graph._nodes.some((n) => {
        const models = n?.properties?.models;
        return Array.isArray(models) && models.some((m) => String(m?.url || "").includes("z_image_turbo"));
    });
}

app.registerExtension({
    name: "EasyColab.PresetWorkflowAutoLoader",
    async setup() {
        if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") {
            return;
        }
        const tryAutoLoad = async (allowEmpty) => {
            if (sessionStorage.getItem("easycolab_preset_autoloaded_v1") === "1") return true;
            const g = app.graph;
            const isStarter = isBuiltInZImageStarterGraph(g);
            const isEmpty = allowEmpty && g && Array.isArray(g._nodes) && g._nodes.length === 0;
            if (!isStarter && !isEmpty) return false;

            sessionStorage.setItem("easycolab_preset_autoloaded_v1", "1");
            try {
                const resp = await fetch("/easycolab/default_workflow", { cache: "no-store" });
                if (!resp.ok) return true;
                const data = await resp.json();
                if (data && data.ok && data.workflow) {
                    const wfTitle = String(data.filename || "EasyColab_Preset").replace(/\\.json$/i, "");
                    await app.loadGraphData(data.workflow, true, true, wfTitle);
                    console.log("[EasyColab] 已自动载入预置中文工作流:", data.filename);
                }
            } catch (err) {
                console.warn("[EasyColab] 自动载入预置工作流跳过:", err);
            }
            return true;
        };

        setTimeout(() => { tryAutoLoad(false); }, 350);
        setTimeout(() => { tryAutoLoad(true); }, 1100);
    },
});
'''
        with open(os.path.join(helper_dir, "__init__.py"), "w", encoding="utf-8") as f_py:
            f_py.write(init_py)
        with open(os.path.join(js_dir, "autoload.js"), "w", encoding="utf-8") as f_js:
            f_js.write(autoload_js)
    except Exception as e:
        print(f"[ComfyUI] 写入 EasyColab 默认工作流引导扩展提示: {e}")


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _safe_symlink_dir
    from .comfy_runtime import _safe_symlink_dir
