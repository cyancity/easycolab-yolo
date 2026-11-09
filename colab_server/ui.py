import gradio as gr
from .config import DEFAULT_OUTPUT_DIR
from .telemetry import get_gpu_and_model_summary_text
from .weights import download_minimax_h3_weights_core
from .loader import load_minimax_h3_to_vram_core
from .video import generate_video, get_history_videos
from .comfy_packs import disconnect_colab_runtime


# ==============================================================================
# 预设提示词模板与 Gradio 界面
# ==============================================================================
PRESET_EXAMPLES = [
    [
        "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        35,
    ],
    [
        "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
        "832 × 480 (16:9 横屏标准 - 推荐)",
        124,
        40,
    ],
]

CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}
.header-banner p {
    margin: 0;
    font-size: 0.95rem;
    color: #cbd5e1;
}
.status-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
"""


WEBUI_HEADER_ZH = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab 多模态 AI 推理与生成服务端</h1>
    <p>支持 <b>ComfyUI 节点工作台 (云盘软链持久化) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · 自动关机看门狗</p>
</div>
"""

WEBUI_HEADER_EN = """
<div class="header-banner">
    <h1>⚡ EasyColab · Google Colab Multimodal AI Inference & Generation Server</h1>
    <p>Supports <b>ComfyUI Node Studio (Google Drive Persistence) · MiniMax-H3 · Qwen3.8-27B (GGUF+MTP / BF16) · Qwen-Image-2.1 · Z-Image-Turbo</b> · Idle Auto-Shutdown Watchdog</p>
</div>
"""

RESOLUTION_CHOICES_ZH = [
    ("832 × 480 (16:9 横屏标准 - 推荐)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 竖屏短视频)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P 高清)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P 竖屏高清)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 方形)", "640 × 640 (1:1 方形)"),
]

RESOLUTION_CHOICES_EN = [
    ("832 × 480 (16:9 Landscape · Recommended)", "832 × 480 (16:9 横屏标准 - 推荐)"),
    ("480 × 832 (9:16 Portrait Short)", "480 × 832 (9:16 竖屏短视频)"),
    ("1280 × 720 (16:9 720P HD)", "1280 × 720 (16:9 720P 高清)"),
    ("720 × 1280 (9:16 720P Portrait)", "720 × 1280 (9:16 720P 竖屏高清)"),
    ("640 × 640 (1:1 Square)", "640 × 640 (1:1 方形)"),
]


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="EasyColab · Colab 多模型 AI 服务端") as demo:
        ui_lang_state = gr.State(value="zh")

        with gr.Row():
            with gr.Column(scale=9):
                header_html = gr.HTML(WEBUI_HEADER_ZH)
            with gr.Column(scale=1, min_width=130):
                lang_toggle_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    sec1_md = gr.Markdown("### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)")
                    gpu_status_text = gr.Textbox(
                        label="硬件、算力消耗与模型检查报告",
                        value=get_gpu_and_model_summary_text(),
                        lines=9,
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    refresh_gpu_btn = gr.Button("🔍 检查 / 刷新模型与算力状态", size="sm")

                    precision_Choice = gr.Radio(
                        choices=[
                            ("BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                        ],
                        value="bf16",
                        label="模型权重精度",
                    )

                    offload_choice = gr.Radio(
                        choices=[
                            ("⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                            ("CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                        ],
                        value="disk",
                        label="显存加载与卸载策略 (VRAM Strategy)",
                    )

                    with gr.Accordion("🔑 Hugging Face Token (可选 / Optional)", open=False):
                        hf_token_input = gr.Textbox(
                            label="HF Access Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
                            type="password",
                        )

                    with gr.Row():
                        download_model_btn = gr.Button(
                            "📥 1. 极速下载模型权重",
                            variant="secondary",
                        )
                        load_model_btn = gr.Button(
                            "🚀 2. 加载模型至显存",
                            variant="primary",
                        )

                with gr.Group():
                    sec2_md = gr.Markdown("### ⚙️ 2. 视频与采样参数")
                    resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTION_CHOICES_ZH,
                        value="832 × 480 (16:9 横屏标准 - 推荐)",
                        label="输出分辨率 (Resolution)",
                    )
                    with gr.Row():
                        num_frames_slider = gr.Slider(
                            minimum=90,
                            maximum=362,
                            step=17,
                            value=124,
                            label="生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)",
                        )
                        fps_slider = gr.Slider(
                            minimum=16,
                            maximum=30,
                            step=1,
                            value=24,
                            label="帧率 (FPS)",
                        )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=60,
                        step=5,
                        value=35,
                        label="采样步数 (Inference Steps，推荐 30~50)",
                    )
                    with gr.Row():
                        seed_input = gr.Number(
                            value=42,
                            precision=0,
                            label="随机种子 (Seed)",
                        )
                        randomize_seed_chk = gr.Checkbox(
                            value=True,
                            label="🎲 每次随机种子",
                        )

                    save_dir_input = gr.Textbox(
                        value=DEFAULT_OUTPUT_DIR,
                        label="📂 视频自动保存路径",
                    )

                with gr.Accordion("💡 Google Colab 算力省额度工具 / Compute Quota Saver", open=True):
                    disconnect_btn = gr.Button("🛑 停止任务并释放 Colab 运行时 (省计算单元)", variant="stop")

            with gr.Column(scale=6):
                task_mode_state = gr.State(value="t2va")

                with gr.Tabs() as mode_tabs:
                    with gr.TabItem("✨ 文生音视频 / T2VA", id="t2va") as tab_t2va:
                        t2va_desc_md = gr.Markdown("输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。")

                    with gr.TabItem("🖼️ 首尾帧生音视频 / FL2VA", id="fl2va") as tab_fl2va:
                        with gr.Row():
                            first_frame_input = gr.Image(
                                label="📌 首帧图片 (First Frame - index 0)",
                                type="pil",
                                height=220,
                            )
                            last_frame_input = gr.Image(
                                label="🏁 尾帧图片 (Last Frame - index -1，可选)",
                                type="pil",
                                height=220,
                            )

                with gr.Group():
                    visual_prompt_input = gr.Textbox(
                        label="🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                        placeholder="描述画面主体、动作、运镜方式、光影风格...",
                        lines=4,
                    )
                    with gr.Row():
                        soundscape_prompt_input = gr.Textbox(
                            label="🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                            placeholder="例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                            lines=2,
                        )
                        music_prompt_input = gr.Textbox(
                            label="🎵 背景配乐风格 (Background Music - 可选)",
                            placeholder="例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                            lines=2,
                        )

                generate_btn = gr.Button(
                    "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                    variant="primary",
                    size="lg",
                )

                with gr.Group():
                    output_video = gr.Video(
                        label="🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)",
                        height=420,
                        autoplay=True,
                    )
                    run_status_box = gr.Textbox(
                        label="运行日志与状态",
                        value="请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」...",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                    )

                with gr.Accordion("📚 提示词一键体验模板 / Preset Prompt Templates", open=True):
                    gr.Examples(
                        examples=PRESET_EXAMPLES,
                        inputs=[
                            visual_prompt_input,
                            soundscape_prompt_input,
                            music_prompt_input,
                            resolution_dropdown,
                            num_frames_slider,
                            steps_slider,
                        ],
                    )

                with gr.Accordion("🗂️ 本次会话历史生成记录 / Session History", open=False):
                    history_gallery = gr.Gallery(
                        label="已生成的视频文件",
                        value=get_history_videos(DEFAULT_OUTPUT_DIR),
                        columns=3,
                        height=240,
                    )
                    refresh_history_btn = gr.Button("🔄 刷新历史记录", size="sm")

        def _toggle_webui_lang(cur_lang: str, cur_prec: str, cur_offload: str, cur_res: str, cur_status: str):
            new_lang = "en" if cur_lang == "zh" else "zh"
            is_en = new_lang == "en"
            default_zh_status = "请先在左侧点击「1. 极速下载模型权重」并「2. 加载模型至显存」..."
            default_en_status = "Click '1. Download Weights' then '2. Load to VRAM' on the left to get started..."
            if is_en and cur_status == default_zh_status:
                next_status = default_en_status
            elif not is_en and cur_status == default_en_status:
                next_status = default_zh_status
            else:
                next_status = cur_status

            return (
                new_lang,
                WEBUI_HEADER_EN if is_en else WEBUI_HEADER_ZH,
                "🌐 简体中文" if is_en else "🌐 English",
                "### 📦 1. Model Weights Center (Download First, Then Load to VRAM)" if is_en else "### 📦 1. 模型权重管理中心 (先下载检查，再加载显存)",
                gr.update(label="Hardware, CU Usage & Model Inspection Report" if is_en else "硬件、算力消耗与模型检查报告"),
                "🔍 Check / Refresh GPU & Model Status" if is_en else "🔍 检查 / 刷新模型与算力状态",
                gr.update(
                    choices=[
                        ("Native BF16 Full Precision (Recommended for G4 96GB)" if is_en else "BF16 原生精度版 (官方满血权重 · G4 96GB 极速推理)", "bf16"),
                    ],
                    value=cur_prec or "bf16",
                    label="Model Precision" if is_en else "模型权重精度",
                ),
                gr.update(
                    choices=[
                        ("⚡ Zero-Copy Fast Mapping + GPU VRAM Resident (Recommended · ~5s)" if is_en else "⚡ 零拷贝极速映射 + GPU 显存常驻 (推荐 · 5秒秒加载)", "disk"),
                        ("Full CPU RAM Preload (Slower · ~3-5 min)" if is_en else "CPU 内存全量预载 (耗时较长 · 约3~5分钟)", "cpu"),
                    ],
                    value=cur_offload or "disk",
                    label="VRAM Loading Strategy" if is_en else "显存加载与卸载策略 (VRAM Strategy)",
                ),
                "📥 1. Download Model Weights" if is_en else "📥 1. 极速下载模型权重",
                "🚀 2. Load Model to VRAM" if is_en else "🚀 2. 加载模型至显存",
                "### ⚙️ 2. Video & Sampling Parameters" if is_en else "### ⚙️ 2. 视频与采样参数",
                gr.update(
                    choices=RESOLUTION_CHOICES_EN if is_en else RESOLUTION_CHOICES_ZH,
                    value=cur_res or "832 × 480 (16:9 横屏标准 - 推荐)",
                    label="Output Resolution" if is_en else "输出分辨率 (Resolution)",
                ),
                gr.update(label="Total Frames (90~362 frames ≈ 4~15s, 17n+5 aligned)" if is_en else "生成总帧数 (90~362帧 ≈ 4~15秒, 17n+5 对齐)"),
                gr.update(label="Frame Rate (FPS)" if is_en else "帧率 (FPS)"),
                gr.update(label="Inference Steps (30~50 recommended)" if is_en else "采样步数 (Inference Steps，推荐 30~50)"),
                gr.update(label="Seed" if is_en else "随机种子 (Seed)"),
                gr.update(label="🎲 Randomize Seed Every Run" if is_en else "🎲 每次随机种子"),
                gr.update(label="📂 Output Save Directory" if is_en else "📂 视频自动保存路径"),
                "🛑 Stop Jobs & Release Colab Runtime (Save CU)" if is_en else "🛑 停止任务并释放 Colab 运行时 (省计算单元)",
                "Describe visual scenes and soundscapes to generate HD video with **32kHz stereo audio**." if is_en else "输入画面描述与声音细节，同步生成高清视频与 **32kHz 立体声音轨**。",
                gr.update(label="📌 First Frame Image (index 0)" if is_en else "📌 首帧图片 (First Frame - index 0)"),
                gr.update(label="🏁 Last Frame Image (index -1, Optional)" if is_en else "🏁 尾帧图片 (Last Frame - index -1，可选)"),
                gr.update(
                    label="🎥 Visual & Camera Prompt" if is_en else "🎥 画面主体与镜头描述 (Visual & Camera Prompt)",
                    placeholder="Describe subject, motion, camera movement, lighting, and visual style..." if is_en else "描述画面主体、动作、运镜方式、光影风格...",
                ),
                gr.update(
                    label="🔊 Soundscape & Dialogue (Optional)" if is_en else "🔊 环境音效与人声描述 (Soundscape & Foley - 可选)",
                    placeholder="e.g., Raindrops hitting pavement, footsteps in puddles, soft spoken dialogue..." if is_en else "例如：雨滴敲击声、脚步踩水声、温柔的女声独白...",
                ),
                gr.update(
                    label="🎵 Background Music Style (Optional)" if is_en else "🎵 背景配乐风格 (Background Music - 可选)",
                    placeholder="e.g., Moody cyberpunk synthwave, soaring orchestral strings..." if is_en else "例如：低沉的赛博朋克合成器电子乐、恢弘管弦乐...",
                ),
                "🎬 Generate Video + Stereo Audio" if is_en else "🎬 开始生成音视频 (Generate Video + Stereo Audio)",
                gr.update(label="🎞️ Output Preview (Unmute player for native stereo audio)" if is_en else "🎞️ 生成结果预览 (请开启播放器声音体验原生立体声)"),
                gr.update(label="Run Status & Logs" if is_en else "运行日志与状态", value=next_status),
                gr.update(label="Generated Videos" if is_en else "已生成的视频文件"),
                "🔄 Refresh History" if is_en else "🔄 刷新历史记录",
            )

        lang_toggle_btn.click(
            fn=_toggle_webui_lang,
            inputs=[ui_lang_state, precision_Choice, offload_choice, resolution_dropdown, run_status_box],
            outputs=[
                ui_lang_state,
                header_html,
                lang_toggle_btn,
                sec1_md,
                gpu_status_text,
                refresh_gpu_btn,
                precision_Choice,
                offload_choice,
                download_model_btn,
                load_model_btn,
                sec2_md,
                resolution_dropdown,
                num_frames_slider,
                fps_slider,
                steps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
                disconnect_btn,
                t2va_desc_md,
                first_frame_input,
                last_frame_input,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                generate_btn,
                output_video,
                run_status_box,
                history_gallery,
                refresh_history_btn,
            ],
        )

        tab_t2va.select(fn=lambda: "t2va", inputs=None, outputs=task_mode_state)
        tab_fl2va.select(fn=lambda: "fl2va", inputs=None, outputs=task_mode_state)

        refresh_gpu_btn.click(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        download_model_btn.click(
            fn=lambda tok: download_minimax_h3_weights_core(tok)[1],
            inputs=[hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        load_model_btn.click(
            fn=lambda prec, off, tok: load_minimax_h3_to_vram_core(prec, off, tok)[1],
            inputs=[precision_Choice, offload_choice, hf_token_input],
            outputs=run_status_box,
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        generate_btn.click(
            fn=generate_video,
            inputs=[
                task_mode_state,
                visual_prompt_input,
                soundscape_prompt_input,
                music_prompt_input,
                first_frame_input,
                last_frame_input,
                resolution_dropdown,
                num_frames_slider,
                steps_slider,
                fps_slider,
                seed_input,
                randomize_seed_chk,
                save_dir_input,
            ],
            outputs=[output_video, run_status_box, seed_input, history_gallery],
        ).then(fn=get_gpu_and_model_summary_text, inputs=None, outputs=gpu_status_text)

        refresh_history_btn.click(
            fn=get_history_videos,
            inputs=save_dir_input,
            outputs=history_gallery,
        )

        disconnect_btn.click(
            fn=disconnect_colab_runtime,
            inputs=None,
            outputs=run_status_box,
        )

    return demo
