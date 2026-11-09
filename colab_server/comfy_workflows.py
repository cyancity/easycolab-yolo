import os
from typing import Optional, List, Dict, Any
import json
from .comfy_presets import (
    _COMFY_STATUS_CACHE,
    COMFY_PRESET_WORKFLOW_PACKS,
    _fetch_official_comfy_template,
    _make_zh_guide_note,
)


def _prune_workflow_dangling_links(wf_data: Dict[str, Any]):
    """确保每个节点的 inputs/outputs 链接表与顶层 links 完全一致，彻底杜绝 LiteGraph 断链警告"""
    node_ids = {n.get("id") for n in wf_data.get("nodes", [])}
    valid_links = [
        lk for lk in wf_data.get("links", [])
        if isinstance(lk, list) and len(lk) >= 5 and lk[1] in node_ids and lk[3] in node_ids
    ]
    wf_data["links"] = valid_links
    valid_lids = {lk[0] for lk in valid_links}
    for n in wf_data.get("nodes", []):
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp.get("link") not in valid_lids:
                inp["link"] = None
        for out in n.get("outputs", []) or []:
            if isinstance(out.get("links"), list):
                out["links"] = [lid for lid in out["links"] if lid in valid_lids]


def _build_and_install_comfy_preset_workflows(
    persist_root: str,
    pack_ids: Optional[List[str]] = None,
) -> List[str]:
    """拉取官方最新模板并转换为全中文标题、内置三步傻瓜注释、G4 高清参数与零缺图报错的开箱即用工作流"""
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    target_set = set(pack_ids) if pack_ids else {p["id"] for p in COMFY_PRESET_WORKFLOW_PACKS}
    installed_files: List[str] = []

    # 1. 万相 Wan2.2 14B (单图 I2V + 首尾帧 FLF2V 4步极速版)
    if "wan22_i2v_flf2v" in target_set:
        try:
            d_i2v = _fetch_official_comfy_template("video_wan2_2_14B_i2v.json")
            kept_nodes = []
            for n in d_i2v.get("nodes", []):
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if n.get("id") == 97 and ntype == "LoadImage":
                    n["title"] = "① 点击此处选择起始图片 (Choose file to upload)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif n.get("id") == 129:
                    n["title"] = "② 填写画面动作与镜头描述 (G4 已预设 1280×720 · 4步极速)"
                    wv = list(n.get("widgets_values") or [])
                    if len(wv) >= 4:
                        wv[0] = "镜头缓慢向前推进，晨曦阳光洒在海面与远山上，云层轻柔流动，水面泛起细腻金光涟漪，电影级光影质感。"
                        wv[1] = 1280
                        wv[2] = 720
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif ntype == "SaveVideo":
                    n["title"] = "③ 生成结果预览与自动保存 (右键可下载视频)"
                    n["widgets_values"] = ["video/Wan2.2_单图生视频", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                kept_nodes.append(n)
            kept_nodes.append(_make_zh_guide_note(
                501,
                [-520, -420],
                "【EasyColab 三步傻瓜出片指南 · 单图生视频】",
                "## 三步直接出片（无需连线！）\n\n"
                "1. **传图**：点击右侧 **「① 点击此处选择起始图片」** 节点底部的上传按钮，选一张你的图片。\n"
                "2. **写动作**：在中间 **「② 填写画面动作与镜头描述」** 第一个文本框里用中文或英文写希望画面怎么动（如：镜头推进、人物转头微笑、发丝随风飘动）。\n"
                "3. **点运行**：点击网页右上角蓝色的 **「运行 (Run)」** 按钮，G4 显卡约 **25 秒** 即可在右侧直接预览成片！\n\n"
                "> 提示：所有生成的视频都会自动同步至你的 Google Drive，也可回到 EasyColab 客户端一键拉取到 Mac 本地。",
            ))
            d_i2v["nodes"] = kept_nodes
            d_i2v["last_node_id"] = max(int(d_i2v.get("last_node_id") or 200), 505)
            d_i2v.setdefault("extra", {})["ds"] = {"scale": 0.72, "offset": [580, 560]}
            _prune_workflow_dangling_links(d_i2v)
            out_1 = os.path.join(wf_dir, "01_Wan2.2_极速单图生视频(4步傻瓜版).json")
            with open(out_1, "w", encoding="utf-8") as wf:
                json.dump(d_i2v, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_1))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 I2V 模板处理异常: {e}")

        try:
            d_flf = _fetch_official_comfy_template("video_wan2_2_14B_flf2v.json")
            # 仅提取官方模板中 4-Step LightX2V LoRA 极速分支连通分量 (原 mode=4)，彻底剔除慢速重复节点并自动激活 (mode=0)
            active_ids = {n["id"] for n in d_flf.get("nodes", []) if n.get("mode") == 4}
            flf_nodes = []
            for n in d_flf.get("nodes", []):
                nid = n.get("id")
                if nid not in active_ids or n.get("type") == "Note":
                    continue
                n["mode"] = 0
                if isinstance(n.get("pos"), list) and len(n["pos"]) >= 2:
                    n["pos"] = [n["pos"][0], n["pos"][1] - 1600]
                if nid == 68 and n.get("type") == "LoadImage":
                    # link 157 -> node 67 slot 5 (start_image)
                    n["title"] = "① 点击上传起始首帧图片 (Start Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 62 and n.get("type") == "LoadImage":
                    # link 158 -> node 67 slot 6 (end_image)
                    n["title"] = "② 点击上传结束尾帧图片 (End Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 6 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "③ 输入首尾帧过渡动作与运镜描述 (正向提示词)"
                    n["widgets_values"] = [
                        "清晨海面的朝阳缓缓升起并化作绚丽的傍晚漫天晚霞，镜头平稳向右推进，云层与海浪自然流动过渡，光影由冷蓝渐变为温暖金红，画面连贯丝滑。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid == 7 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "反向提示词 (保持默认即可)"
                elif nid == 67 and n.get("type") == "WanFirstLastFrameToVideo":
                    n["title"] = "④ 视频尺寸与总帧数 (81帧≈5秒 · 1280×720)"
                    n["widgets_values"] = [1280, 720, 81, 1]
                    n.pop("widgets_values_named", None)
                elif nid == 61 and n.get("type") == "SaveVideo":
                    n["title"] = "⑤ 首尾帧过渡成片预览与自动保存"
                    n["widgets_values"] = ["video/Wan2.2_首尾帧过渡", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                flf_nodes.append(n)

            flf_nodes.append(_make_zh_guide_note(
                502,
                [-430, 490],
                "【EasyColab 三步傻瓜出片指南 · 首尾帧过渡】",
                "## 首尾两张图丝滑变形/运镜（4步极速已开启）\n\n"
                "1. **传首尾两张图**：在右侧 **「① 起始首帧图片」** 和 **「② 结束尾帧图片」** 分别上传开头和结尾画面。\n"
                "2. **写过渡过程**：在 **「③ 输入首尾帧过渡动作与运镜描述」** 写清楚从图 1 怎么变化/运镜到图 2。\n"
                "3. **点右上角运行**：无需按任何快捷键，4 步加速 LoRA 已自动配好，约 **25~35 秒** 直接生成丝滑过渡视频！",
            ))
            d_flf["nodes"] = flf_nodes
            d_flf["links"] = [
                lk for lk in d_flf.get("links", [])
                if lk[1] in active_ids and lk[3] in active_ids
            ]
            zh_group_titles = {
                1: "自动模型加载区 (无需改动)",
                2: "第一步：上传起始首帧图片",
                5: "第二步：上传结束尾帧图片",
                3: "第三步：输入画面过渡描述",
                4: "第四步：视频分辨率与帧数 (默认 1280×720)",
            }
            kept_groups = []
            for g in d_flf.get("groups", []):
                gid = g.get("id")
                if gid in zh_group_titles:
                    b = list(g.get("bounding", [0, 0, 400, 400]))
                    if len(b) >= 2:
                        b[1] = b[1] - 1600
                    g["bounding"] = b
                    g["title"] = zh_group_titles[gid]
                    kept_groups.append(g)
            d_flf["groups"] = kept_groups
            d_flf["last_node_id"] = max(int(d_flf.get("last_node_id") or 200), 505)
            d_flf.setdefault("extra", {})["ds"] = {"scale": 0.62, "offset": [480, 120]}
            _prune_workflow_dangling_links(d_flf)
            out_2 = os.path.join(wf_dir, "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json")
            with open(out_2, "w", encoding="utf-8") as wf:
                json.dump(d_flf, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_2))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Wan2.2 FLF2V 模板处理异常: {e}")

    # 2. MiniMax-H3 Turbo (单图有声视频 + 首尾帧有声视频 8步极速双工作流)
    if "minimax_h3_i2v_flf2v" in target_set:
        try:
            import copy
            d_h3_raw = _fetch_official_comfy_template("video_minimax_h3_i2v.json")

            def _customize_h3(is_dual_frame: bool) -> Dict[str, Any]:
                d_h3 = copy.deepcopy(d_h3_raw)
                nodes_out = []
                for n in d_h3.get("nodes", []):
                    nid = n.get("id")
                    ntype = n.get("type")
                    if ntype == "MarkdownNote" or nid in (119, 120):
                        continue
                    if nid == 114 and ntype == "LoadImage":
                        n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                        n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                        n["widgets_values_named"] = {"image": "easycolab_sample_start.png", "upload": "image"}
                        n["pos"] = [-1320, 4820]
                    elif nid == 115 and ntype == "ResolutionSelector":
                        n["title"] = "画幅比例与清晰度 (0.6≈1056×608 · 0.98=1344×768)"
                        n["widgets_values"] = ["16:9 (Widescreen)", 0.6, 32]
                        n["widgets_values_named"] = {
                            "aspect_ratio": "16:9 (Widescreen)",
                            "megapixels": 0.6,
                            "multiple": 32,
                        }
                        n["pos"] = [-1320, 5660] if is_dual_frame else [-1320, 5260]
                    elif nid == 105:
                        n["title"] = (
                            "③ 填写过渡动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                            if is_dual_frame
                            else "② 填写画面动作 + 音效配乐描述 (8步 Turbo 极速已开启)"
                        )
                        zh_prompt = (
                            "电影级航拍镜头：画面从清晨宁静的蔚蓝海岸平稳推进并过渡至漫天金红晚霞的黄昏海湾，海浪轻拍礁石，光影自然流转。\n"
                            "Audio: 轻柔的海浪拍岸声、温暖的海风吹拂声，伴随舒缓治愈的钢琴与大提琴电影配乐。"
                            if is_dual_frame
                            else "电影级运镜缓慢向前推进，清晨阳光穿透薄雾洒在海面，金光波光粼粼，飞鸟掠过天际。\n"
                                 "Audio: 清脆的海浪声、微风声与远处海鸥鸣叫，搭配温柔空灵的钢琴环境配乐。"
                        )
                        wv = list(n.get("widgets_values") or [])
                        if len(wv) >= 10:
                            wv[0] = zh_prompt
                            wv[9] = True  # 默认启用 8-step Turbo LoRA 极速出片
                        n["widgets_values"] = wv
                        if isinstance(n.get("widgets_values_named"), dict):
                            n["widgets_values_named"]["prompt"] = zh_prompt
                            n["widgets_values_named"]["value"] = True
                    elif nid == 92 and ntype == "SaveVideo":
                        n["title"] = "④ 有声视频预览与自动保存 (请打开播放器音量!)"
                        vid_prefix = "video/MiniMax_H3_首尾帧有声" if is_dual_frame else "video/MiniMax_H3_图生有声"
                        n["widgets_values"] = [vid_prefix, "auto", "auto"]
                        n["widgets_values_named"] = {
                            "filename_prefix": vid_prefix,
                            "format": "auto",
                            "codec": "auto",
                        }
                    nodes_out.append(n)

                links_out = [lk for lk in d_h3.get("links", []) if lk[0] != 228]
                if is_dual_frame:
                    end_node_id = 201
                    new_link_id = 250
                    nodes_out.append({
                        "id": end_node_id,
                        "type": "LoadImage",
                        "pos": [-1320, 5240],
                        "size": [315, 314],
                        "flags": {},
                        "order": 1,
                        "mode": 0,
                        "inputs": [],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [new_link_id]}, {"name": "MASK", "type": "MASK", "links": None}],
                        "title": "② 点击上传结束尾帧图片 (Last Frame)",
                        "properties": {"Node name for S&R": "LoadImage"},
                        "widgets_values": ["easycolab_sample_end.png", "image"],
                        "widgets_values_named": {"image": "easycolab_sample_end.png", "upload": "image"},
                    })
                    for n in nodes_out:
                        if n.get("id") == 105:
                            for inp in n.get("inputs", []):
                                if inp.get("name") == "last_frame":
                                    inp["link"] = new_link_id
                    links_out.append([new_link_id, end_node_id, 0, 105, 1, "IMAGE"])
                    d_h3["last_link_id"] = max(int(d_h3.get("last_link_id") or 230), new_link_id + 1)

                nodes_out.append(_make_zh_guide_note(
                    503,
                    [-1800, 4820],
                    "【EasyColab 三步出片指南 · MiniMax-H3 原生有声视频】",
                    "## 画面 + 立体声音效/配乐一次生成！\n\n"
                    + (
                        "1. **传首尾两张图**：在 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传起止图片。\n"
                        "2. **写画面 + 声音**：在中间 **「③ 过渡动作 + 音效配乐描述」** 写清画面变化，并在 `Audio:` 后写想要的环境音/台词/音乐。\n"
                        "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后记得打开右侧播放器喇叭图标听立体声！"
                        if is_dual_frame
                        else "1. **传首帧图**：在 **「① 起始首帧」** 点击上传你的参考图。\n"
                             "2. **写画面 + 声音**：在中间 **「② 画面动作 + 音效配乐描述」** 描述动作，并在 `Audio:` 后写音效与配乐风格。\n"
                             "3. **点运行**：已默认开启 **8 步 Turbo 加速**，生成后打开右侧视频喇叭即可试听同步立体声！"
                    ),
                ))
                d_h3["nodes"] = nodes_out
                d_h3["links"] = links_out
                d_h3["last_node_id"] = max(int(d_h3.get("last_node_id") or 210), 505)
                d_h3.setdefault("extra", {})["ds"] = {"scale": 0.68, "offset": [1860, -4740]}
                _prune_workflow_dangling_links(d_h3)
                return d_h3

            out_3 = os.path.join(wf_dir, "03_MiniMax-H3_极速图生有声视频(单图带立体声).json")
            with open(out_3, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=False), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_3))

            out_4 = os.path.join(wf_dir, "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json")
            with open(out_4, "w", encoding="utf-8") as wf:
                json.dump(_customize_h3(is_dual_frame=True), wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_4))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] MiniMax-H3 模板处理异常: {e}")

    # 3. LTX-2.3 22B Distilled (首尾帧电影运镜 + 音轨)
    if "ltx23_flf2v" in target_set:
        try:
            d_ltx = _fetch_official_comfy_template("video_ltx2_3_flf2v.json")
            ltx_nodes = []
            for n in d_ltx.get("nodes", []):
                nid = n.get("id")
                ntype = n.get("type")
                if ntype == "MarkdownNote":
                    continue
                if nid == 31 and ntype == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (First Frame)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 39 and ntype == "LoadImage":
                    n["title"] = "② 点击上传结束尾帧图片 (Last Frame)"
                    n["widgets_values"] = ["easycolab_sample_end.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 129:
                    n["title"] = "③ 首尾帧电影运镜与音轨参数 (LTX-2.3 22B Distilled)"
                elif nid == 68 and ntype == "SaveVideo":
                    n["title"] = "④ 电影运镜成片预览与自动保存"
                    n["widgets_values"] = ["video/LTX2.3_首尾帧运镜", "auto", "auto"]
                    n.pop("widgets_values_named", None)
                ltx_nodes.append(n)

            ltx_nodes.append(_make_zh_guide_note(
                504,
                [-490, 2250],
                "【EasyColab 三步傻瓜出片指南 · LTX-2.3 首尾帧运镜】",
                "## 电影级大幅度推拉摇移运镜（自带音效配乐）\n\n"
                "1. **传首尾帧**：在右侧 **「① 起始首帧」** 与 **「② 结束尾帧」** 上传两张不同机位或状态的画面。\n"
                "2. **写运镜轨迹**：在中间 **「③ 首尾帧电影运镜与音轨参数」** 文本框描述镜头怎么移动（建议英文描述运镜 + `Music:` 描述音乐）。\n"
                "3. **点运行**：采用 Distilled 极速蒸馏权重，一键直出高帧率带音轨视频！",
            ))
            d_ltx["nodes"] = ltx_nodes
            d_ltx["last_node_id"] = max(int(d_ltx.get("last_node_id") or 200), 506)
            d_ltx.setdefault("extra", {})["ds"] = {"scale": 0.66, "offset": [560, -2160]}
            _prune_workflow_dangling_links(d_ltx)
            out_5 = os.path.join(wf_dir, "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json")
            with open(out_5, "w", encoding="utf-8") as wf:
                json.dump(d_ltx, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_5))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] LTX-2.3 FLF2V 模板处理异常: {e}")

    # 4. 混元 HunyuanVideo 1.5 (单图 720P -> 1080P 自动潜空间超分)
    if "hunyuan15_i2v_1080p" in target_set:
        try:
            d_hy = _fetch_official_comfy_template("video_hunyuan_video_1.5_720p_i2v.json")
            sr_enable_ids = {108, 111, 109, 113, 114, 117, 118, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141}
            drop_ids = {104, 120, 112, 121, 122, 123, 124, 136}
            hy_nodes = []
            for n in d_hy.get("nodes", []):
                nid = n.get("id")
                if nid in drop_ids:
                    continue
                if nid in sr_enable_ids:
                    n["mode"] = 0
                if nid == 80 and n.get("type") == "LoadImage":
                    n["title"] = "① 点击上传起始首帧图片 (720P → 1080P 超分)"
                    n["widgets_values"] = ["easycolab_sample_start.png", "image"]
                    n.pop("widgets_values_named", None)
                elif nid == 44 and n.get("type") == "CLIPTextEncode":
                    n["title"] = "② 输入画面动作与光影描述 (中英文皆可)"
                    n["widgets_values"] = [
                        "清晨阳光洒在壮阔的海岸山脉上，镜头平稳向前推进，海面波光粼粼，云雾在山谷间缓慢流动，极致写实电影摄影质感。"
                    ]
                    n.pop("widgets_values_named", None)
                elif nid in (127, 131) and n.get("type") == "RandomNoise":
                    wv = list(n.get("widgets_values") or [42, "randomize"])
                    if len(wv) >= 2:
                        wv[1] = "randomize"
                    n["widgets_values"] = wv
                    n.pop("widgets_values_named", None)
                elif nid == 102 and n.get("type") == "SaveVideo":
                    n["title"] = "③ 720P 基础成片预览"
                elif nid == 118 and n.get("type") == "SaveVideo":
                    n["title"] = "④ 1920×1080 超高清成片预览与保存"
                hy_nodes.append(n)

            hy_nodes.append(_make_zh_guide_note(
                506,
                [-1120, 480],
                "【EasyColab 三步傻瓜出片指南 · 混元 1.5 直出 1080P】",
                "## 单图直出 1920×1080 超高清视频（超分管线已激活）\n\n"
                "1. **传图片**：在右侧 **「① 点击上传起始首帧图片」** 选择你的原图。\n"
                "2. **改提示词**：在上方 **「② 输入画面动作与光影描述」** 输入动态描述。\n"
                "3. **点运行**：无需按 `Ctrl+B`，工作流会先生成 720P 动态原片，再自动接力潜空间超分输出 **1920×1080** 超清成片！",
            ))
            d_hy["nodes"] = hy_nodes
            d_hy["links"] = [
                lk for lk in d_hy.get("links", [])
                if lk[1] not in drop_ids and lk[3] not in drop_ids
            ]
            for g in d_hy.get("groups", []):
                if g.get("id") == 3:
                    g["title"] = "第一步：上传起始图片"
                elif g.get("id") == 2:
                    g["title"] = "第二步：输入画面动作描述"
                elif g.get("id") == 8:
                    g["title"] = "1080P 超高清潜空间放大管线 (已默认激活)"
            d_hy["last_node_id"] = max(int(d_hy.get("last_node_id") or 200), 508)
            d_hy.setdefault("extra", {})["ds"] = {"scale": 0.56, "offset": [1180, 120]}
            _prune_workflow_dangling_links(d_hy)
            out_6 = os.path.join(wf_dir, "06_Hunyuan1.5_单图直出1080P超清视频.json")
            with open(out_6, "w", encoding="utf-8") as wf:
                json.dump(d_hy, wf, ensure_ascii=False, indent=2)
            installed_files.append(os.path.basename(out_6))
        except Exception as e:
            print(f"[ComfyUI 工作流生成] Hunyuan 1.5 模板处理异常: {e}")

    if installed_files:
        try:
            with open(os.path.join(wf_dir, ".easycolab_wf_schema_v2"), "w", encoding="utf-8") as sf:
                sf.write("v2")
            ptr_path = os.path.join(persist_root, "user", "default", "easycolab_active_preset.json")
            if (pack_ids and len(pack_ids) == 1) or not os.path.isfile(ptr_path):
                with open(ptr_path, "w", encoding="utf-8") as pf:
                    json.dump({"workflow_filename": installed_files[0]}, pf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = 0.0
    return installed_files


def _get_comfy_preset_packs_status(persist_root: str) -> List[Dict[str, Any]]:
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    results: List[Dict[str, Any]] = []
    for pack in COMFY_PRESET_WORKFLOW_PACKS:
        models = pack["models"]
        installed_m = 0
        for sub, fname, _, _, _ in models:
            p = os.path.join(persist_root, "models", sub, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
                installed_m += 1
        wfs = pack["workflows"]
        installed_wf = sum(1 for w in wfs if os.path.isfile(os.path.join(wf_dir, w)))
        ready = (installed_m == len(models)) and (installed_wf == len(wfs))
        results.append({
            "id": pack["id"],
            "title": pack["title"],
            "badge": pack["badge"],
            "scenario": pack["scenario"],
            "approx_size_gb": pack["approx_size_gb"],
            "workflows": wfs,
            "installed_models": installed_m,
            "total_models": len(models),
            "workflows_installed": installed_wf == len(wfs),
            "ready": ready,
        })
    return results
