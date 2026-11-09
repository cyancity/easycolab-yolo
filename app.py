"""EasyColab Colab 服务端入口。

逻辑都在 colab_server/ 包里；这里按原单文件的执行顺序导入各模块，再启动。
模块顺序即 app.py 原来的语句顺序，导入时的副作用（线程、atexit、环境变量）依赖它，不要调换。
"""
from colab_server import (  # noqa: F401
    state,
    bootstrap,
    kernels,
    qwen_optimize,
    common,
    config,
    hf_download,
    idle,
    model_files,
    telemetry,
    weights,
    gguf_server,
    flash_next_glue,
    vram,
    loader,
    pipeline_opt,
    video,
    comfy_presets,
    comfy_workflows,
    comfy_runtime,
    comfy_sync,
    comfy_start,
    comfy_packs,
    schemas,
    llm_jobs,
    jobs,
    api_auth,
    qwen_stream,
    qwen_prepare,
    routes,
    ui,
    main,
)

for _m in (state, bootstrap, kernels, qwen_optimize, common, config, hf_download, idle, model_files, telemetry, weights, gguf_server, flash_next_glue, vram, loader, pipeline_opt, video, comfy_presets, comfy_workflows, comfy_runtime, comfy_sync, comfy_start, comfy_packs, schemas, llm_jobs, jobs, api_auth, qwen_stream, qwen_prepare, routes, ui, main):
    getattr(_m, "_link_forward", lambda: None)()
from colab_server.main import main

if __name__ == "__main__":
    main()
