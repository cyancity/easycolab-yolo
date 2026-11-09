"""EasyColab 桥接实现包：入口是仓库根的 colab_bridge.py。

先导入 env，保证 UTF-8 输出和系统代理注入在 colab_cli 等依赖导入之前完成。
"""
from . import env  # noqa: F401
