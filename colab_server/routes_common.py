"""路由模块共用：API 路径前缀。"""
from .common import RUNTIME_SECRET


def api_base() -> str:
    return "/k/" + RUNTIME_SECRET  # 启动日志输出的完整地址包含本段
