#!/usr/bin/env python3
"""
Local Headless Google Colab Bridge for EasyColab macOS App.
Powered by google-colab-cli (inspired by FireTable/comfy-colab-kit).
Supports:
  - auth_status : Check OAuth login state, real CU balance, hourly burn rate, and active VMs
  - auth_url    : Generate Google OAuth2 authorization URL for 1-click browser login
  - auth_code   : Exchange Google authorization code and save ~/.config/colab-cli/token.json
  - wake        : Provision/reuse Colab GPU VM (G4/H100/A100/L4/T4), deploy app.py, and return gradio.live URL
  - keepalive   : Refresh Colab runtime proxy & kernel activity while active
  - stop_all    : Unassign and terminate all active Colab VMs on the account (verify 0 CU/hr burn)
"""

# 实现拆在 bridge/ 包里；导入 bridge 时会先完成 UTF-8 输出与系统代理注入，再导入 colab_cli
from bridge.cli import main, run_cli  # noqa: F401


if __name__ == "__main__":
    run_cli()
