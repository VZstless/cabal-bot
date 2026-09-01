# AGENTS.md

## Toolchain

- Package/env manager: **uv** (uv.lock 为准，`.venv` 由 uv 管理)
- Python: 3.10 (`.python-version`)
- 运行任何命令统一用 `uv run ...`，不要直接调用 `.venv/bin/python` 或系统 python
- 依赖改动用 `uv add` / `uv remove`，然后 `uv sync`

## Commands

- `uv sync` — 同步依赖 + 安装本包脚本
- `uv run cabal-train` — 一次性训练音色嵌入（cabal_source/*.wav → assets/se/cabal.pth）
- `uv run cabal-bot` — 启动 Telegram bot（token 在 .env）
- `uv run ruff check src/` / `uv run ruff format src/` — lint / 格式化
- `uv run pyright` — 类型检查

## Project layout

- `src/cabal_bot/` — Telegram bot 主逻辑
- `src/training/` — 音色训练（独立于 bot，`python -m training` 亦可）
- `cabal_source/` — 角色干声（gitignore）
- `checkpoints/` — OpenVoice V2 converter 权重（gitignore，缺失时 cabal-train 自动下载）
- `assets/se/` — 训练出的音色嵌入（gitignore，bot 运行时加载）

## Notes

- setuptools 必须 <82（librosa 0.9.1 依赖 pkg_resources）
- silero VAD 的 torch.hub 信任由 `training.ensure_silero_trusted()` 处理
