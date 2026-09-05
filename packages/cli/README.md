# `@limecloud/lime`

Lime 的官方命令行入口。CLI、TUI 和 Desktop 共用 App Server JSON-RPC、RuntimeCore 与 canonical Thread/Turn/Item，不包含独立任务 runtime。

## 安装

```bash
npm install -g @limecloud/lime
```

根 npm 包只包含 launcher，并通过 optional dependency 安装当前平台的原生载荷；安装阶段不执行网络下载脚本。平台载荷原子包含 `lime`、`app-server`、`code-mode-host`、Windows sandbox helpers 和所需动态库，保证默认 TUI 与 `exec` 都进入同一 App Server 产品链。

当前发布目标为 macOS arm64/x64、Windows x64 与 Linux x64 GNU。尚无真实构建和运行证据的平台会明确拒绝启动，不发布空壳 optional package。

## 命令

```bash
# 默认启动交互式 TUI
lime

# 显式启动 TUI
lime tui

# 非交互运行
lime exec "review this diff"
lime exec --json "review this diff"
lime exec --jsonl "review this diff"

# 只读检查 execpolicy prefix rules
lime execpolicy check --rules ./rules/policy.rules --pretty git push origin main

# 生成 shell completion
lime completion zsh > "${fpath[1]}/_lime"

# 恢复 canonical Thread
lime resume
lime resume <thread-id>

# App Server read/control surfaces
lime thread list
lime thread show <thread-id> --include-turns
lime mcp list
lime skills list
```

`--json` 输出可读的稳定 JSON；`--jsonl` 输出单行 JSON envelope，二者互斥。`completion` 从同一命令树生成 bash、zsh、fish、PowerShell 和 elvish 脚本。

交互式 TUI 支持 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`。可通过默认 `lime`、`lime tui` 或 `lime resume` 的 `--locale <LOCALE>` 指定；未指定时按 `LIME_LOCALE`、`LC_ALL`、`LANG` 解析，未知语言回退到 `en-US`。`exec` 与管理命令的 JSON 合同保持语言无关。

本地连接默认启动同目录或 `PATH` 中的 `app-server`。可用 `LIME_APP_SERVER_BIN` 或 `--app-server <PATH>` 覆盖。需要连接远端 App Server 时使用 Codex 形状的参数：

```bash
lime exec --remote wss://cloud.example/rpc --remote-auth-token-env LIME_REMOTE_TOKEN "review this diff"
lime tui --remote ws://127.0.0.1:4500
```

远端 token 只从环境变量读取；token 连接要求 `wss://` 或 loopback `ws://`。当前仅提供 transport foundation，不代表生产 Cloud endpoint、租户或账号能力已启用。

## 发布

```bash
python3 scripts/build_npm_package.py \
  --package lime \
  --release-version 1.140.0 \
  --pack-output dist/lime-npm-1.140.0.tgz
```

平台包使用 `--package lime-<platform>-<arch> --vendor-src <vendor-root>` staging。`vendor-root` 必须使用 `vendor/<target-triple>/bin` 布局并包含完整 runtime payload。发布工作流先串行发布四个平台版本，最后发布 `@limecloud/lime` 根版本。
