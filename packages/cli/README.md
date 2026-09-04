# `@limecloud/lime`

Lime 的官方命令行入口。CLI、TUI 和 Desktop 共用 App Server JSON-RPC、RuntimeCore 与 canonical Thread/Turn/Item，不包含独立任务 runtime。

## 安装

```bash
npm install -g @limecloud/lime
```

`postinstall` 会从 Lime GitHub Release 下载与当前平台匹配的预编译 `lime` 二进制。源码仓库内找不到预编译产物时，wrapper 可回退到：

- `LIME_CLI_BINARY_PATH` 指定的二进制
- `lime-rs/target/release/lime`
- `cargo run -p cli`

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

本地连接默认启动同目录或 `PATH` 中的 `app-server`。可用 `LIME_APP_SERVER_BIN` 或 `--app-server <PATH>` 覆盖。未来 Cloud 连接只允许在 `app-server-client` transport 边界增加认证远端 transport，不改变 CLI 命令或业务协议。

## 发布

```bash
cargo build --manifest-path "../../lime-rs/Cargo.toml" -p cli --release

npm run build:release -- \
  --binary "../../lime-rs/target/release/lime" \
  --out-dir "./dist"
```

Release asset 命名保持 `lime-<version>-<platform>-<arch>.<archive>`，覆盖当前支持的 macOS、Windows 与 Linux 目标。
