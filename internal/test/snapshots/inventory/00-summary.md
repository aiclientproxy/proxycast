# Codex Snapshot Inventory Summary

扫描基线：`/Users/coso/Documents/dev/rust/codex` @ `4c43465133428898aa84f0bfc02c306ed65fb66a`。

命令：

```bash
find "/Users/coso/Documents/dev/rust/codex/codex-rs" -type f -name '*.snap' | sort
```

- 源文件：`663`
- 去平台后缀的逻辑场景：`658`
- 平台变体：`5`

本目录的分块列表只记录原始相对路径；每个 path 的 Lime scenario/disposition 在
`../03-source-to-scenario-map.md` 中唯一维护。

## Owner 计数

| Codex snapshot owner | 文件数 | inventory 分块 |
| --- | ---: | --- |
| `codex-rs/cli/src/doctor/snapshots` | 1 | [01-cli-core](../inventory/01-cli-core.md) |
| `codex-rs/core/src/context/world_state/snapshots` | 9 | [01-cli-core](../inventory/01-cli-core.md) |
| `codex-rs/core/src/guardian/snapshots` | 3 | [01-cli-core](../inventory/01-cli-core.md) |
| `codex-rs/core/src/session/snapshots` | 1 | [01-cli-core](../inventory/01-cli-core.md) |
| `codex-rs/core/tests/suite/snapshots` | 38 | [01-cli-core](../inventory/01-cli-core.md) |
| `codex-rs/tui/src/app/snapshots` | 6 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/app/tests/snapshots` | 2 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/bottom_pane/request_user_input/snapshots` | 15 | [03-tui-bottom-pane](../inventory/03-tui-bottom-pane.md) |
| `codex-rs/tui/src/bottom_pane/snapshots` | 174 | [03-tui-bottom-pane](../inventory/03-tui-bottom-pane.md) |
| `codex-rs/tui/src/chatwidget/snapshots` | 212 | [02-tui-chat-history](../inventory/02-tui-chat-history.md) |
| `codex-rs/tui/src/chatwidget/tests/snapshots` | 11 | [02-tui-chat-history](../inventory/02-tui-chat-history.md) |
| `codex-rs/tui/src/exec_cell/snapshots` | 1 | [02-tui-chat-history](../inventory/02-tui-chat-history.md) |
| `codex-rs/tui/src/history_cell/snapshots` | 48 | [02-tui-chat-history](../inventory/02-tui-chat-history.md) |
| `codex-rs/tui/src/onboarding/snapshots` | 3 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/render/snapshots` | 1 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/snapshots` | 117 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/status/snapshots` | 19 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |
| `codex-rs/tui/src/streaming/snapshots` | 2 | [04-tui-app-shell](../inventory/04-tui-app-shell.md) |

## 分块校验

四个分块的条目数量之和必须等于 `663`。路径变更后，重新执行扫描并更新本目录、
`../03-source-to-scenario-map.md` 和 `internal/exec-plans/codex-snapshot-frontend-test-plan.md` 的数字。
