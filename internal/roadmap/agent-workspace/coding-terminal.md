# Agent Workspace Coding / TUI 评分卡

> 当前静态分：`2.5 / 5`  
> 目标：验证 Agent Workspace 是否具备 Codex App / Claude Code 级别的真实 coding agent 闭环，而不是只会生成代码片段。

## 1. 5 分标准

| 能力 | 5 分要求 |
| --- | --- |
| Repo 理解 | 能搜索、读文件、定位符号、理解测试与模块边界 |
| Patch 生成 | 以可审查 diff 形式修改文件，不只输出代码块 |
| Diff UI | 多文件 diff、文件列表、行级变更、接受/拒绝/定位 |
| Test loop | 自动运行定向测试，失败后诊断并修复 |
| TUI | 命令 stdout/stderr、exit code、PTY、resize、interrupt、stdin 可见 |
| Git 辅助 | status、diff、commit message、PR context 可作为结构化上下文 |
| 外部 benchmark | 可跑 SWE-bench Lite / Terminal-Bench smoke 子集 |

## 2. 当前证据

| 证据 | 判断 |
| --- | --- |
| Codex app-server | 支持 command/exec、process/spawn、fs、thread、turn、review、diff 等 API |
| CodexMonitor README | 有 git diff、staged/unstaged、PR、file tree、terminal dock、reasoning/tool/diff rendering |
| Lime `internal/roadmap/coding/ui-projection.md` 与 coding 目录 | Agent Workspace coding UI 有路线图，但产品闭环需要实测 |
| Lime 大量 Vitest / smoke / qcloop | 测试基础强，但不等于 Agent 自主 coding loop 已闭环 |
| SWE-bench / Terminal-Bench | 外部 coding/terminal 能力的横向参考 |

## 3. 评分维度

| 评测项 | 当前分 | 必须补证 |
| --- | ---: | --- |
| Code search / repo context | 3.0 | Agent 使用搜索结果定位目标文件 |
| Patch / file write | 3.0 | 实际文件变更与 final answer 一致 |
| Diff review UI | 2.5 | Agent Workspace 中可审查 diff，不只在 git 外部看 |
| Test execution | 2.5 | 定向测试命令、exit code、失败摘要 |
| Failure repair loop | 2.0 | 测试失败后自动修复并重跑 |
| TUI surface | 2.5 | 命令输出、interrupt、exit 可见 |
| SWE-bench compatibility | 1.5 | 至少接入一个 mini fixture |

## 4. P0 实测场景

| 场景 | 输入 | 必须通过 |
| --- | --- | --- |
| `coding-small-fix` | 修一个小 bug | 文件 diff、测试通过、最终说明引用变更 |
| `coding-test-fail-repair` | 故意给失败用例 | 先失败、诊断、修改、重跑通过 |
| `terminal-command-basic` | 运行一个只读命令 | stdout/stderr/exit code 可见 |
| `terminal-interrupt` | 运行长命令后中断 | interrupt 状态正确，terminal 不泄漏 |

## 5. 失败模式

| 失败 | 阻断原因 |
| --- | --- |
| 只输出补丁文本但不改文件 | 不是 coding agent 闭环 |
| 测试命令通过但 UI 不显示 exit code | 用户无法复核 |
| Diff 缺文件路径或行号 | 审查成本过高 |
| terminal 命令无 sandbox/approval 事实 | 安全边界不可证 |

## 6. 下一刀

把 `coding-small-fix` 做成最小 Agent Workspace P0：一个临时 fixture repo、一个确定 bug、一个定向测试、一个 diff UI 断言、一个 Evidence Pack。
