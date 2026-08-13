# 已安排任务 Codex parity 矩阵

状态：`evidence-scoped / no parity claim yet`

## 1. 基准范围

外部代码基准：`/Users/coso/Documents/dev/rust/codex`，审计日期 2026-08-13。

当前能直接验证的源码锚点：

- `codex-rs/app-server-protocol/src/protocol/v2/plugin.rs`
- `codex-rs/app-server/tests/suite/v2/plugin_read.rs`
- `codex-rs/core-plugins/src/remote.rs`

这些证据只证明 Plugin detail scheduled task metadata，不证明截图 UI、CRUD、调度执行与持久化实现。

## 2. 协议 parity

| 语义          | Codex 可验证事实                   | Lime 目标                                   | 结论     |
| ------------- | ---------------------------------- | ------------------------------------------- | -------- |
| 摘要 identity | `key`                              | 本地用 server id；外部模板可保留 source key | 有意差异 |
| 展示名        | `name`                             | `title`（UI 可映射）                        | 等价语义 |
| 指令          | `prompt`                           | `prompt`                                    | 对齐     |
| hourly        | `intervalHours` + optional days    | 同语义，增加 minute/timezone                | 扩展     |
| daily         | `time`                             | 同语义 + timezone                           | 扩展     |
| weekdays      | `time`                             | 同语义 + timezone                           | 扩展     |
| weekly        | `days` + `time`                    | 同语义 + timezone                           | 扩展     |
| weekday       | `MO/TU/WE/TH/FR/SA/SU`             | exact wire 保持相同枚举                     | 对齐     |
| Plugin detail | `scheduledTasks: Option<Vec<...>>` | 后续只作为 draft/template 来源              | 边界明确 |

## 3. 截图行为 parity

| 行为                          | 截图证据         | Lime P0                           |
| ----------------------------- | ---------------- | --------------------------------- |
| 一级“已安排”入口              | 有               | 必须                              |
| 目录搜索 + 全部/已启用/已暂停 | 有               | 必须                              |
| 建议任务                      | 有               | 必须，使用普通模板                |
| 创建菜单：对话创建/手动设置   | 有               | 必须                              |
| 新建详情面板                  | 有               | 必须                              |
| 新聊天/现有聊天               | 有               | 规范为 new_thread/continue_thread |
| 项目、模型、推理              | 已有任务详情可见 | 必须                              |
| 重复、时间、通知              | 有               | 必须                              |
| 运行历史                      | 有               | 必须                              |
| 对话内引导创建                | 有               | 必须，且需结构化确认              |

## 4. 不作 parity 声明的部分

- Codex Desktop 具体组件、CSS、路由和状态管理。
- CRUD method 名称与请求/响应形状。
- 本地数据库 schema。
- 离线补跑、DST、并发、重试和通知实现。
- 新聊天是否每次新建 Thread 的内部策略。
- Screenshot 中模型/推理选择背后的 runtime contract。

这些行为由 Lime current 架构和本需求合同定义，不能声称“复制 Codex 源码”。

## 5. Lime 需要额外承担的能力

- 已有旧 AutomationJob 数据迁移。
- App Server JSON-RPC current 主链与 production mock fail-closed。
- OEM/多 provider model stable id 与 readiness。
- macOS/Windows path、sleep/wake、notification 差异。
- Thread/Turn/Item、Agent Run 和 GUI 历史的一致性。
- 五语种用户文案和 Lime 视觉语言。

## 6. 验证方式

Parity 测试应使用独立 fixture 固化：

- 四种 Codex schedule JSON 的解析/序列化。
- weekday exact wire。
- Plugin scheduledTasks -> ScheduledTaskDraft 映射（接入时）。
- Lime timezone/minute 扩展不会污染或误称 Codex schema。

只有这些逐项通过后，才能在对应矩阵格标记 parity；GUI 相似度不能替代协议证据。
