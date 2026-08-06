## Lime v1.122.0

### 新功能

- 新增 Thread sections：支持创建、重命名、删除和移动会话，并在侧边栏按 section 组织会话。
- App Server v2、Rust thread store、typed client 与 GUI 同步接入 Thread section 的列表、排序和持久化能力。

### 修复

- 修复会话列表与 Agent chat 状态更新时的排序、恢复和历史预览一致性问题。
- 修复 Thread metadata、通知、projection、read model 与 JSON-RPC schema 之间的字段同步边界。

### 优化与重构

- 将会话分组、section 投影和侧边栏操作收口到 current App Server 与 typed gateway，移除旧 pinned 双轨入口。
- 收敛 Thread v2 协议 envelope、schema、生成类型、通知和 canonical store 的实现，保持 Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI 主链一致。

### 测试与质量

- 新增 Thread sections 的 Rust JSON-RPC、thread store、typed gateway、GUI 和恢复行为回归测试。
- 扩展 App Server v2 protocol、Agent chat、sidebar conversation projection 与 current fixture 的 contract 和组件覆盖。
- 增加 macOS 正式签名非代码资源的 timestamp 关闭策略与时间戳服务不可用重试守卫，同时确保嵌套 Mach-O 保留安全时间戳并可通过 notarization。

### 文档

- 同步 v2 Thread projection、App Server 协议边界、重构执行计划和当前架构确认记录。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.122.0`。

**完整变更**: `v1.121.0` -> `v1.122.0`
