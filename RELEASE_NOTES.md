## Lime v1.126.0

### 新功能

- 新增 Code Mode 桌面主链：模型可通过原生 custom `exec` 与 `wait` 编排冻结的工具快照，并支持异步 cell、通知、共享 store、等待、终止与取消。
- 引入独立 `code-mode-host` 进程，在 sandbox-enabled V8 isolate 中执行 JavaScript；App Server 只持有 process client，不在生产路径回退进程内 V8。
- 扩展官方 OpenAI Responses custom tool lowering、模型 `tool_mode` 与 `custom_tools` capability/readiness 门禁。

### 修复

- 修复 provider tool call 的名称、参数类型与 JSON Schema 校验链，畸形调用在 handler 前 fail closed，并保持唯一工具生命周期终态。
- 修复 Code Mode nested tool、notify、cell close、取消与 timeout 的跨进程关联，避免迟到回调串入后续 sampling step。
- 修复 OpenAI Responses、Chat Completions、Anthropic、Gemini、Vertex、Azure 与 Ollama transport 的 canonical request/stream/usage 投影漂移。

### 优化与重构

- 将 Agent session loop 收敛为 thread-owned resources/registry，统一 actor replace、interrupt、shutdown 与 active cell 清理。
- 将 Code Mode V8 provider 限定为 host 内部 owner；dev、Electron assets 与 Windows 构建成组产出 `app-server` 和 `code-mode-host`。
- Electron release manifest 分别记录双 sidecar SHA-256，packaged verifier 强制检查二进制存在性与完整性。

### 测试与质量

- 增加多 provider loopback request capture，验证 endpoint、认证、canonical content、工具定义、generation lowering 与 terminal stream。
- 增加 Code Mode protocol/process、V8 runtime、provider lowering、session lifecycle、sidecar assets 与资源完整性回归。
- 真实 Electron Gate B 已证明 Electron、App Server 与 `code-mode-host` 为独立父子进程，并完成 custom exec 回采样、canonical Tool Item 与 GUI 可见终态。

### 文档

- 更新 Code Mode process owner、provider/tool trust boundary、双 sidecar 构建链、产品范围矩阵与执行计划。
- 明确 remote environment、Codex TUI 与无 Desktop consumer 的 surface 不进入 Lime current 产品链。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.126.0`。

**完整变更**: `v1.125.0` -> `v1.126.0`
