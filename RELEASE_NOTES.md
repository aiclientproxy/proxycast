## Lime v1.138.0

### 新功能

- 新增统一 Composer Controller，首页与会话输入栏共享结构化草稿、提交意图、历史召回以及 start、queue、steer、interrupt 路由语义。
- 新增 App Server `promptHistory/read|append` 持久化历史，提供跨进程 JSONL 分页、文件锁、坏行容错和跨平台日志身份。
- 新增 Desktop capability/readiness 合同、macOS 原生 helper 与 security-scoped bookmark 生命周期，覆盖权限、Launch Services、显示器、HID、设备密钥和打包资源状态。
- 新增跨平台 desktop resource manifest，将 App Server、Code Mode、Windows sandbox helpers 和 macOS native host 纳入统一身份、架构及 SHA-256 校验。

### 修复

- 修复首页预热阶段过早写入正式 local override，导致路由 owner 清空 active draft、Banner/皮肤 Hero 消失且发送无法复用预热 session 的问题；预热现在只准备 session，正式提交时才接管路由。
- 修复 Agnes 官方 `agnes-2.5-pro` endpoint 被识别为不可执行自定义模型的问题，未知 endpoint 继续 fail closed。
- 修复 Composer 压缩 bundle 的初始化时序、异步历史合并、失败草稿保留和大粘贴展开边界。
- 修复线程排队操作、模型恢复、reasoning 状态投影以及窄窗口 Header/Timeline 布局中的漂移和可用性问题。

### 优化与重构

- 将 Composer、prompt history、Desktop capability 和资源完整性分别收敛到唯一 current owner，不新增 Electron 业务后端或 Renderer 持久化事实源。
- 对齐 Codex current reasoning policy，补齐 `persistent` effort、五语言选择器文案以及模型上下文和工具策略的事实源守卫。
- 强化 Windows Squirrel 发布链，安装后执行 Code Mode Gate B 并验证 `Lime.exe -> app-server.exe -> code-mode-host.exe` 进程链。
- 更新 DeepSWE adapter 契约与治理断言，避免测试清单和 current owner 版本漂移。

### 测试与质量

- 扩展 App Server protocol/schema/generated client、Composer、prompt history、Desktop Host、macOS native host、资源 manifest、Windows Squirrel 与 Code Mode Gate B 回归。
- 发布门禁覆盖版本一致性、TypeScript、协议 contracts、受影响前端/Rust 测试、真实 Electron GUI smoke 和 release workflow guard。
- Windows packaged Gate B 由 `windows-2022` release workflow 生成平台证据；本地静态测试不冒充 Windows platform pass。

### 文档

- 更新全局架构、Codex GUI 对齐计划和 Desktop 平台对齐计划，记录 current owner、能力矩阵、证据等级与剩余平台缺口。
- 新增 v1.138.0 发布执行计划，明确候选范围、排除的 Codex 构建参考文件、验证和发布收口步骤。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.138.0`。
- Codex Desktop 实时视觉基线仍受 macOS AppleEvent 权限阻塞；首页 Banner/皮肤 Hero 按产品裁决保留，不纳入对齐。

**完整变更**: `v1.137.0` -> `v1.138.0`
