## Lime v1.136.0

### 新功能

- 新增 Windows 受限执行链：sandbox account、restricted token、ACL lease、Job Object、ConPTY、stdin/输出边界及 Firewall/WFP 网络隔离。
- 新增 Windows sandbox setup/runner sidecar、资源 digest 校验、readiness read-back 和七项真实执行证据矩阵。
- Scheduled Tasks 创建与运行现在冻结 Composer 当前选择的 Provider/模型 opaque route，服务技能和历史任务沿用同一条路由。

### 修复

- 修复 Agent Runtime 第二回合 start/accept 窗口被历史 completed read model 提前收口的问题，确保真实 `turnId` 和终态事件完整到达 GUI。
- 修复 Agent reasoning 与 Markdown 块级投影：仅展示 summary，恢复标题、表格、列表、代码围栏和行内语义。
- 修复 `model/list/updated` 通知被误报为协议漂移、Provider URL tenant/path/query 处理以及 transport 错误脱敏。
- 修复 Windows restricted child 启动、句柄继承、账户权限、桌面/管道生命周期、输出 drain 和 firewall read-back 边界。

### 优化与重构

- 将 Windows 执行面收敛到 `tool-runtime` 单一 current owner，Electron 只负责 sidecar 资源打包与 manifest 校验。
- 重构 Scheduled Tasks、Provider selection、Agent session projection、Markdown normalization 和 App Server stdio transport，保持真实 Electron/App Server JSON-RPC 单一路径。
- 扩展 DeepSWE provider evidence、candidate continuation、failure artifact 和隔离 harness，统一可续跑诊断数据。

### 测试与质量

- 增加 Windows restricted execution 的 workspace ACL、网络阻断、输出上限、stdin handle、ConPTY、world-writable audit 和进程树终止测试。
- 扩展 Agent Runtime、Scheduled Tasks、Provider transport、Markdown/streaming、通知 catalog、Electron resource verifier 与五语言 GUI 回归。
- 通过 TypeScript、协议 contracts、Agent current fixture、真实 Electron GUI smoke、Rust related 测试和格式检查。

### 文档

- 更新架构事实源与 Codex 对齐计划，记录 Windows sandbox 数据流、readiness fail-closed 边界、Provider route 规则和 current/compat/deprecated/dead 分类。
- 新增 reasoning projection 与通知漂移修复执行记录，并保留本次发版候选范围和验证证据。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.136.0`。
- Windows 正式签名、公证、跨平台 installer/release assets 和 live provider/verifier 证据仍需对应平台或 CI runner，本地不作虚假声明。

**完整变更**: `v1.135.0` -> `v1.136.0`
