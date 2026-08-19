## Lime v1.132.0

### 新功能

- 新增 Right Surface Browser Workspace，将用户可见页面与 Agent 操作统一到同一个 Electron `WebContentsView`、tab identity 和 BrowserRoute。
- 新增 Browser 动态工具与 Electron Desktop Host reverse request 闭环，支持页面观察、操作、下载、权限和人工接管状态投影。
- 新增 CodeCell trace/evidence 的 App Server 唯一 owner，支持 lifecycle JSONL 写入、reducer/replay 和 `diagnostics/trace/read` 读取。
- 新增桌面 Deepswe 受控 smoke、contract 与 benchmark harness，补充真实 Electron Gate B 证据入口。

### 修复

- 修复 Browser 打开时 canonical Thread 尚未创建导致 Right Surface 长时间停留在“正在打开网页”的问题。
- 修复 Browser route 的 thread/turn/session/tab/view/webContents identity 校验缺口，过期、跨窗口、重复和未知调用现在 fail closed。
- 修复 CodeCell source/output Item 晚到、yield 跨 Turn、nested invoke 及 terminal 后迟到事件的 trace 关联和关闭行为。
- 修复 provider/model selection、tool inventory、App Server request serialization 与 workspace skill projection 在 current 主链中的边界不一致。

### 优化与重构

- 将旧 Browser Runtime、Canvas Browser、外部 Chrome/CDP、BrowserSessionRef、site adapter 和旧 connector 路径迁出并删除，收敛到 Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI 唯一产品链。
- 重构 Right Surface、Workbench、Artifact、Service Skill 和五语种资源，减少重复状态机与旧 compat 入口。
- 更新 App Server protocol/schema、生成客户端、命令目录、治理清单、架构文档和浏览器路线图。

### 测试与质量

- 补充 BrowserTabHost、Browser Workspace、动态工具、CodeCell trace、协议合同和桌面 harness 定向测试。
- 发布前执行版本一致性、TypeScript 类型检查、协议合同、受影响 Rust/前端测试、GUI smoke、Agent current fixture、治理与本地门禁；实际结果以本次发布执行计划记录为准。

### 文档

- 更新 Browser Workspace 唯一 owner、同 Tab identity、turn cleanup、CodeCell trace owner 和 Gate A/Gate B 证据合同。
- 新增 v1.132.0 发布执行计划，记录 release candidate、验证结果和平台证据边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.132.0`。
- Windows、正式打包签名、公证、发布 asset 和 live provider 证据仍需对应平台或 CI runner，不在本地候选中冒充完成。

**完整变更**: `v1.131.0` -> `v1.132.0`
