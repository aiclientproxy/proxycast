## Lime v1.139.0

### 新功能

- 新增仓库级 `FEATURE-MAP.md`，按用户能力、稳定入口、current owner 与协议边界导航 Agent、Workspace、Provider、MCP、Skills、Plugins、Memory、Artifact、Scheduled Tasks 和 Desktop Host。
- 扩展 macOS native host 与 Windows packaged evidence，覆盖窗口编排、权限、资源身份、Squirrel 安装态和真实 Electron/App Server Gate B 证据。
- 补齐 model-provider reasoning effort 与 provider route 合同，让模型能力、默认选择、切换、readiness 和推理档位继续收敛到统一控制面。

### 修复

- 修复资源管理器生产路径动态加载测试窗口夹具的问题，关闭行为直接使用当前窗口语义。
- 移除服务端对 `tauri://localhost` 与 `tauri.localhost` 的 CORS 放行，只保留当前本地开发入口。
- 修复 Curated Task 结果参考、Memory continuation、模型配置和 provider route 中的事实源漂移。

### 优化与重构

- SceneApp 与 `src-tauri` 全面退役：应用目录统一归 Plugin，业务调用统一进入 Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore / Agent Runtime，结果参考统一归 Curated Task 与 Memory。
- 删除 Tauri capability/schema、SceneApp 专属组件、Hook、网关、文案和正向测试；旧名称只保留在负向回流守卫与历史 evidence。
- 收敛 Rust current crate 的宿主边界表述，移除旧进程内桌面命令、Tauri 状态包装和历史主 crate 叙事。

### 文档

- 以 Lime 自身产品和架构重写 Feature Map，不复制参考项目的能力结构。
- 清理被 current owner 替代的历史发布计划、Plugin v1/v2、旧 refactor、iteration notes 和阶段性研究材料，并同步更新导航与架构文档。
- 本次共删除 98 份 Markdown、1 份 HTML、4 份 JSON 和 16 份旧前端源码/测试；无消费者的旧 A2UI/Tauri 设计一并清退，仍有 current 价值的文档统一改写到 Electron Desktop Host 与 App Server 边界。Git history、Release Notes 与不可变 evidence 承接历史追溯。

### 测试与质量

- 扩展 App Server contracts、Desktop Host、macOS native host、Windows packaged evidence、model-provider、Curated Task、Memory 和治理回归。
- 发布门禁覆盖版本一致性、TypeScript、i18n、协议 contracts、Rust 相关测试、文档边界、治理扫描、真实 Electron GUI smoke 与完整前端续跑。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.139.0`。
- 不新增 compat 或 deprecated owner；SceneApp、`src-tauri`、Tauri capability 与测试窗口夹具均为 `dead / deleted / forbidden-to-restore`。

**完整变更**: `v1.138.0` -> `v1.139.0`
