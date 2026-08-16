## Lime v1.129.0

### 新功能

- 新增候选模型集与 OEM 路由策略，支持按任务族、模态、运行时特性和能力筛选候选，并输出候选数量、路由模式、候选集与策略事实。
- 将语音合成、语音转写和文本嵌入纳入统一 model-provider 能力与 lowering 链路，支持 OpenAI 兼容协议的多模态任务执行。
- 新增媒体任务 worker 的语音合成路由、调度与产物记录能力，统一凭证使用记录和任务状态更新。

### 修复

- 修复无候选、能力缺口和 provider readiness 失败时的路由状态投影，避免错误地保留可用候选或错误的 profile-slot 模式。
- 修复 App Server、RuntimeCore 与前端在模型路由、媒体任务、线程事实和 pending request 数据上的 camelCase/snake_case 边界一致性。
- 修复 Skills、Harness 和工作区能力面中对已移除证据导出入口的依赖，降低旧 API、旧 fixture 和重复状态来源造成的回归。

### 优化与重构

- 将候选模型、OEM policy、provider capability/readiness 与 routing decision 收敛到 RuntimeCore 和 model-provider 的 current owner，删除旧证据 provider/export projection 及其专用 UI/测试。
- 将 Harness 状态面从“问题证据包”切换为线程级 runtime facts，直接展示 turns、items、待处理请求、产物、证据引用和路由决策。
- 精简 App Server protocol schema、客户端类型、脚本 fixture 与治理 catalog，清理已脱离产品链的证据导出和重复兼容路径。

### 测试与质量

- 补充候选模型集、OEM 路由模式、provider 多模态 lowering、媒体任务 artifact 和 current runtime facts 的 Rust/前端回归覆盖。
- 更新 App Server client/protocol、modality、脚本、Harness、治理和五语言资源相关契约；发布门禁结果以本计划实际记录为准。

### 文档

- 更新架构、命令边界、模型路由、App Server/前端集成矩阵、Harness 治理和多模态运行合同文档。
- 新增 `v1.129.0` 发布执行计划并维护执行计划导航。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.129.0`。
- Windows/macOS packaged parity、签名、公证和正式发布资产证据不在本地候选中冒充完成，待 CI/平台流程提供证据。

**完整变更**: `v1.128.0` -> `v1.129.0`
