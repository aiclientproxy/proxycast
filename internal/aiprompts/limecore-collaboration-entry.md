# LimeCore 联动入口说明

当任务同时涉及 `lime` 客户端与服务端能力时，不要只在本仓库里猜测。

跨仓库协作主文档放在：

- 绝对路径：`/Users/coso/Documents/dev/ai/limecloud/limecore/docs/aiprompts/lime-limecore-collaboration.md`
- 在 `limecore` 仓库内位于根目录 `docs/` 下的 `aiprompts/lime-limecore-collaboration.md`

已确认的服务端真实落点：

- `/Users/coso/Documents/dev/ai/limecloud/limecore/services/control-plane-svc`
- `/Users/coso/Documents/dev/ai/limecloud/limecore/services/gateway-svc`
- `/Users/coso/Documents/dev/ai/limecloud/limecore/services/scene-orchestrator-svc`
- `/Users/coso/Documents/dev/ai/limecloud/limecore/services/worker-svc`

固定纠偏：

- 不要再把控制面写成 “`lime` 仓库内待新建本地模块”
- `lime` 仓库默认是消费方，不是控制面唯一背景事实源

## 什么时候先读主文档

遇到下面这些任务时，默认先读 `limecore` 主文档：

- OEM 登录、Google 登录、desktop auth session
- 用户中心、个人资料、会话同步
- AI 服务商页、云端 Provider、默认来源、模型目录
- `client/bootstrap`、`client/session`、`client/profile`
- `client/skills`、`skillCatalog.entries`、`client/service-skills`
- Gateway、命令目录、Service Skill 配置同步
- Codex Desktop 对齐的 App Server WebSocket transport、租户 runtime 路由和 Cloud readiness
- 任何“客户端要不要本地维护一份服务端数据”的判断

## 为什么主文档放在 `limecore`

因为跨仓库联动里的正式事实源更多在服务端：

- 认证与会话
- 客户端 bootstrap
- `client/skills` 统一命令目录
- 用户资料与账户能力
- Provider Offer / 服务目录
- Gateway、App Server transport 与目录/配置策略

客户端仓库更适合作为实现消费方，而不是这些能力的唯一背景事实源。

## 在 `lime` 仓库里继续优先看这些

读完主文档后，如果确认主要是客户端实现，再回到本仓库重点看：

- `src/hooks/useOemCloudAccess.ts`
- `src/lib/api/oemCloudControlPlane.ts`
- `src/lib/api/oemCloudRuntime.ts`
- `src/components/settings-v2/`
- `src/components/agent/`
- `lime-rs/`
- `lime-rs/crates/app-server-client/`

## 默认工作原则

- 服务端已有接口时，优先补客户端接线
- 云事实源不要在客户端长期维护第二份
- `@` / 产品型 `/` 的统一目录优先看 `client/skills.entries`
- Lime 客户端必须保留 seeded / fallback 韧性兜底，不能只靠服务端在线返回
- `limecore` 提供目录、配置和认证后的 App Server transport edge，不承担 `@` / `/scene` / service skill 的执行，也不拥有 Agent runtime
- 能走运行时配置和 `bootstrap.features` 的，不要写死在前端
- 用户界面不要直接暴露 “OEM” 技术概念
- 旧 `client/scenes`、`bootstrap.sceneCatalog` 和 `/scene-api` 已退役；统一命令目录只看 `client/skills.entries` 与插件 Marketplace，不要恢复旧 Scene Runtime 或把目录命中误写成服务端 run
- App Server 会话只走 Lime `app-server-client` -> LimeCore `gateway-svc /v1/app-server` -> control-plane 解析的 tenant-owned Rust App Server；Gateway 不拥有 Thread/Turn/Item 或 Agent loop
