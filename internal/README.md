# internal

## 目录定位

`internal/` 是 Lime 仓库的内部事实源。它承接原本散落在文档目录里的工程规则、路线图、执行计划、研究资料、测试策略、PRD、私有运营材料和长期技术专题。

`../docs/` 现在只作为 Nuxt Content / Docus 文档站包使用，不再承载内部工程事实源。

## 核心入口

- `aiprompts/`：模块级工程导航、架构说明、质量流程、命令边界、治理规则和 Agent 协作规则
- `exec-plans/`：执行计划、进度日志和技术债追踪
- `refactor/`：渐进式重构方案（文件体量治理、目录架构蓝图）
- `roadmap/`：产品、架构、运行时、Plugin、Warp 对齐、i18n 等路线图
- `research/`：外部产品、竞品、协议和工程范式研究
- `prd/`：功能 PRD、工具 PRD 和方案草案
- `test/`、`tests/`、`testing/`：测试策略、场景、manifest、QC 与 E2E 资料
- `develop/`：开发流程、专项技术计划和协作规范
- `design/`：产品设计与交互方案
- `tech/`：跨模块技术蓝图与专题工程文档
- `knowledge/`：Knowledge 相关内部说明
- `bussniss/`、`oem/`、`gongzonghao/`：私有商务、品牌、运营和内容材料

## 当前迁移边界

- `lime-rs/src/**` 与 `lime-rs/src/commands/**` 已物理删除；旧路径只允许出现在负向回流守卫或不可变历史 evidence 中，不得恢复主 crate、启动注册、legacy facade、compat wrapper 或退场 stub。
- Rust 业务逻辑、领域服务、runtime、协议和数据访问的 current owner 位于 `lime-rs/crates/**`。
- 新增 Rust 后端能力必须进入 App Server crates / RuntimeCore / services；窗口、托盘、Dock、updater、shell、deep link 等桌面壳能力进入 Electron Desktop Host。
- 非生成代码超过 `1000` 行时，必须按领域、职责、数据边界或协议边界拆分；无法本轮拆分时，登记 blocker、风险和退出条件，不继续追加新业务逻辑。
- `src/lib/dev-bridge/**` 需要按职责治理，不是整体删除对象：`safeInvoke`、HTTP client、`app_server_handle_json_lines`、bridge availability / event listener capability 是 current renderer bridge；旧命令 policy / no-mock fallback 是迁移期 `compat / deprecated`；已迁旧命令名只能留作 `dead` / `test-only` guard。删不动且跨命令组长期存在的 legacy policy / mock residual 必须回挂 `exec-plans/tech-debt-tracker.md` 的 `CCD-012`，不能只留在聊天、handoff 或临时计划里。
- 相关规则入口：`aiprompts/commands.md`、`aiprompts/governance.md`、`aiprompts/quality-workflow.md`、`roadmap/appserver/README.md`、`exec-plans/production-command-current-migration-plan.md`、`exec-plans/tech-debt-tracker.md`。

## 阅读顺序

1. 先看根目录 `../AGENTS.md`，确认仓库级硬规则。
2. 再看 `aiprompts/README.md`，按场景进入模块级工程文档。
3. 涉及长期任务时，优先查看 `exec-plans/README.md` 和相关路线图。
4. 涉及旧路径迁移、compat / deprecated 收口时，先读 `aiprompts/governance.md`。

## 维护规则

1. 内部长期事实源默认落在本目录，不落在 `../docs/`。
2. 新增一级目录时，同步更新本文件和根 `AGENTS.md` 的导航。
3. 如果某条规则已经可以机械验证，优先补脚本或测试守卫。
4. 私有或暂不公开材料继续遵循 `.gitignore` 的 `internal/` 规则。
