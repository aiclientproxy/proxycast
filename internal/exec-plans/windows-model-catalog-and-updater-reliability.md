# Windows 模型目录与更新可靠性闭环

状态：实现完成；Windows Gate B 待实机验收
日期：2026-07-31

## 目标

1. Windows 用户添加并启用 Provider 模型后，聊天模型选择器必须展示该显式模型。
2. 更新入口只在确认候选版本严格高于当前版本后出现。
3. 更新失败、已是最新和发现新版本使用不同状态，不再把失败或同版本误报成新版本。

## 用户闭环

用户从设置中的 Provider 添加模型和凭证，回到聊天输入区打开模型选择器，能看到并选择刚添加的模型。应用后台检查更新时不打扰用户；只有已确认的新版本可安装时才显示更新入口，失败时提供准确状态和可恢复动作。

## 写集

- `src/components/input-kit/ModelSelector.tsx` 及定向测试
- `src/components/agent/chat/utils/providerModelCompatibility.ts` 及定向测试
- `electron/updateHost.ts` 及定向测试
- `src/components/app-sidebar/AppUpdateEntry.tsx` 及定向测试
- `src/pages/update-notification.tsx` 及定向测试
- `src/i18n/resources/*/common.json` 的更新状态文案
- 本执行记录

避让：App Server protocol、Skills runtime、`internal/aiprompts/architecture.md` 与已有脏执行计划。

## 根因证据

- 模型：Provider 显式模型会被前端临时投影为无权威能力来源的元数据，`ModelSelector` 又把该来源当成路由不可执行并直接过滤，形成“顶部保留旧选择、候选列表为空”。
- 更新：`update-available` 事件没有版本参数，旧逻辑以 `undefined !== currentVersion` 判断有更新；通知页又固定渲染“发现新版本”，未按 `failed / up_to_date / completed` 分支。

## 退出条件

- Provider 显式模型可见，未显式声明且仅靠启发式推断的目录项仍 fail closed。
- 同版本、旧版本、无版本和无效版本均不会产生可安装更新入口。
- 只有严格更高版本进入已下载/可安装状态；失败通知不显示“发现新版本”。
- 定向单测、`typecheck`、`test:contracts`、`verify:app-version` 和 `verify:gui-smoke` 完成或记录明确阻塞。

## 验证记录

- 模型定向回归：`providerModelCompatibility.test.ts` 与 `ModelSelector.test.tsx`，48/48 通过。
- 更新与模型联合定向回归：7 个测试文件，99/99 通过。
- Electron 版本与宿主定向回归：23/23 通过。
- `npm run typecheck`：通过。
- `npm run typecheck:electron`：通过。
- `npm run verify:app-version`：通过，当前版本 `1.117.0`。
- `npm run test:contracts`：通过。
- `npm run verify:gui-smoke`：通过；Electron、preload/IPC、App Server JSON-RPC 与用户可见工作台主路径 smoke 证据：`.lime/qc/project-gates/standalone-shell-01-20260731050343-95334/shell-01-electron-smoke/summary.json`。
- 2026-07-31 最终 GUI smoke：21/21 assertions 通过，console / invoke / preload / page error 均为 0，mock fallback 与 legacy command 命中为 0；证据：`.lime/qc/project-gates/standalone-shell-01-20260731103342-78558/shell-01-electron-smoke/summary.json`。
- updater / About / UpdateNotification / Sidebar 定向回归：51/51 通过；`npm run typecheck:electron` 与 `npm run verify:app-version` 通过，版本 `1.117.0`。
- `npm run smoke:agent-runtime-current-fixture`：通过；Provider 模型选择后的 current Agent 主链、历史、停止继续、typed error success/failure 与 read model terminal 未出现回归，`liveProviderUsed=false`。
- `npm run verify:local`：通过；前端 112/112 个智能测试批次、Rust workspace 测试和内含 GUI smoke 全部通过。
- 线上 Windows x64 `RELEASES` 清单：本机到 `updates.limecloud.com` 连续三次 TLS 握手失败，未取得服务端内容。
- 待实机验收：Windows Squirrel 从 N-1 版本发现更新、下载、重启安装并确认最终版本的 Gate B 闭环。
- 当前完成度：92%；实现、macOS Electron Gate B 与跨层回归已完成，剩余 8% 为 Windows packaged 平台验收，不能由 macOS fixture 代替。

## 状态分类

- current：Provider 配置 -> 模型目录 -> 模型选择器；Electron `autoUpdater` -> 更新会话 -> GUI。
- compat：无新增。
- deprecated：无新增。
- dead / deleted：删除 `undefined !== currentVersion` 伪版本判断，以及版本未知时的用户可见更新入口。

## 架构确认

未改变 Electron Desktop Host、preload/IPC 或 App Server 边界，不属于重大架构变更，无需更新架构图。
