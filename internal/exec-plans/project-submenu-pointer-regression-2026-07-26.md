# 项目子菜单指针回归修复

状态：complete_with_windows_verification_pending

## 目标

修复首页与任务中心项目选择浮层中“新建空白项目 / 使用现有文件夹”无法通过真实鼠标点选的回归，并补稳定守卫，避免后续样式调整再次引入不可命中的空隙。

## 用户闭环

- 用户：从 Lime 首页或任务中心进入项目工作的普通用户。
- 起点与终点：打开项目菜单，展开“添加新项目”，点击新建空白项目或使用现有文件夹，并继续进入对应项目。
- 完成条件：指针从父菜单项移动到右侧子菜单时子菜单不消失；原有项目创建、目录选择与 Workspace 接线保持不变。

## 写集

- `src/components/agent/chat/components/Inputbar/components/InputbarProjectContextBar.tsx`
- `src/components/agent/chat/components/ChatNavbar.test.tsx`
- `internal/exec-plans/project-submenu-pointer-regression-2026-07-26.md`
- `.gitignore`（把本计划加入 versioned artifact 白名单）

只读：项目 Host picker、Workspace API、Popover 基础组件和既有项目 Gate 证据。避让当前工作树中的 Electron、Rust、App Server protocol/runtime 与其它执行计划脏改动。

## 根因

`InputbarProjectContextBar` 把子菜单定位到父项右边界外 `8px`，同时用父项容器的 `onMouseLeave` 立即关闭子菜单。真实 Chrome 指针逐像素移动时，父按钮右边界为 `x=620`、子菜单从 `x=628` 开始；指针到达 `x=620` 时子菜单节点即从 `1` 变为 `0`。现有组件测试直接调用按钮 `.click()`，没有覆盖真实指针跨越路径。

## 修复

- 保留视觉上的 `8px` 间距，把子菜单外层改为从父项右边界开始的透明命中桥，实体菜单放入带左内边距的内层面板。
- 在现有“使用现有文件夹”组件回归中锁定命中桥与实体面板结构，继续断言真实业务回调。
- 不修改项目创建、目录选择、Workspace、Electron IPC 或 App Server 协议。

## 验证计划

- 定向 Vitest：`src/components/agent/chat/components/ChatNavbar.test.tsx`
- 受影响文件 ESLint、Prettier、`git diff --check`
- `npm run typecheck`
- `npm run verify:gui-smoke`
- Gate A：Chrome 对 `http://127.0.0.1:1420/` 逐像素移动并点击两个子菜单入口，确认子菜单持续存在且按钮收到点击。

## 当前证据

- 修复前 Gate A 已稳定复现：右侧存在 `8px` 无命中区域，指针到达父项边界时子菜单卸载。
- 修复后 Gate A：Chrome 在 `deviceScaleFactor=1 / 1.25 / 1.5` 下，命中断层均为 `0px`、实体面板视觉间距均为 `8px`；逐像素移动期间子菜单持续存在，“新建空白项目 / 使用现有文件夹”均收到一次真实 click。
- 当前页面：`http://127.0.0.1:1420/`，browser mirror；独立 Renderer 控制台 error 均来自 3030 bridge 未启动后的 `ERR_CONNECTION_REFUSED` / DevBridge fail-closed，归类为环境噪音，不作为本交互根因。
- Electron CDP：现有 Lime 实例持有单实例锁，新启动的调试 Electron 正常退出；未终止用户实例。
- macOS Gate B-F：`npm run verify:gui-smoke` 通过，evidence 为 `.lime/qc/project-gates/standalone-shell-01-20260726141906-29202/shell-01-electron-smoke/summary.json`，`21/21` assertions、console/page/invoke/mock fallback error 均为 `0`。
- Windows 边界：用户反馈来自 Windows；本次修复位于平台无关 DOM/CSS 命中层，并覆盖 Windows 常见 `100% / 125% / 150%` 缩放对应的 device scale。当前机器为 macOS，未把该证据冒充真实 Windows Electron Gate B。

## 验证结果

- `npx vitest run "src/components/agent/chat/components/ChatNavbar.test.tsx" --silent=passed-only --disableConsoleIntercept`：通过，`1 file / 17 tests`。
- `npx eslint --max-warnings 0 ...`：通过。
- `npx prettier --check ...`：通过。
- `npm run typecheck`：通过。
- `npm run verify:gui-smoke`：通过，Gate B-F `21/21`。
- 窄写集 `git diff --check`：通过。

## 分类与完成度

- `current`：项目选择浮层、透明命中桥、项目创建 / 目录选择现有调用链。
- `compat / deprecated`：本轮未新增、未修改。
- `dead / deleted`：本轮无旧入口恢复。
- 实现与平台无关回归完成度：`100%`；真实 Windows Electron 现场复测完成度：`0%`，需由 Windows 环境确认。
- 本轮整体完成度：`95%`；剩余 `5%` 仅为 Windows 真机确认，不包含待实现代码。

## 退出条件

- 定向组件回归通过。
- 真实指针可以穿过视觉间距并点击两个入口。
- GUI smoke 通过；若真实 Electron 交互因既有实例限制无法完成，明确记录 Gate A / Gate B 证据边界。
