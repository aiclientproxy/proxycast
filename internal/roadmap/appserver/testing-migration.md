# App Server / Electron 测试口径

> 状态：current testing source
> 更新时间：2026-09-02

## 1. 唯一受测主链

```text
Renderer typed gateway
  -> Electron Desktop Host / preload
  -> app_server_handle_json_lines
  -> App Server JSON-RPC
  -> RuntimeCore / domain owner
  -> Thread / Turn / Item projection
  -> GUI
```

Electron 只承担桌面宿主能力和 JSONL 转发，不承接第二套业务后端。生产路径不得回退 renderer mock、Desktop mock 或 legacy command facade。

## 2. 测试分层

| 风险 | 最低证据 |
| --- | --- |
| 纯 selector / parser / projection | 相关 unit test |
| React 组件或 hook | 相关 component test；用户可见文案补五语言断言 |
| App Server method / schema / typed client | `npm run test:contracts` + public JSON-RPC integration |
| Agent loop / Thread / Turn / Item | Rust related/integration + current runtime fixture |
| Electron / preload / IPC | contracts + Gate B Electron fixture |
| Workspace / GUI 主路径 | related tests + `npm run verify:gui-smoke` |
| dead surface 删除 | 负向结构测试 + `npm run governance:legacy-report` |

`src/lib/desktop-host/**` 只允许作为显式 test fixture。它不能证明 production bridge 已命中，也不能替代 Gate B。

`lime-rs/src/commands/**` 已物理删除，只作为负向回流守卫中的旧 Tauri command wrapper 路径；不得恢复 stub、compat wrapper 或业务实现。新增后端能力必须进入 App Server / RuntimeCore current owner，桌面壳能力进入 Electron Desktop Host。

## 3. Gate A 与 Gate B

- Gate A 证明 Renderer 投影、DOM、交互和用户可见状态。
- Gate B 必须证明真实 Electron、preload/IPC、`app_server_handle_json_lines`、App Server、runtime/read model 与 GUI 使用同一 identity。
- `npm run test:e2e` 只是 Vitest 分层入口，不自动等于 Gate B。

## 4. 变更同步

修改跨层命令时，同一变更集必须同步：

1. `app-server-protocol` method、params、result、notification 与 schema。
2. App Server handler 和 current domain owner。
3. `packages/app-server-client` 或 Renderer typed gateway。
4. Electron preload / IPC 白名单，仅当需要宿主能力或 JSONL 转发时。
5. catalog、fixture、mock policy 与负向回流守卫。
6. contracts、相关 Rust/TypeScript 测试和风险匹配的 GUI 证据。

## 5. 验证入口

```bash
npm run typecheck
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run governance:legacy-report
```

默认先运行最贴近改动边界的定向测试，再按风险扩大。完整规则以 `internal/aiprompts/quality-workflow.md` 为准。

## 6. 退出条件

- current 测试不依赖已删除 host、旧命令、旧 runtime 类型或生产 mock fallback。
- command / schema 变更已从 public JSON-RPC 入口验证。
- GUI 主路径已说明 Gate A、Gate B 的实际覆盖范围。
- 已删除 surface 有目录或符号级负向守卫，且 active 文档不再导航到旧实现。
