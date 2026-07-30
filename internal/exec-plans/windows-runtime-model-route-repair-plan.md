# Windows 模型路由与 Skill 安装目录修复

状态：platform-evidence-pending
日期：2026-07-30
负责人：root
预算标签：`budget:tight`
风险等级：P1
完成度：95%

## 主目标

修复 Windows 升级或新装用户选择 `lime-hub/gpt-5.2-pro` 后，Agent turn 在 RuntimeCore admission
阶段返回 `runtime model route is not executable` 的问题，并修复 Provider 实时目录把 `inferred_hint`
模型展示为可切换项导致的同类失败。保持模型能力 provenance fail-closed，不改写用户数据库，不让未知 OEM
模型或普通自定义 Provider 获得推断执行权限。同时修复 Windows 用户从 Skill 管理入口无法打开真实安装目录的
问题，目录定位只消费 App Server 返回的权威本地路径，不在 Renderer 拼接用户目录。

## 根因与当前阶段

- v1.114.0 起，只有 `canonical`、`provider_explicit` capability 可以授权 route。
- Windows Lime Hub 同步和历史模型控制面只持久化 `{ id: "gpt-5.2-pro" }`。
- canonical mapper 未把产品自有的 `lime-hub` 识别为 hosting provider，因此已存在于 bundled registry 的
  `openai/gpt-5.2-pro` 仍被降为 `inferred_hint`，最终触发 `capability_snapshot_missing`。
- `skillManagement/list` 的本地扫描结果没有投影真实安装路径，Renderer 因而无法从 Skill directory id
  可靠解析 Windows 用户目录；同时 `reveal_in_finder` 对目录仍使用文件定位语义，Windows Explorer 无法稳定
  选中或打开该目录。
- 当前阶段：canonical owner、provenance 协议、Renderer executable filter 与 Provider cache taxonomy v3
  已实现并通过本地门禁。macOS 真实 Electron Gate B 已证明 Agnes live directory、真实 UI 模型往返切换、
  clean turn 完成态、权威 read model，以及 Skill 安装目录打开链；等待 Windows packaged build 实机证据闭环。
- 本轮完成：补 Agnes 2.5 官方 canonical 记录；在既有 JSON-RPC response 透传 provenance；实时目录同 ID
  优先使用 API 权威元数据；ModelSelector 仅保留 `canonical / provider_explicit` 可执行项；提升缓存
  taxonomy，自动淘汰升级前持久化的 `inferred_hint` 快照；本地 Skill 增加 `localDirectoryPath` 投影；
  Electron 文件壳对目录使用 `shell.openPath`，普通文件继续使用 `showItemInFolder`。
- 下一刀：在 Windows packaged build 中复现历史 id-only 选择并点击 Skill“打开目录”，验证首回合 route
  admission、Explorer 目录跳转、真实 Electron/preload/IPC/App Server 链与用户可见完成态。

## 写集

- `lime-rs/crates/model-provider/src/canonical/name_builder.rs`
- `lime-rs/crates/model-provider/src/canonical/data/canonical_models.json`
- `lime-rs/crates/services/src/model_registry_service.rs`（仅扩展既有 Agnes conversion 测试）
- `lime-rs/crates/services/src/model_registry_service/runtime_metadata.rs`
- `lime-rs/crates/app-server-protocol/src/protocol/v0/model.rs`
- `lime-rs/crates/app-server/src/local_data_source/model_projection.rs`
- `lime-rs/crates/app-server-protocol/schema/json/**`（生成）
- `packages/app-server-client/src/generated/protocol-types.ts`（生成）
- `src/lib/api/modelRegistry.ts`
- `src/lib/api/modelRegistry.test.ts`
- `src/hooks/useProviderModels.ts`
- `src/hooks/useProviderModels.test.ts`
- `src/components/agent/chat/utils/providerModelCompatibility.ts`
- `src/components/agent/chat/utils/providerModelCompatibility.test.ts`
- `src/components/input-kit/ModelSelector.tsx`
- `src/components/input-kit/ModelSelector.testFixtures.ts`
- `src/components/input-kit/ModelSelector.test.tsx`
- `lime-rs/crates/core/src/models/skill_model.rs`
- `lime-rs/crates/services/src/skill_service.rs`
- `electron/fileShellHost.ts`
- `electron/fileShellHost.test.ts`
- `internal/aiprompts/architecture.md`
- `internal/exec-plans/README.md`
- `internal/exec-plans/windows-runtime-model-route-repair-plan.md`
- `.gitignore`（仅增加本计划的精确跟踪例外）

只读：App Server route resolver、RuntimeCore admission、Electron/App Server bridge。

避让：当前已脏的 `processor/thread.rs`、`processor/turn.rs`、
`tests/model_selection_refresh_jsonrpc.rs` 与其它并行写集。`ModelSelector` 仅在保留既有 `data-testid`
并行改动的前提下做窄改。

`model_registry_service.rs` 已超过 5,700 行，本轮不追加业务逻辑，只把既有 Agnes endpoint conversion
测试扩展为表驱动 2.0/2.5 覆盖。新增 runtime metadata 回归仍放在已拆出的 `runtime_metadata.rs`。
后续再修改该大文件业务逻辑前，退出条件是先按 registry conversion/runtime metadata 职责继续拆分。

`skill_service.rs` 当前为 1,876 行。本轮只在既有本地扫描投影中补权威路径和一条回归，不新增并行扫描器、
路径 resolver 或 Renderer fallback；后续再扩展该文件业务逻辑前，退出条件是先按 local/remote projection
职责拆分。

## Current 主链

```text
Renderer typed gateway / ModelSelector
  -> Electron Desktop Host / app_server_handle_json_lines
  -> App Server JSON-RPC modelProvider/fetchModels + model/list + thread/settings/update
  -> services ModelRegistryService + cache taxonomy
  -> model-provider bundled canonical registry / current wire
  -> RuntimeCore authoritative capability admission
  -> Thread / Turn / Item projection
  -> GUI
```

Electron 只转发 `app_server_handle_json_lines`。本轮不新增 method 或 IPC，但扩展既有
`modelProvider/fetchModels` response schema 与 Renderer typed gateway。

Skill 目录链保持单一 owner：

```text
Renderer skillsApi.revealLocalSkill
  -> app_server_handle_json_lines / skillManagement/list
  -> services SkillService 本地扫描与 localDirectoryPath 投影
  -> Electron reveal_in_finder
  -> shell.openPath(directory) / showItemInFolder(file)
```

## Happy Path

- 输入：Windows 已有数据库包含 `lime-hub` provider 和 id-only `gpt-5.2-pro`。
- 预期：mapper 精确解析为 `openai/gpt-5.2-pro`，metadata provenance 为 `canonical`。
- 预期：snapshot 明确包含 chat、text input/output 和 streaming，turn admission 不再返回
  `capability_snapshot_missing`。
- 预期：Agnes 官方 2.5 Flash / Pro Alpha 进入 ready catalog 并可在已存在会话中切换。
- 预期：Agnes video/image 及其它 id-only 未知模型不会显示为可切换 chat route。
- 预期：本地 Skill 列表返回真实 `localDirectoryPath`，Windows 点击“打开目录”直接打开安装目录；普通文件的
  “在文件夹中显示”语义保持不变，目录打开失败会显式返回错误。
- 失败边界：不在 bundled registry 的 Lime Hub model id 仍停在 RuntimeCore admission，不发 provider 网络请求。
- 不做范围：不放宽普通自定义 Provider、Ollama 或未知 OEM 模型；不增加 mock fallback；不迁移用户数据；
  不在 Renderer 推测 `%APPDATA%`、`USERPROFILE` 或其它平台路径。

## Evidence Layers

| Layer                      | 是否需要 | 证据                                                       |
| -------------------------- | -------- | ---------------------------------------------------------- |
| Unit/domain integration    | 是       | canonical mapper + services runtime metadata 定向测试      |
| App Server contract        | 是       | `npm run test:contracts`，证明现有 method/schema 未漂移    |
| Renderer interaction       | 是       | ModelSelector 点击回归，证明可执行模型提交、非权威模型隐藏 |
| Current fixture            | 是       | `npm run smoke:agent-runtime-current-fixture`              |
| macOS live Electron Gate B | 已完成   | 真实模型往返切换、clean turn/read model 与 Skill 目录打开   |
| Windows packaged Gate B    | 待补     | 当前 macOS 工作区不能替代 Windows 安装包实机证据           |
| Live provider              | 已完成   | 用户要求的 Agnes 2.5 clean turn 完成并返回 `OK`             |

## 必跑命令

```bash
npm run test:rust:related -- \
  lime-rs/crates/app-server-protocol/src/protocol/v0/model.rs \
  lime-rs/crates/app-server/src/local_data_source/model_projection.rs \
  lime-rs/crates/model-provider/src/canonical/data/canonical_models.json \
  lime-rs/crates/model-provider/src/canonical/name_builder.rs \
  lime-rs/crates/services/src/model_registry_service.rs \
  lime-rs/crates/services/src/model_registry_service/runtime_metadata.rs
npm run test:related -- \
  src/lib/api/modelRegistry.ts \
  src/hooks/useProviderModels.ts \
  src/components/agent/chat/utils/providerModelCompatibility.ts \
  src/components/input-kit/ModelSelector.tsx
npx vitest run electron/fileShellHost.test.ts electron/hostCommands.test.ts
npx vitest run src/lib/api/skills.test.ts src/lib/api/fileSystem.test.ts
npm run typecheck:electron
npm run test:rust:related -- \
  lime-rs/crates/core/src/models/skill_model.rs \
  lime-rs/crates/services/src/skill_service.rs
npm run test:contracts
npm run verify:gui-smoke
npm run smoke:agent-runtime-current-fixture
cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check
git diff --check
```

## Agent QC 场景映射

- P0：无；不改变 tool、approval、sandbox、stream terminal 或 read model。
- P1：Agent chat 首回合 route admission。
- P2：`model/list` 目录可执行筛选与 Skill 安装目录宿主打开。
- 允许 current fixture；用户已明确要求本轮执行 live Provider Gate B；不冒充 official Windows release evidence。

## 已完成验证

| 命令                                                                                            | 结果      | 证据范围                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cargo test ... -p model-provider canonical::name_builder::tests::test_map_to_canonical_model`  | 通过，1/1 | 已知 Lime Hub 模型映射到 bundled canonical record；未知模型返回 `None`                                                                                     |
| `cargo test ... -p lime-services lime_hub_declared_canonical_model_keeps_executable_capability` | 通过，1/1 | id-only 模型获得 `canonical` provenance 与 chat/text/streaming snapshot                                                                                    |
| `npm run test:rust:related -- <Skill 两个 Rust 改动路径>`                                      | 通过      | 17 个相关及反向依赖 crate 全绿；Agent Runtime 192、App Server 1,633、Tool Runtime 313 个单测通过                                                           |
| `npm run test:contracts`                                                                        | 通过      | App Server client 286 checks 及 command/harness/modality/scripts/release/docs boundary 全绿                                                                |
| `npx vitest run <4 个模型选择相关测试文件>`                                                     | 通过      | 4 files / 75 tests；2.5 可选、未知 video 不展示、实时 provenance 覆盖本地 hint                                                                            |
| `cargo test ... -p model-provider -p lime-services -p app-server-protocol --lib`                | 通过      | protocol 92、services 220（4 ignored）、model-provider 234；cache taxonomy v2 快照失效与 Agnes 2.5 canonical conversion 通过                             |
| `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`                               | 通过      | Rust 格式无漂移                                                                                                                                            |
| `npx vitest run electron/fileShellHost.test.ts electron/hostCommands.test.ts`                   | 通过      | 2 files / 70 tests；目录走 `openPath`、文件走 `showItemInFolder`、打开失败显式返回                                                                         |
| `npx vitest run src/lib/api/skills.test.ts src/lib/api/fileSystem.test.ts`                     | 通过      | 2 files / 27 tests；Renderer 消费 `localDirectoryPath` 并调用 current `reveal_in_finder`，diagnostic facade 保持 fail-closed                                |
| `npm run typecheck:electron`                                                                    | 通过      | Electron Host/preload 类型检查无漂移                                                                                                                       |
| `npm run verify:gui-smoke`                                                                      | 通过      | 真实 Electron shell、preload/IPC 与 App Server sidecar 初始化通过                                                                                          |
| `npm run smoke:agent-runtime-current-fixture`                                                   | 通过      | macOS controlled Gate B fixture；Electron/preload/IPC/App Server JSON-RPC 命中，legacy/mock fallback/invoke/console error 均为零，`liveProviderUsed=false` |
| macOS live `modelProvider/fetchModels -> model/list`                                            | 通过      | taxonomy v3 首次读取 `fromCache=false`；2.5 Flash/Pro 为 `canonical` 并进入 ready catalog；image/video 保持 `inferred_hint` 且不进入 ready catalog        |
| macOS live 模型切换 Gate B                                                                       | 通过      | thread `019fb385-344b-7bf0-907e-77916ccf4c7c` 经真实 UI 产生两次 `thread/settings/update`；2.0 与 2.5 均为 `electron-ipc/success`，`thread/resume` 权威读回一致 |
| macOS live clean turn + read model                                                              | 通过      | Agnes 2.5 turn `turn_da99c6fc914b49859c01ea87d83d897e` 为 `completed`，`thread/read` 输出 `OK`；GUI 打开最近会话可见 `OK` 与 2.5，输入框可用且无新增 error       |
| macOS live Skill 安装目录 Gate B                                                                | 通过      | 真实 `skillManagement/list` 返回 75 个本地 Skill 且 75 个有 `localDirectoryPath`；`reveal_in_finder` 经 preload/IPC 成功打开目录                            |
| Browser Gate A 目录投影                                                                         | 部分通过  | 临时 Chrome 经 `electron-host` DevBridge 可见两款 2.5 且不见 video；点击提交被浏览器模式 `get_config` bridge gap 阻断，不冒充 Electron preload Gate B     |
| macOS Computer Use                                                                              | 阻塞      | Accessibility 返回 `-1743`，应用列表随后 `-1712` 超时；无法取得真实窗口截图                                                                                |
| `git diff --check`                                                                              | 通过      | 本轮 scoped patch 无空白错误                                                                                                                               |

macOS live 证据证明当前 `electron-host -> app_server_handle_json_lines -> App Server JSON-RPC ->
services/model-provider -> RuntimeCore -> read model -> GUI` 模型链，以及 `skillManagement/list ->
localDirectoryPath -> Electron file shell` 目录链均可执行。取证前错误缓冲中只有两条 App Server 重启期间的
`app-server host is stopping` 历史记录；本轮模型往返、会话恢复与目录打开均无新增 error。
Windows packaged Gate B 仍是唯一未完成交付门槛，不能由 macOS Electron 或单元测试替代。

## 架构确认与分类

- `current`：provenance 透传、Agnes 2.5 canonical records、Provider cache taxonomy v3、selector executable filter、
  `lime-hub -> bundled canonical registry -> RuntimeCore admission`、Skill 本地路径投影与 Electron 文件壳。
- `compat`：无。
- `deprecated`：无。
- `dead / forbidden-to-restore`：provider 名称直接授权、未知模型推断执行、Renderer capability fabrication、
  Renderer 拼接 Skill 安装目录、生产 mock fallback。
- 架构影响：重大但不改变 owner 或依赖方向；已核对 `architecture.md` 第 19 节信任边界。
- 责任开发者确认：`root, 2026-07-30`。

## 完成条件

1. 已有 id-only `lime-hub/gpt-5.2-pro` 无需数据库迁移即可得到 canonical capability。
2. 未知 Lime Hub model id 仍不能映射或执行。
3. 本地 Skill 返回权威安装路径，目录与文件分别使用正确的 Electron shell 语义。
4. Rust related、Electron tests/typecheck、contracts、current fixture、GUI smoke 与格式检查通过。
5. 明确记录 Windows packaged Gate B 未执行时的剩余风险，不把 macOS fixture 冒充平台证据。

当前完成度为 95%：本地实现、协议同步、macOS live directory/ready catalog、真实 UI 模型往返、
Agnes 2.5 clean turn/read model 与 Skill 目录打开均已完成；Windows packaged Gate B 的模型首回合与
Explorer 安装目录跳转仍待补，不能由 macOS 证据替代。
