# Agent Skills Runtime 当前状态

> 更新时间：2026-06-23  
> 口径：先看本文件判断是否继续做 runtime 基础设施；需要细节时再读 `README.md` 和 `execution-skeleton.md`。

## 结论

P0-P5 runtime 骨架已完成，不要再从零重做 Skills Runtime。

当前主链：

```text
Skill roots
  -> AgentSkillSnapshot
  -> skill_search / selector
  -> SkillBodyReader
  -> LimeSkillTool gate
  -> Agent turn execution
  -> timeline / evidence pack
```

专家 Skills 已进入同一主链：

```text
expert.skillRefs
  -> expert_binding selector trigger
  -> SKILL.md body read
  -> turn-scoped LimeSkillTool allowlist
  -> skill_body_read / skill_gate_decision evidence
```

## 已完成

1. 轻量 metadata 发现、排序和 `skill_search`。
2. 显式 `$skill` / `/skill` / `@skill` / `skill://.../SKILL.md` 选择。
3. catalog / scene launch 绑定进入 selector。
4. 隐式高置信匹配、`SKILL.md` 按需读取和 references progressive disclosure。
5. `LimeSkillTool` turn-scoped gate、`allowed_tools` 裁剪和 fail-closed 默认值。
6. telemetry / evidence 记录 `skill_search`、`skill_body_read`、`skill_gate_decision`、真实 invocation。
7. 专家 `skillRefs` 通过 `expert_binding` 跑通 selector / body / gate / evidence。
8. ToolSearch / skill_search 0 命中防循环：0 命中必须终态返回，不触发重复自动搜索。
9. 治理守卫已锁住专家 `skillRefs` 主链，防止回退为只提示、不进 gate/evidence。
10. `needs_registration` 专家恢复动作已能带目标进入 Skills 用户安装页，预填搜索词，并生成项目级 scaffold 草稿自动打开创建表单。
11. 项目级 `skillLocal/scaffold/create` 会同步写 `.lime/registration.json`，生成后的 Skill 可被 `workspaceSkillBindings/list` 投影为 `ready_for_manual_enable`；Skills 工作台创建成功后会刷新已保存技能和 binding readiness；仍不会自动注入 Query Loop 或运行时工具面。
12. `needs_mapping / blocked` 专家恢复动作已具备 UI 骨架：`needs_mapping` 打开“补齐技能目录映射”选择器，`blocked` 打开“替换当前技能引用”选择器，两者都能用已安装 / 目录可识别技能替换问题引用并进入下一轮运行准备。
13. 专家 Skills deterministic 产品闭环已进入 `smoke:agent-runtime-current-fixture` 聚合回归：Expert Plaza 点击、ExpertInfoPanel 修改 skillRefs、下一轮继承 override、`skill_search -> SKILL.md body read -> Skill gate -> Skill invocation`、`runtime.status` side-channel 与 canonical Thread/Turn/Item read-model 复盘均有 Electron fixture 证据；旧 Harness Evidence Pack 导出不再是 current 验收项。
14. `smoke:expert-skills-live-gate` 已作为只读证据门禁落地：默认审计 deterministic summary，缺少显式 live Provider summary 时返回 `pending_live_provider`，不调用真实模型，也不把 fixture 证据误报为 live 验收完成。
15. `smoke:expert-skills-live-runner` 已作为 live Provider 验收骨架落地：默认 fail-fast；显式授权后可归一化已有 live summary，或在 `--execute-live-runtime` 下通过 App Server current JSON-RPC 提交真实 Provider turn，并输出可被 live gate 消费的 summary。

## 当前分类

| Surface | 分类 | 当前口径 |
| --- | --- | --- |
| `AgentSkillSnapshot` / selector / body reader / runtime enable / telemetry | `current` | Agent turn 内 Skills 唯一主链 |
| `skill_search` | `current` | 只搜 metadata，不读 body、不授权 |
| `expert.skillRefs` | `current / declaration` | 专家声明；唯一匹配 native Agent Skill 时进入 `expert_binding` |
| `smoke:expert-skills-live-gate` | `current / read-only evidence gate` | 日常只读门禁；缺 live summary 返回 `pending_live_provider` |
| `smoke:expert-skills-live-runner` | `current / gated live validation runner` | 真实 live 验收入口骨架；默认阻断，显式授权才处理 live summary 或执行 Provider turn |
| `service-skill:*` | `compat / product service` | 不在 App Server 内静默当成本地 `SKILL.md` 执行 |
| `global_registry` / agent extension `loadSkill` | `compat` | 存量适配，不作为新增事实源 |
| 单功能硬编码 `Skill(name)` 首刀 | `deprecated` | 不继续扩张 |
| 绕过 snapshot / selector / `LimeSkillTool` gate 的执行路径 | `dead / forbidden` | 禁止回流 |

## 还差什么

下一刀只做真实质量验收，不做 runtime 基础重构：

1. live Provider 专家真实任务仍未实际执行；现在入口已具备，需显式外部模型调用授权后运行 `npm run smoke:expert-skills-live-runner -- --allow-live-provider --execute-live-runtime`，再由 `smoke:expert-skills-live-gate -- --live-summary <path>` 判定真实模型是否也能先 `skill_search`、再按需读取并授权被选中的 skill。
2. `needs_mapping / blocked` 已能替换问题 ref；真正写回 `catalog locator` / `service_scene_launch.skill_locator` 的后端持久化能力仍未做，但不再阻塞 deterministic 专家 Skills runtime 闭环。
3. `needs_registration` 已完成项目 scaffold registration 与前端刷新骨架；如需宣称完整产品交付，仍应用 live / GUI 手动证据确认“专家缺口 -> 创建 project skill -> readiness 刷新 -> 手动启用 -> 下一轮 evidence”真实链路。

## 最近验证

```bash
cargo fmt --manifest-path "lime-rs/Cargo.toml" --package lime-skills --package app-server --check
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-skills agent_selection
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server agent_skills_context
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server agent_skills_telemetry
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server skill_runtime_enable
npm test -- --run "src/lib/governance/agentSkillsRuntimeBoundary.test.ts"
npx eslint "src/lib/governance/agentSkillsRuntimeBoundary.test.ts" --max-warnings 0
npm test -- --run "src/components/agent/chat/experts/ExpertInfoPanel.test.tsx" "src/components/skills/SkillsWorkspacePage.test.tsx"
npx eslint "src/components/agent/chat/experts/ExpertSkillsSection.tsx" "src/components/agent/chat/experts/ExpertInfoPanel.tsx" "src/components/agent/chat/experts/ExpertInfoPanel.test.tsx" "src/components/agent/chat/AgentChatWorkspace.tsx" "src/components/skills/SkillsWorkspacePage.tsx" "src/components/skills/SkillsWorkspacePage.test.tsx" "src/types/page.ts" --max-warnings 0
git diff --check -- "src/components/agent/chat/experts/ExpertSkillsSection.tsx" "src/components/agent/chat/experts/ExpertInfoPanel.tsx" "src/components/agent/chat/experts/ExpertInfoPanel.test.tsx" "src/components/agent/chat/AgentChatWorkspace.tsx" "src/components/skills/SkillsWorkspacePage.tsx" "src/components/skills/SkillsWorkspacePage.test.tsx" "src/types/page.ts"
npm test -- --run "src/i18n/__tests__/types.test.ts"
cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server --check
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server scaffold
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server workspace_skill
npm test -- --run "src/components/skills/SkillsWorkspacePage.test.tsx"
npx eslint "src/components/skills/SkillsWorkspacePage.tsx" "src/components/skills/SkillsWorkspacePageView.tsx" "src/components/skills/SkillsWorkspacePage.test.tsx" "src/components/skills/SkillsWorkspacePage.testFixtures.tsx" --max-warnings 0
npm test -- --run "src/components/agent/chat/experts/expertSkillRuntimeViewModel.unit.test.ts" "src/components/agent/chat/experts/ExpertInfoPanel.test.tsx"
npm run smoke:agent-runtime-current-fixture
npm test -- --run "scripts/agent-runtime/expert-skills-live-gate.test.mjs"
npm test -- --run "scripts/agent-runtime/expert-skills-live-runner.test.mjs"
node --check "scripts/agent-runtime/expert-skills-live-gate.mjs" && node --check "scripts/agent-runtime/expert-skills-live-gate.test.mjs"
node --check "scripts/agent-runtime/expert-skills-live-runner.mjs" && node --check "scripts/agent-runtime/expert-skills-live-runner.test.mjs"
npx eslint "scripts/agent-runtime/expert-skills-live-gate.mjs" "scripts/agent-runtime/expert-skills-live-gate.test.mjs" --max-warnings 0
npx eslint "scripts/agent-runtime/expert-skills-live-runner.mjs" "scripts/agent-runtime/expert-skills-live-runner.test.mjs" --max-warnings 0
npm run governance:scripts
npm run smoke:expert-skills-live-gate
```

`smoke:expert-skills-live-gate` 当前输出 `pending_live_provider`：deterministic Expert Skills evidence 已通过，live Provider summary artifact 仍缺失。`smoke:expert-skills-live-runner` 已补默认阻断和 summary 归一化守卫，但本轮未执行真实 Provider turn。
