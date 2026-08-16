# Agent Skills Runtime 执行骨架

> 状态：先按骨架推进  
> 更新时间：2026-06-23  
> 原则：先跑通 `发现 -> 选择 -> 读取 -> 授权 -> 执行 -> 复盘` 闭环，再补排序、预算、治理细节。

当前状态入口见：[status.md](status.md)。后续继续前先看状态文件，避免从 P0 重新铺 runtime。

## 1. 主目标

对标 Codex 的 skills progressive disclosure：默认只给模型轻量 metadata；只有任务需要、用户显式点名或产品入口绑定时，才读取对应 `SKILL.md` 并进入会话级授权执行。

## 2. 主链骨架

```text
Skill roots
  -> AgentSkillSnapshot
  -> skill_search / selector
  -> SkillBodyReader
  -> LimeSkillTool gate
  -> Agent turn execution
  -> timeline / evidence pack
```

## 3. 阶段表

| 阶段 | 目标 | 最小交付物 | 当前状态 |
| --- | --- | --- | --- |
| P0 合同 | 固定事实源，不再散落硬编码 | `AgentSkillSnapshot` / metadata / locator 合同 | 骨架已在 `lime-rs/crates/skills/src/agent_snapshot.rs` |
| P1 选择 | 统一显式和隐式候选 | `$skill`、catalog binding、description match 进入同一 selector | 骨架已在 `agent_selection.rs` / `agent_search.rs` |
| P2 读取 | 只读被选中的 skill | `SkillBodyReader` 读取 `SKILL.md` 与必要 references | 骨架已在 `agent_body.rs` / `agent_render.rs` |
| P3 授权 | 执行前过 gate | session allowlist + `allowed_tools` 交集裁剪 | 骨架已在 `skill_tool_gate.rs` / `skill_runtime_enable.rs` |
| P4 复盘 | 让使用过程可解释 | RuntimeEvent / runtime.status / canonical read model 记录 selection、body read、gate、tools | 骨架已在 App Server current event stream 与 Thread/Turn/Item projection 汇总 |
| P5 专家 | 专家 skills 同链路跑通 | Expert Plaza / ExpertInfoPanel `skillRefs` 进入 selector | 骨架已接入：expert refs -> selector -> body read -> gate -> evidence |
| P6 live 验收 | 不把 fixture 证据冒充真实模型质量 | `live-gate` 只读审计 + `live-runner` 显式授权执行 / 归一化 live summary | 骨架已接入：默认阻断真实 Provider，显式授权后输出 gate 可消费 summary |

## 4. 先做的三刀

1. **P5 专家闭环收口**  
   已完成骨架：`expert.skillRefs` 进入现有 selector，能唯一匹配 `AgentSkillSnapshot` 的 skill 会读取 `SKILL.md`、写入本轮 gate allowlist，并产生 `skill_body_read` / `skill_gate_decision` evidence。不新增专家专属执行器。

2. **ToolSearch / skill_search fail-closed 收口**  
   0 命中必须返回终态 metadata，不能触发同义词重试或中间结论自动续写。

3. **文档与守卫对齐**  
   README、fixture smoke、evidence 字段只描述这条 current 主链，删除“另起一条 skill runtime”的暗示。

## 5. 暂缓细节

这些先不阻塞骨架闭环：

- BM25 排序参数
- prompt 文案 polish
- 大量边缘同名冲突策略
- GUI 细节布局
- 全量历史 skill 迁移
- 每个 `@xxx` 命令的逐项精修

## 6. 骨架验收命令

先只跑贴边界验证：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-skills agent_
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent skill
npm run smoke:agent-runtime-current-fixture
npm run smoke:expert-skills-live-gate
```

只有触碰协议 / GUI / catalog 绑定时再补：

```bash
npm run test:contracts
npm run verify:gui-smoke
```

live Provider 专家验收单独显式执行，不进入默认门禁：

```bash
npm run smoke:expert-skills-live-runner -- --allow-live-provider --execute-live-runtime
npm run smoke:expert-skills-live-gate -- --live-summary .lime/qc/expert-skills-live-runner-summary.json
```

## 7. 当前结论

P0-P5 的代码骨架已经存在，下一步不要从零重做；直接做专家 Skills 易用性和 GUI 回归闭环，并把 ToolSearch / skill_search 的 0 命中 fail-closed 作为防循环底线。

## 8. 2026-06-22 快速收口记录

- `current`：`AgentSkillSnapshot` / selector / `SkillBodyReader` / `LimeSkillTool` gate / runtime evidence 是唯一主链。
- `current`：专家 `skillRefs` 作为 `expert_binding` trigger 进入同一主链；只匹配唯一可读 `SKILL.md`，不为专家新增执行分支。
- `current`：专家 `needs_registration` 恢复动作进入 Skills 用户安装页，带目标搜索预填，并生成项目级 scaffold 草稿自动打开创建表单，便于用户直接补齐待注册技能。
- `compat`：`service-skill:*` 专家引用仍按产品服务技能处理；除非映射成 native `SKILL.md` locator，否则不在 App Server 内静默当成本地 skill 执行。
- `pending`：`needs_registration` 还没有自动创建或修复 `workspaceSkillBindings`，不要把搜索预填 / scaffold 预填误判成注册闭环完成。
- `pending/live`：`smoke:expert-skills-live-runner` 已能默认阻断和生成 gate 可消费 summary，但真实 Provider turn 本轮未执行；不能把 deterministic fixture 说成整体目标完成。
- `dead` 防线：ToolSearch / skill_search 0 命中必须 fail-closed，不允许模型靠重复搜索进入自动续写循环。
