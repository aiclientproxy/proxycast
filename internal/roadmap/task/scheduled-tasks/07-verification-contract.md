# 已安排任务验收与验证合同

状态：`normative gate`

## 1. 产品验收矩阵

| 场景           | 必须证明                                                       |
| -------------- | -------------------------------------------------------------- |
| 首次进入       | 一级导航可达，空态/目录态正确，不依赖设置页                    |
| 手动创建       | 字段校验、preview、create、列表/详情即时一致                   |
| 使用 Lime 创建 | 新 Thread -> draft -> 用户确认 -> typed create，未经确认不落盘 |
| 从当前对话创建 | provenance、权限/项目/模型快照正确                             |
| 编辑           | revision 冲突可见，next run 重算，无双写                       |
| 暂停/恢复      | 暂停不 claim，恢复重算，历史保留                               |
| 立即运行       | 返回 run/thread/turn，能打开 live 对话                         |
| 到期运行       | 同一真实 RuntimeCore 主链，非 renderer timer/mock              |
| 历史           | completed/failed/waiting 能打开对应 Thread/Turn                |
| 失败恢复       | 模型/项目/审批/网络错误有明确恢复动作                          |
| 删除           | 定义软删除、历史可追溯、并发运行规则清晰                       |
| 冷启动/休眠    | next run 恢复、补跑策略和幂等成立                              |

## 2. Domain 单元测试

- 四种 schedule 的 next N occurrences。
- IANA timezone、DST 缺失/重复小时。
- weekday 去重、空 days、非法 HH:mm、interval 边界。
- claim 幂等、并发 claim、暂停/删除竞争。
- overlap `skip_if_running`。
- missed/catch-up 24h 边界。
- `Every/Cron/At` 迁移矩阵及不可迁移任务暂停。
- execution snapshot 不含 secret，路径跨平台 normalize。

## 3. Protocol/contract 测试

- method catalog、params/result/schema、Rust/TS generated type 对齐。
- Renderer gateway request/normalize/error fail closed。
- Desktop Host/preload 只转发 JSONL，无任务 CRUD method。
- notification method/shape 与 client dispatcher 对齐。
- 旧 method 完成迁移后加入负向回流断言。

最低命令：

```bash
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/scheduler lime-rs/crates/app-server-protocol lime-rs/crates/app-server
```

## 4. GUI 稳定回归

- VM unit：过滤、分组、schedule formatter、status projection、form request builder。
- Component：创建菜单、表单校验、主从选择、暂停/恢复、历史点击、响应式切换。
- i18n：五语种 key 与长文本布局。
- 可访问性：keyboard、focus return、aria label/live state。

最低命令：

```bash
npm run test:related -- <changed-frontend-files>
npm run verify:gui-smoke
```

## 5. Agent 主链 fixture

先运行：

```bash
npm run smoke:agent-runtime-current-fixture
```

专用 current fixture 必须覆盖：

- scheduled/manual run 创建 Thread/Turn/Item。
- `turn.completed` 驱动 run completed，不依赖 `final_done` timer。
- tool/approval terminal 投影稳定。
- history hydrate 后仍能从 run 打开 Thread。
- renderer/App Server mock backend 命中数为 0。

## 6. Gate A / Gate B

### Gate A

浏览器投影验证布局、状态与交互，不证明真实调度。

### Gate B

真实 Electron fixture 必须证明：

```text
GUI create
  -> preload/IPC
  -> app_server_handle_json_lines
  -> scheduledTask/create
  -> scheduler claim
  -> RuntimeCore thread/start + turn/start
  -> Thread/Turn/Item + Agent Run
  -> GUI history/open thread
```

证据至少记录 taskId/runId/threadId/turnId、method 命中、terminal status、cold restart、mock/legacy 命中为 0。

## 7. 平台矩阵

| 平台场景                 | macOS               | Windows                 |
| ------------------------ | ------------------- | ----------------------- |
| 系统时区/DST             | 必测                | 必测                    |
| sleep/wake reconcile     | 必测                | sleep/resume 必测       |
| cwd/path normalize       | POSIX               | drive/UNC/reparse point |
| OS notification          | Notification Center | Windows notification    |
| app restart/cold restore | 必测                | 必测                    |

macOS 证据不能替代 Windows 语义；无 Windows runner 时状态只能是 platform gap。

## 8. 完整最低门禁

```bash
npm run test:contracts
npm run test:rust:related -- <changed-rust-paths>
npm run smoke:agent-runtime-current-fixture
npm run verify:local
npm run verify:gui-smoke
npm run governance:legacy-report
```

若修改脚本目录，另跑 `npm run governance:scripts`。若全量前端已有中断批次，使用 `npm run test:resume`。

## 9. 可交付结论

只有以下全部成立才能标记 complete：

- 产品验收矩阵全绿。
- protocol/domain/GUI/current fixture 通过。
- 真实 Electron Gate B 通过。
- macOS/Windows 平台矩阵完成或明确记录未完成平台缺口，且不得声称全平台 complete。
- compat 命中为 0，dead 路径被 guard 阻止。
- 架构图已更新并由责任开发者确认。
