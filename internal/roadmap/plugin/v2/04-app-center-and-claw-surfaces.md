# Plugin v2 App Center 与 Claw Surface

状态：`proposed`

## 设计原则

复刻 Codex 的信息架构和状态语义，不复刻其视觉主题。Lime 继续使用当前浅色、安静、桌面应用感的设计语言，不引入 Codex/ChatGPT 的深色背景、品牌渐变或超大圆角。

## 页面类型

App Center 是目录型列表工作台，不是营销 landing page，也不是发布后台。

主对象：插件包。

当前阶段：发现、检查、安装、启停或更新。

主操作：根据状态显示一个明确动作；低频动作进入更多菜单或详情页。

## 页面结构

```text
┌────────────────────────────────────────────────────────────┐
│ Plugins                              [搜索] [来源] [刷新]  │
├──────────────┬─────────────────────────────────────────────┤
│ 全部         │ 已安装 3 / 可用 18                         │
│ 已安装       │                                             │
│ 官方         │ 插件列表                                    │
│ Workspace    │ icon  名称 / 描述 / 来源 / 状态    [主动作] │
│ Repo / Local │ icon  名称 / 描述 / 来源 / 状态    [主动作] │
│ 自定义来源   │                                             │
│ + 添加来源   │                                             │
└──────────────┴─────────────────────────────────────────────┘
```

宽窗口使用左侧来源 rail + 列表；窄窗口把来源折叠为下拉菜单。不要把每个来源做成等权大卡片。

## 顶部工具区

- 标题与安装摘要保持紧凑。
- 搜索框支持名称、描述、keyword、Skill、MCP server。
- 来源筛选优先使用下拉或左侧 rail。
- 刷新使用图标按钮。
- 添加 marketplace 是次级动作，不与安装主动作竞争。
- 不显示大段“插件是什么”说明；帮助信息进入 tooltip/help panel。

## 列表项

每项固定展示：

- 40px 左右 logo；无 logo 时使用稳定生成的 fallback，不复用同一默认图标。
- display name。
- 一行 short description。
- source label。
- installed/enabled/admin/auth/update 中最关键的一个状态。
- 单一主动作。
- 更多菜单。

状态与动作映射：

| 状态              | 主动作         | 次级动作               |
| ----------------- | -------------- | ---------------------- |
| available         | 安装           | 查看详情               |
| installing        | 安装中         | 取消，仅在可安全取消时 |
| installedEnabled  | 在 Claw 中使用 | 禁用、卸载、来源       |
| installedDisabled | 启用           | 卸载、来源             |
| updateAvailable   | 更新           | 在 Claw 中使用、卸载   |
| authRequired      | 连接           | 禁用、卸载             |
| disabledByAdmin   | 无             | 查看策略说明           |
| failed            | 重试           | 查看错误、卸载         |

卡片/行内不同时出现“打开、安装、更新、发布、审核、删除”五六个按钮。

## 来源导航

与 Codex 对齐的来源分区：

- 全部插件
- 已安装
- Lime Bundled/Official
- Workspace
- Shared with me
- Repo marketplace
- Personal/Local
- 用户配置 marketplace
- 添加 marketplace

空来源不必强制显示；Workspace 可在远端加载中显示稳定 skeleton，失败时只影响该来源，不清空其他 marketplace。

## 搜索

- 输入即时过滤当前 projection，不重新触发安装状态读取。
- 搜索为空时展示目录；无结果时显示当前 query 和清空入口。
- source tab 与 query 同时生效。
- 搜索结果仍展示 source 和 policy，不能因过滤丢失安全语境。

## 详情页/抽屉

详情不是卡片里的嵌套卡片。使用独立详情视图或右侧详情 pane：

```text
Header: logo / name / developer / version / primary action
Summary: long description / source / policy / auth
Capabilities: Skills / MCP Apps / Hooks
Examples: default prompts
Trust: website / privacy / terms / digest or signature
Diagnostics: only when blocked or failed
```

详情中的能力列表可以折叠，但外部写入、Hook 事件和授权要求默认可见。

## 安装确认

确认弹窗只在真正安装前出现，内容聚焦影响：

- 来源与开发者
- Read/Write/Interactive 能力
- Skills、MCP/apps、Hooks 数量
- auth-on-install/on-use
- 隐私和条款链接

按钮：`取消`、`安装`。不要把“安装并运行、安装并打开、安装并授权”混成多个竞争按钮；安装后按 auth policy 进入下一状态。

## Claw Composer

### `@` Picker

Picker 分组：

- 已安装插件
- 可安装建议
- 其他现有 mention 类型

每个插件项展示 logo、名称、简述和状态。已安装插件选中后插入结构化 mention；可安装项选中后打开安装确认，不直接插入可执行 mention。

### mention 显示

显示：`@Acme Projects`

结构化值：

```text
kind: plugin
pluginId
sourceIdentity
displayName
version?
```

发送时不得退化成只靠文本正则重新识别。

### 新 thread 提示

安装、启停或更新后，如果当前 thread 的 runtime snapshot 已建立，显示简短提示：“插件将在新对话中生效”。提供“新建对话”动作，不静默修改当前 turn 的 tool schema。

## Claw Right Surface

### 支持 surface

- MCP/App UI：结构化查看、比较、编辑、确认。
- Browser：复用 Browser Right Surface。
- Structured result：表格、列表、详情或 diff。
- File/media preview：复用现有 workspace preview。

### 布局

- Right Surface 与 timeline 保持同一 thread/turn context。
- 顶栏显示插件名、当前工具/资源名、刷新/关闭等明确命令。
- 主内容全高、可滚动，不放在装饰性外层卡片中。
- 窄窗口改为可切换主视图，不压缩到不可读的双栏。
- UI resource 自身不覆盖 Lime 全局标题栏和导航。

### 生命周期

- 打开 surface 不等于重新调用 tool。
- 刷新需要新的 tool/resource read 时，走标准 MCP 调用并记录 item。
- 用户提交写操作时走 approval/elicitation。
- thread 恢复后从 item/read model 恢复 surface descriptor。
- 插件卸载后历史 item 保留只读结果；不能再发起新动作。

## 视觉规则

- 主题继续使用 Lime 当前 color tokens。
- 卡片圆角不超过现有应用中心规范；列表优先，避免大块营销卡。
- 一个插件只使用自己的 logo/brand color 做局部识别，不染色整页。
- 状态色：成功 emerald、信息 sky/slate、提醒 amber、错误 rose/red。
- 主动作使用稳定按钮宽度，异步状态不引发布局跳动。
- 不使用大面积渐变、半透明主表面、嵌套卡片或说明型 hero。

## 响应式约束

至少验证：

- 1024x768：来源可折叠，列表名称与主动作不重叠。
- 1280x800：标准 rail + 列表 + 可选详情 pane。
- 1440x900：提高信息密度，不放大字体。
- 390x844：来源变下拉，详情占主视图，按钮文案可换行但不溢出。

稳定尺寸：

- logo/icon 固定方形。
- 工具按钮固定 `h/w`。
- 状态列与主动作有 min/max width。
- 长名称和长 source label 使用 truncate + title/tooltip。

## 可访问性与国际化

- 所有 icon button 有 `aria-label` 与 `title`。
- tab/segmented control 使用正确 selected state。
- 安装、启停使用 `aria-busy`/disabled 状态并保留可读原因。
- 键盘可完成搜索、切换来源、打开详情、安装和返回。
- 用户文案覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`。
- plugin manifest 内容按原文展示；系统状态和动作本地化。

## 稳定 DOM 合同

建议 test id：

```text
plugin-directory
plugin-source-selector
plugin-search-input
plugin-list
plugin-list-item
plugin-primary-action
plugin-detail
plugin-install-dialog
plugin-mention-option
plugin-right-surface
```

测试使用 identity 属性，例如 `data-plugin-id`、`data-source-kind`、`data-installed`，不依赖样式 class 或中文文本定位关键对象。

## UI 退出条件

- [ ] 用户 5 秒内能看出当前来源、安装状态和下一步动作
- [ ] installed/auth/admin/update 状态不互相覆盖
- [ ] App Center 与 Claw 显示同一 plugin identity 和版本
- [ ] `@plugin` 可安装建议不会伪装为已可用能力
- [ ] Right Surface 从 Thread/Turn/Item 恢复
- [ ] 1024 与移动宽度无文字/按钮重叠
- [ ] 五语言和键盘操作有稳定回归
- [ ] 真实 Electron Gate B 覆盖安装、mention、tool call 与 surface
