# Plugin v2 产品合同

状态：`proposed`

## 产品定义

Plugin 是一个可安装、可治理、可在对话中调用的能力包。它把相关 Skills、MCP servers/apps、Hooks、assets 和展示元数据组织为一个稳定 identity。

Plugin 不是：

- 单独一张应用中心卡片
- 任意网页 URL
- 另一套 Agent runtime
- 只有 UI、没有可调用能力的 iframe
- 把 Skill、MCP 和 Hook 复制进自己的私有协议

## 当前主对象

App Center 的主对象是“插件包”；Claw 的主对象仍是“当前 thread/turn”。插件只向 thread 提供可发现能力，不能把用户从对话主线带到第二套任务系统。

## 核心用户目标

1. 找到可信的插件并理解它能做什么。
2. 看清来源、权限、鉴权时机和外部影响后安装。
3. 在 Claw 中通过自然语言或 `@plugin` 明确调用。
4. 在右侧查看插件返回的结构化 UI、浏览器或结果详情。
5. 随时启停、更新或卸载，并知道残留数据和授权如何处理。

## 来源模型

| 来源       | 说明                                                 | 默认信任                 |
| ---------- | ---------------------------------------------------- | ------------------------ |
| bundled    | 随 Lime 安装包发布并由 release 签名保护              | 高，但仍受运行时权限约束 |
| repo       | 当前 workspace 的 `.agents/plugins/marketplace.json` | 仅当前 workspace         |
| personal   | 用户级 marketplace 和本地插件源                      | 当前用户                 |
| configured | 用户显式添加的 Git/local/npm marketplace             | 按来源和校验结果         |
| workspace  | 工作区管理员或远端目录提供                           | 按 workspace policy      |
| shared     | 其他成员分享或链接安装                               | 需要展示发布者和分享边界 |

来源只决定发现和安装路径，不自动授予 connector、文件、网络、命令或 external action 权限。

## 生命周期状态

### 目录可见性

- `available`：可安装。
- `installedByDefault`：由 bundled/workspace policy 默认安装。
- `notAvailable`：目录中可解释，但当前不能安装。
- `disabledByAdmin`：管理员禁止使用。

### 本地状态

- `notInstalled`
- `installing`
- `installedDisabled`
- `installedEnabled`
- `updateAvailable`
- `uninstalling`
- `failed`

### 鉴权状态

- `notRequired`
- `requiredOnInstall`
- `requiredOnUse`
- `authorizing`
- `authorized`
- `authorizationFailed`

这些状态必须保持正交。一个插件可以“已安装但未授权”，也可以“管理员默认安装但用户禁用”。

## 关键流程

### 目录安装

```text
打开 App Center
  -> 选择来源或搜索
  -> 打开详情
  -> 查看能力、权限、来源与条款
  -> 安装
  -> 必要时授权 connector
  -> 新建或刷新 thread
  -> 在 Claw 中调用
```

### Claw 中安装建议

```text
用户输入 @plugin 或提出明确目标
  -> installed projection 未找到能力
  -> 目录 projection 存在可安装候选
  -> 展示安装建议
  -> 用户确认安装
  -> 必要时授权
  -> 新 thread/turn 生效
```

Agent 可以请求安装，但不能绕过安装确认、管理员策略或 connector 授权。

### 右侧工作区

```text
插件工具返回 structured content / UI resource
  -> RuntimeCore 记录 tool item
  -> GUI projection 解析 surface descriptor
  -> Claw Right Surface 打开对应面板
  -> 用户在同一 thread 继续确认或操作
```

右侧关闭后，对话与 tool item 仍可恢复；右侧不能成为独立于 thread 的隐藏状态源。

## App Center 信息架构

一级结构：

- 全部
- 已安装
- 官方/bundled
- Workspace
- Repo/Local
- 用户添加的 marketplace

主要动作：

- 未安装：`安装`
- 已安装且启用：`在 Claw 中使用`
- 已安装且禁用：`启用`
- 有更新：`更新`
- 详情更多操作：`卸载`、`查看来源`、`管理授权`

不在卡片上同时放置发布、审核、导出、删除数据等低频管理动作。

## 详情披露合同

详情必须展示：

- 名称、图标、开发者、版本
- 一句话能力说明与完整说明
- 来源 marketplace 和安装策略
- Skills
- MCP servers/apps
- Hooks 及触发事件
- 需要的 connector 与鉴权时机
- Read/Write/Interactive 等能力标签
- 网站、隐私政策和服务条款
- 安装/启停/更新/卸载状态

技术细节可以折叠，但外部写入、敏感数据访问和不可逆动作不能隐藏。

## Claw 兼容合同

### Composer

- `@` picker 同时显示已安装插件和明确标注的可安装建议。
- 已安装插件可插入 `plugin://<plugin-id>` 结构化 mention，显示文本使用 `@DisplayName`。
- 未安装插件不能伪装成已可调用；选中后进入安装确认。
- mention identity 必须贯穿 turn request、tool selection、trace 和历史恢复。

### Runtime

- Plugin mention 只增加插件选择上下文，不绕过 Skill/MCP/Tool 选择与权限检查。
- 未显式 mention 时仍可按 Skill 描述和工具 metadata 自动发现。
- 同一插件的 Skills、MCP 与 Hooks 使用同一 plugin identity 和 source authority。

### Right Surface

- 支持 MCP/App UI resource。
- 支持 browser intent 与现有 Browser Right Surface。
- 支持结构化结果、文件预览和审查视图。
- 不支持插件自行启动未受管 iframe、任意本地 server 或 renderer-to-worker 私有协议。

## 启用与安装的语义

- 安装：把受校验的包和来源写入 installed store。
- 启用：允许其能力进入新 thread 的 discovery/activation projection。
- 授权：允许 connector 代表当前身份访问外部系统。
- 运行权限：决定当前 turn 是否可执行文件、网络、命令或外部写操作。

这四件事不得合并为一个“可用”按钮。

## 错误与恢复

每个失败必须给出可执行恢复动作：

| 失败                 | 恢复动作                             |
| -------------------- | ------------------------------------ |
| marketplace 无法读取 | 重试、刷新来源、查看具体来源错误     |
| manifest 无效        | 显示字段/路径错误，不允许安装        |
| 版本不兼容           | 显示所需 host/runtime 版本           |
| connector 未授权     | 进入授权，不重新安装插件             |
| 管理员禁用           | 说明策略来源，不提供无效启用按钮     |
| MCP 启动失败         | 查看连接诊断、重试或禁用该 connector |
| 安装中断             | 回滚 staging，不留下半安装状态       |
| 卸载有残留           | 显示残留位置和重试入口               |

## 首版范围

P0/P1 必须支持：

- bundled/repo/personal/configured local marketplace
- `.codex-plugin/plugin.json`
- skills-only plugin
- MCP plugin，包括 auth-on-install 与 auth-on-use
- Hooks 声明和信任披露
- App Center 目录、详情、安装、启停、卸载
- `@plugin` mention
- Claw Right Surface 的 MCP/App UI

后续再支持：

- public universal directory 发布
- workspace sharing 与 groups
- remote billing/commerce
- scheduled tasks
- 插件作者 analytics

## 产品验收问题

发布前必须能明确回答：

1. 用户正在处理哪个插件、来自哪里？
2. 它为什么可用、不可用或被禁用？
3. 安装后还需要什么授权？
4. 在 Claw 中如何调用？
5. 右侧内容属于哪个 thread/turn/tool item？
6. 卸载后包、配置、授权和数据各自如何处理？
