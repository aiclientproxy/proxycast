# Plugin v2 研究结论

状态：`research-complete`

取证日期：2026-08-04

## 研究问题

本次研究回答四个问题：

1. Codex 现在如何定义、发现、安装和运行插件？
2. ChatGPT/Codex Desktop 的真实安装包如何组织 bundled plugins？
3. Lime 当前实现与 Codex 的关键差异是什么？
4. 哪些差异属于 Lime 产品特性，哪些只是应删除的历史自定义协议？

## 官方文档结论

本次使用当前 Codex Manual 并检索以下官方页面：

- [Build plugins](https://learn.chatgpt.com/docs/build-plugins)
- [Package your plugin](https://developers.openai.com/plugins/build/plugins)
- [Plugin architecture](https://developers.openai.com/plugins/concepts/plugins)
- [Build an MCP server](https://developers.openai.com/plugins/build/mcp-server)
- [Connect and test your plugin](https://developers.openai.com/plugins/deploy/connect-chatgpt)
- [Plugins](https://learn.chatgpt.com/docs/plugins)
- [Skills and plugins](https://learn.chatgpt.com/docs/skills-and-plugins)
- [Plugin controls](https://learn.chatgpt.com/docs/enterprise/apps-and-connectors)

文档给出的稳定结论：

- Plugin 是可安装的能力包，可以包含 Skills、MCP server/apps、Hooks、assets 与展示元数据。
- Skill 适合可复用指令与资源；MCP 适合实时数据、鉴权、受控动作或远端执行；UI 只在结构化交互确有价值时加入 MCP server。
- ChatGPT 与 Codex 使用同一个公共插件目录，但不同产品 surface 的可用性和安装控制仍分开。
- 插件安装不等于 connector 已授权，也不等于 runtime 获得文件、网络或命令权限。
- 安装后应在新 conversation/thread 中验证，以确保 Skills 和工具清单按新状态加载。

## 本机 Desktop 安装包证据

### 应用身份

```text
Application: /Applications/ChatGPT.app
Bundle identifier: com.openai.codex
Version: 26.727.51351
```

虽然应用目录名是 `ChatGPT.app`，bundle identifier 与资源都表明它同时承载 Codex Desktop surface。不能依据 Finder 展示名推断产品边界。

### 关键资源目录

```text
/Applications/ChatGPT.app/Contents/Resources/app.asar
/Applications/ChatGPT.app/Contents/Resources/codex
/Applications/ChatGPT.app/Contents/Resources/codex-code-mode-host
/Applications/ChatGPT.app/Contents/Resources/plugins/openai-bundled
/Applications/ChatGPT.app/Contents/Resources/skills
/Applications/ChatGPT.app/Contents/Resources/native
```

取证时体积约为：

```text
app.asar: 208 MiB
plugins/openai-bundled: 69 MiB
skills: 372 KiB
```

这些目录表明 bundled plugin 不是写死在前端卡片中的 metadata，而是随应用发布的真实包、脚本、Skills、assets 与本地运行资源。

### bundled marketplace

入口文件：

```text
Resources/plugins/openai-bundled/.agents/plugins/marketplace.json
```

取证时包含以下插件：

- `sites`
- `browser`
- `chrome`
- `computer-use`
- `messages`
- `reminders-macos`
- `record-and-replay`
- `latex`
- `deep-research`
- `visualize`

marketplace 只描述来源、安装策略、鉴权时机与类别；插件自己的能力和展示信息位于各包的 `.codex-plugin/plugin.json`。

### 真实插件包样例

Browser 插件使用以下结构：

```text
plugins/browser/
├── .codex-plugin/plugin.json
├── assets/
├── docs/
├── scripts/
└── skills/
    └── control-in-app-browser/SKILL.md
```

manifest 包含稳定 identity、版本、描述、作者、Skills 路径与 `interface` 元数据；能力脚本和大体积依赖留在插件包内，不进入 renderer bundle。

### 用户态目录

本机可见的相关位置包括：

```text
~/Library/Application Support/Codex
~/Library/Application Support/OpenAI/Codex/NativeMessagingHosts
```

前者主要是 Chromium/Electron profile、缓存和 browser partition；后者是 Codex 原生消息桥接位置。它们不是 plugin manifest 事实源。

官方文档给出的本地作者目录约定为：

```text
Repo marketplace:     $REPO_ROOT/.agents/plugins/marketplace.json
Personal marketplace: ~/.agents/plugins/marketplace.json
Common plugin source: $REPO_ROOT/plugins/<name> 或 ~/.codex/plugins/<name>
```

`marketplace.json` 的 `source.path` 才是解析依据，示例目录不是必须硬编码的唯一位置。

## Codex 源码证据

### 领域 owner

| 路径                                                    | 责任                                                    |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `codex-rs/plugin`                                       | manifest、plugin ID、source authority、resolved plugin  |
| `codex-rs/core/plugins`                                 | marketplace discovery、安装、卸载、remote catalog、配置 |
| `codex-rs/utils/plugins`                                | mention syntax、namespace、MCP connector helper         |
| `codex-rs/tools/request_plugin_install.rs`              | agent 发起的插件安装请求                                |
| `codex-rs/app-server/src/request_processors/plugins.rs` | 对 GUI/CLI 暴露 current plugin API                      |
| `codex-rs/tui/src/chatwidget/plugins.rs`                | 列表、搜索、详情、安装、启停、marketplace UI            |

Codex 的 resolver 先生成 inert descriptor，再由上层决定装配哪些能力。路径资源保留 source authority，避免把来自其他 environment 的资源误当成本机路径。

### App Server 方法

Codex current 方法包括：

```text
marketplace/add
marketplace/remove
marketplace/upgrade
plugin/list
plugin/installed
plugin/read
plugin/install
plugin/uninstall
```

`plugin/list` 支持 local、vertical、workspace-directory、shared-with-me、created-by-me-remote 等 marketplace kind；`plugin/read` 返回 Skills、Hooks、Apps、MCP servers 与完整展示信息。

### 产品状态不是单一布尔值

Codex protocol 分开表达：

- `installed`
- `enabled`
- `installPolicy`: `AVAILABLE | INSTALLED_BY_DEFAULT | NOT_AVAILABLE`
- `authPolicy`: `ON_INSTALL | ON_USE`
- `availability`: `AVAILABLE | DISABLED_BY_ADMIN`
- `disabledReason`
- `version` 与 `localVersion`

Lime v2 必须保留这些正交维度，不能再用 `installed/disabled/readiness` 的前端拼装近似替代。

### UI 交互证据

Codex TUI snapshot 和 Desktop `app.asar` 中的真实页面资源共同证明以下 surface 已存在：

- Plugins 目录页与插件详情页
- All / Installed / source marketplace 分区
- 搜索
- 安装数量摘要
- 安装、卸载、启用、禁用
- 管理员分配与管理员禁用
- Skills、Hooks、Apps、MCP servers 披露
- 来源、鉴权时机、隐私和条款披露
- `@plugin` mention 与安装建议

Desktop bundle 中对应资源包括 `plugins-page`、`plugin-detail-page`、`plugin-picker-menu-content`、`plugins-settings` 与 `skills-page`。

## Lime 当前差异

### 包标准分叉

Lime 当前 App Server 要求：

```text
plugin.json
schemaVersion: lime.plugin.package.v1
contributions.runtime
contributions.workbench
```

随后由 `plugin_packages/plugin_manifest.rs` 投影成前端使用的另一份 manifest。这个流程与 Codex `.codex-plugin/plugin.json` 不是同一标准，且该文件已超过 1000 行，混合了解析、projection、兼容和 UI/runtime 推断。

### 前端承担过多事实合并

`src/features/plugin/marketplace/marketplaceRegistryLoader.ts` 同时读取 marketplace、installed state、manifest、setup state、runtime profile 和 readiness，再组合页面 registry。它实际成为第二个插件事实源。

### 自定义 worker 与 UI runtime

现有 `electron/pluginRuntimeTaskHost.ts`、`pluginTaskWorker.ts`、`src/features/plugin/runtime/**` 和 App Server plugin worker runtime 形成了插件专用执行/投影体系。它与 RuntimeCore、MCP、Skills 和 Right Surface 的 current owner 重叠。

### 产品范围膨胀

旧路线图把插件消费、独立应用壳、发布中心、云端上传、历史工作区和内容生产 runtime 同时纳入一个计划，导致 App Center 既像 marketplace，又像应用启动器和发布后台。

## 对 Plugin v2 的直接结论

1. 直接采用 `.codex-plugin/plugin.json`，不设计旧 manifest compat 层。
2. App Server 返回完整 PluginSummary/PluginDetail，Renderer 不再合并 registry。
3. 插件能力装配复用 Skills、MCP、Hooks 与 RuntimeCore owner。
4. `@plugin` mention 是 Claw 的首要调用入口；目录页不是唯一入口。
5. Right Surface 承载 MCP/App UI 或结果投影，不启动第二套插件业务后端。
6. 发布平台与消费者 runtime 分离；v2 先完成本地、repo、bundled 与 remote catalog 消费闭环。
7. 旧文档和 implementation tracker 冻结为历史参考；有效洞察提炼进入 v2，但旧文档不再作为 current 实现依据或独立继续演进。

## 研究限制

当前会话没有暴露 `computer-use` skill 要求的 Node REPL 工具，因此没有通过自动化直接点击本机 ChatGPT/Codex Desktop 的 Plugins 页面。视觉与交互结论来自：

- 官方文档
- 本机真实应用包和 bundle 资源
- Codex TUI snapshot
- Codex App Server protocol 与实现

这足以确定产品结构和协议，但正式 UI 实现前仍需补一轮真实 Desktop 交互对照截图，作为设计输入而不是运行时事实源。
