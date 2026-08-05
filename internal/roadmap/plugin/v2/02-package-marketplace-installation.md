# Plugin v2 包、Marketplace 与安装合同

状态：`proposed`

## 唯一包结构

Plugin v2 只接受 Codex-compatible package：

```text
my-plugin/
├── .codex-plugin/
│   └── plugin.json
├── skills/
│   └── my-workflow/
│       ├── SKILL.md
│       ├── references/
│       ├── scripts/
│       └── assets/
├── .mcp.json                # 可选
├── .app.json                # 可选，兼容 MCP/App connection 映射
├── hooks.json               # 可选
└── assets/                  # 可选，logo/icon/screenshots
```

旧根目录 `plugin.json`、`schemaVersion: lime.plugin.package.v1`、`contributions.runtime`、`contributions.workbench` 不进入 v2 parser。

## Manifest 合同

最小 skills-only 插件：

```json
{
  "name": "meeting-follow-up",
  "version": "1.0.0",
  "description": "Turn meeting notes into decisions and next steps",
  "skills": "./skills/"
}
```

完整示例：

```json
{
  "name": "acme-projects",
  "version": "1.2.0",
  "description": "Review and update Acme projects",
  "author": {
    "name": "Acme"
  },
  "homepage": "https://example.com/docs",
  "repository": "https://example.com/repo",
  "license": "MIT",
  "keywords": ["projects", "planning"],
  "skills": "./skills/",
  "mcpServers": "./.mcp.json",
  "apps": "./.app.json",
  "hooks": "./hooks.json",
  "interface": {
    "displayName": "Acme Projects",
    "shortDescription": "Review projects and blockers",
    "longDescription": "Inspect project status, review blockers, and update approved fields.",
    "developerName": "Acme",
    "category": "Productivity",
    "capabilities": ["Read", "Write", "Interactive"],
    "websiteURL": "https://example.com",
    "privacyPolicyURL": "https://example.com/privacy",
    "termsOfServiceURL": "https://example.com/terms",
    "defaultPrompt": ["Show blockers in my active projects"],
    "brandColor": "#0F766E",
    "composerIcon": "./assets/composer.png",
    "logo": "./assets/logo.png",
    "logoDark": "./assets/logo-dark.png",
    "screenshots": ["./assets/overview.png"]
  }
}
```

## 字段规则

### Identity

- `name`：必填，kebab-case，作为 package identity 与 namespace 基础。
- `version`：发布包必填，严格 semver；本地开发包也建议必填。
- 同一 marketplace 内 `name` 唯一。
- installed identity 不能只用 `name`；至少组合 source identity 与 plugin name，远端包还保留 remote plugin ID。

### Components

- `skills`：包内相对路径，默认发现 `skills/`。
- `mcpServers`：包内 `.mcp.json` 路径或内联 server object。
- `apps`：包内 `.app.json`，指向已注册或可解析的 app/connector declaration。
- `hooks`：包内 hook config；启用前必须做信任和事件披露。
- 所有路径都必须解析到 package root 内，禁止绝对路径、`..` 越界、符号链接逃逸与大小写/Unicode normalization 冲突。

### Interface

- `displayName` 是用户可见名称；缺失时回退 `name`。
- `shortDescription` 用于列表；`longDescription` 用于详情。
- `defaultPrompt` 最多展示 3 条，每条最多 128 字符。
- logo、icon 和 screenshots 必须存在于包内。
- `brandColor` 只用于局部品牌提示，不覆盖 Lime 全局主题。
- 外部写入或交互能力必须来自实际组件声明，不允许只靠 marketing metadata 宣称。

## Marketplace 合同

Repo marketplace：

```text
$REPO_ROOT/.agents/plugins/marketplace.json
```

Personal marketplace：

```text
~/.agents/plugins/marketplace.json
```

示例：

```json
{
  "name": "local-team",
  "interface": {
    "displayName": "Team Plugins"
  },
  "plugins": [
    {
      "name": "acme-projects",
      "source": {
        "source": "local",
        "path": "./plugins/acme-projects"
      },
      "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
      },
      "category": "Productivity"
    }
  ]
}
```

### 支持来源

- `local`：marketplace root 内的相对目录。
- `url`：插件位于 Git repository 根目录。
- `git-subdir`：插件位于 Git repository 子目录，可带 `ref` 或 `sha`。
- `npm`：包名、版本范围和 HTTPS registry；下载时禁止执行 lifecycle scripts。
- `remote`：workspace/public catalog 返回的不可伪造 remote identity。

### Policy

- `installation`: `AVAILABLE | INSTALLED_BY_DEFAULT | NOT_AVAILABLE`
- `authentication`: `ON_INSTALL | ON_USE`
- `products`：仅作为显式 surface override；默认省略。
- `category`：目录分组，不参与权限判断。

## Discovery 顺序

App Server 根据当前 workspace roots 计算 discovery：

1. release-bundled marketplace
2. 当前 repo marketplace
3. personal marketplace
4. 用户显式配置的 marketplace snapshots
5. workspace/remote catalog

顺序只影响展示与冲突诊断，不允许用“后读覆盖前读”静默替换同 identity 包。冲突必须返回 source-aware diagnostics。

Renderer 不读取任何 marketplace 文件，也不自行扫描 `plugins/`。

## Installed Store

installed store 至少保存：

```text
pluginId
pluginName
sourceKind
sourceIdentity
marketplaceIdentity
remotePluginId?
installedVersion
contentDigest
installedAt
enabled
installPolicy
authPolicy
packageRootLocator
componentSummary
```

不要把 connector access token、OAuth refresh token 或用户数据写进 installed record。凭证进入统一 secret/credential owner。

## 安装事务

```text
resolve source
  -> fetch/copy into isolated staging
  -> validate marketplace policy
  -> validate manifest and all paths
  -> compute package digest
  -> inspect Skills/MCP/Apps/Hooks
  -> produce install review
  -> user confirmation
  -> atomic materialization
  -> write installed store
  -> start required auth flow
  -> emit pluginsChanged
```

### Install review

确认界面必须披露：

- 插件名称、版本、开发者和来源
- 安装策略与 auth policy
- Skills 数量与名称
- MCP servers/apps
- Hook 事件与信任状态
- Read/Write/Interactive 能力
- 需要的外部授权
- 包 digest 或签名状态

### 原子性

- staging 与 final package root 必须在同一可原子替换的存储边界。
- 任一步失败都清理 staging，installed store 保持原值。
- 更新使用 side-by-side staging，验证通过后替换，不原地覆盖运行中的包。
- Runtime 使用 version/digest pin；正在运行的 turn 不切换到半更新组件。

## 启用与刷新

- 安装后默认是否启用由 marketplace policy 和用户确认共同决定。
- enabled state 只影响新 thread/turn 的能力 snapshot。
- 改变启用状态后发出 `pluginsChanged`，Claw 提示新 conversation/thread 生效。
- 不在运行中的 turn 动态增删工具定义，避免 model context 与 tool registry 不一致。

## 更新

更新检查比较 source version、installed version 与 local materialized version：

- 同 digest 不重复安装。
- semver 降级需要显式确认。
- source identity 改变视为新安装，不静默更新。
- Git branch/range 可以刷新，但 installed record 必须 pin 到解析后的 commit/digest。
- npm 安装不运行 `preinstall/postinstall` 等生命周期脚本。

## 卸载

卸载分四类资源：

1. package materialization
2. installed/enabled configuration
3. connector authorization
4. plugin-owned user data

默认卸载只删除 1 和 2；3 和 4 必须在 UI 中单独说明和选择。管理员默认安装的插件不可由用户卸载，只能按 policy 允许时禁用。

卸载必须先停止该插件的 MCP process/hooks，等待 terminal 或明确中止，再删除 package。正在运行的 turn 保留历史 item，不因卸载被改写。

## 平台路径

生产代码不得直接拼接 `$HOME`、`~/Library` 或 Windows profile 路径。所有位置通过统一 app paths/platform API 解析：

- bundled source：应用 Resources 内只读目录
- marketplace snapshots：用户 app data/config owner
- installed packages：用户 app data 下的 plugin store
- staging/cache：统一 cache/temp owner
- secrets：系统 keychain/credential owner

macOS 和 Windows 使用相同逻辑 identity，不要求物理路径一致。

## 安全边界

- manifest 与 marketplace 都是不可信输入。
- 路径必须 canonicalize 后验证仍在 authority root 内。
- remote download 必须有大小、超时、content-type、digest/signature 与重定向限制。
- archive 解包防 zip-slip、symlink/hardlink escape、大小炸弹和文件数量炸弹。
- MCP/Hook/Script 的存在不意味着可执行；执行受 runtime policy 和 approval 控制。
- UI resource 使用受控 origin/CSP，不接受任意 `file://` 或 unrestricted localhost。
- 日志只保存 identity、digest、阶段和错误码，不保存 token、完整用户数据或 tool result。

## 明确删除的旧合同

Plugin v2 完成迁移后，下列合同为 `dead / forbidden-to-restore`：

- `lime.plugin.package.v1`
- 根 `plugin.json` 的 `contributions.runtime/workbench`
- `app.runtime.yaml` 作为插件 runtime 主入口
- Renderer 自行 projection manifest
- 以本地完整路径作为跨层 plugin identity
- 安装记录中混合凭证、运行状态和 UI readiness
