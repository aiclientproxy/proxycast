# 数据库模块

本模块负责 SQLite 数据库的初始化、表结构定义和数据迁移。

## 文件索引

| 文件 | 说明 |
|------|------|
| `mod.rs` | 模块入口，数据库初始化 |
| `schema.rs` | 表结构定义和创建 |
| `migration.rs` | 数据迁移逻辑（API Keys、Provider ID 等） |
| `migration_v2.rs` | 统一内容系统迁移（默认项目、话题迁移） |
| `system_providers.rs` | 系统预设 Provider 配置 |
| `dao/` | 数据访问对象层 |

## 数据库表

### 核心表

- `api_key_providers` - API Key Provider 配置
- `api_key_providers` - API Key Provider 配置主表
- `provider_pool_credentials` - 旧凭证池表；启动期只读保留，等待 manifest 驱动迁移
- `providers` - Provider 配置
- `settings` - 应用设置

### Legacy 通用对话迁移面

- `general_chat_sessions` / `general_chat_messages` - 只用于启动期识别并迁移历史安装里的旧表，不再作为新库默认 schema 的一部分

### 功能表

- `mcp_servers` - MCP 服务器配置
- `prompts` - 提示词模板
- `skills` - 技能配置
- `skill_repos` - 技能仓库

## DAO 模块

| 文件 | 说明 |
|------|------|
| `dao/agent.rs` | Agent 会话和消息 DAO |
| `dao/api_key_provider.rs` | API Key Provider DAO |
| `dao/mcp.rs` | MCP 服务器 DAO |
| `dao/prompts.rs` | 提示词 DAO |
| `dao/provider_pool.rs` | 旧凭证池 DAO；不得作为运行时选择入口 |
| `dao/providers.rs` | Provider DAO |
| `dao/skills.rs` | 技能 DAO |

## 数据迁移

### API Keys 迁移

旧 `migrate_api_keys_to_pool()` 只属于历史迁移链。当前启动期不再清理 `provider_pool_credentials` 或 `AppDataRoot/credentials`；运行时不读取旧表选择凭证，旧数据只作为 `deprecated/read-only migration source` 保留，等待独立 maintenance manifest、备份和确认。

- 当前运行时不再把 `provider_pool_credentials` 自动转换为凭证 DTO；历史记录只服务迁移盘点与验证
- 保留使用统计和错误计数
- 标记来源为 `imported`
- 迁移完成后设置 `migrated_api_keys_to_pool` 标记，避免重复迁移

### 统一内容系统迁移 (migration_v2)

`migrate_unified_content_system()` 函数实现统一内容系统的数据迁移：

- **创建默认项目**: 如果不存在 `is_default=true` 的项目，自动创建"默认项目"
- **迁移话题**: 将所有 `project_id` 为 null 的内容迁移到默认项目
- **事务保护**: 迁移过程使用事务，失败时自动回滚
- **幂等性**: 迁移完成后设置标记，避免重复执行

_Requirements: 2.1, 2.2, 2.3, 2.4_

## 使用示例

```rust
use crate::database::{init_database, DbConnection};

// 初始化数据库
let db: DbConnection = init_database()?;

// 使用 DAO 操作数据
let conn = db.lock().unwrap();
let providers = ApiKeyProviderDao::get_all_providers(&conn)?;
```
