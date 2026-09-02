# Workspace 模块

Workspace 是 Lime 应用层的概念，用于组织和管理 AI Agent 的工作上下文。

## 概述

Workspace 是对 Agent 框架 `Session.working_dir` 的命名和配置包装，不修改 Agent 框架本身。

## 设计原则

- **读共享，写隔离** - 只读操作可以共享环境，写操作需要隔离
- **最小有效 context** - 只传递必要的 context，避免 context pollution
- **Workspace = 边界** - 文件系统边界 + context 边界 + 配置边界

## 文件索引

| 文件 | 说明 |
|------|------|
| `mod.rs` | 模块入口，导出公共类型 |
| `types.rs` | 类型定义（Workspace, WorkspaceSettings 等） |
| `manager.rs` | WorkspaceManager 实现 CRUD 操作 |

## 数据模型

```rust
pub struct Workspace {
    pub id: WorkspaceId,
    pub name: String,
    pub workspace_type: WorkspaceType,
    pub root_path: PathBuf,
    pub is_default: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub settings: WorkspaceSettings,
}
```

## 使用示例

```rust
use crate::workspace::WorkspaceManager;

let manager = WorkspaceManager::new(db);

// 创建 workspace
let ws = manager.create("my-project".to_string(), PathBuf::from("/path/to/project"))?;

// 设置为默认
manager.set_default(&ws.id)?;

// 列出所有 workspace
let list = manager.list()?;
```

## 相关文档

- [Workspace 设计文档](../../../../../internal/aiprompts/workspace.md)
