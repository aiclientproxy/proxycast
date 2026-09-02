//! Lime Core Crate
//!
//! 包含纯数据类型、基础模块和无外部业务依赖的独立模块。
//!
//! ## 模块结构
//! - `app_utils`: 应用通用工具函数
//! - `models`: 核心数据模型定义
//! - `data`: 静态数据
//! - `logger`: 日志配置
//! - `errors`: 错误类型定义
//! - `backends`: 后端调用层 Trait
//! - `config`: 配置管理（类型、YAML、热重载、导入导出）
//! - `connect`: Deep Link 协议和中转商注册表
//! - `middleware`: HTTP 中间件（认证、限速）
//! - `session`: 会话管理（限速、粘性路由）
//! - `session_files`: 会话文件存储

pub mod api_host_utils;
pub mod app_bootstrap;
pub mod app_paths;
pub mod app_utils;
pub mod data;
pub mod env_compat;
pub mod image_generation_matcher;
pub mod logger;
pub mod models;
pub mod tray_format;
pub mod tray_menu_meta;
pub mod tray_state;

// 独立业务模块（无主 crate 依赖）
pub mod backends;
pub mod config;
pub mod connect;
pub mod errors;
pub mod middleware;
pub mod session;
pub mod session_files;
pub mod tool_calling;

// 类型模块（纯数据类型，供 database 等模块使用）
pub mod agent;
pub mod general_chat;
pub mod plugin_runtime_token;

// 路由系统
pub mod router;

// 请求处理器核心类型（context, error）
pub mod processor;
pub mod provider_prompt_cache_support;

// WebSocket 核心类型
pub mod websocket;

// 事件发射抽象（供独立 crate 解耦桌面宿主依赖）
pub mod event_emit;

// 网络工具
pub mod network;

// 凭证清理（敏感信息过滤）
pub mod sanitizer;

// 数据层
pub mod content;
pub mod database;
pub mod memory;
pub mod workspace;

// 重新导出常用类型
pub use event_emit::{DynEmitter, EventEmit, NoOpEmitter};
pub use logger::{LogEntry, LogStore, LogStoreConfig, SharedLogStore};
pub use models::provider_type::ProviderType;
pub use models::*;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
