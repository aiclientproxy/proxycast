//! Lime Services Crate
//!
//! 业务服务层，包含所有不依赖 Tauri 的业务逻辑。
//!
//! ## 模块结构
//! - `context_memory_service` - 上下文记忆服务
//! - `sysinfo_service` - 系统信息服务
//! - `update_check_service` - 更新检查服务

#![allow(clippy::type_complexity)]
#![allow(clippy::let_underscore_future)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::await_holding_lock)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_match)]
//! - `voice_config_service` - 语音配置服务
//! - `voice_processor_service` - 语音润色服务
//! - `voice_output_service` - 语音输出服务
//! - `voice_asr_service` - ASR 识别服务
//! - `voice_command_service` - 语音命令业务服务
//! - `voice_recording_service` - 录音状态与设备服务
//! - `machine_id_service` - 机器 ID 服务
//! - `live_sync` - 实时同步
//! - `mcp_sync` - MCP 同步
//! - `prompt_sync` - Prompt 同步
//! - `skill_service` - 技能服务
//! - `backup_service` - 备份服务
//! - `material_service` - 素材服务
//! - `persona_service` - 人设服务
//! - `model_registry_service` - 模型注册服务
//! - `prompt_service` - Prompt 服务
//! - `mcp_service` - MCP 服务
//! - `session_context_service` - 会话上下文服务
//! - `ai_summary_service` - AI 摘要服务
//! - `project_context_builder` - 项目上下文构建器
//! - `api_key_provider_service` - API Key Provider 服务

// 无外部依赖的服务
pub mod context_memory_service;
pub mod project_git_service;
pub mod sysinfo_service;
pub mod update_check_service;
pub mod voice_asr_service;
pub mod voice_command_service;
pub mod voice_config_service;
pub mod voice_output_service;
pub mod voice_processor_service;
pub mod voice_recording_service;

// 依赖 models 的服务
pub mod live_sync;
pub mod machine_id_service;
pub mod mcp_sync;
pub mod prompt_sync;
pub mod skill_service;

// 依赖 database + models 的服务
pub mod backup_service;
pub mod material_service;
pub mod mcp_service;
pub mod model_registry_service;
pub mod persona_service;
pub mod prompt_service;
pub mod usage_statistics_service;
// 依赖其他 services 的服务
pub mod ai_summary_service;
pub mod project_context_builder;
pub mod session_context_service;

// 依赖 providers 的服务
pub mod api_key_provider_service;
pub mod provider_type_mapping;
