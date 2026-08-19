mod automation;
mod connect;
mod diagnostics;
mod gateway;
mod knowledge;
mod mcp;
mod media;
mod memory;
mod model_providers;
mod plugins;
mod right_surface;
mod sessions;
mod skills;
mod usage_stats;
mod voice;
mod workspaces;

pub use automation::AutomationManagementAppDataSource;
pub use automation::AutomationOverviewAppDataSource;
pub use connect::ConnectAppDataSource;
pub use diagnostics::DiagnosticsAppDataSource;
pub use gateway::GatewayAppDataSource;
pub use knowledge::KnowledgeAppDataSource;
pub use mcp::McpAppDataSource;
pub use media::MediaAppDataSource;
pub use memory::MemoryAppDataSource;
pub use model_providers::{ModelCatalogQuery, ModelProviderAppDataSource, ProviderModelCatalog};
pub use plugins::{PluginDataSource, PluginTurnSnapshot};
pub use right_surface::RightSurfaceAppDataSource;
pub use sessions::SessionAppDataSource;
pub use skills::SkillAppDataSource;
pub use skills::WorkspaceSkillBindingAppDataSource;
pub use usage_stats::UsageStatsAppDataSource;
pub use voice::VoiceAppDataSource;
pub use workspaces::WorkspaceAppDataSource;

use super::RuntimeCoreError;

pub trait AppDataSource:
    SessionAppDataSource
    + WorkspaceAppDataSource
    + SkillAppDataSource
    + WorkspaceSkillBindingAppDataSource
    + GatewayAppDataSource
    + MediaAppDataSource
    + VoiceAppDataSource
    + PluginDataSource
    + KnowledgeAppDataSource
    + AutomationOverviewAppDataSource
    + McpAppDataSource
    + AutomationManagementAppDataSource
    + MemoryAppDataSource
    + DiagnosticsAppDataSource
    + UsageStatsAppDataSource
    + ModelProviderAppDataSource
    + ConnectAppDataSource
    + RightSurfaceAppDataSource
    + Send
    + Sync
{
}

impl<T> AppDataSource for T where
    T: SessionAppDataSource
        + WorkspaceAppDataSource
        + SkillAppDataSource
        + WorkspaceSkillBindingAppDataSource
        + GatewayAppDataSource
        + MediaAppDataSource
        + VoiceAppDataSource
        + PluginDataSource
        + KnowledgeAppDataSource
        + AutomationOverviewAppDataSource
        + McpAppDataSource
        + AutomationManagementAppDataSource
        + MemoryAppDataSource
        + DiagnosticsAppDataSource
        + UsageStatsAppDataSource
        + ModelProviderAppDataSource
        + ConnectAppDataSource
        + RightSurfaceAppDataSource
        + Send
        + Sync
{
}

#[derive(Debug, Default)]
pub struct NoopAppDataSource;

pub(super) fn unavailable(operation: &str) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!(
        "{operation} is not available without an app data source"
    ))
}

pub(super) fn requires_current(operation: &str) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!("{operation} requires a current app data source"))
}
