use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Current Windows sandbox capability state exposed by the Desktop control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum WindowsSandboxReadiness {
    Ready,
    NotConfigured,
    UpdateRequired,
}

/// Empty parameters are accepted so the method can be called with `{}` by the
/// Desktop client while the ingress also accepts Codex's omitted-params form.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WindowsSandboxReadinessParams {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WindowsSandboxReadinessResponse {
    pub status: WindowsSandboxReadiness,
}
