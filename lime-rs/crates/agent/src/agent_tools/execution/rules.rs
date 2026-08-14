use super::policy::{
    ToolExecutionPolicy, ToolExecutionRestrictionProfile, ToolExecutionSandboxProfile,
    ToolExecutionWarningPolicy,
};
use crate::agent_tools::catalog::{tool_catalog_entry, APPLY_PATCH_TOOL_NAME};
pub use tool_runtime::execution_rules::{
    NetworkRule, NetworkRuleMatch, ShellCommandRule, ShellCommandRuleMatch,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolPolicyRule {
    pub tool_names: &'static [&'static str],
    pub policy: ToolExecutionPolicy,
}

const DEFAULT_POLICY: ToolExecutionPolicy = ToolExecutionPolicy {
    warning_policy: ToolExecutionWarningPolicy::None,
    restriction_profile: ToolExecutionRestrictionProfile::None,
    sandbox_profile: ToolExecutionSandboxProfile::None,
};

const WORKSPACE_PATH_POLICY: ToolExecutionPolicy = ToolExecutionPolicy {
    restriction_profile: ToolExecutionRestrictionProfile::WorkspacePathRequired,
    ..DEFAULT_POLICY
};

const WORKSPACE_PATH_OPTIONAL_POLICY: ToolExecutionPolicy = ToolExecutionPolicy {
    restriction_profile: ToolExecutionRestrictionProfile::WorkspacePathOptional,
    ..DEFAULT_POLICY
};

const WORKSPACE_SHELL_POLICY: ToolExecutionPolicy = ToolExecutionPolicy {
    warning_policy: ToolExecutionWarningPolicy::ShellCommandRisk,
    restriction_profile: ToolExecutionRestrictionProfile::WorkspaceShellCommand,
    sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
};

const SAFE_HTTPS_URL_POLICY: ToolExecutionPolicy = ToolExecutionPolicy {
    restriction_profile: ToolExecutionRestrictionProfile::SafeHttpsUrlRequired,
    ..DEFAULT_POLICY
};

const WORKSPACE_PATH_TOOLS: &[&str] = &["Read", "view_image"];
const WORKSPACE_SANDBOX_TOOLS: &[&str] = &[APPLY_PATCH_TOOL_NAME];
const WORKSPACE_PATH_OPTIONAL_TOOLS: &[&str] = &["Glob", "Grep"];
const WORKSPACE_SHELL_TOOLS: &[&str] = &["exec_command"];
const SAFE_HTTPS_URL_TOOLS: &[&str] = &["WebFetch"];

const TOOL_POLICY_RULES: &[ToolPolicyRule] = &[
    ToolPolicyRule {
        tool_names: WORKSPACE_PATH_TOOLS,
        policy: WORKSPACE_PATH_POLICY,
    },
    ToolPolicyRule {
        tool_names: WORKSPACE_SANDBOX_TOOLS,
        policy: ToolExecutionPolicy {
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            ..DEFAULT_POLICY
        },
    },
    ToolPolicyRule {
        tool_names: WORKSPACE_PATH_OPTIONAL_TOOLS,
        policy: WORKSPACE_PATH_OPTIONAL_POLICY,
    },
    ToolPolicyRule {
        tool_names: WORKSPACE_SHELL_TOOLS,
        policy: WORKSPACE_SHELL_POLICY,
    },
    ToolPolicyRule {
        tool_names: SAFE_HTTPS_URL_TOOLS,
        policy: SAFE_HTTPS_URL_POLICY,
    },
];

pub(crate) fn default_tool_execution_policy(tool_name: &str) -> ToolExecutionPolicy {
    let Some(catalog_entry) = tool_catalog_entry(tool_name.trim()) else {
        return DEFAULT_POLICY;
    };

    TOOL_POLICY_RULES
        .iter()
        .find(|rule| rule.tool_names.contains(&catalog_entry.name))
        .map(|rule| rule.policy)
        .unwrap_or(DEFAULT_POLICY)
}

#[cfg(test)]
pub(crate) fn tool_policy_rules() -> &'static [ToolPolicyRule] {
    TOOL_POLICY_RULES
}
