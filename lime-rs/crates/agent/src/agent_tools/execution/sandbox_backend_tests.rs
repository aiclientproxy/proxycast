use super::{
    decide_tool_execution, plan_sandbox_backend, SandboxBackend, SandboxBackendPlanInput,
    SandboxBackendPlatform, SandboxBackendStatus, ToolExecutionDecisionInput,
    ToolExecutionDecisionKind, ToolExecutionResolverInput, ToolExecutionSandboxProfile,
};
use serde_json::json;
use std::path::Path;

#[test]
fn test_sandbox_backend_plan_reports_platform_backend_capability() {
    let metadata = json!({
        "workspaceSandbox": {
            "enabled": true,
            "strict": false
        }
    });

    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some("workspace-write"),
        request_metadata: Some(&metadata),
        bypass_restrictions: false,
        platform: SandboxBackendPlatform::Macos,
    });

    assert_eq!(plan.backend, SandboxBackend::Seatbelt);
    if matches!(plan.status, SandboxBackendStatus::Ready) {
        assert_eq!(plan.status, SandboxBackendStatus::Ready);
        assert!(plan.enforced);
        assert_eq!(plan.reason_code, "sandbox_backend_ready");
    } else {
        assert_eq!(plan.status, SandboxBackendStatus::Unavailable);
        assert!(!plan.enforced);
    }
    assert!(plan.required);
    assert!(plan.config.enabled);
    assert!(!plan.config.strict);
    assert_eq!(plan.config.source.label(), "request");
}

#[test]
fn test_sandbox_backend_plan_maps_supported_platforms() {
    let metadata = json!({ "workspaceSandbox": { "enabled": true } });

    let cases = [
        (SandboxBackendPlatform::Macos, SandboxBackend::Seatbelt),
        (SandboxBackendPlatform::Linux, SandboxBackend::LinuxSandbox),
        (
            SandboxBackendPlatform::Windows,
            SandboxBackend::RestrictedToken,
        ),
    ];

    for (platform, expected_backend) in cases {
        let plan = plan_sandbox_backend(SandboxBackendPlanInput {
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            requested_policy: Some("read-only"),
            request_metadata: Some(&metadata),
            bypass_restrictions: false,
            platform,
        });

        assert_eq!(plan.backend, expected_backend);
        if plan.status == SandboxBackendStatus::Ready {
            assert_eq!(plan.status, SandboxBackendStatus::Ready);
            assert!(plan.enforced);
        } else {
            assert_eq!(plan.status, SandboxBackendStatus::Unavailable);
            assert!(!plan.enforced);
        }
    }
}

#[test]
fn test_windows_restricted_token_backend_is_ready_after_platform_evidence() {
    let metadata = json!({ "workspaceSandbox": { "enabled": true } });
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some("workspace-write"),
        request_metadata: Some(&metadata),
        bypass_restrictions: false,
        platform: SandboxBackendPlatform::Windows,
    });

    assert_eq!(plan.backend, SandboxBackend::RestrictedToken);
    assert_eq!(plan.status, SandboxBackendStatus::Ready);
    assert!(plan.enforced);
    assert_eq!(plan.reason_code, "sandbox_backend_ready");
    assert_eq!(
        plan.reason,
        "Windows restricted token backend 可用于当前 shell 工具执行"
    );
}

#[test]
fn test_sandbox_backend_plan_blocks_strict_fallback_when_backend_is_not_enforced() {
    let metadata = json!({
        "workspaceSandbox": {
            "enabled": true,
            "strict": true
        }
    });

    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some("workspace-write"),
        request_metadata: Some(&metadata),
        bypass_restrictions: false,
        platform: SandboxBackendPlatform::Unsupported,
    });

    assert!(plan.strict_fallback_blocks_execution());
    assert_eq!(plan.status, SandboxBackendStatus::Unavailable);
    assert!(!plan.enforced);
}

#[test]
fn test_decide_tool_execution_adds_strict_workspace_sandbox_metadata() {
    let metadata = json!({
        "harness": {
            "workspaceSandbox": {
                "enabled": true,
                "strict": true
            }
        }
    });
    let params = json!({ "command": "pwd" });

    let decision = decide_tool_execution(ToolExecutionDecisionInput {
        tool_name: "exec_command",
        params: &params,
        working_directory: Path::new("/tmp/workspace"),
        surface: "runtime_tool",
        auto_mode: false,
        bypass_restrictions: false,
        approval_policy: Some("never"),
        requested_sandbox_policy: Some("workspace-write"),
        resolver_input: ToolExecutionResolverInput {
            persisted_policy: None,
            request_metadata: Some(&metadata),
        },
    });

    if decision
        .metadata
        .get("sandboxBackendEnforced")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        assert_eq!(decision.kind, ToolExecutionDecisionKind::Allow);
        assert_eq!(
            decision.metadata.get("sandboxBackendStatus"),
            Some(&json!("ready"))
        );
        assert_eq!(
            decision.metadata.get("sandboxBackendEnforced"),
            Some(&json!(true))
        );
    } else {
        assert_eq!(decision.kind, ToolExecutionDecisionKind::SandboxBlocked);
        assert_eq!(
            decision.reason_code,
            "workspace_sandbox_strict_backend_unavailable"
        );
        assert_eq!(
            decision.metadata.get("sandboxBackendEnforced"),
            Some(&json!(false))
        );
    }
    assert_eq!(
        decision.metadata.get("workspaceSandboxEnabled"),
        Some(&json!(true))
    );
    assert_eq!(
        decision.metadata.get("workspaceSandboxStrict"),
        Some(&json!(true))
    );
    assert_eq!(
        decision.metadata.get("workspaceSandboxConfigSource"),
        Some(&json!("runtime"))
    );
    if decision.kind == ToolExecutionDecisionKind::SandboxBlocked {
        assert_eq!(
            decision.metadata.get("sandboxPolicy"),
            Some(&json!("workspace-write"))
        );
    }
}

#[test]
fn test_request_workspace_sandbox_can_disable_persisted_strict_fallback() {
    let metadata = json!({
        "config": {
            "agent": {
                "workspaceSandbox": {
                    "enabled": true,
                    "strict": true
                }
            }
        },
        "workspaceSandbox": {
            "enabled": false,
            "strict": false
        }
    });
    let params = json!({ "command": "pwd" });

    let decision = decide_tool_execution(ToolExecutionDecisionInput {
        tool_name: "exec_command",
        params: &params,
        working_directory: Path::new("/tmp/workspace"),
        surface: "runtime_tool",
        auto_mode: false,
        bypass_restrictions: false,
        approval_policy: Some("never"),
        requested_sandbox_policy: Some("workspace-write"),
        resolver_input: ToolExecutionResolverInput {
            persisted_policy: None,
            request_metadata: Some(&metadata),
        },
    });

    assert_eq!(decision.kind, ToolExecutionDecisionKind::Allow);
    assert_eq!(
        decision.metadata.get("workspaceSandboxEnabled"),
        Some(&json!(false))
    );
    assert_eq!(
        decision.metadata.get("workspaceSandboxStrict"),
        Some(&json!(false))
    );
    assert_eq!(
        decision.metadata.get("workspaceSandboxConfigSource"),
        Some(&json!("request"))
    );
    assert_eq!(
        decision.metadata.get("sandboxBackendStatus"),
        Some(&json!("disabled"))
    );
}

#[test]
fn test_danger_full_access_does_not_require_workspace_sandbox_backend() {
    let metadata = json!({
        "workspaceSandbox": {
            "enabled": true,
            "strict": true
        }
    });
    let params = json!({ "command": "cargo test" });

    let decision = decide_tool_execution(ToolExecutionDecisionInput {
        tool_name: "exec_command",
        params: &params,
        working_directory: Path::new("/tmp/workspace"),
        surface: "runtime_tool",
        auto_mode: false,
        bypass_restrictions: false,
        approval_policy: Some("never"),
        requested_sandbox_policy: Some("danger-full-access"),
        resolver_input: ToolExecutionResolverInput {
            persisted_policy: None,
            request_metadata: Some(&metadata),
        },
    });

    assert_eq!(decision.kind, ToolExecutionDecisionKind::Allow);
    assert_eq!(
        decision.metadata.get("sandboxBackendRequired"),
        Some(&json!(false))
    );
    assert_eq!(
        decision.metadata.get("sandboxBackendStatus"),
        Some(&json!("not_required"))
    );
}
