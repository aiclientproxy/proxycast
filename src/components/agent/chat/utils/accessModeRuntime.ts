import type {
  AgentApprovalPolicy,
  AgentSandboxPolicy,
  AgentSessionExecutionRuntime,
  AgentSessionExecutionRuntimeAccessMode,
} from "@/lib/api/agentExecutionRuntime";
import {
  normalizeAccessMode,
  type AgentAccessMode,
} from "../hooks/agentChatStorage";

export interface AgentAccessRuntimePolicies {
  approvalPolicy: AgentApprovalPolicy;
  sandboxPolicy: AgentSandboxPolicy;
}

export type BuiltinPermissionProfileId =
  | ":read-only"
  | ":workspace"
  | ":danger-full-access";

export function permissionProfileIdFromAccessMode(
  accessMode?: AgentAccessMode | null,
): BuiltinPermissionProfileId {
  switch (normalizeAccessMode(accessMode)) {
    case "read-only":
      return ":read-only";
    case "full-access":
      return ":danger-full-access";
    case "current":
    default:
      return ":workspace";
  }
}

export function createRuntimePoliciesFromAccessMode(
  accessMode?: AgentAccessMode | null,
): AgentAccessRuntimePolicies {
  const resolvedAccessMode = normalizeAccessMode(accessMode);
  switch (resolvedAccessMode) {
    case "read-only":
      return {
        approvalPolicy: "on-request",
        sandboxPolicy: "read-only",
      };
    case "full-access":
      return {
        approvalPolicy: "never",
        sandboxPolicy: "danger-full-access",
      };
    case "current":
    default:
      return {
        approvalPolicy: "on-request",
        sandboxPolicy: "workspace-write",
      };
  }
}

export function createAccessModeFromRuntimePolicies(
  approvalPolicy?: AgentApprovalPolicy | string | null,
  sandboxPolicy?: AgentSandboxPolicy | string | null,
): AgentAccessMode | null {
  const normalizedApprovalPolicy =
    typeof approvalPolicy === "string" ? approvalPolicy.trim() : "";
  const normalizedSandboxPolicy =
    typeof sandboxPolicy === "string" ? sandboxPolicy.trim() : "";

  switch (normalizedSandboxPolicy) {
    case "read-only":
      return !normalizedApprovalPolicy ||
        normalizedApprovalPolicy === "on-request"
        ? "read-only"
        : null;
    case "workspace-write":
      return !normalizedApprovalPolicy ||
        normalizedApprovalPolicy === "on-request"
        ? "current"
        : null;
    case "danger-full-access":
      return !normalizedApprovalPolicy || normalizedApprovalPolicy === "never"
        ? "full-access"
        : null;
    default:
      return null;
  }
}

function normalizeExecutionRuntimeAccessMode(
  value?: AgentSessionExecutionRuntimeAccessMode | string | null,
): AgentAccessMode | null {
  if (value === "read-only" || value === "current" || value === "full-access") {
    return value;
  }
  return null;
}

export function createAccessModeFromExecutionRuntime(
  runtime?: Pick<AgentSessionExecutionRuntime, "recent_access_mode"> | null,
): AgentAccessMode | null {
  return normalizeExecutionRuntimeAccessMode(runtime?.recent_access_mode);
}
