export type AgentToolSurfaceProfile = "core" | "workbench";

export type AgentToolCapability =
  | "planning"
  | "delegation"
  | "web_search"
  | "skill_execution"
  | "session_control"
  | "content_creation"
  | "workspace_io"
  | "execution"
  | "vision";

export type AgentToolLifecycle = "current" | "compat" | "deprecated";

export type AgentToolSourceKind =
  | "agent_builtin"
  | "lime_injected";

export type AgentToolPermissionPlane =
  | "session_allowlist"
  | "parameter_restricted"
  | "caller_filtered";

export type AgentToolExecutionWarningPolicy = "none" | "shell_command_risk";

export type AgentToolExecutionRestrictionProfile =
  | "none"
  | "workspace_path_required"
  | "workspace_path_optional"
  | "workspace_absolute_path_required"
  | "workspace_shell_command"
  | "analyze_image_input"
  | "safe_https_url_required";

export type AgentToolExecutionSandboxProfile = "none" | "workspace_command";
export type AgentToolExecutionPolicySource =
  | "default"
  | "persisted"
  | "runtime";

export type AgentRuntimeExtensionSourceKind =
  | "mcp_bridge"
  | "runtime_extension";

export type AgentRuntimeToolInventoryRuntimeSourceKind =
  | "current_surface"
  | "runtime_extension"
  | "mcp";

export interface AgentRuntimeToolInventoryRequest {
  caller?: string;
  workbench?: boolean;
  metadata?: Record<string, unknown>;
}

export interface AgentRuntimeListWorkspaceSkillBindingsRequest {
  workspaceRoot: string;
  caller?: string;
  workbench?: boolean;
}

export interface AgentRuntimeToolInventorySurface {
  workbench: boolean;
}

export interface AgentRuntimeWorkspaceSkillBindingRequest {
  workspace_root: string;
  caller: string;
  surface: AgentRuntimeToolInventorySurface;
}

export type AgentRuntimeWorkspaceSkillBindingStatus =
  | "ready_for_manual_enable"
  | "blocked";

export interface AgentRuntimeSkillBindingRegistration {
  registrationId?: string;
  registration_id?: string;
  registeredAt?: string;
  registered_at?: string;
  skillDirectory?: string;
  skill_directory?: string;
  registeredSkillDirectory?: string;
  registered_skill_directory?: string;
  sourceDraftId?: string;
  source_draft_id?: string;
  sourceVerificationReportId?: string | null;
  source_verification_report_id?: string | null;
  generatedFileCount?: number;
  generated_file_count?: number;
  permissionSummary?: string[];
  permission_summary?: string[];
}

export interface AgentRuntimeSkillBindingResourceSummary {
  hasScripts?: boolean;
  has_scripts?: boolean;
  hasReferences?: boolean;
  has_references?: boolean;
  hasAssets?: boolean;
  has_assets?: boolean;
}

export interface AgentRuntimeSkillBindingStandardCompliance {
  isStandard?: boolean;
  is_standard?: boolean;
  validationErrors?: string[];
  validation_errors?: string[];
  deprecatedFields?: string[];
  deprecated_fields?: string[];
}

export interface AgentRuntimeWorkspaceSkillBinding {
  key: string;
  name: string;
  description: string;
  directory: string;
  registered_skill_directory: string;
  registration: AgentRuntimeSkillBindingRegistration;
  permission_summary: string[];
  metadata: Record<string, string>;
  allowed_tools: string[];
  resource_summary: AgentRuntimeSkillBindingResourceSummary;
  standard_compliance: AgentRuntimeSkillBindingStandardCompliance;
  runtime_binding_target: string;
  binding_status: AgentRuntimeWorkspaceSkillBindingStatus;
  binding_status_reason: string;
  next_gate: string;
  query_loop_visible: boolean;
  tool_runtime_visible: boolean;
  launch_enabled: boolean;
  runtime_gate: string;
}

export interface AgentRuntimeWorkspaceSkillBindingCounts {
  registered_total: number;
  ready_for_manual_enable_total: number;
  blocked_total: number;
  query_loop_visible_total: number;
  tool_runtime_visible_total: number;
  launch_enabled_total: number;
}

export interface AgentRuntimeWorkspaceSkillBindings {
  request: AgentRuntimeWorkspaceSkillBindingRequest;
  warnings: string[];
  counts: AgentRuntimeWorkspaceSkillBindingCounts;
  bindings: AgentRuntimeWorkspaceSkillBinding[];
}

export interface AgentRuntimeToolInventoryCatalogEntry {
  name: string;
  profiles: AgentToolSurfaceProfile[];
  capabilities: AgentToolCapability[];
  lifecycle: AgentToolLifecycle;
  source: AgentToolSourceKind;
  permission_plane: AgentToolPermissionPlane;
  workspace_default_allow: boolean;
  execution_warning_policy: AgentToolExecutionWarningPolicy;
  execution_warning_policy_source: AgentToolExecutionPolicySource;
  execution_restriction_profile: AgentToolExecutionRestrictionProfile;
  execution_restriction_profile_source: AgentToolExecutionPolicySource;
  execution_sandbox_profile: AgentToolExecutionSandboxProfile;
  execution_sandbox_profile_source: AgentToolExecutionPolicySource;
}

export interface AgentRuntimeToolInventoryNativeEntry {
  name: string;
  description: string;
  catalog_entry_name?: string;
  catalog_source?: AgentToolSourceKind;
  catalog_lifecycle?: AgentToolLifecycle;
  catalog_permission_plane?: AgentToolPermissionPlane;
  catalog_workspace_default_allow?: boolean;
  catalog_execution_warning_policy?: AgentToolExecutionWarningPolicy;
  catalog_execution_warning_policy_source?: AgentToolExecutionPolicySource;
  catalog_execution_restriction_profile?: AgentToolExecutionRestrictionProfile;
  catalog_execution_restriction_profile_source?: AgentToolExecutionPolicySource;
  catalog_execution_sandbox_profile?: AgentToolExecutionSandboxProfile;
  catalog_execution_sandbox_profile_source?: AgentToolExecutionPolicySource;
  deferred_loading: boolean;
  always_visible: boolean;
  allowed_callers: string[];
  tags: string[];
  input_examples_count: number;
  has_output_schema: boolean;
  caller_allowed: boolean;
  visible_in_context: boolean;
}

export interface AgentRuntimeToolInventoryExtensionSurfaceEntry {
  extension_name: string;
  description: string;
  source_kind: AgentRuntimeExtensionSourceKind;
  deferred_loading: boolean;
  allowed_caller?: string;
  available_tools: string[];
  always_expose_tools: string[];
  loaded_tools: string[];
  searchable_tools: string[];
}

export interface AgentRuntimeToolInventoryExtensionToolEntry {
  name: string;
  description: string;
  extension_name?: string;
  source_kind: AgentRuntimeExtensionSourceKind;
  deferred_loading: boolean;
  allowed_caller?: string;
  status: string;
  caller_allowed: boolean;
  visible_in_context: boolean;
}

export interface AgentRuntimeToolInventoryRuntimeEntry {
  name: string;
  description: string;
  source_kind: AgentRuntimeToolInventoryRuntimeSourceKind;
  source_label?: string;
  status?: string;
  catalog_entry_name?: string;
  catalog_source?: AgentToolSourceKind;
  catalog_lifecycle?: AgentToolLifecycle;
  catalog_permission_plane?: AgentToolPermissionPlane;
  catalog_workspace_default_allow?: boolean;
  deferred_loading: boolean;
  always_visible: boolean;
  allowed_callers: string[];
  tags: string[];
  input_examples_count: number;
  has_output_schema: boolean;
  caller_allowed: boolean;
  visible_in_context: boolean;
}

export interface AgentRuntimeToolInventoryMcpEntry {
  server_name: string;
  name: string;
  description: string;
  deferred_loading: boolean;
  always_visible: boolean;
  allowed_callers: string[];
  tags: string[];
  input_examples_count: number;
  has_output_schema: boolean;
  caller_allowed: boolean;
  visible_in_context: boolean;
}

export interface AgentRuntimeToolInventoryCounts {
  catalog_total: number;
  catalog_current_total: number;
  catalog_compat_total: number;
  catalog_deprecated_total: number;
  default_allowed_total: number;
  runtime_total?: number;
  runtime_visible_total?: number;
  native_total: number;
  native_visible_total: number;
  native_catalog_unmapped_total: number;
  extension_surface_total: number;
  extension_mcp_bridge_total: number;
  extension_runtime_total: number;
  extension_tool_total: number;
  extension_tool_visible_total: number;
  mcp_server_total: number;
  mcp_tool_total: number;
  mcp_tool_visible_total: number;
}

export interface AgentRuntimeToolInventory {
  request: {
    caller: string;
    surface: AgentRuntimeToolInventorySurface;
  };
  agent_initialized: boolean;
  warnings: string[];
  mcp_servers: string[];
  default_allowed_tools: string[];
  counts: AgentRuntimeToolInventoryCounts;
  catalog_tools: AgentRuntimeToolInventoryCatalogEntry[];
  native_tools: AgentRuntimeToolInventoryNativeEntry[];
  runtime_tools?: AgentRuntimeToolInventoryRuntimeEntry[];
  extension_surfaces: AgentRuntimeToolInventoryExtensionSurfaceEntry[];
  extension_tools: AgentRuntimeToolInventoryExtensionToolEntry[];
  mcp_tools: AgentRuntimeToolInventoryMcpEntry[];
}
