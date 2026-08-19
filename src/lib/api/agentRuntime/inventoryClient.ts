import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
  METHOD_WORKSPACE_SKILL_BINDINGS_LIST,
} from "../../../../packages/app-server-client/src/protocol";
import type {
  AgentRuntimeListWorkspaceSkillBindingsRequest,
  AgentRuntimeToolInventory,
  AgentRuntimeToolInventoryRequest,
  AgentRuntimeWorkspaceSkillBindings,
} from "./toolInventoryTypes";

export type AgentRuntimeWorkspaceSkillBindingsAppServerClient = Pick<
  AppServerClient,
  "request"
>;

type AppServerWorkspaceSkillBindingsListResponse = {
  bindings: AgentRuntimeWorkspaceSkillBindings;
};

type AppServerToolInventoryReadResponse = {
  inventory: unknown;
};

export interface AgentRuntimeInventoryClientDeps {
  appServerClient?: AgentRuntimeWorkspaceSkillBindingsAppServerClient;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function isOptionalString(value: unknown): value is string | undefined {
  return value === undefined || typeof value === "string";
}

function isOptionalBoolean(value: unknown): value is boolean | undefined {
  return value === undefined || typeof value === "boolean";
}

function isOptionalFiniteNumber(value: unknown): value is number | undefined {
  return (
    value === undefined || (typeof value === "number" && Number.isFinite(value))
  );
}

function hasFiniteNumberField(
  value: Record<string, unknown>,
  field: string,
): boolean {
  return typeof value[field] === "number" && Number.isFinite(value[field]);
}

function isToolInventorySurface(value: unknown): boolean {
  return isRecord(value) && typeof value.workbench === "boolean";
}

function isToolInventoryRequest(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.caller === "string" &&
    isToolInventorySurface(value.surface)
  );
}

const REQUIRED_TOOL_INVENTORY_COUNT_FIELDS = [
  "catalog_total",
  "catalog_current_total",
  "catalog_compat_total",
  "catalog_deprecated_total",
  "default_allowed_total",
  "native_total",
  "native_visible_total",
  "native_catalog_unmapped_total",
  "extension_surface_total",
  "extension_mcp_bridge_total",
  "extension_runtime_total",
  "extension_tool_total",
  "extension_tool_visible_total",
  "mcp_server_total",
  "mcp_tool_total",
  "mcp_tool_visible_total",
];

function isToolInventoryCounts(value: unknown): boolean {
  return (
    isRecord(value) &&
    REQUIRED_TOOL_INVENTORY_COUNT_FIELDS.every((field) =>
      hasFiniteNumberField(value, field),
    ) &&
    isOptionalFiniteNumber(value.runtime_total) &&
    isOptionalFiniteNumber(value.runtime_visible_total)
  );
}

function isCatalogToolEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    Array.isArray(value.profiles) &&
    Array.isArray(value.capabilities) &&
    typeof value.lifecycle === "string" &&
    typeof value.source === "string" &&
    typeof value.permission_plane === "string" &&
    typeof value.workspace_default_allow === "boolean" &&
    typeof value.execution_warning_policy === "string" &&
    typeof value.execution_warning_policy_source === "string" &&
    typeof value.execution_restriction_profile === "string" &&
    typeof value.execution_restriction_profile_source === "string" &&
    typeof value.execution_sandbox_profile === "string" &&
    typeof value.execution_sandbox_profile_source === "string"
  );
}

function isNativeToolEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    typeof value.description === "string" &&
    isOptionalString(value.catalog_entry_name) &&
    isOptionalString(value.catalog_source) &&
    isOptionalString(value.catalog_lifecycle) &&
    isOptionalString(value.catalog_permission_plane) &&
    isOptionalBoolean(value.catalog_workspace_default_allow) &&
    isOptionalString(value.catalog_execution_warning_policy) &&
    isOptionalString(value.catalog_execution_warning_policy_source) &&
    isOptionalString(value.catalog_execution_restriction_profile) &&
    isOptionalString(value.catalog_execution_restriction_profile_source) &&
    isOptionalString(value.catalog_execution_sandbox_profile) &&
    isOptionalString(value.catalog_execution_sandbox_profile_source) &&
    typeof value.deferred_loading === "boolean" &&
    typeof value.always_visible === "boolean" &&
    isStringArray(value.allowed_callers) &&
    isStringArray(value.tags) &&
    hasFiniteNumberField(value, "input_examples_count") &&
    typeof value.has_output_schema === "boolean" &&
    typeof value.caller_allowed === "boolean" &&
    typeof value.visible_in_context === "boolean"
  );
}

function isExtensionSurfaceEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.extension_name === "string" &&
    typeof value.description === "string" &&
    typeof value.source_kind === "string" &&
    typeof value.deferred_loading === "boolean" &&
    isOptionalString(value.allowed_caller) &&
    isStringArray(value.available_tools) &&
    isStringArray(value.always_expose_tools) &&
    isStringArray(value.loaded_tools) &&
    isStringArray(value.searchable_tools)
  );
}

function isExtensionToolEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    typeof value.description === "string" &&
    isOptionalString(value.extension_name) &&
    typeof value.source_kind === "string" &&
    typeof value.deferred_loading === "boolean" &&
    isOptionalString(value.allowed_caller) &&
    typeof value.status === "string" &&
    typeof value.caller_allowed === "boolean" &&
    typeof value.visible_in_context === "boolean"
  );
}

function isRuntimeToolEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    typeof value.description === "string" &&
    typeof value.source_kind === "string" &&
    isOptionalString(value.source_label) &&
    isOptionalString(value.status) &&
    isOptionalString(value.catalog_entry_name) &&
    isOptionalString(value.catalog_source) &&
    isOptionalString(value.catalog_lifecycle) &&
    isOptionalString(value.catalog_permission_plane) &&
    isOptionalBoolean(value.catalog_workspace_default_allow) &&
    typeof value.deferred_loading === "boolean" &&
    typeof value.always_visible === "boolean" &&
    isStringArray(value.allowed_callers) &&
    isStringArray(value.tags) &&
    hasFiniteNumberField(value, "input_examples_count") &&
    typeof value.has_output_schema === "boolean" &&
    typeof value.caller_allowed === "boolean" &&
    typeof value.visible_in_context === "boolean"
  );
}

function isMcpToolEntry(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.server_name === "string" &&
    typeof value.name === "string" &&
    typeof value.description === "string" &&
    typeof value.deferred_loading === "boolean" &&
    typeof value.always_visible === "boolean" &&
    isStringArray(value.allowed_callers) &&
    isStringArray(value.tags) &&
    hasFiniteNumberField(value, "input_examples_count") &&
    typeof value.has_output_schema === "boolean" &&
    typeof value.caller_allowed === "boolean" &&
    typeof value.visible_in_context === "boolean"
  );
}

function isArrayOf(
  value: unknown,
  predicate: (item: unknown) => boolean,
): boolean {
  return Array.isArray(value) && value.every((item) => predicate(item));
}

function assertToolInventoryShape(
  value: unknown,
): asserts value is AgentRuntimeToolInventory {
  if (
    !isRecord(value) ||
    !isToolInventoryRequest(value.request) ||
    typeof value.agent_initialized !== "boolean" ||
    !isStringArray(value.warnings) ||
    !isStringArray(value.mcp_servers) ||
    !isStringArray(value.default_allowed_tools) ||
    !isToolInventoryCounts(value.counts) ||
    !isArrayOf(value.catalog_tools, isCatalogToolEntry) ||
    !isArrayOf(value.native_tools, isNativeToolEntry) ||
    (value.runtime_tools !== undefined &&
      !isArrayOf(value.runtime_tools, isRuntimeToolEntry)) ||
    !isArrayOf(value.extension_surfaces, isExtensionSurfaceEntry) ||
    !isArrayOf(value.extension_tools, isExtensionToolEntry) ||
    !isArrayOf(value.mcp_tools, isMcpToolEntry)
  ) {
    throw new Error(
      "App Server agentSession/toolInventory/read did not return tool inventory",
    );
  }
}

export function createInventoryClient({
  appServerClient = new AppServerClient(),
}: AgentRuntimeInventoryClientDeps = {}) {
  async function getAgentRuntimeToolInventory(
    request: AgentRuntimeToolInventoryRequest = {},
  ): Promise<AgentRuntimeToolInventory> {
    const response =
      await appServerClient.request<AppServerToolInventoryReadResponse>(
        METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
        toolInventoryParamsFromRequest(request),
      );
    const result = response.result.inventory;
    assertToolInventoryShape(result);
    return result;
  }

  async function listWorkspaceSkillBindings(
    request: AgentRuntimeListWorkspaceSkillBindingsRequest,
  ): Promise<AgentRuntimeWorkspaceSkillBindings> {
    const workspaceRoot = request.workspaceRoot.trim();
    if (!workspaceRoot) {
      throw new Error(
        "workspaceRoot is required to list App Server workspace skill bindings",
      );
    }

    const response =
      await appServerClient.request<AppServerWorkspaceSkillBindingsListResponse>(
        METHOD_WORKSPACE_SKILL_BINDINGS_LIST,
        workspaceSkillBindingsListParamsFromRequest(request, workspaceRoot),
      );
    if (!response.result.bindings) {
      throw new Error(
        "App Server workspaceSkillBindings/list did not return bindings",
      );
    }
    return response.result.bindings;
  }

  return {
    getAgentRuntimeToolInventory,
    listWorkspaceSkillBindings,
  };
}

export const { getAgentRuntimeToolInventory, listWorkspaceSkillBindings } =
  createInventoryClient();

function workspaceSkillBindingsListParamsFromRequest(
  request: AgentRuntimeListWorkspaceSkillBindingsRequest,
  workspaceRoot: string,
) {
  return {
    workspaceRoot,
    ...(request.caller ? { caller: request.caller } : {}),
    ...(request.workbench === undefined
      ? {}
      : { workbench: request.workbench }),
  };
}

function toolInventoryParamsFromRequest(
  request: AgentRuntimeToolInventoryRequest,
) {
  return {
    ...(request.caller ? { caller: request.caller } : {}),
    ...(request.workbench === undefined
      ? {}
      : { workbench: request.workbench }),
    ...(request.metadata === undefined ? {} : { metadata: request.metadata }),
  };
}
