import type {
  McpContent,
  McpPromptResult,
  McpResourceContent,
  McpResourceDefinition,
  McpResourceListResult,
  McpResourceTemplateDefinition,
  McpServer,
  McpServerOAuthLoginResponse,
  McpToolResult,
} from "./mcpTypes";

export function assertArrayField<T>(
  method: string,
  response: unknown,
  field: string,
): T[] {
  if (
    !response ||
    typeof response !== "object" ||
    !Array.isArray((response as Record<string, unknown>)[field])
  ) {
    throw new Error(`${method} did not return ${field}`);
  }
  return (response as Record<string, T[]>)[field];
}

function assertRecord(
  method: string,
  response: unknown,
  description: string,
): Record<string, unknown> {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    throw new Error(`${method} did not return ${description}`);
  }
  return response as Record<string, unknown>;
}

export function assertServerListResponse(
  method: string,
  response: unknown,
): void {
  assertArrayField<McpServer>(method, response, "servers");
}

export function assertLifecycleResponse(
  method: string,
  response: unknown,
): void {
  const record = assertRecord(method, response, "empty lifecycle result");
  if (Object.keys(record).length > 0) {
    throw new Error(`${method} did not return empty lifecycle result`);
  }
}

export function assertEmptyResponse(method: string, response: unknown): void {
  const record = assertRecord(method, response, "empty result");
  if (Object.keys(record).length > 0) {
    throw new Error(`${method} did not return empty result`);
  }
}

export function assertOAuthLoginResponse(
  method: string,
  response: unknown,
): McpServerOAuthLoginResponse {
  const record = assertRecord(method, response, "OAuth login response");
  if (
    typeof record.authorizationUrl !== "string" ||
    typeof record.state !== "string"
  ) {
    throw new Error(`${method} did not return OAuth login response`);
  }
  return response as McpServerOAuthLoginResponse;
}

function isMcpContent(value: unknown): value is McpContent {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const record = value as Record<string, unknown>;
  if (record.type === "text") {
    return typeof record.text === "string";
  }
  if (record.type === "image") {
    return (
      typeof record.data === "string" && typeof record.mime_type === "string"
    );
  }
  if (record.type === "resource") {
    return (
      typeof record.uri === "string" &&
      (record.text === undefined || typeof record.text === "string") &&
      (record.blob === undefined || typeof record.blob === "string")
    );
  }
  return false;
}

export function assertMcpToolResult(
  method: string,
  response: unknown,
): McpToolResult {
  const record = assertRecord(method, response, "tool result");
  if (
    !Array.isArray(record.content) ||
    typeof record.is_error !== "boolean" ||
    !record.content.every(isMcpContent)
  ) {
    throw new Error(`${method} did not return tool result`);
  }
  return response as McpToolResult;
}

export function assertMcpPromptResult(
  method: string,
  response: unknown,
): McpPromptResult {
  const record = assertRecord(method, response, "prompt result");
  const hasValidDescription =
    record.description === undefined || typeof record.description === "string";
  const hasValidMessages =
    Array.isArray(record.messages) &&
    record.messages.every((message) => {
      if (!message || typeof message !== "object" || Array.isArray(message)) {
        return false;
      }
      const messageRecord = message as Record<string, unknown>;
      return (
        typeof messageRecord.role === "string" &&
        isMcpContent(messageRecord.content)
      );
    });
  if (!hasValidDescription || !hasValidMessages) {
    throw new Error(`${method} did not return prompt result`);
  }
  return response as McpPromptResult;
}

export function assertMcpResourceContent(
  method: string,
  response: unknown,
): McpResourceContent {
  const record = assertRecord(method, response, "resource content");
  if (
    typeof record.uri !== "string" ||
    (record.mime_type !== undefined && typeof record.mime_type !== "string") ||
    (record.text !== undefined && typeof record.text !== "string") ||
    (record.blob !== undefined && typeof record.blob !== "string") ||
    (record.meta !== undefined &&
      (!record.meta ||
        typeof record.meta !== "object" ||
        Array.isArray(record.meta)))
  ) {
    throw new Error(`${method} did not return resource content`);
  }
  return response as McpResourceContent;
}

export function assertMcpResourceListResponse(
  method: string,
  response: unknown,
): McpResourceListResult {
  const resources = assertArrayField<McpResourceDefinition>(
    method,
    response,
    "resources",
  );
  const record = response as Record<string, unknown>;
  const resourceTemplates = record.resourceTemplates;
  if (resourceTemplates !== undefined && !Array.isArray(resourceTemplates)) {
    throw new Error(`${method} did not return resourceTemplates`);
  }
  return {
    resources,
    resourceTemplates: (resourceTemplates ?? []) as McpResourceTemplateDefinition[],
  };
}
