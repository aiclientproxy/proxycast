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

function lowerCodexMcpContent(value: unknown): McpContent | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  if (record.type === "text" && typeof record.text === "string") {
    return { type: "text", text: record.text };
  }
  if (
    record.type === "image" &&
    typeof record.data === "string" &&
    (typeof record.mimeType === "string" ||
      typeof record.mime_type === "string")
  ) {
    return {
      type: "image",
      data: record.data,
      mime_type: (record.mimeType ?? record.mime_type) as string,
    };
  }
  if (record.type === "resource" && typeof record.uri === "string") {
    const text = record.text;
    const blob = record.blob;
    if (
      (text !== undefined && typeof text !== "string") ||
      (blob !== undefined && typeof blob !== "string")
    ) {
      return null;
    }
    return {
      type: "resource",
      uri: record.uri,
      ...(text === undefined ? {} : { text }),
      ...(blob === undefined ? {} : { blob }),
    };
  }
  return null;
}

export function assertMcpServerToolResult(
  method: string,
  response: unknown,
): McpToolResult {
  const record = assertRecord(method, response, "tool result");
  if (!Array.isArray(record.content)) {
    throw new Error(`${method} did not return tool result`);
  }
  const content = record.content.map(lowerCodexMcpContent);
  if (content.some((item): item is null => item === null)) {
    throw new Error(`${method} did not return canonical MCP content`);
  }
  if (
    record.isError !== undefined &&
    record.isError !== null &&
    typeof record.isError !== "boolean"
  ) {
    throw new Error(`${method} did not return isError`);
  }
  return {
    content: content as McpContent[],
    structuredContent: record.structuredContent,
    is_error: record.isError === true,
  };
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

export function assertMcpServerResourceContent(
  method: string,
  response: unknown,
): McpResourceContent {
  const record = assertRecord(method, response, "resource contents");
  if (!Array.isArray(record.contents) || record.contents.length === 0) {
    throw new Error(`${method} did not return resource contents`);
  }
  if (record.contents.length !== 1) {
    throw new Error(`${method} returned multiple resource contents`);
  }
  const content = record.contents[0];
  if (!content || typeof content !== "object" || Array.isArray(content)) {
    throw new Error(`${method} did not return resource content`);
  }
  const contentRecord = content as Record<string, unknown>;
  if (typeof contentRecord.uri !== "string") {
    throw new Error(`${method} did not return resource URI`);
  }
  const hasText = typeof contentRecord.text === "string";
  const hasBlob = typeof contentRecord.blob === "string";
  if (hasText === hasBlob) {
    throw new Error(`${method} did not return text or blob content`);
  }
  const mimeType = contentRecord.mimeType ?? contentRecord.mime_type;
  if (
    mimeType !== undefined &&
    mimeType !== null &&
    typeof mimeType !== "string"
  ) {
    throw new Error(`${method} did not return resource MIME type`);
  }
  const meta = contentRecord._meta ?? contentRecord.meta;
  if (
    meta !== undefined &&
    meta !== null &&
    (typeof meta !== "object" || Array.isArray(meta))
  ) {
    throw new Error(`${method} did not return resource metadata`);
  }
  return {
    uri: contentRecord.uri,
    ...(mimeType === undefined || mimeType === null
      ? {}
      : { mime_type: mimeType as string }),
    ...(hasText ? { text: contentRecord.text as string } : {}),
    ...(hasBlob ? { blob: contentRecord.blob as string } : {}),
    ...(meta === undefined || meta === null
      ? {}
      : { meta: meta as Record<string, unknown> }),
  };
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
    resourceTemplates: (resourceTemplates ??
      []) as McpResourceTemplateDefinition[],
  };
}
