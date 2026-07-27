import type { AppServerAgentEvent } from "@/lib/api/appServer";
import type {
  AgentMessageContent,
  AgentMessageContentImage,
} from "../agentProtocol";
import {
  normalizeRecord,
  readBoolean,
  readFiniteNumber,
  readString,
} from "./appServerEventPayloadUtils";

type ItemStatus = "in_progress" | "completed" | "failed";

const TERMINAL_ITEM_STATUSES = new Set(["completed", "failed"]);
const TOOL_ITEM_TYPES = new Set([
  "mcpToolCall",
  "dynamicToolCall",
  "collabAgentToolCall",
]);

/**
 * Project a Codex v2 ThreadItem. Durable identity and lifecycle timestamps
 * come from the raw event envelope; the canonical item only supplies its
 * tagged-union content.
 */
export function readCanonicalThreadItem(
  item: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> | null {
  const id = readString(item, "id");
  const type = readString(item, "type");
  const route = readItemRoute(event);
  if (!id || !type || !route) {
    return null;
  }

  const status = itemStatus(item, event, type);
  if (!status) {
    return null;
  }
  const base = {
    id,
    thread_id: route.threadId,
    turn_id: route.turnId,
    sequence: event.sequence,
    status,
    started_at: event.timestamp,
    updated_at: event.timestamp,
    ...(isTerminalItemStatus(status) ? { completed_at: event.timestamp } : {}),
    ...(item.metadata && typeof item.metadata === "object"
      ? { metadata: item.metadata }
      : {}),
  };

  switch (type) {
    case "userMessage": {
      const content = readUserMessageContent(item.content);
      if (!content) {
        return null;
      }
      return {
        ...base,
        type: "user_message",
        content: content.text,
        content_parts: content.parts,
        client_id: readString(item, "clientId"),
      };
    }
    case "agentMessage": {
      const text = readString(item, "text");
      if (text === undefined) {
        return null;
      }
      return {
        ...base,
        type: "agent_message",
        text,
        phase: readString(item, "phase"),
        memoryCitation: item.memoryCitation,
      };
    }
    case "plan": {
      const text = readString(item, "text");
      if (text === undefined) {
        return null;
      }
      const derivedMetadata = canonicalPlanMetadata(id, route.turnId);
      const existingMetadata = normalizeRecord(item.metadata) ?? {};
      return {
        ...base,
        type: "plan",
        text,
        ...(derivedMetadata
          ? { metadata: { ...derivedMetadata, ...existingMetadata } }
          : {}),
      };
    }
    case "reasoning": {
      const summary = readStringList(item.summary);
      const content = readStringList(item.content);
      return {
        ...base,
        type: "reasoning",
        text: summary.join("\n\n"),
        summary,
        content,
      };
    }
    case "commandExecution": {
      const command = readString(item, "command");
      const cwd = readString(item, "cwd");
      if (command === undefined || cwd === undefined) {
        return null;
      }
      return {
        ...base,
        type: "command_execution",
        command,
        cwd,
        process_id: readString(item, "processId"),
        source: readString(item, "source"),
        command_actions: Array.isArray(item.commandActions)
          ? item.commandActions
          : [],
        aggregated_output: readString(item, "aggregatedOutput"),
        exit_code: readFiniteNumber(item, "exitCode"),
        duration_ms: readFiniteNumber(item, "durationMs"),
        error: readDisplayText(item.error),
      };
    }
    case "fileChange": {
      const changes = Array.isArray(item.changes) ? item.changes : [];
      const fileStatus = readString(item, "status");
      const paths = changes.flatMap((change) => {
        const record = normalizeRecord(change);
        const path = readString(record ?? {}, "path");
        return path ? [path] : [];
      });
      return {
        ...base,
        type: "patch",
        text: JSON.stringify(changes),
        changes,
        file_status: fileStatus,
        paths,
        success: status === "completed",
      };
    }
    case "mcpToolCall":
    case "dynamicToolCall":
    case "collabAgentToolCall":
      return readCanonicalToolThreadItem(item, event);
    case "subAgentActivity":
      return {
        ...base,
        type: "subagent_activity",
        status_label: readString(item, "kind") ?? type,
        session_id: readString(item, "agentThreadId"),
        metadata: { agentPath: readString(item, "agentPath") },
      };
    case "contextCompaction":
      return {
        ...base,
        type: "context_compaction",
        stage: isTerminalItemStatus(status) ? "completed" : "started",
      };
    case "imageView": {
      const uri = readString(item, "path") ?? "";
      return {
        ...base,
        type: "media",
        uri,
        mime_type: imageMimeType(uri),
      };
    }
    case "webSearch":
      return readCanonicalWebSearchThreadItem(item, event, status);
    case "imageGeneration": {
      const generationStatus = readString(item, "status");
      const result = readString(item, "result");
      if (!generationStatus || result === undefined) {
        return null;
      }
      return {
        ...base,
        type: "image_generation",
        generation_status: generationStatus,
        revised_prompt: readString(item, "revisedPrompt"),
        result,
        saved_path: readString(item, "savedPath"),
      };
    }
    case "hookPrompt":
    case "sleep":
    case "enteredReviewMode":
    case "exitedReviewMode":
      return {
        ...base,
        type: "extension",
        name: type,
        data: item,
      };
    default:
      return null;
  }
}

function canonicalPlanMetadata(
  itemId: string,
  turnId: string,
): Record<string, string> | null {
  const prefix = `plan_${turnId}_`;
  if (!itemId.startsWith(prefix)) {
    return null;
  }
  const revisionId = itemId.slice(prefix.length).trim();
  if (!revisionId) {
    return null;
  }
  const source = revisionId.startsWith("update_plan:")
    ? "update_plan"
    : revisionId.startsWith("proposed_plan:")
      ? "proposed_plan"
      : "thread_item";
  return { revisionId, source };
}

export function readCanonicalToolThreadItem(
  item: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> | null {
  const type = readString(item, "type");
  const id = readString(item, "id");
  const toolName = readString(item, "tool");
  const route = readItemRoute(event);
  if (!id || !type || !TOOL_ITEM_TYPES.has(type) || !toolName || !route) {
    return null;
  }

  const metadata = readCanonicalToolMetadata(item, type);
  if (!metadata) {
    return null;
  }

  const status = itemStatus(item, event, type);
  if (!status) {
    return null;
  }
  const output = readToolOutput(item);
  const error = readDisplayText(item.error);
  const explicitSuccess = readBoolean(item, "success");

  return {
    id,
    thread_id: route.threadId,
    turn_id: route.turnId,
    sequence: event.sequence,
    status,
    started_at: event.timestamp,
    updated_at: event.timestamp,
    ...(isTerminalItemStatus(status) ? { completed_at: event.timestamp } : {}),
    type: "tool_call",
    tool_name: toolName,
    arguments: item.arguments,
    output,
    structured_content: readStructuredContent(item),
    output_truncated: readBoolean(item, "truncated"),
    duration_ms: readFiniteNumber(item, "durationMs"),
    success:
      explicitSuccess ??
      (status === "completed" ? true : status === "failed" ? false : undefined),
    error,
    metadata,
  };
}

function readCanonicalToolMetadata(
  item: Record<string, unknown>,
  type: string,
): Record<string, unknown> | null {
  const id = readString(item, "id");
  if (!id) {
    return null;
  }

  if (type === "mcpToolCall") {
    const server = readString(item, "server");
    if (!server || !Object.prototype.hasOwnProperty.call(item, "arguments")) {
      return null;
    }
    const appContext = readOptionalRecord(item.appContext);
    if (
      item.appContext !== undefined &&
      item.appContext !== null &&
      (!appContext || !readString(appContext, "connectorId"))
    ) {
      return null;
    }
    const result = readOptionalRecord(item.result);
    if (
      item.result !== undefined &&
      item.result !== null &&
      (!result || !Array.isArray(result.content))
    ) {
      return null;
    }
    return omitUndefined({
      callId: id,
      canonical_type: type,
      server,
      app_context: appContext,
      plugin_id: readString(item, "pluginId"),
      result_content: result?.content,
      result_meta: result?._meta,
    });
  }

  if (type === "dynamicToolCall") {
    if (!Object.prototype.hasOwnProperty.call(item, "arguments")) {
      return null;
    }
    if (
      item.contentItems !== undefined &&
      item.contentItems !== null &&
      !Array.isArray(item.contentItems)
    ) {
      return null;
    }
    return omitUndefined({
      callId: id,
      canonical_type: type,
      namespace: readString(item, "namespace"),
      content_items: Array.isArray(item.contentItems)
        ? item.contentItems
        : undefined,
    });
  }

  const senderThreadId = readString(item, "senderThreadId");
  const receiverThreadIds = readRequiredStringList(item.receiverThreadIds);
  const agentsStates = readOptionalRecord(item.agentsStates);
  if (!senderThreadId || !receiverThreadIds || !agentsStates) {
    return null;
  }
  return omitUndefined({
    callId: id,
    canonical_type: type,
    sender_thread_id: senderThreadId,
    receiver_thread_ids: receiverThreadIds,
    prompt: readString(item, "prompt"),
    model: readString(item, "model"),
    reasoning_effort: readString(item, "reasoningEffort"),
    agents_states: agentsStates,
  });
}

function readCanonicalWebSearchThreadItem(
  item: Record<string, unknown>,
  event: AppServerAgentEvent,
  status: ItemStatus,
): Record<string, unknown> | null {
  const id = readString(item, "id");
  const query = readString(item, "query");
  const route = readItemRoute(event);
  const action = readOptionalRecord(item.action);
  const results = item.results;
  if (
    !id ||
    query === undefined ||
    !route ||
    (item.action !== undefined && item.action !== null && !action) ||
    (results !== undefined && results !== null && !Array.isArray(results))
  ) {
    return null;
  }
  return {
    id,
    thread_id: route.threadId,
    turn_id: route.turnId,
    sequence: event.sequence,
    status,
    started_at: event.timestamp,
    updated_at: event.timestamp,
    ...(isTerminalItemStatus(status) ? { completed_at: event.timestamp } : {}),
    type: "web_search",
    query,
    action: action ? readString(action, "type") : undefined,
    action_data: action,
    results: Array.isArray(results) ? results : undefined,
  };
}

function readItemRoute(
  event: AppServerAgentEvent,
): { threadId: string; turnId: string } | null {
  const threadId = event.threadId?.trim();
  const turnId = event.turnId?.trim();
  return threadId && turnId ? { threadId, turnId } : null;
}

function itemStatus(
  item: Record<string, unknown>,
  event: AppServerAgentEvent,
  type: string,
): ItemStatus | null {
  const explicit = readString(item, "status");
  if (type === "commandExecution" || type === "fileChange") {
    return normalizeItemStatus(explicit, true);
  }
  if (TOOL_ITEM_TYPES.has(type)) {
    return normalizeItemStatus(explicit, false);
  }
  return statusFromEventType(event.type);
}

function normalizeItemStatus(
  status: string | undefined,
  allowDeclined: boolean,
): ItemStatus | null {
  switch (status) {
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "inProgress":
      return "in_progress";
    case "declined":
      return allowDeclined ? "failed" : null;
    default:
      return null;
  }
}

function statusFromEventType(type: string): ItemStatus | null {
  switch (type) {
    case "item.started":
      return "in_progress";
    case "item.completed":
      return "completed";
    default:
      return null;
  }
}

function readUserMessageContent(
  value: unknown,
): { text: string; parts: AgentMessageContent[] } | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const parts: AgentMessageContent[] = [];
  const textParts: string[] = [];
  for (const part of value) {
    const record = normalizeRecord(part);
    if (!record) {
      return null;
    }
    switch (readString(record, "type")) {
      case "text": {
        const text = readString(record, "text");
        if (text === undefined) {
          return null;
        }
        if ("textElements" in record) {
          return null;
        }
        const textElements = readTextElements(record.text_elements, text);
        if (!textElements) {
          return null;
        }
        parts.push({
          type: "text",
          text,
          ...(textElements.length > 0 ? { text_elements: textElements } : {}),
        });
        textParts.push(text);
        break;
      }
      case "image": {
        const url = readString(record, "url");
        const detail = readImageDetail(record.detail);
        if (!url || detail === null) {
          return null;
        }
        parts.push(imageContentReference(url, undefined, detail));
        break;
      }
      case "localImage": {
        const path = readString(record, "path");
        const detail = readImageDetail(record.detail);
        if (!path || detail === null) {
          return null;
        }
        parts.push(imageContentReference(path, path, detail));
        break;
      }
      case "skill":
      case "mention": {
        const type = readString(record, "type");
        const name = readString(record, "name");
        const path = readString(record, "path");
        if ((type !== "skill" && type !== "mention") || !name || !path) {
          return null;
        }
        parts.push({ type, name, path });
        break;
      }
      default:
        return null;
    }
  }
  return { text: textParts.join(""), parts };
}

function imageContentReference(
  uri: string,
  sourcePath?: string,
  detail?: AgentMessageContentImage["detail"],
): AgentMessageContentImage {
  const dataUrl = /^data:(image\/[\w.+-]+);base64,([\s\S]*)$/iu.exec(uri);
  if (dataUrl) {
    return {
      type: "image",
      mime_type: dataUrl[1].toLowerCase(),
      data: dataUrl[2],
      ...(detail ? { detail } : {}),
    };
  }
  const content: AgentMessageContentImage = {
    type: "image",
    mime_type: imageMimeType(uri),
    data: "",
    uri,
    ...(detail ? { detail } : {}),
  };
  if (sourcePath) {
    content.source_path = sourcePath;
  }
  return content;
}

function imageMimeType(value: string): string {
  const normalized = value.split(/[?#]/u, 1)[0]?.toLowerCase() ?? "";
  if (normalized.endsWith(".png")) return "image/png";
  if (normalized.endsWith(".jpg") || normalized.endsWith(".jpeg")) {
    return "image/jpeg";
  }
  if (normalized.endsWith(".gif")) return "image/gif";
  if (normalized.endsWith(".webp")) return "image/webp";
  if (normalized.endsWith(".svg")) return "image/svg+xml";
  if (normalized.endsWith(".avif")) return "image/avif";
  return "image/*";
}

function readImageDetail(
  value: unknown,
): AgentMessageContentImage["detail"] | null {
  if (value === undefined || value === null) {
    return undefined;
  }
  return value === "auto" ||
    value === "low" ||
    value === "high" ||
    value === "original"
    ? value
    : null;
}

function readTextElements(
  value: unknown,
  text: string,
): NonNullable<
  Extract<AgentMessageContent, { type: "text" }>["text_elements"]
> | null {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    return null;
  }
  const byteLength = new TextEncoder().encode(text).length;
  const elements = [];
  for (const entry of value) {
    const record = normalizeRecord(entry);
    const byteRange = normalizeRecord(record?.byteRange);
    const start = byteRange?.start;
    const end = byteRange?.end;
    const placeholder = record?.placeholder;
    if (
      !record ||
      !byteRange ||
      typeof start !== "number" ||
      typeof end !== "number" ||
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      !Number.isInteger(start) ||
      !Number.isInteger(end) ||
      start < 0 ||
      end < start ||
      end > byteLength ||
      (placeholder !== undefined &&
        placeholder !== null &&
        typeof placeholder !== "string")
    ) {
      return null;
    }
    elements.push({
      byte_range: { start, end },
      ...(typeof placeholder === "string" ? { placeholder } : {}),
    });
  }
  return elements;
}

function readToolOutput(item: Record<string, unknown>): string | undefined {
  const result = normalizeRecord(item.result);
  if (result) {
    return readDisplayText(result.output ?? result.content ?? result);
  }
  if (Array.isArray(item.contentItems)) {
    return readDisplayText(item.contentItems);
  }
  return undefined;
}

function readStructuredContent(
  item: Record<string, unknown>,
): unknown | undefined {
  const result = normalizeRecord(item.result);
  return result?.structuredContent;
}

function readDisplayText(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    const text = value
      .map((entry) => readDisplayText(entry))
      .filter((entry): entry is string => Boolean(entry))
      .join("");
    return text || undefined;
  }
  const record = normalizeRecord(value);
  if (!record) {
    return undefined;
  }
  return (
    readString(record, "text", "output", "content", "message") ?? undefined
  );
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is string => typeof entry === "string");
}

function readRequiredStringList(value: unknown): string[] | null {
  return Array.isArray(value) &&
    value.every((entry): entry is string => typeof entry === "string")
    ? value
    : null;
}

function readOptionalRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  return value === undefined || value === null
    ? undefined
    : normalizeRecord(value);
}

function omitUndefined(
  value: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  );
}

function isTerminalItemStatus(status: string): boolean {
  return TERMINAL_ITEM_STATUSES.has(status);
}
