import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import { normalizeActionArguments } from "@/lib/api/agentActionArguments";
import type { ActionRequired, AgentThreadItem } from "../../types";
import { mapItemStatus } from "./statusMapping";
import { stringifyResponse } from "./textFormatting";

function metadataRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function itemMetadataRecord(item: AgentThreadItem): Record<string, unknown> {
  return {
    ...(metadataRecord(item.metadata) || {}),
    sequence: item.sequence,
  };
}

function mergeResultMetadata(
  metadata: Record<string, unknown> | undefined,
  extra?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!metadata && !extra) {
    return undefined;
  }
  return {
    ...(metadata || {}),
    ...(extra || {}),
  };
}

function stripCommandOutputWrapper(
  value: string | undefined,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const lines = value.split(/\r?\n/);
  if (!/^Exit code:\s*-?\d+\s*$/.test(lines[0] || "")) {
    return value;
  }
  const outputIndex = lines.findIndex((line) => line.trim() === "Output:");
  if (outputIndex < 0) {
    return value;
  }
  return lines.slice(outputIndex + 1).join("\n");
}

function toQuestionOptions(
  options: Array<{ label: string; description?: string }> | undefined,
) {
  return options?.map((option) => ({
    label: option.label,
    description: option.description,
  }));
}

function actionRequiredScope(item: AgentThreadItem): ActionRequired["scope"] {
  return {
    ...(item.thread_id ? { threadId: item.thread_id } : {}),
    turnId: item.turn_id,
  };
}

export function toActionRequired(item: AgentThreadItem): ActionRequired | null {
  if (item.type === "approval_request") {
    return {
      requestId: item.request_id,
      actionType: "tool_confirmation",
      toolName: item.tool_name,
      arguments: normalizeActionArguments(item.arguments),
      prompt: item.prompt,
      scope: actionRequiredScope(item),
      status: item.status === "completed" ? "submitted" : "pending",
      submittedResponse: stringifyResponse(item.response),
      submittedUserData: item.response,
    };
  }

  if (item.type === "request_user_input") {
    return {
      requestId: item.request_id,
      actionType:
        item.action_type === "elicitation" ? "elicitation" : "ask_user",
      prompt: item.prompt,
      questions: item.questions?.map((question) => ({
        question: question.question,
        header: question.header,
        options: toQuestionOptions(question.options),
        multiSelect: question.multi_select,
      })),
      scope: actionRequiredScope(item),
      status: item.status === "completed" ? "submitted" : "pending",
      submittedResponse: stringifyResponse(item.response),
      submittedUserData: item.response,
    };
  }

  return null;
}

export function toToolCallState(item: AgentThreadItem): ToolCallState | null {
  switch (item.type) {
    case "tool_call": {
      const metadata = itemMetadataRecord(item);
      const itemRecord = item as AgentThreadItem & {
        output_preview?: string;
        outputPreview?: string;
      };
      const outputText =
        item.output ??
        itemRecord.output_preview ??
        itemRecord.outputPreview ??
        "";
      const structuredContent =
        item.structuredContent ?? item.structured_content;
      return {
        id: item.id,
        name: item.tool_name,
        arguments:
          item.arguments === undefined
            ? undefined
            : JSON.stringify(item.arguments, null, 2),
        status: mapItemStatus(item.status),
        result:
          item.output !== undefined ||
          itemRecord.output_preview !== undefined ||
          itemRecord.outputPreview !== undefined ||
          item.error !== undefined ||
          structuredContent !== undefined ||
          item.metadata !== undefined
            ? {
                success:
                  item.success ??
                  (item.status === "completed" && item.error === undefined),
                output: outputText,
                error: item.error,
                structuredContent,
                structured_content: structuredContent,
                metadata,
              }
            : undefined,
        metadata,
        startTime: new Date(item.started_at),
        endTime: item.completed_at ? new Date(item.completed_at) : undefined,
      };
    }
    case "command_execution": {
      const commandOutput = stripCommandOutputWrapper(item.aggregated_output);
      const metadata = {
        ...itemMetadataRecord(item),
        ...(item.command_status
          ? { command_status: item.command_status }
          : {}),
      };
      return {
        id: item.id,
        name: "exec_command",
        arguments: JSON.stringify(
          { command: item.command, cwd: item.cwd },
          null,
          2,
        ),
        status: mapItemStatus(item.status),
        result:
          item.aggregated_output !== undefined ||
          item.error !== undefined ||
          item.exit_code !== undefined ||
          item.command_status !== undefined ||
          item.metadata !== undefined
            ? {
                success:
                  item.status === "completed" &&
                  item.command_status !== "declined" &&
                  item.command_status !== "failed" &&
                  item.error === undefined &&
                  (item.exit_code === undefined || item.exit_code === 0),
                output: commandOutput || "",
                error: item.error,
                metadata: mergeResultMetadata(metadataRecord(item.metadata), {
                  sequence: item.sequence,
                  ...(item.command_status
                    ? { command_status: item.command_status }
                    : {}),
                  ...(item.exit_code !== undefined
                    ? { exit_code: item.exit_code }
                    : {}),
                  cwd: item.cwd,
                }),
              }
            : undefined,
        metadata,
        startTime: new Date(item.started_at),
        endTime: item.completed_at ? new Date(item.completed_at) : undefined,
      };
    }
    case "patch": {
      const metadata = itemMetadataRecord(item);
      return {
        id: item.id,
        name: "apply_patch",
        arguments: JSON.stringify(
          { paths: item.paths ?? item.summary ?? [] },
          null,
          2,
        ),
        status: mapItemStatus(item.status),
        result:
          item.stdout !== undefined ||
          item.stderr !== undefined ||
          item.success !== undefined ||
          item.metadata !== undefined
            ? {
                success:
                  item.success ??
                  (item.status === "completed" && item.stderr === undefined),
                output: item.stdout || item.text || "",
                error: item.stderr,
                metadata,
              }
            : undefined,
        metadata,
        startTime: new Date(item.started_at),
        endTime: item.completed_at ? new Date(item.completed_at) : undefined,
      };
    }
    case "web_search": {
      const metadata = itemMetadataRecord(item);
      return {
        id: item.id,
        name: "web_search",
        arguments:
          item.query !== undefined || item.action !== undefined
            ? JSON.stringify(
                { action: item.action || "web_search", query: item.query },
                null,
                2,
              )
            : undefined,
        status: mapItemStatus(item.status),
        result:
          item.output !== undefined || metadata
            ? {
                success: item.status !== "failed",
                output: item.output || "",
                metadata,
              }
            : undefined,
        metadata,
        startTime: new Date(item.started_at),
        endTime: item.completed_at ? new Date(item.completed_at) : undefined,
      };
    }
    case "hook": {
      const metadata = itemMetadataRecord(item);
      const entriesText =
        item.entries?.map((entry) => `${entry.kind}: ${entry.text}`).join("\n") ??
        "";
      return {
        id: item.id,
        name: "hook",
        arguments: JSON.stringify(
          {
            run_id: item.run_id,
            event_name: item.event_name,
            source_path: item.source_path,
            target_item_id: item.target_item_id,
          },
          null,
          2,
        ),
        status: mapItemStatus(item.status),
        result:
          item.output !== undefined ||
          item.status_message !== undefined ||
          item.entries !== undefined ||
          item.metadata !== undefined
            ? {
                success: item.status !== "failed",
                output: item.output || entriesText || item.status_message || "",
                error: item.status === "failed" ? item.status_message : undefined,
                metadata,
              }
            : undefined,
        metadata,
        startTime: new Date(item.started_at),
        endTime: item.completed_at ? new Date(item.completed_at) : undefined,
      };
    }
    default:
      return null;
  }
}
