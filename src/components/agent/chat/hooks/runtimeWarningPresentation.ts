import { resolveRequiredAgentChatCopy } from "../utils/agentChatCopy";

export const ARTIFACT_DOCUMENT_REPAIRED_WARNING_CODE =
  "artifact_document_repaired";
export const ARTIFACT_DOCUMENT_FAILED_WARNING_CODE = "artifact_document_failed";
export const ARTIFACT_DOCUMENT_PERSIST_FAILED_WARNING_CODE =
  "artifact_document_persist_failed";
export const SKILL_NOT_AVAILABLE_WARNING_CODE = "skill_not_available";
export const SKILL_LOAD_FAILED_WARNING_CODE = "skill_load_failed";
export const MENTION_NOT_AVAILABLE_WARNING_CODE = "mention_not_available";
const UNKNOWN_APP_SERVER_NOTIFICATION_PREFIX =
  "unknown_app_server_notification:";
const UNPROJECTED_APP_SERVER_NOTIFICATION_PREFIX =
  "unprojected_app_server_notification:";

export type RuntimeWarningToastLevel = "info" | "warning" | "error";

export interface RuntimeWarningToastPresentation {
  level: RuntimeWarningToastLevel;
  message: string;
  shouldToast: boolean;
}

export function resolveRuntimeWarningToastPresentation(params: {
  code?: string | null;
  message?: string | null;
}): RuntimeWarningToastPresentation {
  const code = typeof params.code === "string" ? params.code.trim() : "";
  const message =
    typeof params.message === "string" && params.message.trim()
      ? params.message.trim()
      : "收到运行提醒";

  if (code.startsWith(UNKNOWN_APP_SERVER_NOTIFICATION_PREFIX)) {
    return {
      level: "warning",
      message: resolveRequiredAgentChatCopy("runtimeWarning.unknownNotification", {
        method: code.slice(UNKNOWN_APP_SERVER_NOTIFICATION_PREFIX.length),
      }),
      shouldToast: true,
    };
  }
  if (code.startsWith(UNPROJECTED_APP_SERVER_NOTIFICATION_PREFIX)) {
    return {
      level: "warning",
      message: resolveRequiredAgentChatCopy(
        "runtimeWarning.unprojectedNotification",
        {
          method: code.slice(UNPROJECTED_APP_SERVER_NOTIFICATION_PREFIX.length),
        },
      ),
      shouldToast: true,
    };
  }

  switch (code) {
    case ARTIFACT_DOCUMENT_REPAIRED_WARNING_CODE:
      return {
        level: "info",
        message: "已整理为可继续编辑的文稿，可继续查看或补全结构。",
        shouldToast: false,
      };
    case ARTIFACT_DOCUMENT_FAILED_WARNING_CODE:
      return {
        level: "warning",
        message: "结构化文稿未完整生成，已保留一份可继续编辑的恢复稿。",
        shouldToast: true,
      };
    case ARTIFACT_DOCUMENT_PERSIST_FAILED_WARNING_CODE:
      return {
        level: "warning",
        message: "文稿未能保存到工作区，当前结果仍保留在对话中。",
        shouldToast: true,
      };
    case SKILL_NOT_AVAILABLE_WARNING_CODE:
      return {
        level: "warning",
        message: resolveRequiredAgentChatCopy(
          "runtimeWarning.skillNotAvailable",
        ),
        shouldToast: true,
      };
    case SKILL_LOAD_FAILED_WARNING_CODE:
      return {
        level: "warning",
        message: resolveRequiredAgentChatCopy("runtimeWarning.skillLoadFailed"),
        shouldToast: true,
      };
    case MENTION_NOT_AVAILABLE_WARNING_CODE:
      return {
        level: "warning",
        message: resolveRequiredAgentChatCopy(
          "runtimeWarning.mentionNotAvailable",
        ),
        shouldToast: true,
      };
    default:
      return {
        level: "warning",
        message,
        shouldToast: true,
      };
  }
}
