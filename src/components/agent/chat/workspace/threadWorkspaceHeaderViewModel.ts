import type { TaskStatus, Topic } from "../hooks/agentChatShared";
import { resolveTaskCenterTopicTitle } from "./taskCenterTabProjection";

export interface ThreadWorkspaceHeaderViewModel {
  sessionId: string;
  title: string;
  status: TaskStatus | null;
  workingDirectory: string | null;
  canAcceptDirectInput: boolean | null;
  onRename?: (title: string) => Promise<void>;
  onArchive?: () => Promise<void>;
  onFork?: () => Promise<void>;
}

interface BuildThreadWorkspaceHeaderViewModelParams {
  sessionId?: string | null;
  currentSessionTitle?: string | null;
  initialSessionId?: string | null;
  initialSessionName?: string | null;
  topic?: Topic | null;
  sessionWorkingDirectory?: string | null;
  projectRootPath?: string | null;
  canonicalThreadStatus?: string | null;
  canAcceptDirectInput?: boolean | null;
  onRename?: (title: string) => Promise<void>;
  onArchive?: () => Promise<void>;
  onFork?: () => Promise<void>;
  isSending?: boolean;
  pendingActionCount?: number;
  untitledTaskLabel: string;
}

function normalizeText(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function resolveThreadWorkspaceStatus({
  topicStatus,
  canonicalThreadStatus,
  isSending,
  pendingActionCount,
}: {
  topicStatus?: TaskStatus | null;
  canonicalThreadStatus?: string | null;
  isSending?: boolean;
  pendingActionCount?: number;
}): TaskStatus | null {
  if ((pendingActionCount ?? 0) > 0) {
    return "waiting";
  }
  const normalizedCanonicalStatus = canonicalThreadStatus?.trim().toLowerCase();
  switch (normalizedCanonicalStatus) {
    case "queued":
      return "queued";
    case "active":
    case "running":
    case "interrupting":
      return "running";
    case "blocked":
    case "waiting":
    case "waiting_input":
    case "waiting_request":
      return "waiting";
    case "failed":
    case "system_error":
      return "failed";
    case "cancelled":
    case "canceled":
    case "completed":
    case "done":
    case "idle":
      return "done";
    default:
      break;
  }
  if (isSending) {
    return "running";
  }
  return topicStatus ?? null;
}

export function buildThreadWorkspaceHeaderViewModel({
  sessionId,
  currentSessionTitle,
  initialSessionId,
  initialSessionName,
  topic,
  sessionWorkingDirectory,
  projectRootPath,
  canonicalThreadStatus,
  canAcceptDirectInput,
  onRename,
  onArchive,
  onFork,
  isSending,
  pendingActionCount,
  untitledTaskLabel,
}: BuildThreadWorkspaceHeaderViewModelParams): ThreadWorkspaceHeaderViewModel | null {
  const normalizedSessionId = normalizeText(sessionId);
  if (!normalizedSessionId) {
    return null;
  }

  const initialTitleFallback =
    normalizedSessionId === normalizeText(initialSessionId)
      ? normalizeText(initialSessionName)
      : null;
  const title = resolveTaskCenterTopicTitle(
    normalizeText(topic?.title) ?? normalizeText(currentSessionTitle),
    initialTitleFallback ?? untitledTaskLabel,
  );
  const workingDirectory =
    normalizeText(sessionWorkingDirectory) ??
    normalizeText(topic?.workingDir) ??
    normalizeText(projectRootPath);

  return {
    sessionId: normalizedSessionId,
    title,
    status: resolveThreadWorkspaceStatus({
      topicStatus: topic?.status,
      canonicalThreadStatus,
      isSending,
      pendingActionCount,
    }),
    workingDirectory,
    canAcceptDirectInput: canAcceptDirectInput ?? null,
    ...(onRename ? { onRename } : {}),
    ...(onArchive ? { onArchive } : {}),
    ...(onFork ? { onFork } : {}),
  };
}
