import type { CanvasStateUnion } from "@/components/workspace/canvas/canvasUtils";
import type { DocumentVersion } from "@/components/workspace/canvas/canvasUtils";
import type { StepStatus } from "@/lib/workspace/workbenchContract";
import type {
  GeneralWorkbenchRunTodoItem,
  GeneralWorkbenchRunTerminalItem,
  GeneralWorkbenchRunState as BackendGeneralWorkbenchRunState,
} from "@/lib/api/executionRun";
import type { GeneralWorkbenchDocumentState } from "@/lib/api/project";
import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import type { SkillDetailInfo } from "@/lib/api/skill-execution";
import { extractArtifactProtocolPathsFromValue } from "@/lib/artifact-protocol";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import type { TopicBranchStatus } from "../hooks/useTopicBranchBoard";
import { parseSkillSlashCommand } from "../hooks/skillCommand";
import { CONTENT_POST_SKILL_KEY } from "../utils/contentPostSkill";
import type { Message } from "../types";
import type { TaskFile } from "../components/TaskFiles";
import { isPrimaryResultDocumentPath } from "./resultFilePriority";

export const GENERAL_WORKBENCH_DOCUMENT_META_KEY =
  "general_workbench_document_v1";
export const LEGACY_GENERAL_WORKBENCH_DOCUMENT_META_KEY =
  "theme_workbench_document_v1";
export const MAX_PERSISTED_DOCUMENT_VERSIONS = 40;
export const SOCIAL_ARTICLE_SKILL_KEY = CONTENT_POST_SKILL_KEY;
export const GENERAL_WORKBENCH_ACTIVE_RUN_MAX_AGE_MS = 45 * 1000;
export const GENERAL_WORKBENCH_HISTORY_PAGE_SIZE = 20;

function resolveThemeWorkbenchRunStepStatus(
  status: "queued" | "running" | "success" | "error" | "canceled" | "timeout",
): StepStatus {
  if (status === "running") {
    return "active";
  }
  if (status === "queued") {
    return "pending";
  }
  if (status === "success") {
    return "completed";
  }
  return "error";
}

function parseThemeWorkbenchToolArguments(
  argumentsJson?: string,
): Record<string, unknown> {
  if (!argumentsJson) {
    return {};
  }

  try {
    const parsed = JSON.parse(argumentsJson);
    return parsed && typeof parsed === "object"
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function truncateThemeWorkbenchLabel(value: string, limit = 28): string {
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function resolveThemeWorkbenchTextArg(
  args: Record<string, unknown>,
  keys: string[],
): string {
  for (const key of keys) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
    if (Array.isArray(value)) {
      const firstString = value.find(
        (item): item is string =>
          typeof item === "string" && item.trim().length > 0,
      );
      if (firstString) {
        return firstString.trim();
      }
    }
  }
  return "";
}

function getThemeWorkbenchFileLabel(pathValue: string): string {
  const normalized = pathValue.trim();
  if (!normalized) {
    return "主稿文件";
  }
  const segments = normalized.split(/[/\\]/).filter(Boolean);
  if (segments.length >= 2) {
    return `${segments[segments.length - 2]}/${segments[segments.length - 1]}`;
  }
  return segments[0] || normalized;
}

function resolveThemeWorkbenchToolTaskTitle(toolCall: ToolCallState): string {
  const normalized = toolCall.name.trim().toLowerCase();
  const args = parseThemeWorkbenchToolArguments(toolCall.arguments);
  const queryValue = resolveThemeWorkbenchTextArg(args, [
    "query",
    "q",
    "keyword",
    "pattern",
    "text",
  ]);
  const urlValue = resolveThemeWorkbenchTextArg(args, ["url", "href"]);
  const elementValue = resolveThemeWorkbenchTextArg(args, [
    "element",
    "name",
    "label",
    "ref",
  ]);

  if (normalized.includes("social_generate_cover_image")) {
    const size = resolveThemeWorkbenchTextArg(args, ["size"]);
    return size ? `生成封面图（${size}）` : "生成封面图";
  }
  if (normalized.includes("create_image_generation_task")) {
    return "提交配图任务";
  }
  if (normalized === "video_generate") {
    return "提交视频任务";
  }
  if (normalized.includes("write_file") || normalized.includes("create_file")) {
    const pathValue = extractArtifactProtocolPathsFromValue(args)[0] ?? "";
    return pathValue
      ? `写入 ${getThemeWorkbenchFileLabel(pathValue)}`
      : "写入主稿文件";
  }
  if (normalized.includes("websearch")) {
    return queryValue
      ? `检索 ${truncateThemeWorkbenchLabel(queryValue)}`
      : "检索参考资料";
  }
  if (
    normalized.includes("browser_navigate") ||
    (normalized.includes("navigate") && urlValue)
  ) {
    return urlValue
      ? `打开 ${truncateThemeWorkbenchLabel(urlValue, 36)}`
      : "打开网页";
  }
  if (normalized.includes("browser_click") || normalized === "click") {
    return elementValue
      ? `点击「${truncateThemeWorkbenchLabel(elementValue, 20)}」`
      : "点击页面元素";
  }
  if (normalized.includes("browser_hover") || normalized === "hover") {
    return elementValue
      ? `定位「${truncateThemeWorkbenchLabel(elementValue, 20)}」`
      : "定位页面元素";
  }
  if (normalized.includes("browser_type") || normalized === "type") {
    return elementValue
      ? `填写「${truncateThemeWorkbenchLabel(elementValue, 20)}」`
      : queryValue
        ? `填写 ${truncateThemeWorkbenchLabel(queryValue, 18)}`
        : "填写页面内容";
  }
  if (
    normalized.includes("browser_select_option") ||
    normalized.includes("select_option")
  ) {
    const value = resolveThemeWorkbenchTextArg(args, [
      "value",
      "values",
      "option",
    ]);
    return value
      ? `选择 ${truncateThemeWorkbenchLabel(value, 20)}`
      : elementValue
        ? `选择「${truncateThemeWorkbenchLabel(elementValue, 20)}」`
        : "选择页面选项";
  }
  if (
    normalized.includes("browser_press_key") ||
    normalized.includes("press_key")
  ) {
    const keyValue = resolveThemeWorkbenchTextArg(args, ["key"]);
    return keyValue ? `触发按键 ${keyValue}` : "触发页面快捷键";
  }
  if (normalized.includes("browser_drag") || normalized.includes("drag")) {
    const endValue = resolveThemeWorkbenchTextArg(args, [
      "endElement",
      "endRef",
    ]);
    return endValue
      ? `拖拽到「${truncateThemeWorkbenchLabel(endValue, 18)}」`
      : "拖拽页面元素";
  }
  if (
    normalized.includes("browser_snapshot") ||
    normalized.includes("screenshot")
  ) {
    return elementValue
      ? `分析页面区域：${truncateThemeWorkbenchLabel(elementValue, 20)}`
      : urlValue
        ? `分析页面 ${truncateThemeWorkbenchLabel(urlValue, 30)}`
        : "分析页面内容";
  }
  if (normalized.includes("bash") || normalized.includes("shell")) {
    const commandValue = resolveThemeWorkbenchTextArg(args, ["command", "cmd"]);
    const commandProbe = commandValue.toLowerCase();
    if (commandProbe.includes("ffmpeg")) {
      return "处理音视频素材";
    }
    if (commandProbe.includes("curl") || commandProbe.includes("wget")) {
      return "下载远程资源";
    }
    if (
      commandProbe.includes("python") ||
      commandProbe.includes("node") ||
      commandProbe.includes("tsx") ||
      commandProbe.includes("npm")
    ) {
      return "执行自动化脚本";
    }
    return commandValue
      ? `执行命令：${truncateThemeWorkbenchLabel(commandValue, 22)}`
      : "执行终端命令";
  }
  if (normalized.includes("browser")) {
    return urlValue
      ? `采集 ${truncateThemeWorkbenchLabel(urlValue, 36)}`
      : elementValue
        ? `处理页面元素：${truncateThemeWorkbenchLabel(elementValue, 20)}`
        : "采集网页信息";
  }
  return toolCall.name.replace(/[_-]+/g, " ").trim() || "执行工具";
}

function resolveThemeWorkbenchPrimaryTaskTitle(
  skillName: string,
  detail?: SkillDetailInfo | null,
): string {
  if (skillName === SOCIAL_ARTICLE_SKILL_KEY) {
    return "生成内容主稿";
  }

  const displayName = detail?.display_name?.trim();
  if (displayName) {
    return displayName;
  }

  return skillName.replace(/[_-]+/g, " ").trim() || "执行任务";
}

function extractThemeWorkbenchWorkflowMarkerIndex(
  content: string,
): number | null {
  const matches = [...content.matchAll(/\*\*步骤\s+(\d+)\/(\d+):/g)];
  if (matches.length === 0) {
    return null;
  }
  const last = matches[matches.length - 1];
  const value = Number(last[1]);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value - 1;
}

function findLatestThemeWorkbenchExecution(messages: Message[]): {
  assistantMessage: Message;
  skillName: string | null;
} | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") {
      continue;
    }

    const hasToolCalls = (message.toolCalls?.length || 0) > 0;
    const hasPendingAction =
      message.actionRequests?.some(
        (request) => request.status !== "submitted",
      ) || false;
    if (!message.isThinking && !hasToolCalls && !hasPendingAction) {
      continue;
    }

    let skillName: string | null = null;
    for (let userIndex = index - 1; userIndex >= 0; userIndex -= 1) {
      const candidate = messages[userIndex];
      if (candidate.role !== "user") {
        continue;
      }
      skillName = parseSkillSlashCommand(candidate.content)?.skillName || null;
      break;
    }

    return {
      assistantMessage: message,
      skillName,
    };
  }

  return null;
}

function buildThemeWorkbenchLiveWorkflowSteps(
  messages: Message[],
  skillDetailMap: Record<string, SkillDetailInfo | null>,
  isSending: boolean,
): Array<{ id: string; title: string; status: StepStatus }> {
  const activeExecution = findLatestThemeWorkbenchExecution(messages);
  if (!activeExecution) {
    return [];
  }

  const { assistantMessage, skillName } = activeExecution;
  if (!skillName) {
    return [];
  }

  const skillDetail = skillDetailMap[skillName] || null;
  const workflowSteps = skillDetail?.workflow_steps || [];
  if (workflowSteps.length > 0) {
    const latestAssistantContent =
      messages
        .slice()
        .reverse()
        .find((m) => m.role === "assistant")?.content || "";
    const activeIndex =
      extractThemeWorkbenchWorkflowMarkerIndex(latestAssistantContent) ?? 0;
    return workflowSteps.map((step, index) => ({
      id: step.id,
      title: step.name,
      status:
        index < activeIndex
          ? ("completed" as StepStatus)
          : index === activeIndex
            ? ("active" as StepStatus)
            : ("pending" as StepStatus),
    }));
  }

  const toolCalls = assistantMessage.toolCalls || [];
  const steps: Array<{ id: string; title: string; status: StepStatus }> = [];
  const primaryTaskTitle = resolveThemeWorkbenchPrimaryTaskTitle(
    skillName,
    skillDetail,
  );
  const hasRunningTool = toolCalls.some(
    (toolCall) => toolCall.status === "running",
  );
  const hasFailedTool = toolCalls.some(
    (toolCall) => toolCall.status === "failed",
  );
  const hasCompletedPrimaryWrite = toolCalls.some((toolCall) => {
    if (toolCall.status !== "completed") {
      return false;
    }
    const normalizedName = toolCall.name.trim().toLowerCase();
    return (
      normalizedName.includes("write_file") ||
      normalizedName.includes("create_file")
    );
  });

  steps.push({
    id: `${skillName}:primary`,
    title: primaryTaskTitle,
    status: hasCompletedPrimaryWrite
      ? ("completed" as StepStatus)
      : hasFailedTool
        ? ("error" as StepStatus)
        : toolCalls.length > 0
          ? ("completed" as StepStatus)
          : assistantMessage.isThinking || isSending
            ? ("active" as StepStatus)
            : ("pending" as StepStatus),
  });

  toolCalls.forEach((toolCall, index) => {
    steps.push({
      id: toolCall.id || `${skillName}:tool:${index}`,
      title: resolveThemeWorkbenchToolTaskTitle(toolCall),
      status:
        toolCall.status === "running"
          ? ("active" as StepStatus)
          : toolCall.status === "completed"
            ? ("completed" as StepStatus)
            : ("error" as StepStatus),
    });
  });

  if (isSending && toolCalls.length > 0 && !hasRunningTool) {
    steps.push({
      id: `${skillName}:finalize`,
      title: "整理最终结果",
      status: "active",
    });
  }

  return steps;
}

function resolveThemeWorkbenchQueueItemTitle(
  item: GeneralWorkbenchRunTodoItem,
  skillDetailMap: Record<string, SkillDetailInfo | null>,
): string {
  const sourceRef = resolveGeneralWorkbenchSkillSourceRef(item);
  if (sourceRef) {
    return resolveThemeWorkbenchPrimaryTaskTitle(
      sourceRef,
      skillDetailMap[sourceRef],
    );
  }
  return item.title?.trim() || "执行任务";
}

export function resolveGeneralWorkbenchSkillSourceRef(
  item:
    | GeneralWorkbenchRunTodoItem
    | GeneralWorkbenchRunTerminalItem
    | { source?: string | null; source_ref?: string | null },
): string | null {
  if ((item.source || "").trim() !== "skill") {
    return null;
  }
  const sourceRef = item.source_ref?.trim();
  return sourceRef || null;
}

interface PersistedGeneralWorkbenchDocument {
  versions: DocumentVersion[];
  currentVersionId: string;
  versionStatusMap: Record<string, TopicBranchStatus>;
}

function isTopicBranchStatus(value: unknown): value is TopicBranchStatus {
  return (
    value === "in_progress" ||
    value === "pending" ||
    value === "merged" ||
    value === "candidate"
  );
}

function normalizeDocumentVersion(value: unknown): DocumentVersion | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Record<string, unknown>;
  const id = typeof candidate.id === "string" ? candidate.id.trim() : "";
  const content =
    typeof candidate.content === "string" ? candidate.content : "";
  const createdAt =
    typeof candidate.createdAt === "number"
      ? candidate.createdAt
      : typeof candidate.created_at === "number"
        ? candidate.created_at
        : NaN;
  const description =
    typeof candidate.description === "string"
      ? candidate.description
      : undefined;
  const metadata =
    candidate.metadata && typeof candidate.metadata === "object"
      ? (candidate.metadata as DocumentVersion["metadata"])
      : undefined;

  if (!id || Number.isNaN(createdAt)) {
    return null;
  }

  return {
    id,
    content,
    createdAt,
    description,
    metadata,
  };
}

export function buildPersistedGeneralWorkbenchDocument(
  state: CanvasStateUnion,
  statusMap: Record<string, TopicBranchStatus>,
): PersistedGeneralWorkbenchDocument | null {
  if (state.type !== "document" || state.versions.length === 0) {
    return null;
  }

  const normalizedVersions = state.versions
    .map((version) => normalizeDocumentVersion(version))
    .filter((version): version is DocumentVersion => !!version);

  if (normalizedVersions.length === 0) {
    return null;
  }

  const latestVersions = normalizedVersions.slice(
    -MAX_PERSISTED_DOCUMENT_VERSIONS,
  );
  const versionIdSet = new Set(latestVersions.map((version) => version.id));
  let currentVersionId = state.currentVersionId;

  if (!versionIdSet.has(currentVersionId)) {
    currentVersionId =
      latestVersions[latestVersions.length - 1]?.id || latestVersions[0].id;
  }

  const persistedVersions = latestVersions.map((version) =>
    version.id === currentVersionId ? { ...version, content: "" } : version,
  );

  const versionStatusMap = Object.fromEntries(
    Object.entries(statusMap).filter(
      ([versionId, status]) =>
        versionIdSet.has(versionId) && isTopicBranchStatus(status),
    ),
  ) as Record<string, TopicBranchStatus>;

  return {
    versions: persistedVersions,
    currentVersionId,
    versionStatusMap,
  };
}

export function readPersistedGeneralWorkbenchDocument(
  metadata?: Record<string, unknown>,
): PersistedGeneralWorkbenchDocument | null {
  const raw =
    metadata?.[GENERAL_WORKBENCH_DOCUMENT_META_KEY] ??
    metadata?.[LEGACY_GENERAL_WORKBENCH_DOCUMENT_META_KEY];
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const candidate = raw as Record<string, unknown>;
  const versionsRaw = Array.isArray(candidate.versions)
    ? candidate.versions
    : [];
  const versions = versionsRaw
    .map((version) => normalizeDocumentVersion(version))
    .filter((version): version is DocumentVersion => !!version)
    .slice(-MAX_PERSISTED_DOCUMENT_VERSIONS);
  if (versions.length === 0) {
    return null;
  }

  const versionIdSet = new Set(versions.map((version) => version.id));
  const currentVersionIdRaw = candidate.currentVersionId;
  const currentVersionId =
    typeof currentVersionIdRaw === "string" &&
    versionIdSet.has(currentVersionIdRaw)
      ? currentVersionIdRaw
      : versions[versions.length - 1]?.id || versions[0].id;

  const statusRaw = candidate.versionStatusMap;
  const statusEntries =
    statusRaw && typeof statusRaw === "object" ? statusRaw : {};
  const versionStatusMap = Object.fromEntries(
    Object.entries(statusEntries).filter(
      ([versionId, status]) =>
        versionIdSet.has(versionId) && isTopicBranchStatus(status),
    ),
  ) as Record<string, TopicBranchStatus>;

  return {
    versions,
    currentVersionId,
    versionStatusMap,
  };
}

export function applyBackendGeneralWorkbenchDocumentState(
  state: CanvasStateUnion,
  backendState: GeneralWorkbenchDocumentState,
  currentBody: string,
): {
  state: CanvasStateUnion;
  statusMap: Record<string, TopicBranchStatus>;
} | null {
  if (state.type !== "document" || backendState.versions.length === 0) {
    return null;
  }

  const versions = backendState.versions
    .map((version, index) => ({
      id: version.id,
      content: version.is_current ? currentBody : "",
      createdAt: version.created_at,
      description: version.description?.trim() || `版本 ${index + 1}`,
    }))
    .slice(-MAX_PERSISTED_DOCUMENT_VERSIONS);

  if (versions.length === 0) {
    return null;
  }

  const currentVersion =
    versions.find(
      (version) => version.id === backendState.current_version_id,
    ) || versions[versions.length - 1];

  const statusMap = Object.fromEntries(
    backendState.versions
      .filter(
        (
          version,
        ): version is GeneralWorkbenchDocumentState["versions"][number] & {
          status: TopicBranchStatus;
        } => isTopicBranchStatus(version.status),
      )
      .map((version) => [version.id, version.status]),
  ) as Record<string, TopicBranchStatus>;

  return {
    state: {
      ...state,
      versions,
      currentVersionId: currentVersion.id,
      content: currentVersion.content,
    },
    statusMap,
  };
}

export function inferGeneralWorkbenchGateFromQueueItem(
  queueItem: GeneralWorkbenchRunTodoItem | null,
): {
  key: "topic_select" | "write_mode" | "publish_confirm";
  title: string;
  description: string;
} {
  const gateKey = queueItem?.gate_key;
  if (gateKey === "publish_confirm") {
    return {
      key: "publish_confirm",
      title: "发布闸门",
      description: queueItem?.title || "正在准备发布前检查与平台适配结果。",
    };
  }
  if (gateKey === "topic_select") {
    return {
      key: "topic_select",
      title: "选题闸门",
      description: queueItem?.title || "正在整理选题方向并生成可确认方案。",
    };
  }
  if (gateKey === "write_mode") {
    return {
      key: "write_mode",
      title: "写作闸门",
      description: queueItem?.title || "正在执行主稿写作与插图生成流程。",
    };
  }

  if (!queueItem) {
    return {
      key: "topic_select",
      title: "选题闸门",
      description: "正在整理选题方向并生成可确认方案。",
    };
  }

  const probe =
    `${queueItem.title} ${queueItem.source_ref || ""} ${queueItem.source}`.toLowerCase();
  const looksLikePublish =
    /publish|adapt|distribution|release|发布|分发|平台适配/.test(probe);
  if (looksLikePublish) {
    return {
      key: "publish_confirm",
      title: "发布闸门",
      description: queueItem.title || "正在准备发布前检查与平台适配结果。",
    };
  }

  const looksLikeTopic = /topic|research|trend|idea|选题|方向|调研|洞察/.test(
    probe,
  );
  if (looksLikeTopic) {
    return {
      key: "topic_select",
      title: "选题闸门",
      description: queueItem.title || "正在整理选题方向并生成可确认方案。",
    };
  }

  return {
    key: "write_mode",
    title: "写作闸门",
    description: queueItem.title || "正在执行主稿写作与插图生成流程。",
  };
}

export function resolveGeneralWorkbenchGateByKey(
  gateKey: "topic_select" | "write_mode" | "publish_confirm",
  fallbackTitle?: string,
): {
  key: "topic_select" | "write_mode" | "publish_confirm";
  title: string;
  description: string;
} {
  if (gateKey === "publish_confirm") {
    return {
      key: "publish_confirm",
      title: "发布闸门",
      description: fallbackTitle || "正在准备发布前检查与平台适配结果。",
    };
  }
  if (gateKey === "topic_select") {
    return {
      key: "topic_select",
      title: "选题闸门",
      description: fallbackTitle || "正在整理选题方向并生成可确认方案。",
    };
  }
  return {
    key: "write_mode",
    title: "写作闸门",
    description: fallbackTitle || "正在执行主稿写作与插图生成流程。",
  };
}

export function formatGeneralWorkbenchRunTimeLabel(
  raw: string | null | undefined,
): string {
  if (!raw) {
    return "--:--";
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return "--:--";
  }
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatGeneralWorkbenchRunDurationLabel(
  startedAt: string | null | undefined,
  finishedAt: string | null | undefined,
): string | undefined {
  if (!startedAt || !finishedAt) {
    return undefined;
  }

  const started = new Date(startedAt);
  const finished = new Date(finishedAt);
  if (Number.isNaN(started.getTime()) || Number.isNaN(finished.getTime())) {
    return undefined;
  }

  const durationMs = finished.getTime() - started.getTime();
  if (durationMs < 0) {
    return undefined;
  }
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  }
  return `${Math.floor(durationMs / 60000)}m${Math.round(
    (durationMs % 60000) / 1000,
  )}s`;
}

export function resolveGeneralWorkbenchApplyTargetByGateKey(
  gateKey: "topic_select" | "write_mode" | "publish_confirm" | "idle",
): string {
  if (gateKey === "topic_select") {
    return "选题池";
  }
  if (gateKey === "publish_confirm") {
    return "发布产物";
  }
  if (gateKey === "write_mode") {
    return "版本主稿";
  }
  return "主稿内容";
}

function extractExecutionIdFromSocialToolId(toolCallId: string): string | null {
  const normalized = toolCallId.trim();
  if (!normalized.startsWith("social-write-")) {
    return null;
  }
  const match = normalized.match(/^social-write-(.+)-[0-9a-f]{8}$/i);
  const executionId = match?.[1]?.trim();
  if (!executionId) {
    return null;
  }
  return executionId;
}

export function resolveExecutionIdCandidatesForActivityLog(
  log: SidebarActivityLog,
): string[] {
  const candidates: string[] = [];
  const pushCandidate = (value?: string | null) => {
    const normalized = value?.trim();
    if (!normalized) {
      return;
    }
    if (!candidates.includes(normalized)) {
      candidates.push(normalized);
    }
  };

  pushCandidate(log.executionId);
  pushCandidate(log.messageId);

  const normalizedLogId = log.id.trim();
  if (normalizedLogId) {
    let toolCallIdProbe = normalizedLogId;
    if (log.messageId) {
      const messagePrefix = `${log.messageId}-`;
      if (normalizedLogId.startsWith(messagePrefix)) {
        toolCallIdProbe = normalizedLogId.slice(messagePrefix.length);
      }
    }
    pushCandidate(extractExecutionIdFromSocialToolId(toolCallIdProbe));
  }

  return candidates;
}

export function isGeneralWorkbenchPrimaryDocumentArtifact(
  fileName: string,
): boolean {
  return isPrimaryResultDocumentPath(fileName);
}

const DOCUMENT_FILE_EXTENSIONS = new Set([
  "csv",
  "doc",
  "docx",
  "htm",
  "html",
  "json",
  "md",
  "mdx",
  "odt",
  "pdf",
  "ppt",
  "pptx",
  "rtf",
  "text",
  "toml",
  "tsv",
  "txt",
  "xls",
  "xlsx",
  "xml",
  "yaml",
  "yml",
]);

function inferTaskFileType(fileName: string): TaskFile["type"] {
  const normalized = fileName.trim().toLowerCase();
  const extension = normalized.split(".").pop() || "";

  if (DOCUMENT_FILE_EXTENSIONS.has(extension)) {
    return "document";
  }
  if (
    ["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "ico"].includes(
      extension,
    )
  ) {
    return "image";
  }
  if (
    ["mp3", "wav", "aac", "flac", "m4a", "ogg", "mid", "midi"].includes(
      extension,
    )
  ) {
    return "audio";
  }
  if (["mp4", "mov", "avi", "mkv", "webm"].includes(extension)) {
    return "video";
  }
  return "other";
}

export function looksLikeSocialPublishPayload(content: string): boolean {
  const trimmed = content.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return false;
  }

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    return (
      typeof parsed.article_path === "string" ||
      typeof parsed.cover_meta_path === "string" ||
      Array.isArray(parsed.pipeline) ||
      Array.isArray(parsed.recommended_channels)
    );
  } catch {
    return false;
  }
}

function looksLikeGeneralWorkbenchErrorPayload(content: string): boolean {
  const normalized = content.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  return (
    normalized.startsWith("ran into this error:") ||
    normalized.startsWith("request failed:") ||
    normalized.includes(
      "please retry if you think this is a transient or recoverable error.",
    ) ||
    normalized.includes("api key not valid")
  );
}

export function isCorruptedGeneralWorkbenchDocumentContent(
  content?: string | null,
): boolean {
  if (typeof content !== "string") {
    return false;
  }

  return (
    looksLikeSocialPublishPayload(content) ||
    looksLikeGeneralWorkbenchErrorPayload(content)
  );
}

export function resolveTaskFileType(
  fileName: string,
  content?: string | null,
): TaskFile["type"] {
  const inferredType = inferTaskFileType(fileName);
  if (
    inferredType === "document" &&
    isCorruptedGeneralWorkbenchDocumentContent(content)
  ) {
    return "other";
  }
  return inferredType;
}

export function normalizeSessionTaskFileType(
  fileType: string,
  fileName: string,
  content?: string | null,
): TaskFile["type"] {
  const normalized = fileType.trim().toLowerCase();
  if (
    normalized === "document" ||
    normalized === "image" ||
    normalized === "audio" ||
    normalized === "video" ||
    normalized === "other"
  ) {
    const resolvedByContent = resolveTaskFileType(fileName, content);
    if (normalized === "document" && resolvedByContent !== "document") {
      return resolvedByContent;
    }
    return normalized;
  }
  return resolveTaskFileType(fileName, content);
}

export function isRenderableTaskFile(
  file: Pick<TaskFile, "name" | "type">,
  isThemeWorkbench: boolean,
): boolean {
  if (file.type !== "document") {
    return false;
  }
  if (!isThemeWorkbench) {
    return true;
  }
  return isGeneralWorkbenchPrimaryDocumentArtifact(file.name);
}

export function buildGeneralWorkbenchWorkflowSteps(
  messages: Message[],
  backendRunState: BackendGeneralWorkbenchRunState | null,
  isSending: boolean,
  skillDetailMap: Record<string, SkillDetailInfo | null>,
): Array<{ id: string; title: string; status: StepStatus }> {
  const liveSteps = buildThemeWorkbenchLiveWorkflowSteps(
    messages,
    skillDetailMap,
    isSending,
  );
  if (liveSteps.length > 0) {
    return liveSteps;
  }

  const queueItems = backendRunState?.queue_items || [];
  if (queueItems.length > 0) {
    if (queueItems.length === 1) {
      const item = queueItems[0];
      const sourceRef = resolveGeneralWorkbenchSkillSourceRef(item);
      const workflowSteps = sourceRef
        ? skillDetailMap[sourceRef]?.workflow_steps || []
        : [];
      if (workflowSteps.length > 0) {
        const latestAssistantContent =
          messages
            .slice()
            .reverse()
            .find((m) => m.role === "assistant")?.content || "";
        const activeIndex =
          extractThemeWorkbenchWorkflowMarkerIndex(latestAssistantContent) ?? 0;
        return workflowSteps.map((step, index) => ({
          id: `${item.run_id}-${step.id}`,
          title: step.name,
          status:
            index < activeIndex
              ? ("completed" as StepStatus)
              : index === activeIndex
                ? ("active" as StepStatus)
                : ("pending" as StepStatus),
        }));
      }
    }
    return queueItems.map((item) => ({
      id: item.run_id,
      title: resolveThemeWorkbenchQueueItemTitle(item, skillDetailMap),
      status: resolveThemeWorkbenchRunStepStatus(item.status),
    }));
  }

  const latestTerminal = backendRunState?.latest_terminal;
  if (latestTerminal && backendRunState?.run_state !== "auto_running") {
    return [
      {
        id: latestTerminal.run_id,
        title: resolveThemeWorkbenchQueueItemTitle(
          latestTerminal,
          skillDetailMap,
        ),
        status: resolveThemeWorkbenchRunStepStatus(latestTerminal.status),
      },
    ];
  }

  return [];
}

export function loadPersistedBoolean(key: string, fallback = false): boolean {
  try {
    const stored = localStorage.getItem(key);
    if (stored == null) {
      return fallback;
    }

    try {
      const parsed = JSON.parse(stored);
      return typeof parsed === "boolean" ? parsed : fallback;
    } catch {
      return stored === "true";
    }
  } catch {
    return fallback;
  }
}

export function savePersistedBoolean(key: string, value: boolean) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore write errors
  }
}

export function isCanvasStateEmpty(state: CanvasStateUnion | null): boolean {
  if (!state) return true;

  switch (state.type) {
    case "document":
      return !state.content || state.content.trim() === "";
    case "video":
      return !state.prompt || state.prompt.trim() === "";
    case "design":
      return state.document.layers.length === 0;
    default:
      return true;
  }
}

export function serializeCanvasStateForSync(state: CanvasStateUnion): string {
  switch (state.type) {
    case "document":
      return state.content || "";
    case "video":
      return state.prompt || "";
    case "design":
      return JSON.stringify(state.document, null, 2);
    default:
      return JSON.stringify(state);
  }
}

export function isSyncContentEmpty(content: string): boolean {
  return !content || content === "[]" || content === "{}";
}

export function resolveGeneralWorkbenchRecentTerminals(
  state: BackendGeneralWorkbenchRunState | null,
): GeneralWorkbenchRunTerminalItem[] {
  if (!state) {
    return [];
  }

  const rawTerminals =
    Array.isArray(state.recent_terminals) && state.recent_terminals.length > 0
      ? state.recent_terminals
      : state.latest_terminal
        ? [state.latest_terminal]
        : [];

  const seenRunIds = new Set<string>();
  return rawTerminals.filter((item) => {
    const runId = item.run_id?.trim();
    if (!runId || seenRunIds.has(runId)) {
      return false;
    }
    seenRunIds.add(runId);
    return true;
  });
}

export function mergeGeneralWorkbenchTerminalItems(
  ...groups: GeneralWorkbenchRunTerminalItem[][]
): GeneralWorkbenchRunTerminalItem[] {
  const merged: GeneralWorkbenchRunTerminalItem[] = [];
  const seenRunIds = new Set<string>();

  groups.forEach((items) => {
    items.forEach((item) => {
      const runId = item.run_id?.trim();
      if (!runId || seenRunIds.has(runId)) {
        return;
      }
      seenRunIds.add(runId);
      merged.push(item);
    });
  });

  return merged;
}

export function buildGeneralWorkbenchRunStateSignature(
  state: BackendGeneralWorkbenchRunState | null,
): string {
  if (!state) {
    return "null";
  }

  const queueSignature = (state.queue_items || [])
    .map((item) =>
      [
        item.run_id,
        item.execution_id || "",
        item.status,
        item.gate_key || "",
        item.source || "",
        item.source_ref || "",
      ].join(":"),
    )
    .join("|");

  const terminalSignature = resolveGeneralWorkbenchRecentTerminals(state)
    .map((item) =>
      [
        item.run_id,
        item.execution_id || "",
        item.status,
        item.gate_key || "",
        item.source || "",
        item.source_ref || "",
      ].join(":"),
    )
    .join("|");

  return [
    state.run_state,
    state.current_gate_key || "",
    queueSignature,
    terminalSignature,
  ].join("||");
}
