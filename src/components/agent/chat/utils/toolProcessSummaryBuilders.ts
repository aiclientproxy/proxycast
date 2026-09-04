import { resolveRequiredAgentChatCopy } from "./agentChatCopy";
import { resolveContentWorkbenchToolCopy } from "./contentWorkbenchToolCopy";
import {
  getHostnameFromUrl,
  isUnifiedWebSearchToolName,
  resolveSearchResultPreviewItemsFromText,
} from "./searchResultPreview";
import { normalizeSiteToolResultSummary } from "./siteToolResultSummary";
import {
  getToolDisplayInfo,
  isBrowserToolName,
  normalizeToolNameKey,
} from "./toolDisplayInfo";
import {
  normalizeToolSearchResultSummary,
  resolveUserFacingToolSearchItemLabel,
} from "./toolSearchResultSummary";
import {
  buildGenericPostSummary as buildGenericPostSummaryBase,
  buildKnownPreSummary,
  buildPreSummaryByFamily,
} from "./toolProcessGenericSummary";
import type { ToolProcessStatus } from "./toolProcessSummaryTypes";
import {
  normalizeNarrativeSubject,
  resolvePhasedProcessSummaryCopy,
} from "./toolProcessSummaryCopy";
import {
  resolveToolProcessFactsFamily,
  resolveToolProcessFactsSubject,
} from "./toolProcessSummaryMetadata";
import {
  asRecord,
  normalizeArgumentsRecord,
  readString,
  resolveToolSubject,
  shorten,
} from "./toolProcessSummaryText";

export {
  buildVisionToolSummary,
  normalizeNarrativeSubject,
} from "./toolProcessSummaryCopy";

function resolveUrlLabel(
  args: Record<string, unknown>,
  metadata: Record<string, unknown> | null,
): string | null {
  const rawUrl =
    readString(args, ["url", "pageUrl", "page_url", "href"]) ||
    readString(metadata, ["url", "pageUrl", "page_url", "href"]);
  if (!rawUrl) {
    return null;
  }

  const hostname = shorten(getHostnameFromUrl(rawUrl), 48);
  return hostname || shorten(rawUrl, 64);
}

function buildToolSearchPreSummary(args: Record<string, unknown>): string {
  const query = readString(args, ["query", "q"]) || "";
  if (/^(?:select|tool|tools|name|tag):/i.test(query)) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.toolSearch.confirmAvailable",
    );
  }
  if (query) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.toolSearch.findForQuery",
      { query: shorten(query, 32) },
    );
  }
  return resolveRequiredAgentChatCopy(
    "toolCall.processSummary.toolSearch.confirmAvailable",
  );
}

export function buildToolSearchPostSummary(output: string): string | null {
  const summary = normalizeToolSearchResultSummary(output);
  if (!summary) {
    return null;
  }

  const toolNames = summary.tools
    .slice(0, 2)
    .map((item) => resolveUserFacingToolSearchItemLabel(item.name))
    .filter(Boolean);
  const prefix = resolveRequiredAgentChatCopy(
    "toolCall.processSummary.toolSearch.confirmedCount",
    { count: summary.count },
  );

  if (toolNames.length === 0) {
    return prefix;
  }

  return resolveRequiredAgentChatCopy(
    "toolCall.processSummary.toolSearch.confirmedWithTools",
    {
      countLabel: prefix,
      tools: toolNames.join(" · "),
    },
  );
}

export function buildWebSearchPostSummary(output: string): string | null {
  const items = resolveSearchResultPreviewItemsFromText(output);
  if (items.length === 0) {
    return null;
  }

  return resolveRequiredAgentChatCopy(
    "toolCall.processSummary.webSearch.sourcesFound",
    { count: items.length },
  );
}

export function buildFetchSearchFailureSummary(
  family: "fetch" | "search",
): string {
  return family === "fetch"
    ? resolveRequiredAgentChatCopy("toolCall.processSummary.fetch.unavailable")
    : resolveRequiredAgentChatCopy(
        "toolCall.processSummary.search.unavailable",
      );
}

function resolveSiteProjectTargetCopy(params: {
  source?: string;
  projectId?: string;
}): string {
  if (params.source === "context_project") {
    return resolveRequiredAgentChatCopy(
      "toolCall.siteResult.target.currentProject",
    );
  }
  if (params.source === "explicit_project") {
    return resolveRequiredAgentChatCopy(
      "toolCall.siteResult.target.selectedProject",
    );
  }
  if (params.projectId?.trim()) {
    return resolveRequiredAgentChatCopy("toolCall.siteResult.target.project", {
      projectId: params.projectId.trim(),
    });
  }
  return resolveRequiredAgentChatCopy("toolCall.siteResult.target.generic");
}

export function buildSitePostSummary(metadata: unknown): string | null {
  const summary = normalizeSiteToolResultSummary(metadata);
  if (!summary) {
    return null;
  }

  if (summary.saveErrorMessage) {
    return resolveRequiredAgentChatCopy("toolCall.siteResult.saveError", {
      message: shorten(summary.saveErrorMessage, 56),
    });
  }

  if (summary.savedContent?.title) {
    return resolveRequiredAgentChatCopy("toolCall.siteResult.saved", {
      target: resolveSiteProjectTargetCopy({
        source: summary.savedBy,
        projectId: summary.savedProjectId || summary.savedContent.projectId,
      }),
      title: summary.savedContent.title,
    });
  }

  if (summary.savedContent?.markdownRelativePath) {
    return resolveRequiredAgentChatCopy("toolCall.siteResult.markdownExported");
  }

  if (summary.saveSkippedProjectId) {
    return resolveRequiredAgentChatCopy("toolCall.siteResult.saveSkipped", {
      target: resolveSiteProjectTargetCopy({
        source: summary.saveSkippedBy,
        projectId: summary.saveSkippedProjectId,
      }),
    });
  }

  return null;
}

const LIME_TASK_SUMMARY_LABELS: Partial<
  Record<string, { key: string; defaultValue: string }>
> = {
  videogenerate: {
    key: "label.videoGeneration",
    defaultValue: "Video generation",
  },
  limecreateaudiogenerationtask: {
    key: "label.audioGeneration",
    defaultValue: "Voice generation",
  },
  limecreatetranscriptiontask: {
    key: "label.transcription",
    defaultValue: "Transcription",
  },
  limecreatebroadcastgenerationtask: {
    key: "label.broadcastGeneration",
    defaultValue: "Broadcast generation",
  },
  limecreatecovergenerationtask: {
    key: "label.coverGeneration",
    defaultValue: "Cover generation",
  },
  limecreateresourcesearchtask: {
    key: "label.resourceSearch",
    defaultValue: "Asset search",
  },
  limecreatemodalresourcesearchtask: {
    key: "label.resourceSearch",
    defaultValue: "Asset search",
  },
  limecreateimagegenerationtask: {
    key: "label.imageGeneration",
    defaultValue: "Image generation",
  },
  limecreateurlparsetask: {
    key: "label.urlParse",
    defaultValue: "URL parsing",
  },
  limecreatetypesettingtask: {
    key: "label.typesetting",
    defaultValue: "Typesetting",
  },
};

const DIRECT_CONTENT_SUMMARY_LABELS: Partial<
  Record<string, { key: string; defaultValue: string }>
> = {
  socialgeneratecoverimage: {
    key: "label.coverImage",
    defaultValue: "cover image",
  },
  generateimage: {
    key: "label.image",
    defaultValue: "image",
  },
};

function buildLimeTaskSummary(
  phase: "pre" | "post",
  normalizedName: string,
  subject: string | null,
): string | null {
  const directLabel = DIRECT_CONTENT_SUMMARY_LABELS[normalizedName];
  if (directLabel) {
    const normalizedSubject = normalizeNarrativeSubject(subject);
    const label = resolveContentWorkbenchToolCopy(
      directLabel.key,
      directLabel.defaultValue,
    );
    return normalizedSubject
      ? resolveContentWorkbenchToolCopy(`summary.direct.${phase}WithSubject`, {
          subject: normalizedSubject,
          label,
        })
      : resolveContentWorkbenchToolCopy(`summary.direct.${phase}`, { label });
  }

  const taskLabel = LIME_TASK_SUMMARY_LABELS[normalizedName];
  if (!taskLabel) {
    return null;
  }

  const normalizedSubject = normalizeNarrativeSubject(subject);
  const label = resolveContentWorkbenchToolCopy(
    taskLabel.key,
    taskLabel.defaultValue,
  );

  return normalizedSubject
    ? resolveContentWorkbenchToolCopy(`summary.task.${phase}WithSubject`, {
        subject: normalizedSubject,
        label,
      })
    : resolveContentWorkbenchToolCopy(`summary.task.${phase}`, { label });
}

function buildSiteToolSummary(
  phase: "pre" | "post",
  normalizedName: string,
  subject: string | null,
): string | null {
  const normalizedSubject = normalizeNarrativeSubject(subject, [
    resolveRequiredAgentChatCopy("toolCall.subject.siteCapability"),
    resolveRequiredAgentChatCopy("toolCall.subject.siteCapabilityCatalog"),
    resolveRequiredAgentChatCopy("toolCall.subject.siteAdapter"),
  ]);

  if (normalizedName === "limesitelist") {
    return resolvePhasedProcessSummaryCopy(
      "toolCall.processSummary.siteCapability.list",
      phase,
      null,
    );
  }

  if (normalizedName === "limesiterecommend") {
    return resolvePhasedProcessSummaryCopy(
      "toolCall.processSummary.siteCapability.recommend",
      phase,
      normalizedSubject,
    );
  }

  if (normalizedName === "limesitesearch") {
    return resolvePhasedProcessSummaryCopy(
      "toolCall.processSummary.siteCapability.search",
      phase,
      normalizedSubject,
    );
  }

  if (normalizedName === "limesiteinfo") {
    return resolvePhasedProcessSummaryCopy(
      "toolCall.processSummary.siteCapability.info",
      phase,
      normalizedSubject,
    );
  }

  if (normalizedName === "limesiterun") {
    return resolvePhasedProcessSummaryCopy(
      "toolCall.processSummary.siteCapability.run",
      phase,
      normalizedSubject,
    );
  }

  return null;
}

function buildBrowserPreSummary(
  normalizedName: string,
  args: Record<string, unknown>,
  metadata: Record<string, unknown> | null,
): string | null {
  const urlLabel = resolveUrlLabel(args, metadata);
  const target =
    readString(args, ["selector", "element", "target", "label", "text"]) ||
    readString(metadata, ["selector", "element", "target", "label", "text"]);

  if (normalizedName.includes("navigate") || normalizedName.includes("goto")) {
    return urlLabel
      ? resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.openUrl",
          { target: urlLabel },
        )
      : resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.openTargetPage",
        );
  }

  if (
    normalizedName.includes("snapshot") ||
    normalizedName.includes("screenshot")
  ) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.browser.capturePageState",
    );
  }

  if (normalizedName.includes("click")) {
    return target
      ? resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.operateTarget",
          { target: shorten(target, 28) },
        )
      : resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.operateElement",
        );
  }

  if (
    normalizedName.includes("fill") ||
    normalizedName.includes("type") ||
    normalizedName.includes("selectoption") ||
    normalizedName.includes("presskey")
  ) {
    return target
      ? resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.fillTarget",
          { target: shorten(target, 28) },
        )
      : resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.continueOperation",
        );
  }

  if (
    normalizedName.includes("evaluate") ||
    normalizedName.includes("runtime")
  ) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.browser.readPageInfo",
    );
  }

  return resolveRequiredAgentChatCopy(
    "toolCall.processSummary.browser.viewPageState",
  );
}

export function buildBrowserPostSummary(
  normalizedName: string,
  args: Record<string, unknown>,
  metadata: Record<string, unknown> | null,
): string | null {
  const urlLabel = resolveUrlLabel(args, metadata);

  if (normalizedName.includes("navigate") || normalizedName.includes("goto")) {
    return urlLabel
      ? resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.openedUrl",
          { target: urlLabel },
        )
      : resolveRequiredAgentChatCopy(
          "toolCall.processSummary.browser.openedTargetPage",
        );
  }

  if (
    normalizedName.includes("snapshot") ||
    normalizedName.includes("screenshot")
  ) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.browser.snapshotCaptured",
    );
  }

  if (
    normalizedName.includes("evaluate") ||
    normalizedName.includes("runtime")
  ) {
    return resolveRequiredAgentChatCopy(
      "toolCall.processSummary.browser.stateCaptured",
    );
  }

  return resolveRequiredAgentChatCopy(
    "toolCall.processSummary.browser.operationCompleted",
  );
}

export function buildGenericPostSummary(params: {
  toolName: string;
  status: ToolProcessStatus;
  subject: string | null;
  factsFamily?: ReturnType<typeof getToolDisplayInfo>["family"] | null;
}): string | null {
  const normalizedName = normalizeToolNameKey(params.toolName);
  const normalizedSubject = normalizeNarrativeSubject(params.subject);

  return buildGenericPostSummaryBase({
    ...params,
    displayFamily: params.factsFamily,
    limeTaskSummary: buildLimeTaskSummary(
      "post",
      normalizedName,
      normalizedSubject,
    ),
    siteToolSummary: buildSiteToolSummary(
      "post",
      normalizedName,
      normalizedSubject,
    ),
  });
}

export function buildGenericPreSummary(params: {
  toolName: string;
  argumentsValue?: string | Record<string, unknown>;
  metadata?: unknown;
  factsSubject?: string | null;
  factsFamily?: ReturnType<typeof getToolDisplayInfo>["family"] | null;
}): string | null {
  const { toolName, argumentsValue, metadata } = params;
  const normalizedName = normalizeToolNameKey(toolName);
  const args = normalizeArgumentsRecord(argumentsValue);
  const metadataRecord = asRecord(metadata);
  const subject =
    params.factsSubject ||
    resolveToolProcessFactsSubject(metadata) ||
    resolveToolSubject(toolName, argumentsValue);
  const factsFamily =
    params.factsFamily || resolveToolProcessFactsFamily(metadata);
  const normalizedSubject = normalizeNarrativeSubject(subject);
  const query =
    readString(args, ["query", "q", "pattern", "search_query"]) ||
    readString(metadataRecord, ["query", "q", "pattern", "search_query"]) ||
    subject;

  if (normalizedName === "toolsearch") {
    return buildToolSearchPreSummary(args);
  }

  if (isUnifiedWebSearchToolName(toolName)) {
    return query
      ? resolveRequiredAgentChatCopy(
          "toolCall.processSummary.webSearch.searchFirstWithQuery",
          { query: shorten(query, 36) },
        )
      : resolveRequiredAgentChatCopy(
          "toolCall.processSummary.webSearch.searchFirst",
        );
  }

  if (
    factsFamily === "browser" ||
    (!factsFamily && isBrowserToolName(normalizedName))
  ) {
    return buildBrowserPreSummary(normalizedName, args, metadataRecord);
  }

  const limeTaskSummary = buildLimeTaskSummary(
    "pre",
    normalizedName,
    normalizedSubject,
  );
  const siteToolSummary = buildSiteToolSummary(
    "pre",
    normalizedName,
    normalizedSubject,
  );
  const knownSummary = buildKnownPreSummary({
    normalizedName,
    normalizedSubject,
    subject,
    limeTaskSummary,
    siteToolSummary,
  });
  if (knownSummary) {
    return knownSummary;
  }

  const display = getToolDisplayInfo(toolName, "running");
  return buildPreSummaryByFamily({
    displayFamily: factsFamily || display.family,
    normalizedName,
    normalizedSubject,
    query,
    args,
    urlLabel: resolveUrlLabel(args, metadataRecord),
  });
}
