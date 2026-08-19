import type { AgentToolCallState } from "@/lib/api/agentProtocol";
import type {
  ArtifactDocumentSource,
  ArtifactDocumentSourceLink,
  ArtifactDocumentV1,
} from "@/lib/artifact-document";
import { extractArtifactProtocolPathsFromValue } from "@/lib/artifact-protocol";
import { resolveSearchResultPreviewItemsFromText } from "./searchResultPreview";

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalizeText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, "/").trim();
}

function fileNameFromPath(path: string): string {
  const normalized = normalizePath(path);
  const segments = normalized.split("/");
  return segments[segments.length - 1] || normalized;
}

function parseLooseJsonValue(raw?: string): unknown {
  const normalized = raw?.trim();
  if (!normalized) {
    return undefined;
  }

  const candidates = [normalized];
  const fenced = normalized.match(/```(?:json)?\s*([\s\S]*?)\s*```/i)?.[1];
  if (fenced) {
    candidates.unshift(fenced.trim());
  }

  const objectBlock = normalized.match(/\{[\s\S]*\}/)?.[0];
  if (objectBlock) {
    candidates.unshift(objectBlock.trim());
  }

  const arrayBlock = normalized.match(/\[[\s\S]*\]/)?.[0];
  if (arrayBlock) {
    candidates.unshift(arrayBlock.trim());
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      continue;
    }
  }

  return undefined;
}

function sourceIdentity(source: ArtifactDocumentSource): string {
  const locatorUrl = normalizeText(source.locator?.url)?.toLowerCase();
  if (locatorUrl) {
    return `url:${locatorUrl}`;
  }
  const locatorPath = normalizeText(source.locator?.path)?.toLowerCase();
  if (locatorPath) {
    return `path:${locatorPath}`;
  }

  return [
    normalizeText(source.id)?.toLowerCase(),
    normalizeText(source.type)?.toLowerCase(),
    normalizeText(source.label)?.toLowerCase(),
    normalizeText(source.snippet)?.toLowerCase(),
  ]
    .filter(Boolean)
    .join("::");
}

function sourceLinkIdentity(link: ArtifactDocumentSourceLink): string {
  return [
    link.blockId,
    link.sourceId || "",
    link.sourceType,
    link.sourceRef,
    normalizeText(link.label) || "",
  ].join("::");
}

function mergeSource(
  current: ArtifactDocumentSource,
  incoming: ArtifactDocumentSource,
): ArtifactDocumentSource {
  return {
    ...current,
    ...incoming,
    id: current.id || incoming.id,
    type: incoming.type || current.type,
    label: incoming.label || current.label,
    ...(current.locator || incoming.locator
      ? {
          locator: {
            ...(current.locator || {}),
            ...(incoming.locator || {}),
          },
        }
      : {}),
    snippet: incoming.snippet || current.snippet,
    reliability: incoming.reliability || current.reliability,
  };
}

export function mergeArtifactDocumentSources(
  existing: ArtifactDocumentSource[],
  incoming: ArtifactDocumentSource[],
): ArtifactDocumentSource[] {
  if (incoming.length === 0) {
    return existing;
  }

  const merged: ArtifactDocumentSource[] = [];
  const indexByIdentity = new Map<string, number>();

  for (const source of existing) {
    const identity = sourceIdentity(source);
    indexByIdentity.set(identity, merged.length);
    merged.push(source);
  }

  for (const source of incoming) {
    const identity = sourceIdentity(source);
    const currentIndex = indexByIdentity.get(identity);
    if (currentIndex === undefined) {
      indexByIdentity.set(identity, merged.length);
      merged.push(source);
      continue;
    }

    merged[currentIndex] = mergeSource(merged[currentIndex], source);
  }

  return merged;
}

function mergeArtifactDocumentSourceLinks(
  existing: ArtifactDocumentSourceLink[],
  incoming: ArtifactDocumentSourceLink[],
): ArtifactDocumentSourceLink[] {
  if (incoming.length === 0) {
    return existing;
  }

  const merged: ArtifactDocumentSourceLink[] = [];
  const seen = new Set<string>();

  for (const link of [...existing, ...incoming]) {
    const identity = sourceLinkIdentity(link);
    if (seen.has(identity)) {
      continue;
    }
    seen.add(identity);
    merged.push(link);
  }

  return merged;
}

export function mergeArtifactDocuments(
  primary: ArtifactDocumentV1 | null,
  secondary: ArtifactDocumentV1 | null,
): ArtifactDocumentV1 | null {
  if (!primary) {
    return secondary;
  }
  if (!secondary || primary === secondary) {
    return primary;
  }

  const sourceLinks = mergeArtifactDocumentSourceLinks(
    primary.metadata.sourceLinks || [],
    secondary.metadata.sourceLinks || [],
  );

  return {
    ...secondary,
    ...primary,
    summary: primary.summary || secondary.summary,
    sources: mergeArtifactDocumentSources(primary.sources, secondary.sources),
    metadata: {
      ...secondary.metadata,
      ...primary.metadata,
      ...(sourceLinks.length > 0 ? { sourceLinks } : {}),
    },
  };
}

export function mergeSourcesIntoArtifactDocument(
  document: ArtifactDocumentV1 | null,
  sources: ArtifactDocumentSource[],
): ArtifactDocumentV1 | null {
  if (!document || sources.length === 0) {
    return document;
  }

  return {
    ...document,
    sources: mergeArtifactDocumentSources(document.sources, sources),
  };
}

function hasStructuredSourceHints(value: unknown, depth = 0): boolean {
  if (depth > 4) {
    return false;
  }

  if (Array.isArray(value)) {
    return value.some((item) => hasStructuredSourceHints(item, depth + 1));
  }

  const record = asRecord(value);
  if (!record) {
    return false;
  }

  for (const key of [
    "sources",
    "citations",
    "results",
    "items",
    "data",
    "url",
    "href",
    "link",
    "target_url",
    "targetUrl",
    "page_info",
    "pageInfo",
    "last_page_info",
    "lastPageInfo",
  ]) {
    if (record[key] !== undefined) {
      return true;
    }
  }

  return Object.values(record).some((item) =>
    hasStructuredSourceHints(item, depth + 1),
  );
}

function isSearchLikeToolName(toolName: string): boolean {
  const normalized = toolName.toLowerCase().replace(/[\s_-]+/g, "");
  return normalized.includes("search") || normalized.includes("webquery");
}

function isBrowserToolName(toolName: string): boolean {
  return toolName.trim().toLowerCase().startsWith("browser__");
}

function buildFileSource(path: string): ArtifactDocumentSource {
  const normalizedPath = normalizePath(path);
  return {
    id: `file:${normalizedPath}`,
    type: "file",
    label: fileNameFromPath(normalizedPath),
    locator: {
      path: normalizedPath,
    },
    reliability: "primary",
  };
}

export function extractArtifactDocumentSourcesFromToolCall(
  toolCall: Pick<AgentToolCallState, "id" | "name" | "arguments" | "result">,
): ArtifactDocumentSource[] {
  const result = toolCall.result;
  if (!result) {
    return [];
  }

  const collected: ArtifactDocumentSource[] = [];
  const parsedArguments = parseLooseJsonValue(toolCall.arguments);
  const parsedOutput = parseLooseJsonValue(result.output);
  const metadataRecord = asRecord(result.metadata);
  const browserTool = isBrowserToolName(toolCall.name);

  const pathCandidates = new Set<string>();
  for (const path of extractArtifactProtocolPathsFromValue(parsedArguments)) {
    pathCandidates.add(path);
  }
  for (const path of extractArtifactProtocolPathsFromValue(metadataRecord)) {
    pathCandidates.add(path);
  }
  for (const path of extractArtifactProtocolPathsFromValue(parsedOutput)) {
    pathCandidates.add(path);
  }
  for (const path of pathCandidates) {
    collected.push(buildFileSource(path));
  }

  const shouldExtractWebSources =
    isSearchLikeToolName(toolCall.name) ||
    browserTool ||
    hasStructuredSourceHints(metadataRecord) ||
    hasStructuredSourceHints(parsedOutput);
  if (shouldExtractWebSources) {
    const previewInputs = new Set<string>();
    if (normalizeText(result.output)) {
      previewInputs.add(result.output);
    }
    if (metadataRecord && hasStructuredSourceHints(metadataRecord)) {
      previewInputs.add(JSON.stringify(metadataRecord));
    }
    if (parsedOutput && hasStructuredSourceHints(parsedOutput)) {
      previewInputs.add(JSON.stringify(parsedOutput));
    }

    for (const rawText of previewInputs) {
      for (const item of resolveSearchResultPreviewItemsFromText(rawText)) {
        collected.push({
          id: `web:${item.url}`,
          type: browserTool ? "web" : "search_result",
          label: item.title,
          locator: {
            url: item.url,
          },
          snippet: item.snippet,
          reliability: "secondary",
        });
      }
    }
  }

  return mergeArtifactDocumentSources([], collected);
}

export function collectArtifactDocumentSourcesFromToolCalls(
  toolCalls: Array<
    Pick<AgentToolCallState, "id" | "name" | "arguments" | "result">
  >,
): ArtifactDocumentSource[] {
  return toolCalls.reduce<ArtifactDocumentSource[]>((allSources, toolCall) => {
    const nextSources = extractArtifactDocumentSourcesFromToolCall(toolCall);
    if (nextSources.length === 0) {
      return allSources;
    }
    return mergeArtifactDocumentSources(allSources, nextSources);
  }, []);
}
