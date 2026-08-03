import type { AgentThreadItem } from "../types";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(
  record: Record<string, unknown> | null,
  keys: string[],
): string | null {
  for (const key of keys) {
    const value = record?.[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function normalizeMarker(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, ".");
}

function isReadMarker(value: string | null): boolean {
  if (!value) {
    return false;
  }

  return new Set(["read", "file.read", "read.file", "file.read.item"]).has(
    normalizeMarker(value),
  );
}

export function isTimelineReadOnlyFileArtifact(
  item: AgentThreadItem,
): boolean {
  if (item.type !== "file_artifact") {
    return false;
  }

  const metadata = asRecord(item.metadata);
  const sourceProvenance =
    asRecord(metadata?.source_provenance) ||
    asRecord(metadata?.sourceProvenance);
  const source = item.source.trim();
  const markers = [
    source,
    readString(metadata, [
      "source",
      "eventClass",
      "event_class",
      "operation",
      "action",
      "toolName",
      "tool_name",
      "sourceEventType",
      "source_event_type",
    ]),
    readString(sourceProvenance, [
      "operation",
      "action",
      "toolName",
      "tool_name",
      "sourceEventType",
      "source_event_type",
    ]),
  ];

  return markers.some(isReadMarker);
}
