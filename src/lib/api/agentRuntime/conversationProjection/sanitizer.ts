import type { AgentThreadItem } from "../../agentProtocol";

export const MAX_PROJECTION_OUTPUT_BYTES = 256 * 1024;
export const PROJECTION_OUTPUT_TRUNCATION_MARKER =
  "[... earlier output truncated ...]\n";

export function boundProjectionOutput(value: string): string {
  if (utf8ByteLength(value) <= MAX_PROJECTION_OUTPUT_BYTES) {
    return value;
  }

  const retainedBytes = Math.max(
    0,
    MAX_PROJECTION_OUTPUT_BYTES -
      utf8ByteLength(PROJECTION_OUTPUT_TRUNCATION_MARKER),
  );
  return `${PROJECTION_OUTPUT_TRUNCATION_MARKER}${trailingUtf8Bytes(
    value,
    retainedBytes,
  )}`;
}

function utf8ByteLength(value: string): number {
  let length = 0;
  for (const character of value) {
    length += utf8CodePointLength(character.codePointAt(0) ?? 0);
  }
  return length;
}

function trailingUtf8Bytes(value: string, maxBytes: number): string {
  let bytes = 0;
  let start = value.length;
  while (start > 0) {
    let candidateStart = start - 1;
    const codeUnit = value.charCodeAt(candidateStart);
    if (
      codeUnit >= 0xdc00 &&
      codeUnit <= 0xdfff &&
      candidateStart > 0 &&
      value.charCodeAt(candidateStart - 1) >= 0xd800 &&
      value.charCodeAt(candidateStart - 1) <= 0xdbff
    ) {
      candidateStart -= 1;
    }

    const candidateBytes = utf8CodePointLength(
      value.codePointAt(candidateStart) ?? 0,
    );
    if (bytes + candidateBytes > maxBytes) {
      break;
    }
    bytes += candidateBytes;
    start = candidateStart;
  }
  return value.slice(start);
}

function utf8CodePointLength(codePoint: number): number {
  if (codePoint <= 0x7f) return 1;
  if (codePoint <= 0x7ff) return 2;
  if (codePoint <= 0xffff) return 3;
  return 4;
}

export function mergeProjectionOutput(
  current: string | undefined,
  value: string,
  mode: "append" | "replace" = "append",
): string {
  return boundProjectionOutput(
    mode === "replace" ? value : `${current ?? ""}${value}`,
  );
}

export function sanitizeProjectionItem(item: AgentThreadItem): AgentThreadItem {
  switch (item.type) {
    case "command_execution":
      return item.aggregated_output === undefined
        ? item
        : {
            ...item,
            aggregated_output: boundProjectionOutput(item.aggregated_output),
          };
    case "tool_call":
    case "web_search":
      return item.output === undefined
        ? item
        : { ...item, output: boundProjectionOutput(item.output) };
    case "patch":
      return {
        ...item,
        ...(item.stdout === undefined
          ? {}
          : { stdout: boundProjectionOutput(item.stdout) }),
        ...(item.stderr === undefined
          ? {}
          : { stderr: boundProjectionOutput(item.stderr) }),
      };
    default:
      return item;
  }
}
