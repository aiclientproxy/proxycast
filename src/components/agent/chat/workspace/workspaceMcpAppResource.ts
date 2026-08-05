import type { EmbeddedBrowserHtmlParams } from "@/lib/api/embeddedBrowser";
import type { McpResourceContent } from "@/lib/api/mcp";

export type WorkspaceMcpAppResourceErrorCode =
  | "invalidMimeType"
  | "missingHtml"
  | "uriMismatch";

export class WorkspaceMcpAppResourceError extends Error {
  constructor(readonly code: WorkspaceMcpAppResourceErrorCode) {
    super(code);
    this.name = "WorkspaceMcpAppResourceError";
  }
}

export function buildWorkspaceMcpAppHtmlParams({
  content,
  expectedUri,
  viewId,
}: {
  content: McpResourceContent;
  expectedUri: string;
  viewId: string;
}): EmbeddedBrowserHtmlParams {
  if (content.uri !== expectedUri) {
    throw new WorkspaceMcpAppResourceError("uriMismatch");
  }
  if (!isMcpAppHtmlMimeType(content.mime_type)) {
    throw new WorkspaceMcpAppResourceError("invalidMimeType");
  }
  if (typeof content.text !== "string" || !content.text.trim()) {
    throw new WorkspaceMcpAppResourceError("missingHtml");
  }
  return {
    viewId,
    csp: readMcpAppCsp(content.meta),
    html: content.text,
    source: "mcpApp",
    sourceUri: expectedUri,
  };
}

function isMcpAppHtmlMimeType(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  const [type, ...parameters] = value
    .toLowerCase()
    .split(";")
    .map((part) => part.trim());
  return (
    type === "text/html" &&
    parameters.some((parameter) => parameter === "profile=mcp-app")
  );
}

function readMcpAppCsp(
  meta: Record<string, unknown> | undefined,
): EmbeddedBrowserHtmlParams["csp"] {
  const ui = asRecord(meta?.ui);
  const csp = asRecord(ui?.csp);
  if (!csp) {
    return undefined;
  }
  return {
    baseUriDomains: readStringArray(csp.baseUriDomains),
    connectDomains: readStringArray(csp.connectDomains),
    frameDomains: readStringArray(csp.frameDomains),
    resourceDomains: readStringArray(csp.resourceDomains),
  };
}

function readStringArray(value: unknown): string[] | undefined {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
