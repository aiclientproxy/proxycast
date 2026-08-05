type HostArgs = Record<string, unknown> | null | undefined;

export const EMBEDDED_BROWSER_HTML_MAX_BYTES = 1024 * 1024;

export interface EmbeddedBrowserHtmlPayload {
  csp: EmbeddedBrowserMcpAppCsp;
  html: string;
  source: "mcpApp";
  sourceUri: string;
}

export interface EmbeddedBrowserMcpAppCsp {
  baseUriDomains: string[];
  connectDomains: string[];
  frameDomains: string[];
  resourceDomains: string[];
}

export function readEmbeddedBrowserHtmlPayload(
  args: HostArgs,
): EmbeddedBrowserHtmlPayload {
  const record = readRecord(args);
  if (record?.source !== "mcpApp") {
    throw new Error("embedded browser HTML source 必须是 mcpApp。");
  }

  const sourceUri = readRequiredString(record, "sourceUri");
  if (!isMcpAppResourceUri(sourceUri)) {
    throw new Error("MCP App sourceUri 必须使用 ui:// 协议。");
  }

  const html = readRequiredString(record, "html", false);
  const byteLength = Buffer.byteLength(html, "utf8");
  if (byteLength > EMBEDDED_BROWSER_HTML_MAX_BYTES) {
    throw new Error(
      `MCP App HTML 超过 ${EMBEDDED_BROWSER_HTML_MAX_BYTES} 字节上限。`,
    );
  }

  return {
    csp: readMcpAppCsp(record?.csp),
    html,
    source: "mcpApp",
    sourceUri,
  };
}

export function buildEmbeddedBrowserHtmlDataUrl(
  payload: Pick<EmbeddedBrowserHtmlPayload, "csp" | "html">,
): string {
  const html = injectMcpAppContentSecurityPolicy(payload.html, payload.csp);
  return `data:text/html;charset=utf-8;base64,${Buffer.from(html, "utf8").toString("base64")}`;
}

function readMcpAppCsp(value: unknown): EmbeddedBrowserMcpAppCsp {
  const record = readRecord(value);
  return {
    baseUriDomains: readDomainList(record?.baseUriDomains, "baseUriDomains", [
      "http:",
      "https:",
    ]),
    connectDomains: readDomainList(record?.connectDomains, "connectDomains", [
      "http:",
      "https:",
      "ws:",
      "wss:",
    ]),
    frameDomains: readDomainList(record?.frameDomains, "frameDomains", [
      "http:",
      "https:",
    ]),
    resourceDomains: readDomainList(
      record?.resourceDomains,
      "resourceDomains",
      ["http:", "https:"],
    ),
  };
}

function readDomainList(
  value: unknown,
  field: string,
  protocols: readonly string[],
): string[] {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value) || value.length > 32) {
    throw new Error(`MCP App CSP ${field} 必须是最多 32 项的数组。`);
  }
  return Array.from(
    new Set(
      value.map((entry) => normalizeCspOrigin(entry, field, protocols)),
    ),
  );
}

function normalizeCspOrigin(
  value: unknown,
  field: string,
  protocols: readonly string[],
): string {
  if (typeof value !== "string" || value.length > 2048) {
    throw new Error(`MCP App CSP ${field} 包含无效 origin。`);
  }
  const trimmed = value.trim();
  const wildcard = /^(https?):\/\/\*\.(.+)$/i.exec(trimmed);
  const parseTarget = wildcard ? `${wildcard[1]}://${wildcard[2]}` : trimmed;
  try {
    const url = new URL(parseTarget);
    if (
      !protocols.includes(url.protocol) ||
      url.username ||
      url.password ||
      url.pathname !== "/" ||
      url.search ||
      url.hash ||
      !url.hostname
    ) {
      throw new Error("invalid origin");
    }
    return wildcard
      ? `${url.protocol}//* .${url.host}`.replace("* .", "*.")
      : url.origin;
  } catch {
    throw new Error(`MCP App CSP ${field} 包含无效 origin。`);
  }
}

function injectMcpAppContentSecurityPolicy(
  html: string,
  csp: EmbeddedBrowserMcpAppCsp,
): string {
  const resourceSources = csp.resourceDomains.join(" ");
  const policy = [
    "default-src 'none'",
    `script-src 'unsafe-inline'${appendSources(resourceSources)}`,
    `style-src 'unsafe-inline'${appendSources(resourceSources)}`,
    `img-src data: blob:${appendSources(resourceSources)}`,
    `font-src data:${appendSources(resourceSources)}`,
    `media-src data: blob:${appendSources(resourceSources)}`,
    `connect-src ${sourcesOrNone(csp.connectDomains)}`,
    `frame-src ${sourcesOrNone(csp.frameDomains)}`,
    `base-uri ${sourcesOrNone(csp.baseUriDomains)}`,
    "object-src 'none'",
    "form-action 'none'",
  ].join("; ");
  const tag = `<meta http-equiv="Content-Security-Policy" content="${escapeHtmlAttribute(policy)}">`;
  const headMatch = /<head(?:\s[^>]*)?>/i.exec(html);
  if (headMatch?.index !== undefined) {
    const offset = headMatch.index + headMatch[0].length;
    return `${html.slice(0, offset)}${tag}${html.slice(offset)}`;
  }
  return `${tag}${html}`;
}

function appendSources(value: string): string {
  return value ? ` ${value}` : "";
}

function sourcesOrNone(values: readonly string[]): string {
  return values.length > 0 ? values.join(" ") : "'none'";
}

function escapeHtmlAttribute(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
}

function isMcpAppResourceUri(value: string): boolean {
  try {
    return new URL(value).protocol === "ui:";
  } catch {
    return false;
  }
}

function readRequiredString(
  record: Record<string, unknown> | null,
  key: string,
  trim = true,
): string {
  const value = record?.[key];
  if (typeof value !== "string") {
    throw new Error(`embedded browser ${key} 不能为空。`);
  }
  const normalized = trim ? value.trim() : value;
  if (!normalized.trim()) {
    throw new Error(`embedded browser ${key} 不能为空。`);
  }
  return normalized;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
