import type { WebContents } from "electron";

const MAX_AX_NODES = 250;

export interface BrowserObservationNode {
  backendNodeId: unknown;
  childIds: unknown[];
  ignored: boolean;
  name: unknown;
  nodeId: unknown;
  role: unknown;
  value: unknown;
}

export interface BrowserPageObservation {
  currentIndex: unknown;
  nodes: BrowserObservationNode[];
  pageRevision: number;
  snapshotId: string;
  title: string;
  truncated: boolean;
  url: string;
}

export async function observeBrowserPage(
  webContents: WebContents,
  identity: { pageRevision: number; snapshotId: string },
): Promise<BrowserPageObservation> {
  const [history, accessibility] = await Promise.all([
    webContents.debugger.sendCommand("Page.getNavigationHistory"),
    webContents.debugger.sendCommand("Accessibility.getFullAXTree"),
  ]);
  const historyRecord = asRecord(history);
  const accessibilityRecord = asRecord(accessibility);
  const nodes = Array.isArray(accessibilityRecord?.nodes)
    ? accessibilityRecord.nodes.slice(0, MAX_AX_NODES).map(normalizeAxNode)
    : [];
  return {
    title: webContents.getTitle(),
    url: webContents.getURL(),
    currentIndex: historyRecord?.currentIndex ?? null,
    nodes,
    pageRevision: identity.pageRevision,
    snapshotId: identity.snapshotId,
    truncated:
      Array.isArray(accessibilityRecord?.nodes) &&
      accessibilityRecord.nodes.length > MAX_AX_NODES,
  };
}

export async function describeBrowserNode(
  webContents: WebContents,
  backendNodeId: number,
): Promise<string> {
  const response = asRecord(
    await webContents.debugger.sendCommand("DOM.describeNode", {
      backendNodeId,
      depth: 0,
    }),
  );
  const node = asRecord(response?.node);
  const attributes = Array.isArray(node?.attributes)
    ? node.attributes.map(String).join(" ")
    : "";
  return [node?.nodeName, node?.localName, attributes]
    .filter(Boolean)
    .join(" ")
    .slice(0, 500);
}

export async function browserNodeCenter(
  webContents: WebContents,
  backendNodeId: number,
): Promise<{ x: number; y: number }> {
  const response = asRecord(
    await webContents.debugger.sendCommand("DOM.getBoxModel", {
      backendNodeId,
    }),
  );
  const model = asRecord(response?.model);
  const content = Array.isArray(model?.content)
    ? model.content.map(Number).filter(Number.isFinite)
    : [];
  if (content.length < 8) {
    throw new Error("Browser target is not actionable");
  }
  const xs = [content[0], content[2], content[4], content[6]];
  const ys = [content[1], content[3], content[5], content[7]];
  return {
    x: xs.reduce((sum, value) => sum + value, 0) / xs.length,
    y: ys.reduce((sum, value) => sum + value, 0) / ys.length,
  };
}

function normalizeAxNode(value: unknown): BrowserObservationNode {
  const node = asRecord(value) ?? {};
  return {
    backendNodeId: node.backendDOMNodeId ?? null,
    childIds: Array.isArray(node.childIds) ? node.childIds : [],
    ignored: node.ignored === true,
    name: axValue(node.name),
    nodeId: node.nodeId ?? null,
    role: axValue(node.role),
    value: axValue(node.value),
  };
}

function axValue(value: unknown): unknown {
  return asRecord(value)?.value ?? null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
