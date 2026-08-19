import {
  consumeWorkspaceRightSurfacePending,
  requestWorkspaceRightSurface,
  type WorkspaceRightSurfaceAppServerClient,
} from "./workspaceRightSurface";

export interface BrowserWorkspaceIdentity {
  browserSessionId: string;
  tabId: string;
}

export interface BrowserWorkspaceOwner {
  runtimeSessionId: string;
  threadId: string;
}

export interface BrowserWorkspaceIdentityDeps {
  appServerClient?: WorkspaceRightSurfaceAppServerClient;
}

export async function openBrowserWorkspaceIdentity(params: {
  runtimeSessionId?: string | null;
  threadId: string;
}, deps: BrowserWorkspaceIdentityDeps = {}): Promise<BrowserWorkspaceIdentity> {
  return await requestBrowserIdentity(
    {
      action: "open",
      runtimeSessionId: params.runtimeSessionId,
      threadId: params.threadId,
    },
    deps,
  );
}

export async function createBrowserWorkspaceTabIdentity(params: {
  browserSessionId: string;
  runtimeSessionId?: string | null;
  threadId: string;
}, deps: BrowserWorkspaceIdentityDeps = {}): Promise<BrowserWorkspaceIdentity> {
  return await requestBrowserIdentity(
    {
      action: "createTab",
      browserSessionId: params.browserSessionId,
      runtimeSessionId: params.runtimeSessionId,
      threadId: params.threadId,
    },
    deps,
  );
}

async function requestBrowserIdentity(
  params: {
    action: "open" | "createTab";
    browserSessionId?: string;
    runtimeSessionId?: string | null;
    threadId: string;
  },
  deps: BrowserWorkspaceIdentityDeps,
): Promise<BrowserWorkspaceIdentity> {
  const response = await requestWorkspaceRightSurface(
    {
      surfaceKind: "browser",
      origin: "renderer",
      priority: "background",
      reason: `browser_identity_${params.action}`,
      sessionId: params.runtimeSessionId ?? null,
      ttlMs: 30_000,
      metadata: {
        browser: {
          action: params.action,
          ...(params.browserSessionId
            ? { browserSessionId: params.browserSessionId }
            : {}),
          threadId: params.threadId,
        },
      },
    },
    deps,
  );
  try {
    const identity = readBrowserIdentity(response.pending.metadata);
    if (!identity) {
      throw new Error("App Server did not return a canonical Browser identity");
    }
    return identity;
  } finally {
    await consumeWorkspaceRightSurfacePending(
      { requestId: response.requestId },
      deps,
    );
  }
}

function readBrowserIdentity(value: unknown): BrowserWorkspaceIdentity | null {
  if (!isRecord(value)) {
    return null;
  }
  const browser = value.browser;
  if (!isRecord(browser)) {
    return null;
  }
  const browserSessionId = readNonEmptyString(browser.browserSessionId);
  const tabId = readNonEmptyString(browser.tabId);
  return browserSessionId && tabId ? { browserSessionId, tabId } : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
