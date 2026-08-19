import {
  getClientBootstrap,
  type OemCloudBootstrapResponse,
} from "@/lib/api/oemCloudControlPlane";
import { DEFAULT_OEM_CLOUD_DESKTOP_OAUTH_NEXT_PATH } from "@/lib/api/oemCloudRuntime";
import {
  setOemCloudBootstrapSnapshot,
  setStoredOemCloudSessionState,
} from "@/lib/oemCloudSession";
import { syncSkillCatalogFromBootstrapPayload } from "@/lib/skillCatalogBootstrap";
import { syncServiceSkillCatalogFromBootstrapPayload } from "@/lib/serviceSkillCatalogBootstrap";

export const OEM_CLOUD_OAUTH_COMPLETED_EVENT = "lime:oem-cloud-oauth-completed";

export interface OemCloudDesktopOAuthCallbackPayload {
  tenantId: string | null;
  token: string | null;
  nextPath: string;
  error: string | null;
}

export interface OemCloudDesktopOAuthCompletedDetail {
  tenantId: string;
  nextPath: string;
  provider: "google";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim();
  return normalized ? normalized : null;
}

function normalizeAppPath(value: unknown, fallback: string): string {
  const normalized = normalizeText(value);
  if (!normalized) {
    return fallback;
  }

  if (normalized === "/") {
    return normalized;
  }

  return normalized.startsWith("/") ? normalized : `/${normalized}`;
}

function setWindowSessionToken(token: string | null): void {
  if (typeof window === "undefined") {
    return;
  }

  if (!token) {
    delete window.__LIME_SESSION_TOKEN__;
    return;
  }

  window.__LIME_SESSION_TOKEN__ = token;
}

function setWindowRuntimeTenantId(tenantId: string | null): void {
  if (typeof window === "undefined") {
    return;
  }

  const currentRuntime = isRecord(window.__LIME_OEM_CLOUD__)
    ? window.__LIME_OEM_CLOUD__
    : {};

  if (!tenantId) {
    window.__LIME_OEM_CLOUD__ = {
      ...currentRuntime,
    };
    return;
  }

  window.__LIME_OEM_CLOUD__ = {
    ...currentRuntime,
    tenantId,
  };
}

function restoreWindowOauthContext(previous: {
  runtime: unknown;
  sessionToken: unknown;
}): void {
  if (typeof window === "undefined") {
    return;
  }

  if (previous.runtime === undefined) {
    delete window.__LIME_OEM_CLOUD__;
  } else {
    window.__LIME_OEM_CLOUD__ = previous.runtime;
  }

  if (previous.sessionToken === undefined) {
    delete window.__LIME_SESSION_TOKEN__;
  } else {
    window.__LIME_SESSION_TOKEN__ = previous.sessionToken;
  }
}

export function parseOemCloudDesktopOAuthCallbackUrl(
  value: string,
): OemCloudDesktopOAuthCallbackPayload | null {
  try {
    const parsed = new URL(value);
    const normalizedPath = parsed.pathname.replace(/\/+$/, "") || "/";
    if (
      parsed.protocol !== "lime:" ||
      parsed.hostname !== "oauth" ||
      normalizedPath !== "/callback"
    ) {
      return null;
    }

    return {
      tenantId: normalizeText(parsed.searchParams.get("tenantId")),
      token: normalizeText(parsed.searchParams.get("token")),
      nextPath: normalizeAppPath(
        parsed.searchParams.get("next"),
        DEFAULT_OEM_CLOUD_DESKTOP_OAUTH_NEXT_PATH,
      ),
      error: normalizeText(parsed.searchParams.get("error")),
    };
  } catch {
    return null;
  }
}

export function dispatchOemCloudOAuthCompleted(
  detail: OemCloudDesktopOAuthCompletedDetail,
): void {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent<OemCloudDesktopOAuthCompletedDetail>(
      OEM_CLOUD_OAUTH_COMPLETED_EVENT,
      {
        detail,
      },
    ),
  );
}

export async function completeOemCloudDesktopOAuthLogin(
  payload: OemCloudDesktopOAuthCallbackPayload,
): Promise<OemCloudBootstrapResponse> {
  if (!payload.tenantId || !payload.token) {
    throw new Error("桌面 OAuth 回调缺少 tenantId 或 token");
  }

  const previousWindowState =
    typeof window === "undefined"
      ? {
          runtime: undefined,
          sessionToken: undefined,
        }
      : {
          runtime: window.__LIME_OEM_CLOUD__,
          sessionToken: window.__LIME_SESSION_TOKEN__,
        };

  setWindowRuntimeTenantId(payload.tenantId);
  setWindowSessionToken(payload.token);

  try {
    const bootstrap = await getClientBootstrap(payload.tenantId);
    const nextSession = {
      ...bootstrap.session,
      token: bootstrap.session.token ?? payload.token,
    };
    const nextBootstrap = {
      ...bootstrap,
      session: nextSession,
    };

    setStoredOemCloudSessionState(nextSession);
    setOemCloudBootstrapSnapshot(nextBootstrap);
    syncSkillCatalogFromBootstrapPayload(nextBootstrap);
    syncServiceSkillCatalogFromBootstrapPayload(nextBootstrap);
    dispatchOemCloudOAuthCompleted({
      tenantId: payload.tenantId,
      nextPath: payload.nextPath,
      provider: "google",
    });

    return nextBootstrap;
  } catch (error) {
    restoreWindowOauthContext(previousWindowState);
    throw error;
  }
}
