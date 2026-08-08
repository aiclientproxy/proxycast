import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const hoisted = vi.hoisted(() => ({
  controlPlaneMocks: {
    getClientBootstrap: vi.fn(),
    getClientCloudActivation: vi.fn(),
    getClientOrder: vi.fn(),
    getClientCreditTopupOrder: vi.fn(),
    getClientProviderOffer: vi.fn(),
    listClientProviderOfferModels: vi.fn(),
    updateClientProviderPreference: vi.fn(),
    createClientOrder: vi.fn(),
    createClientOrderCheckout: vi.fn(),
    createClientCreditTopupOrder: vi.fn(),
    createClientCreditTopupOrderCheckout: vi.fn(),
    createClientAccessToken: vi.fn(),
    rotateClientAccessToken: vi.fn(),
    revokeClientAccessToken: vi.fn(),
    createClientDesktopAuthSession: vi.fn(),
    pollClientDesktopAuthSession: vi.fn(),
    loginClientByPassword: vi.fn(),
    logoutClient: vi.fn(),
    sendClientAuthEmailCode: vi.fn(),
    verifyClientAuthEmailCode: vi.fn(),
  },
  shellOpenMock: vi.fn(),
  systemBrowserMocks: {
    openExternalUrlWithSystemBrowser: vi.fn(),
    startOemCloudOAuthCallbackBridge: vi.fn(),
  },
  desktopRuntimeMocks: {
    hasDesktopHostInvokeCapability: vi.fn(),
    hasDesktopHostRuntimeMarkers: vi.fn(),
  },
  desktopAuthMocks: {
    completeOemCloudDesktopOAuthLogin: vi.fn(),
  },
}));

export const controlPlaneMocks = hoisted.controlPlaneMocks;
export const shellOpenMock = hoisted.shellOpenMock;
export const systemBrowserMocks = hoisted.systemBrowserMocks;
export const desktopRuntimeMocks = hoisted.desktopRuntimeMocks;
export const desktopAuthMocks = hoisted.desktopAuthMocks;

vi.mock("@/lib/api/oemCloudControlPlane", () => {
  class MockOemCloudControlPlaneError extends Error {
    status: number;

    constructor(message: string, status = 500) {
      super(message);
      this.status = status;
    }
  }

  return {
    OemCloudControlPlaneError: MockOemCloudControlPlaneError,
    getClientBootstrap: hoisted.controlPlaneMocks.getClientBootstrap,
    getClientCloudActivation:
      hoisted.controlPlaneMocks.getClientCloudActivation,
    getClientOrder: hoisted.controlPlaneMocks.getClientOrder,
    getClientCreditTopupOrder:
      hoisted.controlPlaneMocks.getClientCreditTopupOrder,
    getClientProviderOffer: hoisted.controlPlaneMocks.getClientProviderOffer,
    listClientProviderOfferModels:
      hoisted.controlPlaneMocks.listClientProviderOfferModels,
    updateClientProviderPreference:
      hoisted.controlPlaneMocks.updateClientProviderPreference,
    createClientOrder: hoisted.controlPlaneMocks.createClientOrder,
    createClientOrderCheckout:
      hoisted.controlPlaneMocks.createClientOrderCheckout,
    createClientCreditTopupOrder:
      hoisted.controlPlaneMocks.createClientCreditTopupOrder,
    createClientCreditTopupOrderCheckout:
      hoisted.controlPlaneMocks.createClientCreditTopupOrderCheckout,
    createClientAccessToken: hoisted.controlPlaneMocks.createClientAccessToken,
    rotateClientAccessToken: hoisted.controlPlaneMocks.rotateClientAccessToken,
    revokeClientAccessToken: hoisted.controlPlaneMocks.revokeClientAccessToken,
    createClientDesktopAuthSession:
      hoisted.controlPlaneMocks.createClientDesktopAuthSession,
    pollClientDesktopAuthSession:
      hoisted.controlPlaneMocks.pollClientDesktopAuthSession,
    loginClientByPassword: hoisted.controlPlaneMocks.loginClientByPassword,
    logoutClient: hoisted.controlPlaneMocks.logoutClient,
    sendClientAuthEmailCode:
      hoisted.controlPlaneMocks.sendClientAuthEmailCode,
    verifyClientAuthEmailCode:
      hoisted.controlPlaneMocks.verifyClientAuthEmailCode,
  };
});

vi.mock("@/lib/desktop-host/shell", () => ({
  open: hoisted.shellOpenMock,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_EVENT: "oem-cloud-oauth-callback",
  openExternalUrlWithSystemBrowser:
    hoisted.systemBrowserMocks.openExternalUrlWithSystemBrowser,
  startOemCloudOAuthCallbackBridge:
    hoisted.systemBrowserMocks.startOemCloudOAuthCallbackBridge,
}));

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostInvokeCapability: hoisted.desktopRuntimeMocks.hasDesktopHostInvokeCapability,
  hasDesktopHostRuntimeMarkers: hoisted.desktopRuntimeMocks.hasDesktopHostRuntimeMarkers,
}));

vi.mock("@/lib/oemCloudDesktopAuth", () => ({
  OEM_CLOUD_OAUTH_COMPLETED_EVENT: "lime:oem-cloud-oauth-completed",
  completeOemCloudDesktopOAuthLogin:
    hoisted.desktopAuthMocks.completeOemCloudDesktopOAuthLogin,
}));

vi.mock("@/lib/serviceSkillCatalogBootstrap", () => ({
  syncServiceSkillCatalogFromBootstrapPayload: vi.fn(),
}));

vi.mock("@/lib/skillCatalogBootstrap", () => ({
  syncSkillCatalogFromBootstrapPayload: vi.fn(),
}));

vi.mock("@/lib/api/skillCatalog", () => ({
  clearSkillCatalogCache: vi.fn(),
}));

vi.mock("@/lib/siteAdapterCatalogBootstrap", () => ({
  syncSiteAdapterCatalogFromBootstrapPayload: vi.fn(),
  clearSiteAdapterCatalogCache: vi.fn(),
}));

import { useOemCloudAccess } from "./useOemCloudAccess";

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

let mountedHarness: MountedHarness | null = null;
export let latestState: ReturnType<typeof useOemCloudAccess> | null = null;

export function mountHookHarness() {
  function HookHarness() {
    latestState = useOemCloudAccess();
    return (
      <div data-testid="hook-state">
        {latestState.initializing
          ? "initializing"
          : latestState.session?.session.id || "anonymous"}
      </div>
    );
  }

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedHarness = { container, root };

  act(() => {
    root.render(<HookHarness />);
  });

  return { container, root };
}

export function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return {
    promise,
    resolve,
    reject,
  };
}

export function createOpenedWindow() {
  return {
    closed: false,
    opener: {},
    close: vi.fn(),
    document: {
      title: "",
      body: {
        innerHTML: "",
      },
    },
    location: {
      assign: vi.fn(),
    },
  } as unknown as Window;
}

export async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

export function useOemCloudAccessTestLifecycle() {
  beforeEach(async () => {
    await changeLimeLocale("en-US");
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;

    latestState = null;
    window.localStorage.clear();
    delete window.__LIME_SESSION_TOKEN__;
    delete window.__LIME_BOOTSTRAP__;
    window.__LIME_OEM_CLOUD__ = {
      enabled: true,
      baseUrl: "https://user.limeai.run",
      tenantId: "tenant-0001",
    };

    controlPlaneMocks.getClientCloudActivation.mockResolvedValue({
      gateway: {
        basePath: "https://llm.limeai.run",
        openAIBaseUrl: "https://llm.limeai.run/v1",
        anthropicBaseUrl: "https://llm.limeai.run",
      },
      llmBaseUrl: "https://llm.limeai.run",
      openAIBaseUrl: "https://llm.limeai.run/v1",
      anthropicBaseUrl: "https://llm.limeai.run",
      readiness: {
        status: "no_api_key",
        title: "还没有可用 API Key",
        canInvoke: false,
        blockers: ["api_key"],
        steps: [],
      },
      pendingPayment: null,
      paymentConfigs: [],
      plans: [],
      subscription: null,
      creditAccount: null,
      creditsDashboard: null,
      topupPackages: [],
      usageDashboard: null,
      billingDashboard: null,
      providerOffers: [],
      selectedOffer: null,
      providerModels: [],
      providerPreference: null,
      accessTokens: [],
      activeAccessToken: { hasActive: false, token: null },
      orders: [],
      creditTopupOrders: [],
    });
    controlPlaneMocks.getClientProviderOffer.mockResolvedValue(null);
    controlPlaneMocks.getClientOrder.mockResolvedValue(null);
    controlPlaneMocks.getClientCreditTopupOrder.mockResolvedValue(null);
    controlPlaneMocks.listClientProviderOfferModels.mockResolvedValue([]);
    controlPlaneMocks.updateClientProviderPreference.mockResolvedValue(null);
    controlPlaneMocks.createClientOrder.mockResolvedValue(null);
    controlPlaneMocks.createClientOrderCheckout.mockResolvedValue(null);
    controlPlaneMocks.createClientCreditTopupOrder.mockResolvedValue(null);
    controlPlaneMocks.createClientCreditTopupOrderCheckout.mockResolvedValue(
      null,
    );
    controlPlaneMocks.createClientAccessToken.mockResolvedValue(null);
    controlPlaneMocks.rotateClientAccessToken.mockResolvedValue(null);
    controlPlaneMocks.revokeClientAccessToken.mockResolvedValue(null);
    controlPlaneMocks.createClientDesktopAuthSession.mockResolvedValue(null);
    controlPlaneMocks.pollClientDesktopAuthSession.mockResolvedValue(null);
    controlPlaneMocks.loginClientByPassword.mockResolvedValue(null);
    controlPlaneMocks.logoutClient.mockResolvedValue(undefined);
    controlPlaneMocks.sendClientAuthEmailCode.mockResolvedValue(null);
    controlPlaneMocks.verifyClientAuthEmailCode.mockResolvedValue(null);
    desktopAuthMocks.completeOemCloudDesktopOAuthLogin.mockResolvedValue({});
    shellOpenMock.mockResolvedValue(undefined);
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockReset();
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockResolvedValue(
      undefined,
    );
    systemBrowserMocks.startOemCloudOAuthCallbackBridge.mockReset();
    systemBrowserMocks.startOemCloudOAuthCallbackBridge.mockResolvedValue({
      callbackUrl: "http://127.0.0.1:18081/oauth/callback",
    });
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
  });

  afterEach(async () => {
    vi.clearAllMocks();
    vi.useRealTimers();
    window.localStorage.clear();
    delete window.__LIME_SESSION_TOKEN__;
    delete window.__LIME_BOOTSTRAP__;
    delete window.__LIME_OEM_CLOUD__;

    if (mountedHarness) {
      act(() => {
        mountedHarness?.root.unmount();
      });
      mountedHarness.container.remove();
      mountedHarness = null;
    }
    await changeLimeLocale("zh-CN");
  });
}
