import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildOemCloudLoginUrl,
  createExternalBrowserOpenTarget,
  openExternalUrl,
  startOemCloudLogin,
} from "@/lib/oemCloudLoginLauncher";
import { setStoredOemCloudSessionState } from "@/lib/oemCloudSession";

const shellOpenMock = vi.hoisted(() => vi.fn());
const controlPlaneMocks = vi.hoisted(() => ({
  createClientDesktopAuthSession: vi.fn(),
  pollClientDesktopAuthSession: vi.fn(),
}));
const systemBrowserMocks = vi.hoisted(() => ({
  openExternalUrlWithSystemBrowser: vi.fn(),
  startOemCloudOAuthCallbackBridge: vi.fn(),
}));
const desktopRuntimeMocks = vi.hoisted(() => ({
  hasDesktopHostInvokeCapability: vi.fn(),
  hasDesktopHostRuntimeMarkers: vi.fn(),
}));
const devBridgeMocks = vi.hoisted(() => ({
  isDevBridgeAvailable: vi.fn(),
  safeListen: vi.fn(),
}));

vi.mock("@/lib/api/oemCloudControlPlane", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/oemCloudControlPlane")>();

  return {
    ...actual,
    createClientDesktopAuthSession:
      controlPlaneMocks.createClientDesktopAuthSession,
    pollClientDesktopAuthSession:
      controlPlaneMocks.pollClientDesktopAuthSession,
  };
});

vi.mock("@/lib/desktop-host/shell", () => ({
  open: shellOpenMock,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_EVENT: "oem-cloud-oauth-callback",
  openExternalUrlWithSystemBrowser:
    systemBrowserMocks.openExternalUrlWithSystemBrowser,
  startOemCloudOAuthCallbackBridge:
    systemBrowserMocks.startOemCloudOAuthCallbackBridge,
}));

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostInvokeCapability:
    desktopRuntimeMocks.hasDesktopHostInvokeCapability,
  hasDesktopHostRuntimeMarkers:
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers,
}));

vi.mock("@/lib/dev-bridge", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/dev-bridge")>();
  return {
    ...actual,
    isDevBridgeAvailable: devBridgeMocks.isDevBridgeAvailable,
    safeListen: devBridgeMocks.safeListen,
  };
});

function createOpenedWindow() {
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

describe("oemCloudLoginLauncher", () => {
  beforeEach(() => {
    localStorage.clear();
    shellOpenMock.mockReset();
    shellOpenMock.mockResolvedValue(undefined);
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockReset();
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockResolvedValue(
      undefined,
    );
    systemBrowserMocks.startOemCloudOAuthCallbackBridge.mockReset();
    systemBrowserMocks.startOemCloudOAuthCallbackBridge.mockResolvedValue({
      callbackUrl: "http://127.0.0.1:18081/oauth/callback",
    });
    controlPlaneMocks.createClientDesktopAuthSession.mockReset();
    controlPlaneMocks.pollClientDesktopAuthSession.mockReset();
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReset();
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReset();
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(false);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(false);
    devBridgeMocks.isDevBridgeAvailable.mockReset();
    devBridgeMocks.isDevBridgeAvailable.mockReturnValue(false);
    devBridgeMocks.safeListen.mockReset();
    devBridgeMocks.safeListen.mockResolvedValue(() => undefined);
  });

  afterEach(() => {
    localStorage.clear();
    Object.defineProperty(window, "parent", {
      configurable: true,
      value: window,
    });
    vi.restoreAllMocks();
  });

  it("Desktop Host 可用时应优先通过 native 命令打开系统浏览器", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    const browserTarget = {
      navigate: vi.fn(),
      close: vi.fn(),
    };

    await openExternalUrl("https://user.limeai.run/login", { browserTarget });

    expect(
      systemBrowserMocks.openExternalUrlWithSystemBrowser,
    ).toHaveBeenCalledWith("https://user.limeai.run/login");
    expect(shellOpenMock).not.toHaveBeenCalled();
    expect(browserTarget.close).toHaveBeenCalledTimes(1);
    expect(browserTarget.navigate).not.toHaveBeenCalled();
  });

  it("native 打开命令不可用时不回退 Desktop Host shell open", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockRejectedValue(
      new Error("unknown command"),
    );

    await expect(
      openExternalUrl("https://user.limeai.run/login"),
    ).rejects.toThrow("系统浏览器打开失败：unknown command");

    expect(shellOpenMock).not.toHaveBeenCalled();
  });

  it("Desktop Host shell open 失败时应抛错且不回退成假成功", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    systemBrowserMocks.openExternalUrlWithSystemBrowser.mockRejectedValue(
      new Error("native denied"),
    );
    shellOpenMock.mockRejectedValue(new Error("permission denied"));
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(null);

    await expect(
      openExternalUrl("https://user.limeai.run/login", {
        copy: {
          systemBrowserOpenFailedWithMessage: (message) =>
            `System browser failed from copy: ${message}`,
        },
      }),
    ).rejects.toThrow("System browser failed from copy: native denied");

    expect(windowOpenSpy).not.toHaveBeenCalled();
  });

  it("浏览器场景应先预打开空白页，再导航到登录 URL", async () => {
    const openedWindow = createOpenedWindow();
    const windowOpenSpy = vi
      .spyOn(window, "open")
      .mockReturnValueOnce(openedWindow);

    const browserTarget = createExternalBrowserOpenTarget({
      openingTitle: "Opening login",
      openingBody: "Opening login body",
    });

    expect(browserTarget).not.toBeNull();
    expect(windowOpenSpy).toHaveBeenCalledWith("about:blank", "_blank");
    expect(openedWindow.document.title).toBe("Opening login");
    expect(openedWindow.document.body.innerHTML).toBe("Opening login body");

    await openExternalUrl("https://user.limeai.run/login", { browserTarget });

    expect(openedWindow.location.assign).toHaveBeenCalledWith(
      "https://user.limeai.run/login",
    );
    expect(
      systemBrowserMocks.openExternalUrlWithSystemBrowser,
    ).not.toHaveBeenCalled();
    expect(shellOpenMock).not.toHaveBeenCalled();
    expect(windowOpenSpy).toHaveBeenCalledTimes(1);
  });

  it("Desktop Host 桌面登录应启动本机回调桥并把 callbackUrl 传给 desktop auth session", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    controlPlaneMocks.createClientDesktopAuthSession.mockResolvedValue({
      authSessionId: "auth-session-001",
      deviceCode: "device-001",
      tenantId: "tenant-0001",
      clientId: "desktop-client",
      clientName: "Lime Desktop",
      provider: "google",
      desktopRedirectUri: "http://127.0.0.1:18081/oauth/callback",
      status: "pending_login",
      expiresInSeconds: 600,
      pollIntervalSeconds: 1,
      authorizeUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });
    controlPlaneMocks.pollClientDesktopAuthSession.mockReturnValue(
      new Promise(() => undefined),
    );

    await startOemCloudLogin(
      {
        baseUrl: "https://user.limeai.run",
        controlPlaneBaseUrl: "https://user.limeai.run/api",
        sceneBaseUrl: "https://user.limeai.run/scene-api",
        gatewayBaseUrl: "https://llm.limeai.run",
        tenantId: "tenant-0001",
        sessionToken: null,
        hubProviderName: null,
        loginPath: "/login",
        desktopClientId: "desktop-client",
        desktopOauthRedirectUrl: "lime://oauth/callback",
        desktopOauthNextPath: "/welcome",
        pluginSignatureTrustRoots: [],
      },
      { waitForCompletion: false },
    );

    expect(
      systemBrowserMocks.startOemCloudOAuthCallbackBridge,
    ).toHaveBeenCalledTimes(1);
    expect(
      controlPlaneMocks.createClientDesktopAuthSession,
    ).toHaveBeenCalledWith("tenant-0001", {
      clientId: "desktop-client",
      provider: "google",
      desktopRedirectUri: "http://127.0.0.1:18081/oauth/callback",
    });
  });

  it("DevBridge 浏览器模式应使用宿主命令打开登录页并启用本机回调桥", async () => {
    devBridgeMocks.isDevBridgeAvailable.mockReturnValue(true);
    controlPlaneMocks.createClientDesktopAuthSession.mockResolvedValue({
      authSessionId: "auth-session-001",
      deviceCode: "device-001",
      tenantId: "tenant-0001",
      clientId: "desktop-client",
      clientName: "Lime Desktop",
      provider: "google",
      desktopRedirectUri: "http://127.0.0.1:18081/oauth/callback",
      status: "pending_login",
      expiresInSeconds: 600,
      pollIntervalSeconds: 1,
      authorizeUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });
    controlPlaneMocks.pollClientDesktopAuthSession.mockReturnValue(
      new Promise(() => undefined),
    );

    await startOemCloudLogin(
      {
        baseUrl: "https://user.limeai.run",
        controlPlaneBaseUrl: "https://user.limeai.run/api",
        sceneBaseUrl: "https://user.limeai.run/scene-api",
        gatewayBaseUrl: "https://llm.limeai.run",
        tenantId: "tenant-0001",
        sessionToken: null,
        hubProviderName: null,
        loginPath: "/login",
        desktopClientId: "desktop-client",
        desktopOauthRedirectUrl: "lime://oauth/callback",
        desktopOauthNextPath: "/welcome",
        pluginSignatureTrustRoots: [],
      },
      { waitForCompletion: false },
    );

    expect(
      systemBrowserMocks.startOemCloudOAuthCallbackBridge,
    ).toHaveBeenCalledTimes(1);
    expect(
      controlPlaneMocks.createClientDesktopAuthSession,
    ).toHaveBeenCalledWith("tenant-0001", {
      clientId: "desktop-client",
      provider: "google",
      desktopRedirectUri: "http://127.0.0.1:18081/oauth/callback",
    });
    expect(
      systemBrowserMocks.openExternalUrlWithSystemBrowser,
    ).toHaveBeenCalledWith(
      "https://user.limeai.run/oauth/desktop/device-001/signin",
    );
    expect(shellOpenMock).not.toHaveBeenCalled();
  });

  it("浏览器弹窗被拦截时应抛出可感知错误", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);

    await expect(
      openExternalUrl("https://user.limeai.run/login", {
        copy: {
          popupBlocked: "Popup blocked from copy",
        },
      }),
    ).rejects.toThrow("Popup blocked from copy");
  });

  it("构建云端登录页时应携带租户、桌面回跳和返回路径", () => {
    const loginUrl = buildOemCloudLoginUrl({
      baseUrl: "https://user.limeai.run",
      loginPath: "/login",
      tenantId: "tenant-0001",
      desktopOauthRedirectUrl: "lime://oauth/callback",
      desktopOauthNextPath: "/welcome",
    });

    const parsedUrl = new URL(loginUrl);

    expect(parsedUrl.origin).toBe("https://user.limeai.run");
    expect(parsedUrl.pathname).toBe("/login");
    expect(parsedUrl.searchParams.get("tenant")).toBe("tenant-0001");
    expect(parsedUrl.searchParams.get("tenantId")).toBe("tenant-0001");
    expect(parsedUrl.searchParams.get("redirectUrl")).toBe(
      "lime://oauth/callback",
    );
    expect(parsedUrl.searchParams.get("redirect")).toBe("/welcome");
    expect(parsedUrl.searchParams.get("next")).toBe("/welcome");
  });

  it("桌面 OAuth 回调返回内部租户 ID 且本地会话 slug 命中时应完成登录", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    controlPlaneMocks.createClientDesktopAuthSession.mockResolvedValue({
      authSessionId: "auth-session-001",
      deviceCode: "device-001",
      tenantId: "tenant-0514",
      clientId: "desktop-client",
      clientName: "Lime Desktop",
      provider: "google",
      desktopRedirectUri: "lime://oauth/callback",
      status: "pending_login",
      expiresInSeconds: 600,
      pollIntervalSeconds: 2,
      authorizeUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });
    controlPlaneMocks.pollClientDesktopAuthSession.mockReturnValue(
      new Promise(() => undefined),
    );

    const loginPromise = startOemCloudLogin({
      baseUrl: "https://user.limeai.run",
      controlPlaneBaseUrl: "https://user.limeai.run/api",
      sceneBaseUrl: "https://user.limeai.run/scene-api",
      gatewayBaseUrl: "https://llm.limeai.run",
      tenantId: "tenant-0001",
      sessionToken: null,
      hubProviderName: null,
      loginPath: "/login",
      desktopClientId: "desktop-client",
      desktopOauthRedirectUrl: "lime://oauth/callback",
      desktopOauthNextPath: "/welcome",
      pluginSignatureTrustRoots: [],
    });

    await vi.waitFor(() => {
      expect(
        systemBrowserMocks.openExternalUrlWithSystemBrowser,
      ).toHaveBeenCalledWith(
        "https://user.limeai.run/oauth/desktop/device-001/signin",
      );
    });

    setStoredOemCloudSessionState({
      token: "session-token",
      tenant: { id: "tenant-0514", slug: "tenant-0001" },
      user: { id: "user-001" },
      session: { id: "session-001", provider: "google" },
    });
    window.dispatchEvent(
      new CustomEvent("lime:oem-cloud-oauth-completed", {
        detail: {
          tenantId: "tenant-0514",
          nextPath: "/welcome",
          provider: "google",
        },
      }),
    );

    await expect(loginPromise).resolves.toEqual({
      mode: "desktop_auth",
      openedUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });
  });

  it("桌面 OAuth 可只等待浏览器打开并在后台继续同步登录", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    controlPlaneMocks.createClientDesktopAuthSession.mockResolvedValue({
      authSessionId: "auth-session-001",
      deviceCode: "device-001",
      tenantId: "tenant-0001",
      clientId: "desktop-client",
      clientName: "Lime Desktop",
      provider: "google",
      desktopRedirectUri: "lime://oauth/callback",
      status: "pending_login",
      expiresInSeconds: 600,
      pollIntervalSeconds: 2,
      authorizeUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });
    controlPlaneMocks.pollClientDesktopAuthSession.mockReturnValue(
      new Promise(() => undefined),
    );

    await expect(
      startOemCloudLogin(
        {
          baseUrl: "https://user.limeai.run",
          controlPlaneBaseUrl: "https://user.limeai.run/api",
          sceneBaseUrl: "https://user.limeai.run/scene-api",
          gatewayBaseUrl: "https://llm.limeai.run",
          tenantId: "tenant-0001",
          sessionToken: null,
          hubProviderName: null,
          loginPath: "/login",
          desktopClientId: "desktop-client",
          desktopOauthRedirectUrl: "lime://oauth/callback",
          desktopOauthNextPath: "/welcome",
        pluginSignatureTrustRoots: [],
        },
        { waitForCompletion: false },
      ),
    ).resolves.toEqual({
      mode: "desktop_auth",
      openedUrl: "https://user.limeai.run/oauth/desktop/device-001/signin",
    });

    expect(
      systemBrowserMocks.openExternalUrlWithSystemBrowser,
    ).toHaveBeenCalledWith(
      "https://user.limeai.run/oauth/desktop/device-001/signin",
    );
    expect(shellOpenMock).not.toHaveBeenCalled();
    expect(controlPlaneMocks.pollClientDesktopAuthSession).toHaveBeenCalledWith(
      "device-001",
    );

    window.dispatchEvent(
      new CustomEvent("lime:oem-cloud-oauth-completed", {
        detail: {
          tenantId: "tenant-0001",
          nextPath: "/welcome",
          provider: "google",
        },
      }),
    );
    await Promise.resolve();
  });

  it("requestLogin fallback 打开普通登录页后应等待本机回调完成", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    controlPlaneMocks.createClientDesktopAuthSession.mockRejectedValue(
      new Error("desktop client not found"),
    );

    const loginPromise = startOemCloudLogin(
      {
        baseUrl: "https://user.limeai.run",
        controlPlaneBaseUrl: "https://user.limeai.run/api",
        sceneBaseUrl: "https://user.limeai.run/scene-api",
        gatewayBaseUrl: "https://llm.limeai.run",
        tenantId: "tenant-0001",
        sessionToken: null,
        hubProviderName: null,
        loginPath: "/login",
        desktopClientId: "desktop-client",
        desktopOauthRedirectUrl: "lime://oauth/callback",
        desktopOauthNextPath: "/welcome",
        pluginSignatureTrustRoots: [],
      },
      { waitForCompletion: true },
    );

    await vi.waitFor(() => {
      expect(
        systemBrowserMocks.openExternalUrlWithSystemBrowser,
      ).toHaveBeenCalledTimes(1);
    });
    const openedUrl = systemBrowserMocks.openExternalUrlWithSystemBrowser.mock
      .calls[0]?.[0] as string;
    expect(new URL(openedUrl).searchParams.get("redirectUrl")).toBe(
      "http://127.0.0.1:18081/oauth/callback",
    );

    setStoredOemCloudSessionState({
      token: "session-token",
      tenant: { id: "tenant-0001", slug: "tenant-0001" },
      user: { id: "user-001" },
      session: { id: "session-001", provider: "google" },
    });
    window.dispatchEvent(
      new CustomEvent("lime:oem-cloud-oauth-completed", {
        detail: {
          tenantId: "tenant-0001",
          nextPath: "/welcome",
          provider: "google",
        },
      }),
    );

    await expect(loginPromise).resolves.toEqual({
      mode: "login_url",
      openedUrl,
    });
  });
});
