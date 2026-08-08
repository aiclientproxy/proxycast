import { getElectronHostBridge } from "@/lib/electron-host";

type UrlCallback = (urls: string[]) => void;
type UnlistenFn = () => void;

function isTestEnvironment(): boolean {
  return Boolean(import.meta.env?.MODE === "test" || import.meta.env?.VITEST);
}

function assertTestDeepLinkFixture(apiName: string): void {
  if (isTestEnvironment()) {
    return;
  }
  throw new Error(
    `[Mock] ${apiName} 只能在测试环境使用；生产 Deep Link 能力必须进入 Electron Desktop Host IPC。`,
  );
}

export async function onOpenUrl(handler: UrlCallback): Promise<UnlistenFn> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.deepLink) {
    return electronHost.deepLink.onOpenUrl(handler);
  }

  assertTestDeepLinkFixture("deepLink.onOpenUrl");
  console.log("[Mock] Deep link onOpenUrl registered");

  if (typeof window !== "undefined") {
    const urlParams = new URLSearchParams(window.location.search);
    const deepLinkUrl = urlParams.get("lime");

    if (deepLinkUrl) {
      console.log("[Mock] Deep link URL from params:", deepLinkUrl);
      setTimeout(() => handler([deepLinkUrl]), 100);
    }
  }

  return () => {
    console.log("[Mock] Deep link unlisten");
  };
}

export async function getUrls(): Promise<string[]> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.deepLink) {
    return electronHost.deepLink.getUrls();
  }

  assertTestDeepLinkFixture("deepLink.getUrls");
  console.log("[Mock] Deep link getUrls");

  if (typeof window !== "undefined") {
    const urlParams = new URLSearchParams(window.location.search);
    const deepLinkUrl = urlParams.get("lime");
    return deepLinkUrl ? [deepLinkUrl] : [];
  }

  return [];
}

export async function getCurrent(): Promise<string[] | null> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.deepLink) {
    return electronHost.deepLink.getCurrent();
  }

  const urls = await getUrls();
  return urls.length > 0 ? urls : null;
}
