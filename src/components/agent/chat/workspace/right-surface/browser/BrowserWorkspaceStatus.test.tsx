import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { limeI18nResources } from "@/i18n/createI18n";
import { SUPPORTED_LOCALES, type SupportedLocale } from "@/i18n/locales";
import type {
  BrowserTabDownloadEvent,
  BrowserTabPermissionRequestEvent,
} from "@/lib/api/browserTab";
import {
  BrowserWorkspaceDownloadShelf,
  BrowserWorkspaceErrorBanner,
  BrowserWorkspaceHostUnavailable,
  BrowserWorkspaceLoading,
  BrowserWorkspacePermissionBanner,
} from "./BrowserWorkspaceStatus";

const browserKeys = [
  "agentChat.browserWorkspace.hostUnavailableTitle",
  "agentChat.browserWorkspace.hostUnavailableBody",
  "agentChat.browserWorkspace.loading",
  "agentChat.browserWorkspace.loadFailedTitle",
  "agentChat.browserWorkspace.loadFailedBody",
  "agentChat.browserWorkspace.permissionBlockedTitle",
  "agentChat.browserWorkspace.permissionBlockedBody",
  "agentChat.browserWorkspace.downloadComplete",
] as const;

const permission: BrowserTabPermissionRequestEvent = {
  browserSessionId: "browser-session-1",
  decision: "blocked",
  embeddingOrigin: "https://example.com",
  ownerWebContentsId: 41,
  permission: "geolocation",
  requestingUrl: "https://example.com/",
  requestId: "permission-1",
  tabId: "browser-session-1:user:primary",
  threadId: "thread-1",
  url: "https://example.com/",
  viewId: "browser:browser-session-1:user:primary",
  webContentsId: 101,
  windowId: 7,
};

const download: BrowserTabDownloadEvent = {
  browserSessionId: "browser-session-1",
  canResume: false,
  downloadId: "download-1",
  filename: "quarterly-report.pdf",
  mimeType: "application/pdf",
  ownerWebContentsId: 41,
  receivedBytes: 100,
  state: "completed",
  tabId: "browser-session-1:user:primary",
  threadId: "thread-1",
  totalBytes: 100,
  url: "https://example.com/quarterly-report.pdf",
  viewId: "browser:browser-session-1:user:primary",
  webContentsId: 101,
  windowId: 7,
};

function translate(locale: SupportedLocale) {
  const resource = limeI18nResources[locale].agent as Record<string, string>;
  return (key: string, options?: Record<string, unknown>) => {
    const value = resource[key] ?? key;
    return value.replace(/\{\{(\w+)\}\}/g, (_match, name: string) =>
      String(options?.[name] ?? `{{${name}}}`),
    );
  };
}

describe("BrowserWorkspaceStatus", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("为五个界面语言提供 Browser 状态文案，并保留插值", () => {
    for (const locale of SUPPORTED_LOCALES) {
      const resource = limeI18nResources[locale].agent as Record<
        string,
        string
      >;
      for (const key of browserKeys) {
        expect(resource[key], `${locale} 缺少 ${key}`).toBeTypeOf("string");
        expect(resource[key], `${locale} 回退成 key: ${key}`).not.toBe(key);
      }

      const t = translate(locale);
      expect(
        t("agentChat.browserWorkspace.downloadComplete", download),
      ).toContain(download.filename);
      expect(
        t("agentChat.browserWorkspace.permissionBlockedBody", {
          source: permission.requestingUrl,
        }),
      ).toContain(permission.requestingUrl);
    }
  });

  it("为 loading、错误、权限和下载状态提供稳定语义与布局护栏", () => {
    const t = translate("zh-CN");
    act(() => {
      root.render(
        <>
          <BrowserWorkspaceHostUnavailable t={t} />
          <BrowserWorkspaceLoading t={t} />
          <BrowserWorkspaceErrorBanner
            error={{
              body: "页面无法打开",
              source: "load",
              title: "页面加载失败",
            }}
          />
          <BrowserWorkspaceErrorBanner
            error={{
              body: "Host 不可用",
              source: "host",
              title: "浏览器宿主不可用",
            }}
          />
          <BrowserWorkspacePermissionBanner permission={permission} t={t} />
          <BrowserWorkspaceDownloadShelf download={download} t={t} />
        </>,
      );
    });

    expect(
      container
        .querySelector('[data-browser-workspace-status="host-unavailable"]')
        ?.getAttribute("role"),
    ).toBe("alert");
    expect(
      container
        .querySelector('[data-browser-workspace-status="loading"]')
        ?.getAttribute("aria-live"),
    ).toBe("polite");
    expect(
      container
        .querySelector('[data-browser-workspace-status="load-error"]')
        ?.getAttribute("data-browser-error-source"),
    ).toBe("load");
    expect(
      container
        .querySelector('[data-browser-workspace-status="host-error"]')
        ?.getAttribute("data-browser-error-source"),
    ).toBe("host");
    expect(
      container
        .querySelector('[data-browser-workspace-status="permission-blocked"]')
        ?.getAttribute("role"),
    ).toBe("alert");
    expect(
      container
        .querySelector('[data-browser-workspace-status="download-completed"]')
        ?.getAttribute("aria-live"),
    ).toBe("polite");

    for (const selector of [
      '[data-browser-workspace-status="load-error"]',
      '[data-browser-workspace-status="permission-blocked"]',
      '[data-browser-workspace-status="download-completed"]',
    ]) {
      expect(container.querySelector(selector)?.className).toContain(
        "shrink-0",
      );
      expect(container.querySelector(selector)?.className).not.toContain(
        "absolute",
      );
    }
    expect(
      container.querySelector(
        '[data-browser-workspace-status="permission-blocked"]',
      )?.textContent,
    ).toContain(permission.requestingUrl);
    expect(
      container.querySelector(
        '[data-browser-workspace-status="download-completed"]',
      )?.textContent,
    ).toContain(download.filename);
  });
});
