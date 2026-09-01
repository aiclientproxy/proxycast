import { describe, expect, it, vi } from "vitest";
import {
  installMainWindowMediaPermissionHandler,
  shouldAllowMainWindowMediaPermission,
} from "./mainWindowMediaPermissions";

describe("mainWindowMediaPermissions", () => {
  it("只允许主窗口请求音频和视频媒体权限", () => {
    const mainWebContents = { id: 1 };
    const requestWebContents = { id: 1 };

    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "media",
        details: { mediaTypes: ["audio"] },
      }),
    ).toBe(true);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "microphone",
      }),
    ).toBe(true);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "media",
        details: { mediaTypes: ["video"] },
      }),
    ).toBe(true);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "camera",
      }),
    ).toBe(true);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "media",
        details: { mediaTypes: ["audio", "video"] },
      }),
    ).toBe(true);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "media",
        details: { mediaTypes: ["unknown"] },
      }),
    ).toBe(false);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents: { id: 2 },
        permission: "media",
        details: { mediaTypes: ["audio"] },
      }),
    ).toBe(false);
    expect(
      shouldAllowMainWindowMediaPermission({
        mainWebContents,
        requestWebContents,
        permission: "notifications",
      }),
    ).toBe(false);
  });

  it("安装到 default session 后按主窗口身份回调", () => {
    const setPermissionRequestHandler = vi.fn();
    installMainWindowMediaPermissionHandler({
      session: { setPermissionRequestHandler },
      getMainWindow: () => ({ webContents: { id: 8 } }),
    });

    const handler = setPermissionRequestHandler.mock.calls[0]?.[0];
    const callback = vi.fn();
    handler?.({ id: 8 }, "media", callback, { mediaTypes: ["audio"] });

    expect(callback).toHaveBeenCalledWith(true);
  });
});
