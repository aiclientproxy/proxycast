import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  requestChatHostOpenPath,
  requestChatHostSavePath,
} from "./chatHostCapabilities";
import { open, save } from "@/lib/desktop-host/plugin-dialog";
import { selectPluginDirectory } from "@/lib/api/plugins";
import { getElectronHostBridge } from "@/lib/electron-host";

vi.mock("@/lib/desktop-host/plugin-dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

vi.mock("@/lib/api/plugins", () => ({
  selectPluginDirectory: vi.fn(),
}));

vi.mock("@/lib/electron-host", () => ({
  getElectronHostBridge: vi.fn(),
}));

describe("chatHostCapabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getElectronHostBridge).mockReturnValue(null);
  });

  it("Electron dialog 可用时应原样转发目录选择", async () => {
    const options = {
      directory: true,
      multiple: false as const,
      title: "选择工作区",
    };
    vi.mocked(getElectronHostBridge).mockReturnValue({
      dialog: {},
    } as never);
    vi.mocked(open).mockResolvedValueOnce("/workspace/project");

    await expect(requestChatHostOpenPath(options)).resolves.toBe(
      "/workspace/project",
    );
    expect(open).toHaveBeenCalledWith(options);
    expect(selectPluginDirectory).not.toHaveBeenCalled();
  });

  it("浏览器镜像应通过真实 Desktop Host 命令选择目录", async () => {
    vi.mocked(selectPluginDirectory).mockResolvedValueOnce({
      path: "/workspace/project",
      cancelled: false,
    });

    await expect(
      requestChatHostOpenPath({
        directory: true,
        multiple: false,
        title: "选择工作区",
      }),
    ).resolves.toBe("/workspace/project");
    expect(selectPluginDirectory).toHaveBeenCalledWith({
      title: "选择工作区",
    });
    expect(open).not.toHaveBeenCalled();
  });

  it("浏览器镜像取消目录选择时应返回 null", async () => {
    vi.mocked(selectPluginDirectory).mockResolvedValueOnce({
      path: null,
      cancelled: true,
    });

    await expect(
      requestChatHostOpenPath({ directory: true, multiple: false }),
    ).resolves.toBeNull();
  });

  it("单文件选择应保持原 Dialog 路径", async () => {
    const options = {
      directory: false,
      multiple: false as const,
      filters: [{ name: "文档", extensions: ["md", "txt"] }],
    };
    vi.mocked(open).mockResolvedValueOnce(null);

    await expect(requestChatHostOpenPath(options)).resolves.toBeNull();
    expect(open).toHaveBeenCalledWith(options);
    expect(selectPluginDirectory).not.toHaveBeenCalled();
  });

  it("原样转发保存路径 options 与结果", async () => {
    const options = {
      defaultPath: "artifact.md",
      filters: [{ name: "Markdown", extensions: ["md"] }],
      title: "导出文档",
    };
    vi.mocked(save).mockResolvedValueOnce("/workspace/artifact.md");

    await expect(requestChatHostSavePath(options)).resolves.toBe(
      "/workspace/artifact.md",
    );
    expect(save).toHaveBeenCalledWith(options);
  });
});
