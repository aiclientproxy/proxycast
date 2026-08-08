import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  requestChatHostOpenPath,
  requestChatHostSavePath,
} from "./chatHostCapabilities";
import { open, save } from "@/lib/desktop-host/dialog";

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

describe("chatHostCapabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("Electron dialog 可用时应原样转发目录选择", async () => {
    const options = {
      directory: true,
      multiple: false as const,
      title: "选择工作区",
    };
    vi.mocked(open).mockResolvedValueOnce("/workspace/project");

    await expect(requestChatHostOpenPath(options)).resolves.toBe(
      "/workspace/project",
    );
    expect(open).toHaveBeenCalledWith(options);
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
