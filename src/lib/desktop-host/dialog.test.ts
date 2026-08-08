import { beforeEach, describe, expect, it, vi } from "vitest";

import { open, save } from "./dialog";

function clearElectronBridge(): void {
  delete (window as any).electronAPI;
}

describe("desktop-host/dialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    clearElectronBridge();
  });

  it("Electron dialog bridge 可用时委托真实 bridge", async () => {
    const dialogOpen = vi.fn().mockResolvedValue("/real/file.txt");
    const dialogSave = vi.fn().mockResolvedValue("/real/export.skills");
    (window as any).electronAPI = {
      invoke: vi.fn(),
      dialog: {
        open: dialogOpen,
        save: dialogSave,
      },
    };

    await expect(open({ multiple: false })).resolves.toBe("/real/file.txt");
    await expect(save({ title: "Export" })).resolves.toBe(
      "/real/export.skills",
    );

    expect(dialogOpen).toHaveBeenCalledWith({ multiple: false });
    expect(dialogSave).toHaveBeenCalledWith({ title: "Export" });
  });

  it("测试夹具不能伪造本机目录路径", async () => {
    await expect(open({ directory: true, multiple: false })).rejects.toThrow(
      "Native directory dialog is unavailable in browser preview.",
    );
  });

  it("测试环境允许普通文件选择夹具", async () => {
    await expect(open({ multiple: false })).resolves.toBe(
      "/mock/path/to/file.txt",
    );
  });

  it("测试环境允许 Skill 安装包导出保存夹具", async () => {
    await expect(
      save({
        filters: [{ name: "Skill package", extensions: ["skills", "skill"] }],
      }),
    ).resolves.toBe("/mock/path/to/saved/file.skills");
  });

  it("非测试环境无 Electron dialog bridge 时 fail-closed", async () => {
    vi.stubEnv("MODE", "production");
    vi.stubEnv("VITEST", "");

    try {
      await expect(open({ multiple: false })).rejects.toThrow(
        "dialog.open 只能在测试环境使用",
      );
      await expect(save({ title: "Export" })).rejects.toThrow(
        "dialog.save 只能在测试环境使用",
      );
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
