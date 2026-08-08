import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import {
  createDirectoryAtPath,
  createFileAtPath,
  deletePath,
  getFileIconDataUrl,
  getFileManagerLocations,
  listDirectory,
  readFilePreview,
  renamePath,
} from "./fileBrowser";

const {
  appServerCreateDirectoryMock,
  appServerCopyMock,
  appServerGetMetadataMock,
  appServerReadDirectoryMock,
  appServerReadFileMock,
  appServerRemoveMock,
  appServerWriteFileMock,
} = vi.hoisted(() => ({
  appServerCreateDirectoryMock: vi.fn(),
  appServerCopyMock: vi.fn(),
  appServerGetMetadataMock: vi.fn(),
  appServerReadDirectoryMock: vi.fn(),
  appServerReadFileMock: vi.fn(),
  appServerRemoveMock: vi.fn(),
  appServerWriteFileMock: vi.fn(),
}));

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn(() => ({
    createDirectory: appServerCreateDirectoryMock,
    copy: appServerCopyMock,
    getMetadata: appServerGetMetadataMock,
    readDirectory: appServerReadDirectoryMock,
    readFile: appServerReadFileMock,
    remove: appServerRemoveMock,
    writeFile: appServerWriteFileMock,
  })),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

describe("fileBrowser API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("应通过 App Server current 主链获取目录列表与文件预览", async () => {
    appServerReadDirectoryMock.mockResolvedValueOnce({
      result: {
        entries: [
          {
            fileName: "Lime.app",
            isDirectory: true,
            isFile: false,
          },
        ],
      },
    });
    appServerGetMetadataMock.mockResolvedValueOnce({
      result: {
        isDirectory: true,
        isFile: false,
        isSymlink: false,
        createdAtMs: 0,
        modifiedAtMs: 1,
      },
    });
    appServerReadFileMock.mockResolvedValueOnce({
      result: { dataBase64: "aGVsbG8=" },
    });

    await expect(listDirectory("/Applications")).resolves.toEqual(
      expect.objectContaining({
        path: "/Applications",
        entries: [
          expect.objectContaining({
            name: "Lime.app",
            path: "/Applications/Lime.app",
            modifiedAt: 1,
          }),
        ],
      }),
    );
    await expect(readFilePreview("/tmp/demo.txt", 1024)).resolves.toEqual(
      expect.objectContaining({ path: "/tmp/demo.txt", content: "hello" }),
    );
    expect(appServerReadDirectoryMock).toHaveBeenCalledWith({
      path: "/Applications",
    });
    expect(appServerGetMetadataMock).toHaveBeenCalledWith({
      path: "/Applications/Lime.app",
    });
    expect(appServerReadFileMock).toHaveBeenCalledWith({
      path: "/tmp/demo.txt",
    });
  });

  it("文件预览应在 renderer 解码、截断文本并识别二进制", async () => {
    appServerReadFileMock
      .mockResolvedValueOnce({ result: { dataBase64: "aGVsbG8=" } })
      .mockResolvedValueOnce({ result: { dataBase64: "AP8=" } });

    await expect(readFilePreview("/tmp/demo.txt", 2)).resolves.toEqual({
      path: "/tmp/demo.txt",
      content: "he",
      isBinary: false,
      size: 5,
      error: null,
    });
    await expect(readFilePreview("/tmp/demo.bin", 1024)).resolves.toEqual({
      path: "/tmp/demo.bin",
      content: null,
      isBinary: true,
      size: 2,
      error: null,
    });
  });

  it("应代理文件增删改命令", async () => {
    appServerWriteFileMock.mockResolvedValueOnce({ result: {} });
    appServerCreateDirectoryMock.mockResolvedValueOnce({ result: {} });
    appServerGetMetadataMock.mockResolvedValueOnce({
      result: { isDirectory: false },
    });
    appServerCopyMock.mockResolvedValueOnce({ result: {} });
    appServerRemoveMock.mockResolvedValue({ result: {} });

    await expect(createFileAtPath("/tmp/demo.txt")).resolves.toBeUndefined();
    await expect(
      createDirectoryAtPath("/tmp/demo-dir"),
    ).resolves.toBeUndefined();
    await expect(
      renamePath("/tmp/demo.txt", "/tmp/demo2.txt"),
    ).resolves.toBeUndefined();
    await expect(deletePath("/tmp/demo2.txt", false)).resolves.toBeUndefined();

    expect(appServerWriteFileMock).toHaveBeenCalledWith({
      path: "/tmp/demo.txt",
      dataBase64: "",
    });
    expect(appServerCreateDirectoryMock).toHaveBeenCalledWith({
      path: "/tmp/demo-dir",
      recursive: true,
    });
    expect(appServerGetMetadataMock).toHaveBeenCalledWith({
      path: "/tmp/demo.txt",
    });
    expect(appServerCopyMock).toHaveBeenCalledWith({
      sourcePath: "/tmp/demo.txt",
      destinationPath: "/tmp/demo2.txt",
      recursive: false,
    });
    expect(appServerRemoveMock).toHaveBeenNthCalledWith(1, {
      path: "/tmp/demo.txt",
      recursive: false,
      force: false,
    });
    expect(appServerRemoveMock).toHaveBeenNthCalledWith(2, {
      path: "/tmp/demo2.txt",
      recursive: false,
      force: false,
    });
  });

  it("文件写命令应透传 App Server RPC 错误", async () => {
    const error = new Error("fs/writeFile failed");
    appServerWriteFileMock.mockRejectedValueOnce(error);

    await expect(createFileAtPath("/tmp/demo.txt")).rejects.toThrow(error);
  });

  it("创建目录时应原样传递 Windows 原生路径", async () => {
    appServerCreateDirectoryMock.mockResolvedValueOnce({ result: {} });
    const windowsPath = String.raw`C:\Users\demo\workspace\new-folder`;

    await expect(createDirectoryAtPath(windowsPath)).resolves.toBeUndefined();

    expect(appServerCreateDirectoryMock).toHaveBeenCalledWith({
      path: windowsPath,
      recursive: true,
    });
  });

  it("应代理文件管理器快捷入口命令", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce([
      {
        id: "downloads",
        label: "下载",
        path: "/Users/demo/Downloads",
        kind: "downloads",
      },
    ]);

    await expect(getFileManagerLocations()).resolves.toEqual([
      expect.objectContaining({ id: "downloads", label: "下载" }),
    ]);
    expect(safeInvoke).toHaveBeenCalledWith("get_file_manager_locations");
  });

  it("文件管理器快捷入口遇到 Electron empty diagnostic list 时应 fail closed", async () => {
    const diagnosticList: unknown[] = [];
    Object.defineProperty(diagnosticList, "__diagnostic", {
      value: {
        command: "get_file_manager_locations",
        source: "electron-empty-diagnostic",
        status: "degraded",
      },
      enumerable: false,
    });

    vi.mocked(safeInvoke).mockResolvedValueOnce(diagnosticList);

    await expect(getFileManagerLocations()).rejects.toThrow(
      "get_file_manager_locations 尚未接入真实文件管理 current 通道，收到 electron-empty-diagnostic 诊断返回。",
    );
  });

  it("文件管理器快捷入口遇到 mock-like payload 或缺字段项时应 fail closed", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({ success: true })
      .mockResolvedValueOnce([{ id: "downloads" }]);

    await expect(getFileManagerLocations()).rejects.toThrow(
      "get_file_manager_locations did not return file manager locations",
    );
    await expect(getFileManagerLocations()).rejects.toThrow(
      "get_file_manager_locations did not return file manager locations",
    );
  });

  it("应代理文件图标异步读取命令", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce("data:image/png;base64,abc");

    await expect(getFileIconDataUrl("/Applications/Lime.app")).resolves.toBe(
      "data:image/png;base64,abc",
    );
    expect(safeInvoke).toHaveBeenCalledWith("get_file_icon_data_url", {
      path: "/Applications/Lime.app",
    });
  });

  it("文件图标异步读取允许 null，但错误 payload 应 fail closed", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce({ success: true })
      .mockResolvedValueOnce({ error: "failed" });

    await expect(getFileIconDataUrl("/tmp/missing.txt")).resolves.toBeNull();
    await expect(getFileIconDataUrl("/Applications/Lime.app")).rejects.toThrow(
      "get_file_icon_data_url did not return file icon data URL",
    );
    await expect(getFileIconDataUrl("/Applications/Lime.app")).rejects.toThrow(
      "get_file_icon_data_url did not return file icon data URL",
    );
  });
});
