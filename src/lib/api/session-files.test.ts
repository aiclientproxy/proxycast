import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import {
  deleteFile,
  getOrCreateSession,
  importDocument,
  listFiles,
  readFile,
  resolveFilePath,
  saveFile,
  updateSessionMeta,
} from "./session-files";

const {
  appServerDeleteSessionFileMock,
  appServerGetOrCreateSessionFileMock,
  appServerListSessionFilesMock,
  readFilePreviewMock,
  appServerReadSessionFileMock,
  appServerResolveSessionFilePathMock,
  appServerSaveSessionFileMock,
  appServerUpdateSessionFileMetaMock,
} = vi.hoisted(() => ({
  appServerDeleteSessionFileMock: vi.fn(),
  appServerGetOrCreateSessionFileMock: vi.fn(),
  appServerListSessionFilesMock: vi.fn(),
  readFilePreviewMock: vi.fn(),
  appServerReadSessionFileMock: vi.fn(),
  appServerResolveSessionFilePathMock: vi.fn(),
  appServerSaveSessionFileMock: vi.fn(),
  appServerUpdateSessionFileMetaMock: vi.fn(),
}));

vi.mock("@/lib/api/appServer", () => ({
  createAppServerClient: vi.fn(() => ({
    deleteSessionFile: appServerDeleteSessionFileMock,
    getOrCreateSessionFile: appServerGetOrCreateSessionFileMock,
    listSessionFiles: appServerListSessionFilesMock,
    readSessionFile: appServerReadSessionFileMock,
    resolveSessionFilePath: appServerResolveSessionFilePathMock,
    saveSessionFile: appServerSaveSessionFileMock,
    updateSessionFileMeta: appServerUpdateSessionFileMetaMock,
  })),
}));

vi.mock("@/lib/api/fileBrowser", () => ({
  readFilePreview: readFilePreviewMock,
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

describe("session-files API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("session-scoped 写读链走 App Server sessionFile current 方法", async () => {
    appServerGetOrCreateSessionFileMock.mockResolvedValueOnce({
      result: {
        meta: {
          sessionId: "session-1",
          createdAt: 1,
          updatedAt: 2,
          fileCount: 0,
          totalSize: 0,
        },
      },
    });
    appServerUpdateSessionFileMetaMock.mockResolvedValueOnce({
      result: {
        meta: {
          sessionId: "session-1",
          title: "新会话",
          theme: "article",
          createdAt: 1,
          updatedAt: 3,
          fileCount: 0,
          totalSize: 0,
        },
      },
    });
    appServerListSessionFilesMock.mockResolvedValueOnce({
      result: { files: [{ name: "article.md", fileType: "document" }] },
    });
    appServerReadSessionFileMock.mockResolvedValueOnce({
      result: { content: "# title" },
    });
    appServerResolveSessionFilePathMock.mockResolvedValueOnce({
      result: { path: "/tmp/session-1/files/article.md" },
    });
    appServerSaveSessionFileMock.mockResolvedValueOnce({
      result: { file: { name: "article.md", fileType: "document" } },
    });
    appServerDeleteSessionFileMock.mockResolvedValueOnce({ result: {} });

    await expect(getOrCreateSession("session-1")).resolves.toMatchObject({
      sessionId: "session-1",
      fileCount: 0,
    });
    await expect(
      updateSessionMeta("session-1", {
        title: "新会话",
        theme: "article",
      }),
    ).resolves.toMatchObject({
      sessionId: "session-1",
      title: "新会话",
      theme: "article",
    });
    await expect(listFiles("session-1")).resolves.toEqual([
      { name: "article.md", fileType: "document" },
    ]);
    await expect(readFile("session-1", "article.md")).resolves.toBe("# title");
    await expect(resolveFilePath("session-1", "article.md")).resolves.toBe(
      "/tmp/session-1/files/article.md",
    );
    await expect(
      saveFile("session-1", "article.md", "# title", { kind: "draft" }),
    ).resolves.toEqual({ name: "article.md", fileType: "document" });
    await expect(deleteFile("session-1", "article.md")).resolves.toBe(
      undefined,
    );

    expect(appServerGetOrCreateSessionFileMock).toHaveBeenCalledWith({
      sessionId: "session-1",
    });
    expect(appServerUpdateSessionFileMetaMock).toHaveBeenCalledWith({
      sessionId: "session-1",
      title: "新会话",
      theme: "article",
    });
    expect(appServerListSessionFilesMock).toHaveBeenCalledWith({
      sessionId: "session-1",
    });
    expect(appServerReadSessionFileMock).toHaveBeenCalledWith({
      sessionId: "session-1",
      fileName: "article.md",
    });
    expect(appServerResolveSessionFilePathMock).toHaveBeenCalledWith({
      sessionId: "session-1",
      fileName: "article.md",
    });
    expect(appServerSaveSessionFileMock).toHaveBeenCalledWith({
      sessionId: "session-1",
      fileName: "article.md",
      content: "# title",
      metadata: { kind: "draft" },
    });
    expect(appServerDeleteSessionFileMock).toHaveBeenCalledWith({
      sessionId: "session-1",
      fileName: "article.md",
    });

    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Document Import 走 App Server file preview，不再调用旧导入命令", async () => {
    readFilePreviewMock.mockResolvedValueOnce({
      path: "/tmp/doc.md",
      content: "document text",
      isBinary: false,
      size: 13,
      error: null,
    });

    await expect(importDocument("/tmp/doc.md")).resolves.toBe("document text");

    expect(readFilePreviewMock).toHaveBeenCalledWith(
      "/tmp/doc.md",
      2 * 1024 * 1024,
    );
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "import_document",
      expect.anything(),
    );
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "import_document_to_session",
      expect.anything(),
    );
  });

  it("Document Import 对 App Server file preview 非文本形态 fail closed", async () => {
    readFilePreviewMock.mockResolvedValueOnce({
      path: "/tmp/doc.md",
      content: undefined,
      isBinary: false,
      size: 0,
      error: null,
    });

    await expect(importDocument("/tmp/doc.md")).rejects.toThrow(
      "fs/readFile did not return document text",
    );
  });

  it("Document Import 对 App Server file preview 错误与二进制文件 fail closed", async () => {
    readFilePreviewMock
      .mockResolvedValueOnce({
        path: "/tmp/doc.md",
        content: null,
        isBinary: false,
        size: 0,
        error: "too large",
      })
      .mockResolvedValueOnce({
        path: "/tmp/image.png",
        content: null,
        isBinary: true,
        size: 8,
        error: null,
      });

    await expect(importDocument("/tmp/doc.md")).rejects.toThrow("too large");
    await expect(importDocument("/tmp/image.png")).rejects.toThrow(
      "当前文稿导入只支持文本文件",
    );
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "import_document",
      expect.anything(),
    );
  });
});
