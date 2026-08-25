import { describe, expect, it, vi } from "vitest";
import type { Project } from "@limecloud/app-server-client";
import {
  assignThreadProject,
  createProjectDirectoryEntry,
  listProjectDirectory,
  readThreadProjectId,
  type ProjectDirectoryAppServerClient,
} from "./projectDirectory";

function project(id: string, position: number): Project {
  return {
    id,
    name: `Project ${position}`,
    roots: [{ path: `/workspace/${id}` }],
    metadata: {},
    position,
    createdAt: 1,
    updatedAt: 1,
  };
}

describe("projectDirectory", () => {
  it("应通过 project/list 的 opaque cursor 读取完整目录", async () => {
    const first = project("project-1", 0);
    const second = project("project-2", 1);
    const request = vi
      .fn()
      .mockResolvedValueOnce({
        result: { data: [first], nextCursor: "opaque-page-2" },
      })
      .mockResolvedValueOnce({
        result: { data: [second], nextCursor: null },
      });
    const client = { request } as ProjectDirectoryAppServerClient;

    await expect(listProjectDirectory(client)).resolves.toEqual([
      first,
      second,
    ]);
    expect(request).toHaveBeenNthCalledWith(1, "project/list", { limit: 100 });
    expect(request).toHaveBeenNthCalledWith(2, "project/list", {
      cursor: "opaque-page-2",
      limit: 100,
    });
  });

  it("应从 canonical thread/read 恢复归属并通过 metadata update 写入或清空", async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce({
        result: { thread: { id: "thread-1", projectId: "project-1" } },
      })
      .mockResolvedValueOnce({
        result: { thread: { id: "thread-1", projectId: "project-2" } },
      })
      .mockResolvedValueOnce({
        result: { thread: { id: "thread-1", projectId: null } },
      });
    const client = { request } as ProjectDirectoryAppServerClient;

    await expect(readThreadProjectId(" thread-1 ", client)).resolves.toBe(
      "project-1",
    );
    await expect(
      assignThreadProject("thread-1", " project-2 ", client),
    ).resolves.toBe("project-2");
    await expect(assignThreadProject("thread-1", null, client)).resolves.toBe(
      null,
    );

    expect(request).toHaveBeenNthCalledWith(1, "thread/read", {
      threadId: "thread-1",
      includeTurns: false,
    });
    expect(request).toHaveBeenNthCalledWith(2, "thread/metadata/update", {
      threadId: "thread-1",
      projectId: "project-2",
    });
    expect(request).toHaveBeenNthCalledWith(3, "thread/metadata/update", {
      threadId: "thread-1",
      projectId: "",
    });
  });

  it("应使用 project/create 创建当前工作区目录项并拒绝损坏响应", async () => {
    const created = project("project-created", 0);
    const request = vi.fn().mockResolvedValue({ result: { project: created } });
    const client = { request } as ProjectDirectoryAppServerClient;

    await expect(
      createProjectDirectoryEntry(
        {
          idempotencyKey: "workspace-root-key",
          name: " Lime ",
          rootPath: " /workspace/lime ",
        },
        client,
      ),
    ).resolves.toEqual(created);
    expect(request).toHaveBeenCalledWith("project/create", {
      idempotencyKey: "workspace-root-key",
      name: "Lime",
      roots: [{ path: "/workspace/lime" }],
    });

    const invalidClient = {
      request: vi.fn().mockResolvedValue({
        result: { data: [], nextCursor: 42 },
      }),
    } as ProjectDirectoryAppServerClient;
    await expect(listProjectDirectory(invalidClient)).rejects.toThrow(
      "invalid nextCursor",
    );

    const invalidProjectClient = {
      request: vi.fn().mockResolvedValue({
        result: {
          data: [{ ...created, roots: undefined }],
          nextCursor: null,
        },
      }),
    } as ProjectDirectoryAppServerClient;
    await expect(listProjectDirectory(invalidProjectClient)).rejects.toThrow(
      "invalid project",
    );
  });

  it("应拒绝循环 cursor，避免目录读取无限循环", async () => {
    const client = {
      request: vi.fn().mockResolvedValue({
        result: { data: [project("project-1", 0)], nextCursor: "same" },
      }),
    } as ProjectDirectoryAppServerClient;

    await expect(listProjectDirectory(client)).rejects.toThrow(
      "repeated nextCursor",
    );
  });
});
