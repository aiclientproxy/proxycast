import { describe, expect, it, vi } from "vitest";
import {
  createThreadSectionClient,
  isPinnedThreadSession,
  PINNED_THREAD_SECTION,
  type ThreadSectionClient,
} from "./threadSections";

function threadSectionTransportMock() {
  return {
    createThreadSection: vi.fn(),
    deleteThreadSection: vi.fn(),
    listThreadSections: vi.fn(),
    moveThreadToSection: vi.fn(),
    updateThreadSection: vi.fn(),
  } as unknown as ThreadSectionClient;
}

describe("threadSections", () => {
  it("应通过 current App Server gateway 移入或移出 section", async () => {
    const transport = threadSectionTransportMock();
    vi.mocked(transport.moveThreadToSection).mockResolvedValue({
      result: {},
    } as never);
    const client = createThreadSectionClient(transport);

    await client.moveThreadToSection({
      threadId: " thread-1 ",
      sectionId: ` ${PINNED_THREAD_SECTION.id} `,
    });
    await client.moveThreadToSection({
      threadId: "thread-1",
      sectionId: null,
    });

    expect(transport.moveThreadToSection).toHaveBeenNthCalledWith(1, {
      threadId: "thread-1",
      sectionId: PINNED_THREAD_SECTION.id,
    });
    expect(transport.moveThreadToSection).toHaveBeenNthCalledWith(2, {
      threadId: "thread-1",
      sectionId: null,
    });
  });

  it("应按内置 section id 判断置顶状态，并拒绝空 threadId", async () => {
    const transport = threadSectionTransportMock();
    const client = createThreadSectionClient(transport);

    expect(
      isPinnedThreadSession({ section: { ...PINNED_THREAD_SECTION } }),
    ).toBe(true);
    expect(isPinnedThreadSession({ section: undefined })).toBe(false);
    await expect(
      client.moveThreadToSection({ threadId: " ", sectionId: null }),
    ).rejects.toThrow("threadId is required");
    await expect(
      client.moveThreadToSection({ threadId: "thread-1", sectionId: " " }),
    ).rejects.toThrow("sectionId must be null or a non-empty section id");
    expect(transport.moveThreadToSection).not.toHaveBeenCalled();
  });

  it("应分页读取 section catalog 并保持服务端顺序", async () => {
    const transport = threadSectionTransportMock();
    vi.mocked(transport.listThreadSections)
      .mockResolvedValueOnce({
        result: {
          data: [{ id: PINNED_THREAD_SECTION.id, name: "Pinned" }],
          nextCursor: "cursor-2",
        },
      } as never)
      .mockResolvedValueOnce({
        result: {
          data: [{ id: "section-active", name: "Active" }],
          nextCursor: null,
        },
      } as never);
    const client = createThreadSectionClient(transport);

    await expect(client.listThreadSections()).resolves.toEqual([
      { id: PINNED_THREAD_SECTION.id, name: "Pinned" },
      { id: "section-active", name: "Active" },
    ]);
    expect(transport.listThreadSections).toHaveBeenNthCalledWith(1, {
      limit: 100,
    });
    expect(transport.listThreadSections).toHaveBeenNthCalledWith(2, {
      cursor: "cursor-2",
      limit: 100,
    });
  });

  it("应通过唯一 typed gateway 创建、重命名和删除 section", async () => {
    const transport = threadSectionTransportMock();
    vi.mocked(transport.createThreadSection).mockResolvedValue({
      result: { section: { id: "section-active", name: "Active" } },
    } as never);
    vi.mocked(transport.updateThreadSection).mockResolvedValue({
      result: { section: { id: "section-active", name: "Current" } },
    } as never);
    vi.mocked(transport.deleteThreadSection).mockResolvedValue({
      result: {},
    } as never);
    const client = createThreadSectionClient(transport);

    await expect(
      client.createThreadSection({ name: " Active " }),
    ).resolves.toEqual({ id: "section-active", name: "Active" });
    await expect(
      client.updateThreadSection({
        sectionId: " section-active ",
        name: " Current ",
      }),
    ).resolves.toEqual({ id: "section-active", name: "Current" });
    await client.deleteThreadSection({ sectionId: " section-active " });

    expect(transport.createThreadSection).toHaveBeenCalledWith({
      name: "Active",
    });
    expect(transport.updateThreadSection).toHaveBeenCalledWith({
      sectionId: "section-active",
      name: "Current",
    });
    expect(transport.deleteThreadSection).toHaveBeenCalledWith({
      sectionId: "section-active",
    });
  });

  it("应在网络调用前拒绝空 section 名称与 id", async () => {
    const transport = threadSectionTransportMock();
    const client = createThreadSectionClient(transport);

    await expect(client.createThreadSection({ name: " " })).rejects.toThrow(
      "section name is required",
    );
    await expect(
      client.updateThreadSection({ sectionId: " ", name: "Active" }),
    ).rejects.toThrow("section id is required");
    await expect(
      client.deleteThreadSection({ sectionId: " " }),
    ).rejects.toThrow("section id is required");
    expect(transport.createThreadSection).not.toHaveBeenCalled();
    expect(transport.updateThreadSection).not.toHaveBeenCalled();
    expect(transport.deleteThreadSection).not.toHaveBeenCalled();
  });
});
