import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  findButtonContaining,
  getTextarea,
  mockGetProject,
  mockSearchProjectFiles,
  renderHarness,
  typeMentionAndWait,
} from "./CharacterMention.testFixtures";

describe("CharacterMention project files", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("在项目内查询文件并只替换当前 @token", async () => {
    vi.useFakeTimers();
    mockGetProject.mockResolvedValue({
      id: "project-1",
      rootPath: "/workspace",
    });
    mockSearchProjectFiles.mockResolvedValue([
      {
        root: "/workspace",
        path: "docs/product brief.md",
        match_type: "file",
        file_name: "product brief.md",
        score: 84,
        indices: [5, 6, 7],
      },
    ]);
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      projectId: "project-1",
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "compare @brief with @other");
    act(() => {
      textarea.setSelectionRange(14, 14);
      textarea.dispatchEvent(new Event("keyup", { bubbles: true }));
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120);
      await Promise.resolve();
    });

    expect(mockGetProject).toHaveBeenCalledWith("project-1");
    expect(mockSearchProjectFiles).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "brief",
        rootPath: "/workspace",
        cancellationToken: expect.stringContaining("composer-project-files-"),
      }),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(document.body.textContent).toContain("项目文件");
    const fileButton = findButtonContaining(
      "product brief.md",
      "docs/product brief.md",
    );
    expect(fileButton).toBeTruthy();

    act(() => {
      fileButton?.click();
    });

    expect(onChangeSpy).toHaveBeenLastCalledWith(
      'compare "docs/product brief.md" with @other',
    );
  });

  it("没有项目时不调用文件搜索", async () => {
    vi.useFakeTimers();
    const container = renderHarness({ projectId: null });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@app");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120);
    });

    expect(mockGetProject).not.toHaveBeenCalled();
    expect(mockSearchProjectFiles).not.toHaveBeenCalled();
  });

  it("旧查询晚到时不得覆盖当前文件结果", async () => {
    vi.useFakeTimers();
    mockGetProject.mockResolvedValue({
      id: "project-1",
      rootPath: "/workspace",
    });
    let resolveFirst: ((value: unknown[]) => void) | undefined;
    let resolveSecond: ((value: unknown[]) => void) | undefined;
    mockSearchProjectFiles
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );
    const container = renderHarness({ projectId: "project-1" });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@app");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120);
      await Promise.resolve();
    });
    await typeMentionAndWait(textarea, "@read");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120);
      await Promise.resolve();
      resolveSecond?.([
        {
          root: "/workspace",
          path: "README.md",
          match_type: "file",
          file_name: "README.md",
          score: 90,
          indices: [0, 1],
        },
      ]);
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("README.md");

    await act(async () => {
      resolveFirst?.([
        {
          root: "/workspace",
          path: "src/app.rs",
          match_type: "file",
          file_name: "app.rs",
          score: 84,
          indices: [4, 5, 6],
        },
      ]);
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("README.md");
    expect(document.body.textContent).not.toContain("src/app.rs");
  });
});
