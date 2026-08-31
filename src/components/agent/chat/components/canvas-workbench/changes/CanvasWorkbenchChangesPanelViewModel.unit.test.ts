import { describe, expect, it } from "vitest";
import {
  buildCanvasWorkbenchChangeFileTree,
  buildCanvasWorkbenchChangeDiffLines,
  buildCanvasWorkbenchGitApplyPatch,
  countCanvasWorkbenchDiffStats,
  countCanvasWorkbenchChangeItemsStats,
  inferCanvasWorkbenchChangeKind,
  findChangeItemForSelection,
  isPendingChangeItem,
  flattenCanvasWorkbenchChangeFileTree,
  parseCanvasWorkbenchGitPatchToChangeItems,
  resolveCanvasWorkbenchSelectedChangeItem,
  resolveChangeDisplayMeta,
  resolveChangeItemDisplayName,
  resolveChangeStatusClassName,
  resolveChangeStatusCopyKey,
  summarizeCanvasWorkbenchChanges,
} from "./CanvasWorkbenchChangesPanelViewModel";
import { resolveCanvasWorkbenchChangeTreeNodeMeta } from "./CanvasWorkbenchChangeTreeMeta";

describe("CanvasWorkbenchChangesPanelViewModel", () => {
  it("应按 selection path / target path / 标题匹配变更项", () => {
    const items = [
      {
        id: "a",
        path: "src/app.ts",
        absolutePath: "/workspace/src/app.ts",
        status: "completed" as const,
      },
      {
        id: "b",
        path: "README.md",
        displayName: "README.md",
        status: "in_progress" as const,
      },
    ];

    expect(
      findChangeItemForSelection(items, {
        title: "App",
        selectionPath: "/workspace/SRC/APP.ts",
        target: { kind: "empty", title: "empty" },
      })?.id,
    ).toBe("a");
    expect(
      findChangeItemForSelection(items, {
        title: "README.md",
        target: { kind: "empty", title: "empty" },
      })?.id,
    ).toBe("b");
    expect(findChangeItemForSelection(items, null)).toBeNull();
  });

  it("应优先选中带证据的 runtime 变更项而不是默认预览项", () => {
    const items = [
      {
        id: "preview",
        path: "greeting.ts",
        source: "runtime" as const,
        preview: "export function greeting() returns Hello Lime Workbench.",
        status: "completed" as const,
      },
      {
        id: "runtime",
        path: "src/coding-target.ts",
        source: "runtime" as const,
        preview: "export const codingWorkbenchSmoke = true;",
        status: "completed" as const,
        changeKind: "modified" as const,
      },
    ];

    expect(
      resolveCanvasWorkbenchSelectedChangeItem({
        items,
        selectedChangeId: null,
        activeSelectionChangeItem: items[0],
        preferRuntimeEvidenceDefault: true,
      })?.id,
    ).toBe("runtime");
    expect(
      resolveCanvasWorkbenchSelectedChangeItem({
        items,
        selectedChangeId: "preview",
        activeSelectionChangeItem: items[1],
        preferRuntimeEvidenceDefault: true,
      })?.id,
    ).toBe("preview");
  });

  it("应推导变更项展示名、pending 状态和状态样式", () => {
    expect(
      resolveChangeItemDisplayName({
        id: "a",
        path: "/workspace/src/app.ts",
        displayName: "  ",
      }),
    ).toBe("app.ts");
    expect(
      isPendingChangeItem({
        id: "a",
        path: "src/app.ts",
        status: "in_progress",
      }),
    ).toBe(true);
    expect(
      resolveChangeStatusCopyKey({
        id: "a",
        path: "src/app.ts",
        status: "failed",
      }),
    ).toBe("agentChat.canvasWorkbench.coding.changes.status.failed");
    expect(
      resolveChangeStatusClassName({
        id: "a",
        path: "src/app.ts",
        status: "completed",
      }),
    ).toContain("emerald");
    expect(
      resolveChangeStatusClassName({ id: "a", path: "src/app.ts" }),
    ).toContain("slate");
  });

  it("应构建可筛选的文件树并统计 diff 增删", () => {
    const tree = buildCanvasWorkbenchChangeFileTree(
      [
        {
          id: "layout",
          path: "src/components/Layout.tsx",
          status: "completed",
          changeKind: "modified",
          previousContent: "old\nsame",
          currentContent: "new\nsame\nextra",
        },
        {
          id: "readme",
          path: "README.md",
          preview: "usage",
          status: "completed",
        },
      ],
      "layout",
    );

    expect(tree).toHaveLength(1);
    expect(tree[0]).toMatchObject({
      type: "folder",
      name: "src",
      depth: 0,
    });
    const srcNode = tree[0];
    expect(srcNode.type).toBe("folder");
    if (srcNode.type === "folder") {
      expect(srcNode.children[0]).toMatchObject({
        type: "folder",
        name: "components",
      });
      const componentsNode = srcNode.children[0];
      expect(componentsNode.type).toBe("folder");
      if (componentsNode.type === "folder") {
        expect(componentsNode.children[0]).toMatchObject({
          type: "file",
          name: "Layout.tsx",
          path: "src/components/Layout.tsx",
        });
      }
    }
    expect(
      countCanvasWorkbenchDiffStats([
        { type: "context", value: "a" },
        { type: "add", value: "b" },
        { type: "remove", value: "c" },
      ]),
    ).toEqual({ additions: 1, removals: 1 });
  });

  it("应归一化文件变更类型并给出展示元数据", () => {
    expect(
      inferCanvasWorkbenchChangeKind({
        id: "new",
        path: "src/new.ts",
        changeKind: "create",
      }),
    ).toBe("added");
    expect(
      inferCanvasWorkbenchChangeKind({
        id: "deleted",
        path: "src/old.ts",
        previousContent: "old",
        currentContent: null,
      }),
    ).toBe("deleted");
    expect(
      resolveChangeDisplayMeta({
        id: "rename",
        path: "src/new-name.ts",
        changeKind: "renamed",
      }),
    ).toMatchObject({
      kind: "renamed",
      shortLabelKey:
        "agentChat.canvasWorkbench.coding.changes.kindShort.renamed",
    });
  });

  it("应从变更项生成可用于 git apply 的 patch", () => {
    const patch = buildCanvasWorkbenchGitApplyPatch([
      {
        id: "modified",
        path: "src/app.ts",
        previousContent: "const value = 1;\n",
        currentContent: "const value = 2;\n",
      },
      {
        id: "added",
        path: "src/new.ts",
        previousContent: null,
        currentContent: "export const ok = true;\n",
      },
      {
        id: "deleted",
        path: "src/old.ts",
        previousContent: "legacy\n",
        currentContent: null,
      },
    ]);

    expect(patch).toContain("diff --git a/src/app.ts b/src/app.ts");
    expect(patch).toContain("--- a/src/app.ts");
    expect(patch).toContain("+++ b/src/app.ts");
    expect(patch).toContain("-const value = 1;");
    expect(patch).toContain("+const value = 2;");
    expect(patch).toContain("new file mode 100644");
    expect(patch).toContain("--- /dev/null");
    expect(patch).toContain("+++ b/src/new.ts");
    expect(patch).toContain("deleted file mode 100644");
    expect(patch).toContain("--- a/src/old.ts");
    expect(patch).toContain("+++ /dev/null");
  });

  it("应从变更项生成带行号的 diff 并汇总所有文件统计", () => {
    const items = [
      {
        id: "a",
        path: "src/app.ts",
        previousContent: "old\nsame\n",
        currentContent: "new\nsame\n",
      },
      {
        id: "b",
        path: "docs/new.md",
        previousContent: null,
        currentContent: "hello\n",
      },
    ];

    const diffLines = buildCanvasWorkbenchChangeDiffLines(items[0]);

    expect(diffLines[0]).toMatchObject({
      type: "remove",
      oldLineNumber: 1,
      newLineNumber: null,
    });
    expect(diffLines[1]).toMatchObject({
      type: "add",
      oldLineNumber: null,
      newLineNumber: 1,
    });
    expect(countCanvasWorkbenchChangeItemsStats(items)).toEqual({
      additions: 3,
      removals: 1,
    });
    expect(
      flattenCanvasWorkbenchChangeFileTree(
        buildCanvasWorkbenchChangeFileTree(items),
      ),
    ).toHaveLength(2);
  });

  it("应把 changes workbench 的文件数与增删行汇总到同一摘要", () => {
    expect(
      summarizeCanvasWorkbenchChanges([
        {
          id: "a",
          path: "src/a.ts",
          previousContent: "old\n",
          currentContent: "new\nextra\n",
        },
        {
          id: "b",
          path: "src/b.ts",
          previousContent: null,
          currentContent: "added\n",
        },
      ]),
    ).toEqual({
      fileCount: 2,
      diffStats: { additions: 4, removals: 1 },
    });
  });

  it("应为文件树目录聚合子级变更标识", () => {
    const tree = buildCanvasWorkbenchChangeFileTree([
      {
        id: "modified",
        path: "src/app.ts",
        changeKind: "modified",
      },
      {
        id: "added",
        path: "src/components/new.tsx",
        changeKind: "added",
      },
    ]);

    const srcNode = tree.find((node) => node.type === "folder");

    expect(srcNode).toBeTruthy();
    expect(resolveCanvasWorkbenchChangeTreeNodeMeta(srcNode!)?.kind).toBe(
      "added",
    );
    if (srcNode?.type === "folder") {
      const fileNode = srcNode.children.find(
        (node) => node.type === "file" && node.path === "src/app.ts",
      );
      expect(resolveCanvasWorkbenchChangeTreeNodeMeta(fileNode!)?.kind).toBe(
        "modified",
      );
    }
  });

  it("应把后端 Git patch 解析为可渲染的文件变更项", () => {
    const items = parseCanvasWorkbenchGitPatchToChangeItems(
      [
        "diff --git a/src/App.tsx b/src/App.tsx",
        "--- a/src/App.tsx",
        "+++ b/src/App.tsx",
        "@@ -1 +1 @@",
        "-old",
        "+new",
        "diff --git a/docs/new.md b/docs/new.md",
        "new file mode 100644",
        "--- /dev/null",
        "+++ b/docs/new.md",
        "@@ -0,0 +1 @@",
        "+hello",
      ].join("\n"),
    );

    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({
      path: "src/App.tsx",
      displayName: "App.tsx",
      source: "git",
      changeKind: "modified",
      hunkStartLine: 1,
      hunkEndLine: 1,
    });
    expect(items[0].diffLines).toEqual([
      { type: "remove", value: "old", oldLineNumber: 1, newLineNumber: null },
      { type: "add", value: "new", oldLineNumber: null, newLineNumber: 1 },
    ]);
    expect(items[1]).toMatchObject({
      path: "docs/new.md",
      displayName: "new.md",
      changeKind: "added",
      hunkStartLine: 1,
      hunkEndLine: 1,
    });
    expect(items[1].diffLines).toEqual([
      { type: "add", value: "hello", oldLineNumber: null, newLineNumber: 1 },
    ]);
  });
});
