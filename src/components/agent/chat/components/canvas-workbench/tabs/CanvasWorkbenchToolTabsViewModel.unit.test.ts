import { describe, expect, it } from "vitest";
import { buildCanvasWorkbenchToolTabProjection } from "./CanvasWorkbenchToolTabsViewModel";

const copy: Record<string, string> = {
  "agentChat.canvasWorkbench.coding.tabs.changes": "审查",
  "agentChat.canvasWorkbench.coding.tabs.logs": "日志",
  "agentChat.canvasWorkbench.coding.tabs.outputs": "输出",
  "agentChat.canvasWorkbench.newTabs.terminal": "终端",
  "agentChat.canvasWorkbench.newTabs.files": "文件",
  "agentChat.canvasWorkbench.newTabs.terminalTab": "终端",
  "agentChat.canvasWorkbench.newTabs.filesTab": "打开文件",
};

function t(key: string): string {
  return copy[key] ?? key;
}

describe("CanvasWorkbenchToolTabsViewModel", () => {
  it("应把审查作为固定首 tab，并按变更数量显示 badge", () => {
    const projection = buildCanvasWorkbenchToolTabProjection({
      changeItemCount: 120,
      documentDiffLineCount: 8,
      failedChangeItemCount: 0,
      openedToolTabs: [],
      translateWorkbench: t,
    });

    expect(projection.primaryTabs[0]).toMatchObject({
      key: "changes",
      label: "审查",
      badge: "99+",
      badgeTone: "sky",
    });
  });

  it("失败变更应让审查 badge 使用错误语义", () => {
    const projection = buildCanvasWorkbenchToolTabProjection({
      changeItemCount: 2,
      documentDiffLineCount: 0,
      failedChangeItemCount: 1,
      openedToolTabs: [],
      translateWorkbench: t,
    });

    expect(projection.primaryTabs[0]?.badgeTone).toBe("rose");
  });

  it("应把已打开的工具 tab 投影为可关闭 tab", () => {
    const projection = buildCanvasWorkbenchToolTabProjection({
      changeItemCount: 0,
      documentDiffLineCount: 0,
      failedChangeItemCount: 0,
      openedToolTabs: [
        { id: "terminal:1", kind: "terminal", sequence: 1 },
        { id: "project-files:1", kind: "project-files", sequence: 1 },
        { id: "terminal:2", kind: "terminal", sequence: 2 },
      ],
      translateWorkbench: t,
    });

    expect(projection.primaryTabs.slice(1)).toEqual([
      { key: "terminal:1", label: "终端", closable: true },
      { key: "project-files:1", label: "打开文件", closable: true },
      { key: "terminal:2", label: "终端 2", closable: true },
    ]);
  });

  it("coding 模式应在存在投影视图时显示输出和日志 tab", () => {
    const projection = buildCanvasWorkbenchToolTabProjection({
      changeItemCount: 1,
      documentDiffLineCount: 0,
      failedChangeItemCount: 0,
      utilityTabs: {
        outputs: true,
        logs: true,
      },
      openedToolTabs: [],
      translateWorkbench: t,
    });

    expect(projection.primaryTabs.map((tab) => tab.key)).toEqual([
      "changes",
      "outputs",
      "logs",
    ]);
    expect(projection.primaryTabs.slice(1)).toMatchObject([
      { key: "outputs", label: "输出" },
      { key: "logs", label: "日志" },
    ]);
  });

  it("应集中定义新增 tab 下拉动作", () => {
    const projection = buildCanvasWorkbenchToolTabProjection({
      changeItemCount: 0,
      documentDiffLineCount: 0,
      failedChangeItemCount: 0,
      openedToolTabs: [],
      translateWorkbench: t,
    });

    expect(projection.newTabActions).toEqual([
      { key: "terminal", label: "终端", shortcut: "^`" },
      { key: "project-files", label: "文件", shortcut: "⌘P" },
    ]);
  });
});
