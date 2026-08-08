import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  createAlignedRuntimeToolInventory,
  createToolInventory,
  renderExpandedPanel as renderPanel,
} from "./HarnessStatusPanel.testFixtures";

describe("HarnessStatusPanel tools", () => {
  it("存在工具库存时应展示工具与权限区块及来源统计", () => {
    renderPanel({
      toolInventory: createToolInventory(),
    });

    expect(document.body.textContent).toContain("工具与权限");
    expect(document.body.textContent).toContain("工具库存");
    expect(document.body.textContent).toContain("运行时覆盖");
    expect(document.body.textContent).toContain("持久化覆盖");
    expect(document.body.textContent).toContain("默认策略");
    expect(document.body.textContent).toContain("实际 Runtime 工具面");
    expect(document.body.textContent).toContain("Catalog 工具");
  });

  it("runtime tool surface 应在工具库存中展示 subagents/plan current gaps", () => {
    renderPanel({
      toolInventory: createToolInventory(),
    });

    const summary = document.body.querySelector(
      '[data-testid="harness-runtime-tool-capability-summary"]',
    ) as HTMLElement | null;

    expect(summary?.textContent).toContain("Runtime 能力摘要");
    expect(summary?.textContent).toContain("来源 runtime_tools");
    expect(summary?.textContent).toContain("WebSearch 未接通");
    expect(summary?.textContent).toContain("子任务核心 tools 缺 1 项");
    expect(summary?.textContent).toContain("Subagents 协作 tools 缺 3 项");
    expect(summary?.textContent).toContain("Plan current tool 缺 1 项");
    expect(summary?.textContent).toContain("SendMessage");
    expect(summary?.textContent).toContain(
      "TeamCreate / TeamDelete / ListPeers",
    );
    expect(summary?.textContent).toContain("update_plan");
  });

  it("runtime tool surface 应在工具库存中展示已接通的 subagents/plan current surface", () => {
    renderPanel({
      toolInventory: createAlignedRuntimeToolInventory(),
    });

    const summary = document.body.querySelector(
      '[data-testid="harness-runtime-tool-capability-summary"]',
    ) as HTMLElement | null;

    expect(summary?.textContent).toContain("来源 runtime_tools");
    expect(summary?.textContent).toContain("WebSearch 已接通");
    expect(summary?.textContent).toContain("子任务核心 tools 已接通");
    expect(summary?.textContent).toContain("Subagents 协作 tools 已接通");
    expect(summary?.textContent).toContain("Plan current tool 已接通");
    expect(summary?.textContent).toContain(
      "当前 runtime current surface 已覆盖 WebSearch、子任务、Subagents 协作与 Plan 主链。",
    );
  });

  it("工具库存应支持按来源筛选 catalog 条目", () => {
    renderPanel({
      toolInventory: createToolInventory(),
    });

    const runtimeFilterButton = document.body.querySelector(
      'button[aria-label="工具库存筛选：运行时覆盖"]',
    ) as HTMLButtonElement | null;

    act(() => {
      runtimeFilterButton?.click();
    });

    const inventorySection = document.body.querySelector(
      '[data-harness-section="inventory"]',
    ) as HTMLElement | null;

    expect(inventorySection?.textContent).toContain("Catalog 工具");
    expect(inventorySection?.textContent).toContain("1 / 3");
    expect(inventorySection?.textContent).toContain("bash");
    expect(inventorySection?.textContent).not.toContain("write");
  });

  it("工具库存加载失败时应展示错误并支持手动刷新", () => {
    const onRefreshToolInventory = vi.fn();

    renderPanel({
      toolInventoryLoading: true,
      toolInventoryError: "读取失败",
      onRefreshToolInventory,
    });

    expect(document.body.textContent).toContain(
      "正在读取当前工具库存与权限策略",
    );
    expect(document.body.textContent).toContain("读取失败");

    const refreshButton = document.body.querySelector(
      'button[aria-label="刷新工具库存"]',
    ) as HTMLButtonElement | null;

    act(() => {
      refreshButton?.click();
    });

    expect(onRefreshToolInventory).toHaveBeenCalledTimes(1);
  });
});
