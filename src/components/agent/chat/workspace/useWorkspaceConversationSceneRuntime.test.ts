import React from "react";
import { describe, expect, it, vi } from "vitest";
import {
  createBaseParams,
  getRenderedSceneProps,
} from "./useWorkspaceConversationSceneRuntime.testFixtures";

describe("useWorkspaceConversationSceneRuntime", () => {
  it("通用 Claw 双栏场景应继续同步 stacked/split 布局状态", () => {
    const params = createBaseParams();
    const setCanvasWorkbenchLayoutMode = params.setCanvasWorkbenchLayoutMode;

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.canvasWorkbenchLayoutProps.onLayoutModeChange).toBe(
      setCanvasWorkbenchLayoutMode,
    );
  });

  it("应向画布壳透传关闭动作", () => {
    const handleCloseCanvasWorkbench = vi.fn();
    const params = createBaseParams({
      canvasScene: {
        ...createBaseParams().canvasScene,
        handleCloseCanvasWorkbench,
      },
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.canvasWorkbenchLayoutProps.onClose).toBe(
      handleCloseCanvasWorkbench,
    );
  });

  it("主题工作台场景不应再向外回写 stacked/split 布局状态", () => {
    const params = createBaseParams({
      activeTheme: "general",
      isThemeWorkbench: true,
      isSpecializedThemeMode: true,
      layoutMode: "canvas",
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(
      sceneProps.canvasWorkbenchLayoutProps.onLayoutModeChange,
    ).toBeUndefined();
  });

  it("生成场景应继续向页面层透传顶栏上下文变体", () => {
    const params = createBaseParams({
      navbarContextVariant: "task-center",
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.navbarContextVariant).toBe("task-center");
  });

  it("存在 Harness 入口时应透传顶栏按钮文案", () => {
    const params = createBaseParams({
      showHarnessToggle: true,
      harnessToggleLabel: "Harness",
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.harnessToggleLabel).toBe("Harness");
  });

  it("Task Center 隐藏旧顶栏动作时仍应保留 Harness 开关能力", () => {
    const handleToggleHarnessPanel = vi.fn();
    const params = createBaseParams({
      navbarContextVariant: "task-center",
      suppressNavbarUtilityActions: true,
      showHarnessToggle: false,
      navbarHarnessPanelVisible: false,
      handleToggleHarnessPanel,
      harnessToggleLabel: "Harness",
    });

    const sceneProps = getRenderedSceneProps(params);

    expect(sceneProps.navbarContextVariant).toBe("task-center");
    expect(sceneProps.showHarnessToggle).toBe(false);
    expect(sceneProps.onToggleHarnessPanel).toBe(handleToggleHarnessPanel);
    expect(sceneProps.harnessToggleLabel).toBe("Harness");
  });

  it("首页空态应继续透传 service skills 与选择回调", () => {
    const onSelectServiceSkill = vi.fn();
    const serviceSkills = [
      {
        id: "daily-trend-briefing",
        title: "每日趋势摘要",
      },
    ];
    const params = createBaseParams({
      serviceSkills,
      onSelectServiceSkill,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.landingSurface.emptyStateProps.serviceSkills).toBe(
      serviceSkills,
    );
    expect(sceneProps.landingSurface.emptyStateProps.onSelectServiceSkill).toBe(
      onSelectServiceSkill,
    );
  });

  it("应把做法执行摘要卡透传给 WorkspaceConversationScene", () => {
    const sceneAppExecutionSummaryCard = React.createElement(
      "div",
      { "data-testid": "sceneapp-summary-card-probe" },
      "sceneapp summary",
    );
    const params = createBaseParams({
      sceneAppExecutionSummaryCard,
    });

    const sceneProps = getRenderedSceneProps(params);

    expect(sceneProps.landingSurface.sceneAppExecutionSummaryCard).toBe(
      sceneAppExecutionSummaryCard,
    );
  });

  it("parent-owned thread 不应暴露手动上下文压缩入口", () => {
    const params = createBaseParams({
      sessionId: "session-child",
      threadRead: {
        thread_id: "thread-child",
        can_accept_direct_input: false,
      },
    });

    const sceneProps = getRenderedSceneProps(params);

    expect(sceneProps.showContextCompactionAction).toBe(false);
    expect(sceneProps.onCompactContext).toBeUndefined();
  });
});
