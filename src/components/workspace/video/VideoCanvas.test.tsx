import React, { act, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoCanvas } from "./VideoCanvas";
import { createInitialVideoState, type VideoCanvasState } from "./types";
import {
  clickButtonByTitle,
  cleanupMountedRoots,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  emitCanvasImageInsertRequest,
  onCanvasImageInsertAck,
  type CanvasImageInsertAck,
} from "@/lib/canvasImageInsertBus";

const { mockGetProviders, mockToastSuccess } = vi.hoisted(() => ({
  mockGetProviders: vi.fn(),
  mockToastSuccess: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    success: mockToastSuccess,
  },
}));

vi.mock("@/hooks/useGlobalMediaGenerationDefaults", () => ({
  useGlobalMediaGenerationDefaults: () => ({
    mediaDefaults: {
      video: {
        preferredProviderId: "openai",
        preferredModelId: "sora-2-pro",
      },
    },
    loading: false,
  }),
}));

vi.mock("@/lib/api/apiKeyProvider", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/apiKeyProvider")>();

  return {
    ...actual,
    apiKeyProviderApi: {
      ...actual.apiKeyProviderApi,
      getProviders: (...args: unknown[]) => mockGetProviders(...args),
    },
  };
});

vi.mock("./VideoSidebar", () => ({
  VideoSidebar: ({
    state,
    availableModels,
  }: {
    state: VideoCanvasState;
    availableModels: string[];
  }) => (
    <div data-testid="video-sidebar-state">
      {state.providerId}/{state.model}/{availableModels.join(",")}
    </div>
  ),
}));

vi.mock("./VideoWorkspace", () => ({
  VideoWorkspace: ({ state }: { state: VideoCanvasState }) => (
    <div data-testid="video-workspace-state">
      {state.providerId}/{state.model}
    </div>
  ),
}));

const mountedRoots: MountedRoot[] = [];

function ControlledVideoCanvas({
  onObservedStateChange,
}: {
  onObservedStateChange?: (state: VideoCanvasState) => void;
}) {
  const [state, setState] = useState<VideoCanvasState>(() =>
    createInitialVideoState("测试视频任务"),
  );

  return (
    <VideoCanvas
      state={state}
      projectId="project-video-1"
      onStateChange={(nextState) => {
        onObservedStateChange?.(nextState);
        setState(nextState);
      }}
    />
  );
}

describe("VideoCanvas 全局默认模型", () => {
  beforeEach(async () => {
    setupReactActEnvironment();
    await changeLimeLocale("zh-CN");
    localStorage.clear();
    vi.clearAllMocks();
    mockGetProviders.mockResolvedValue([
      {
        id: "doubao",
        name: "豆包视频",
        enabled: true,
        api_key_count: 1,
        models: [{ id: "seedance-1-5-pro-251215" }],
      },
      {
        id: "openai",
        name: "OpenAI Video",
        enabled: true,
        api_key_count: 1,
        models: [{ id: "sora-2" }, { id: "sora-2-pro" }],
      },
    ]);
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
    localStorage.clear();
  });

  it("provider/model 为空时应优先采用 workspace_preferences.media_defaults.video", async () => {
    const observedStateChanges: VideoCanvasState[] = [];

    const mounted = mountHarness(
      ControlledVideoCanvas,
      {
        onObservedStateChange: (state: VideoCanvasState) => {
          observedStateChanges.push(state);
        },
      },
      mountedRoots,
    );

    await flushEffects(8);

    expect(mockGetProviders).toHaveBeenCalledTimes(1);
    expect(observedStateChanges[observedStateChanges.length - 1]).toMatchObject(
      {
        providerId: "openai",
        model: "sora-2-pro",
      },
    );
    expect(
      mounted.container.querySelector("[data-testid='video-sidebar-state']")
        ?.textContent,
    ).toContain("openai/sora-2-pro");
    expect(
      mounted.container.querySelector("[data-testid='video-workspace-state']")
        ?.textContent,
    ).toContain("openai/sora-2-pro");
  });

  it("英文界面应使用 workspace namespace 渲染侧栏折叠按钮 title", async () => {
    await changeLimeLocale("en-US");

    const mounted = mountHarness(
      ControlledVideoCanvas,
      {
        onObservedStateChange: vi.fn(),
      },
      mountedRoots,
    );

    await flushEffects(8);

    const collapseButton = clickButtonByTitle(
      mounted.container,
      "Collapse sidebar",
    );
    expect(collapseButton).toBeInstanceOf(HTMLButtonElement);
    await flushEffects();
    expect(
      mounted.container.querySelector('button[title="Expand sidebar"]'),
    ).toBeInstanceOf(HTMLButtonElement);
  });

  it("英文界面画布图片插入反馈应使用 workspace namespace", async () => {
    await changeLimeLocale("en-US");
    const observedStateChanges: VideoCanvasState[] = [];
    const observedAcks: CanvasImageInsertAck[] = [];
    const unsubscribeAck = onCanvasImageInsertAck((ack) => {
      observedAcks.push(ack);
    });

    mountHarness(
      ControlledVideoCanvas,
      {
        onObservedStateChange: (state) => {
          observedStateChanges.push(state);
        },
      },
      mountedRoots,
    );
    await flushEffects(8);

    const request = await act(async () => {
      const emittedRequest = emitCanvasImageInsertRequest({
        projectId: "project-video-1",
        canvasType: "video",
        source: "gallery",
        image: {
          id: "image-start-frame",
          previewUrl: "asset://preview.png",
          contentUrl: "asset://start-frame.png",
        },
      });
      await Promise.resolve();
      return emittedRequest;
    });
    await flushEffects(4);
    unsubscribeAck();

    expect(mockToastSuccess).toHaveBeenCalledWith(
      "Set as the video start reference image",
    );
    expect(observedStateChanges[observedStateChanges.length - 1]).toMatchObject(
      {
        startImage: "asset://start-frame.png",
      },
    );
    expect(
      observedAcks.find((ack) => ack.requestId === request.requestId),
    ).toMatchObject({
      success: true,
      canvasType: "video",
      locationLabel: "Start-frame reference image",
    });
  });
});
