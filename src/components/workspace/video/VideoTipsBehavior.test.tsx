import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PromptInput } from "./PromptInput";
import { VideoSidebar } from "./VideoSidebar";
import { VideoWorkspace } from "./VideoWorkspace";
import { createInitialVideoState } from "./types";
import {
  cleanupMountedRoots,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import { changeLimeLocale } from "@/i18n/createI18n";

const mountedRoots: MountedRoot[] = [];

function getBodyText(): string {
  return document.body.textContent ?? "";
}

function queryTipButton(ariaLabel: string): HTMLButtonElement | null {
  return (
    Array.from(
      document.body.querySelectorAll<HTMLButtonElement>("button[aria-label]"),
    ).find((button) => button.getAttribute("aria-label") === ariaLabel) ?? null
  );
}

function expectTipContentOnHover(ariaLabel: string, content: string): void {
  const button = queryTipButton(ariaLabel);

  expect(button).not.toBeNull();
  expect(getBodyText()).not.toContain(content);

  act(() => {
    button?.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
  });

  expect(getBodyText()).toContain(content);

  act(() => {
    button?.dispatchEvent(new MouseEvent("mouseout", { bubbles: true }));
  });

  expect(getBodyText()).not.toContain(content);
}

describe("视频工作台 tips 收口", () => {
  beforeEach(async () => {
    setupReactActEnvironment();
    await changeLimeLocale("zh-CN");
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
  });

  it("提示词输入区只在 tip 交互时展示解释文案", async () => {
    mountHarness(
      PromptInput,
      {
        state: createInitialVideoState(),
        onStateChange: vi.fn(),
        onGenerate: vi.fn(),
      },
      mountedRoots,
    );
    await flushEffects();

    expect(getBodyText()).not.toContain(
      "先写主体、场景和运动方式，再补充光线、氛围或镜头语言",
    );
    expect(getBodyText()).not.toContain("按 Enter 直接生成");

    expectTipContentOnHover(
      "提示词说明",
      "先写主体、场景和运动方式，再补充光线、氛围或镜头语言",
    );
    expect(
      document.body.querySelector("button[aria-label='快捷键说明']"),
    ).toBeNull();
    expect(getBodyText()).not.toContain("按 Enter 直接生成");
  });

  it("提示词输入区英文界面应使用 workspace namespace 文案", async () => {
    await changeLimeLocale("en-US");
    mountHarness(
      PromptInput,
      {
        state: {
          ...createInitialVideoState(),
          duration: 8,
          endImage: "asset://end.png",
          model: "veo-3.1",
          startImage: "asset://start.png",
        },
        onStateChange: vi.fn(),
        onGenerate: vi.fn(),
      },
      mountedRoots,
    );
    await flushEffects();

    expect(getBodyText()).toContain("VIDEO STUDIO");
    expect(getBodyText()).toContain(
      "Describe the scene, camera, and pacing you want to generate",
    );
    expect(getBodyText()).toContain("Model veo-3.1");
    expect(getBodyText()).toContain("8s");
    expect(getBodyText()).toContain("2 reference image(s)");
    expect(
      document.body.querySelector("textarea")?.getAttribute("placeholder"),
    ).toContain("At dusk by the sea");
    expect(getBodyText()).toContain("Generate video");

    expectTipContentOnHover(
      "Prompt guidance",
      "Start with the subject, scene, and motion",
    );
    expect(
      document.body.querySelector("button[aria-label='Shortcut guidance']"),
    ).toBeNull();
    expect(getBodyText()).not.toContain(
      "Start with the subject, scene, and motion",
    );
    expect(getBodyText()).not.toContain(
      "Press Enter to generate, Shift + Enter for a new line.",
    );
  });

  it("左侧参数栏只在 tip 交互时展示解释文案", async () => {
    mountHarness(
      VideoSidebar,
      {
        state: createInitialVideoState(),
        providers: [],
        availableModels: [],
        onStateChange: vi.fn(),
      },
      mountedRoots,
    );
    await flushEffects();

    expect(getBodyText()).not.toContain(
      "先确定模型，再补参考图和输出规格。这里保持轻量控制，主创作仍留在右侧画布。",
    );
    expect(getBodyText()).not.toContain(
      "需要复现某次结果时再固定种子；探索阶段保持随机即可。",
    );
    expect(getBodyText()).not.toContain(
      "提示词优先写清主体、场景、镜头运动和光线。",
    );

    expectTipContentOnHover(
      "生成参数说明",
      "先确定模型，再补参考图和输出规格。这里保持轻量控制，主创作仍留在右侧画布。",
    );
    expect(
      document.body.querySelector("button[aria-label='提示词建议']"),
    ).toBeNull();
  });

  it("左侧参数栏英文界面应使用 workspace namespace 概览与创作 Tips 文案", async () => {
    await changeLimeLocale("en-US");
    mountHarness(
      VideoSidebar,
      {
        state: {
          ...createInitialVideoState(),
          aspectRatio: "16:9",
          duration: 12,
          resolution: "1080p",
          startImage: "data:image/png;base64,start",
        },
        providers: [],
        availableModels: [],
        onStateChange: vi.fn(),
      },
      mountedRoots,
    );
    await flushEffects();

    expect(getBodyText()).toContain("VIDEO CONTROL");
    expect(getBodyText()).toContain("Generation parameters");
    expect(getBodyText()).toContain("Current model");
    expect(getBodyText()).toContain("Configure a video model");
    expect(getBodyText()).toContain("No video models available");
    expect(getBodyText()).toContain("Configure a video-capable provider first");
    expect(getBodyText()).toContain("Output spec");
    expect(getBodyText()).toContain("16:9 · 1080p");
    expect(getBodyText()).toContain("Duration");
    expect(getBodyText()).toContain("12 sec");
    expect(getBodyText()).toContain("Reference images");
    expect(getBodyText()).toContain("Ready references: 1");
    expect(getBodyText()).toContain("Start frame");
    expect(
      document.body.querySelector("img[alt='Start frame preview']"),
    ).not.toBeNull();
    expect(getBodyText()).toContain("Change");
    expect(getBodyText()).toContain(
      "Drag and drop to replace the current image",
    );
    expect(getBodyText()).toContain("End frame");
    expect(getBodyText()).toContain("Add image");
    expect(getBodyText()).toContain(
      "Drag or click to upload a reference image",
    );
    expect(getBodyText()).toContain("Choose image");
    expect(getBodyText()).toContain("Aspect ratio");
    expect(getBodyText()).toContain("Adaptive");
    expect(getBodyText()).toContain("Resolution");
    expect(getBodyText()).toContain("Seed");
    expect(
      document.body.querySelector("input[placeholder='Random']"),
    ).not.toBeNull();
    expect(
      document.body.querySelector("button[title='Random seed']"),
    ).not.toBeNull();
    expect(getBodyText()).toContain("Generate audio");
    expect(getBodyText()).toContain("Fixed camera");
    expect(getBodyText()).not.toContain("Creation tips");
    expect(getBodyText()).not.toContain("Parameter pacing");
    const expectedTips = [
      [
        "Generation parameter guidance",
        "Choose the model first, then add reference images and output specs.",
      ],
      [
        "Opening frame guidance",
        "Use it to lock the opening composition",
      ],
      [
        "Ending frame guidance",
        "Use it to constrain the final shot",
      ],
      ["Duration guidance", "Start with 4 to 8 seconds"],
      ["Seed guidance", "Lock the seed only"],
      [
        "Generate audio guidance",
        "Turn this on only when you need ambience",
      ],
      ["Fixed camera guidance", "Reduce camera movement"],
    ] as const;

    expectedTips.forEach(([ariaLabel, content]) => {
      expectTipContentOnHover(ariaLabel, content);
    });

    expect(
      document.body.querySelector("button[aria-label='Prompt guidance']"),
    ).toBeNull();
    expect(
      document.body.querySelector(
        "button[aria-label='Parameter pacing guidance']",
      ),
    ).toBeNull();
    expect(getBodyText()).not.toContain("Keep this panel lightweight");
    expect(getBodyText()).not.toContain(
      "Use it to lock the opening composition",
    );
    expect(getBodyText()).not.toContain("Use it to constrain the final shot");
    expect(getBodyText()).not.toContain("Start with 4 to 8 seconds");
    expect(getBodyText()).not.toContain("Lock the seed only");
    expect(getBodyText()).not.toContain(
      "Turn this on only when you need ambience",
    );
    expect(getBodyText()).not.toContain("Reduce camera movement");
    expect(getBodyText()).not.toContain(
      "Prioritize the subject, scene, camera motion, and lighting",
    );
    expect(getBodyText()).not.toContain(
      "Lock the model and aspect ratio first",
    );
  });

  it("左侧参数栏英文界面应使用 workspace namespace 模型面板文案", async () => {
    await changeLimeLocale("en-US");
    mountHarness(
      VideoSidebar,
      {
        state: {
          ...createInitialVideoState(),
          model: "sora-2-pro",
          providerId: "openai",
        },
        providers: [
          {
            id: "openai",
            name: "OpenAI",
            models: ["sora-2-pro"],
          },
        ],
        availableModels: [],
        onStateChange: vi.fn(),
      },
      mountedRoots,
    );
    await flushEffects();

    expect(getBodyText()).toContain("Model");
    expect(getBodyText()).toContain("Sora-2-Pro");

    expectTipContentOnHover(
      "Model guidance",
      "Model capabilities determine the available resolutions",
    );

    const trigger = document.body.querySelector(
      "button[title='Choose video model']",
    );
    expect(trigger).toBeInstanceOf(HTMLButtonElement);

    await act(async () => {
      trigger?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(getBodyText()).toContain("MODEL LIBRARY");
    expect(getBodyText()).toContain("Choose a video model");
    expect(getBodyText()).toContain("Review model capabilities");
    expect(getBodyText()).toContain("20 credits / sec · est. 80 for 4s");
    expect(getBodyText()).toContain(
      "Sora-2 Pro takes about 2 minutes to generate",
    );
  });

  it("主工作台只在 tip 交互时展示解释文案", async () => {
    mountHarness(
      VideoWorkspace,
      {
        state: createInitialVideoState(),
        onStateChange: vi.fn(),
        projectId: null,
      },
      mountedRoots,
    );
    await flushEffects(6);

    expect(getBodyText()).not.toContain(
      "用一句清晰的场景描述启动视频生成，再逐步补充镜头运动、情绪和画面锚点。",
    );
    expect(getBodyText()).not.toContain("请先在左侧选择视频服务");

    expectTipContentOnHover(
      "视频创作说明",
      "用一句清晰的场景描述启动视频生成，再逐步补充镜头运动、情绪和画面锚点。",
    );
    expectTipContentOnHover("当前模型说明", "请先在左侧选择视频服务");
  });

  it("主工作台英文界面应使用 workspace namespace 首屏与摘要卡文案", async () => {
    await changeLimeLocale("en-US");
    mountHarness(
      VideoWorkspace,
      {
        state: createInitialVideoState(),
        onStateChange: vi.fn(),
        projectId: null,
      },
      mountedRoots,
    );
    await flushEffects(6);

    expect(getBodyText()).toContain("VIDEO WORKBENCH");
    expect(getBodyText()).toContain("Video creation");
    expect(getBodyText()).toContain("Current model");
    expect(getBodyText()).toContain("Select a model");
    expect(getBodyText()).toContain("Output spec");
    expect(getBodyText()).toContain("Reference images");
    expect(getBodyText()).toContain("Text-to-video only");
    expect(getBodyText()).toContain("Task sync");
    expect(getBodyText()).toContain("Select a project first");

    expectTipContentOnHover(
      "Video creation guidance",
      "Start video generation with one clear scene description",
    );
    expectTipContentOnHover(
      "Current model guidance",
      "Select a video provider from the left panel first.",
    );
    expectTipContentOnHover(
      "Output spec guidance",
      "Duration 5 sec",
    );
  });
});
