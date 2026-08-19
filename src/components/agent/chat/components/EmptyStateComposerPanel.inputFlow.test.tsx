import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  mockEmptyStateCharacterMention,
  renderPanel,
  updateTextareaValue,
} from "./EmptyStateComposerPanel.testFixtures";

describe("EmptyStateComposerPanel", () => {
  it("不应重新接入 input restore request 子面板 fallback", () => {
    const source = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      ),
      "utf8",
    );

    expect(source).not.toContain("inputRestoreRequest");
    expect(source).not.toContain("InterruptedInputRestoreRequest");
  });

  it("输入为空时不再展示 Tab 起手建议，也不拦截 Tab 输入", async () => {
    const container = renderPanel();

    expect(
      container.querySelector('[data-testid="home-input-tab-suggestion"]'),
    ).toBeNull();

    const textarea = container.querySelector("textarea");
    await act(async () => {
      textarea?.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Tab",
          bubbles: true,
          cancelable: true,
        }),
      );
      await Promise.resolve();
    });

    expect(
      (container.querySelector("textarea") as HTMLTextAreaElement).value,
    ).toBe("");
  }, 15_000);

  it("Shift+Tab 保持焦点切换，不填入起手建议", () => {
    const container = renderPanel();

    const textarea = container.querySelector("textarea");
    act(() => {
      textarea?.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Tab",
          shiftKey: true,
          bubbles: true,
          cancelable: true,
        }),
      );
    });

    expect(
      (container.querySelector("textarea") as HTMLTextAreaElement).value,
    ).toBe("");
  });

  it("引导帮助模式应展示可关闭的上下文 badge 并隐藏 Tab 起手建议", () => {
    const onClearGuideHelp = vi.fn();
    const container = renderPanel({
      guideHelpActive: true,
      guideHelpLabel: "Lime 引导帮助",
      onClearGuideHelp,
    });

    expect(
      container.querySelector('[data-testid="home-guide-help-active-badge"]')
        ?.textContent,
    ).toContain("Lime 引导帮助");
    expect(
      container.querySelector('[data-testid="home-guide-help-toolbar-badge"]')
        ?.textContent,
    ).toContain("引导帮助");
    expect(
      container.querySelector('[data-testid="home-input-tab-suggestion"]'),
    ).toBeNull();

    const closeButton = container.querySelector(
      '[data-testid="home-guide-help-active-badge"] button',
    ) as HTMLButtonElement | null;
    act(() => {
      closeButton?.click();
    });

    expect(onClearGuideHelp).toHaveBeenCalledTimes(1);

    const toolbarCloseButton = container.querySelector(
      '[data-testid="home-guide-help-toolbar-badge"]',
    ) as HTMLButtonElement | null;
    act(() => {
      toolbarCloseButton?.click();
    });

    expect(onClearGuideHelp).toHaveBeenCalledTimes(2);
  });

  it("应将 onPaste 绑定到输入框", () => {
    const onPaste = vi.fn();
    const container = renderPanel({ onPaste });
    const textarea = container.querySelector("textarea");

    expect(textarea).toBeTruthy();

    act(() => {
      textarea?.dispatchEvent(new Event("paste", { bubbles: true }));
    });

    expect(onPaste).toHaveBeenCalledTimes(1);
  });

  it("发送时应把本地草稿显式传给首页发送链", () => {
    const onSend = vi.fn();
    const container = renderPanel({ onSend });
    const textarea = container.querySelector("textarea");

    updateTextareaValue(textarea, "帮我快速开一个新对话");

    const sendButton = container.querySelector(
      'button[title="发送"]',
    ) as HTMLButtonElement | null;

    expect(sendButton?.disabled).toBe(false);

    act(() => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSend).toHaveBeenCalledWith(
      "帮我快速开一个新对话",
      {
        goalEnabled: false,
        planEnabled: false,
        subagentEnabled: false,
      },
      expect.objectContaining({
        triggerSource: "button",
        triggeredAt: expect.any(Number),
      }),
    );
    expect(
      (container.querySelector("textarea") as HTMLTextAreaElement).value,
    ).toBe("");
  });

  it("插件 chip 激活但输入未含触发词时发送应补齐插件触发词", async () => {
    const onSend = vi.fn();
    const plugin = {
      pluginId: "content-factory-app",
      displayName: "写文章",
      trigger: "@写文章",
      description: "生成文章草稿",
    };
    const container = renderPanel({
      input: "帮我整理项目资料",
      onSend,
      pluginSuggestions: [plugin],
    });
    const mentionProps = mockEmptyStateCharacterMention.mock.calls.at(-1)?.[0];

    await act(async () => {
      mentionProps?.onSelectPlugin?.(plugin, undefined, {
        inputOverride: "帮我整理项目资料",
        preserveInputOverride: true,
      });
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="inputbar-plugin-badge"]')
        ?.textContent,
    ).toContain("写文章");
    expect(
      (container.querySelector("textarea") as HTMLTextAreaElement).value,
    ).toBe("帮我整理项目资料");

    const sendButton = container.querySelector(
      'button[title="发送"]',
    ) as HTMLButtonElement | null;

    act(() => {
      sendButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSend).toHaveBeenCalledWith(
      "@写文章 帮我整理项目资料",
      {
        goalEnabled: false,
        planEnabled: false,
        subagentEnabled: false,
      },
      expect.objectContaining({
        triggerSource: "button",
        triggeredAt: expect.any(Number),
      }),
    );
  });

  it("发送准备中应禁用首页发送入口并展示忙碌态", () => {
    const onSend = vi.fn();
    const container = renderPanel({
      input: "请帮我梳理首页首次发送链路",
      isLoading: true,
      disabled: true,
      onSend,
    });

    const textarea = container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement | null;
    const runningButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("正在输出"),
    ) as HTMLButtonElement | undefined;

    expect(textarea?.disabled).toBe(true);
    expect(runningButton).toBeTruthy();
    expect(runningButton?.disabled).toBe(true);
    expect(container.querySelector('button[title="发送"]')).toBeNull();

    act(() => {
      runningButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSend).not.toHaveBeenCalled();
  });

  it("运行中应把空态停止按钮接到 onStop", () => {
    const onStop = vi.fn();
    const container = renderPanel({
      input: "请帮我梳理首页首次发送链路",
      isLoading: true,
      disabled: true,
      onStop,
    });

    const stopButton = container.querySelector(
      'button[title="停止"]',
    ) as HTMLButtonElement | null;

    expect(stopButton).toBeTruthy();
    expect(stopButton?.disabled).toBe(false);

    act(() => {
      stopButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("有待发送图片时应显示预览并支持删除", () => {
    const onRemoveImage = vi.fn();
    const container = renderPanel({
      pendingImages: [
        {
          data: "aGVsbG8=",
          mediaType: "image/png",
        },
      ],
      onRemoveImage,
    });

    expect(container.querySelector('img[alt="预览 1"]')).toBeTruthy();

    const removeButton = container.querySelector(
      'button[aria-label="移除图片 1"]',
    ) as HTMLButtonElement | null;

    expect(removeButton).toBeTruthy();

    act(() => {
      removeButton?.click();
    });

    expect(onRemoveImage).toHaveBeenCalledWith(0);
  });

});
