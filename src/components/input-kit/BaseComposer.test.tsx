import React, { useState } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BaseComposer, type BaseComposerSendMetadata } from "./BaseComposer";
import { ComposerController, LARGE_PASTE_CHAR_THRESHOLD } from "./ComposerController";

interface RenderResult {
  container: HTMLDivElement;
  root: Root;
  onSend: ReturnType<
    typeof vi.fn<(metadata?: BaseComposerSendMetadata) => void>
  >;
  onStop: ReturnType<typeof vi.fn<() => void>>;
}

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

interface HarnessProps {
  initialText?: string;
  isLoading?: boolean;
  disabled?: boolean;
  hasAdditionalContent?: boolean;
  deferSendOnEnter?: boolean;
  sendOnPointerDown?: boolean;
  controller?: ComposerController;
  onSend: (metadata?: BaseComposerSendMetadata) => void;
  onStop: () => void;
}

const Harness: React.FC<HarnessProps> = ({
  initialText = "",
  isLoading = false,
  disabled = false,
  hasAdditionalContent = false,
  deferSendOnEnter = false,
  sendOnPointerDown = false,
  controller,
  onSend,
  onStop,
}) => {
  const [text, setText] = useState(initialText);

  return (
    <BaseComposer
      text={text}
      setText={setText}
      onSend={onSend}
      onStop={onStop}
      isLoading={isLoading}
      disabled={disabled}
      hasAdditionalContent={hasAdditionalContent}
      deferSendOnEnter={deferSendOnEnter}
      sendOnPointerDown={sendOnPointerDown}
      controller={controller}
      placeholder="输入内容"
    >
      {({
        textareaRef,
        textareaProps,
        onPrimaryAction,
        onPrimaryActionStart,
        isPrimaryDisabled,
      }) => (
        <div>
          <textarea
            data-testid="composer-textarea"
            ref={textareaRef}
            {...textareaProps}
          />
          <button
            data-testid="composer-primary"
            onPointerDown={onPrimaryActionStart}
            onClick={onPrimaryAction}
            disabled={isPrimaryDisabled}
          >
            action
          </button>
        </div>
      )}
    </BaseComposer>
  );
};

const renderHarness = (props: Partial<HarnessProps> = {}): RenderResult => {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const onSend = vi.fn<(metadata?: BaseComposerSendMetadata) => void>();
  const onStop = vi.fn<() => void>();

  act(() => {
    root.render(
      <Harness
        initialText={props.initialText}
        isLoading={props.isLoading}
        disabled={props.disabled}
        hasAdditionalContent={props.hasAdditionalContent}
        deferSendOnEnter={props.deferSendOnEnter}
        sendOnPointerDown={props.sendOnPointerDown}
        controller={props.controller}
        onSend={onSend}
        onStop={onStop}
      />,
    );
  });

  mountedRoots.push({ root, container });
  return { container, root, onSend, onStop };
};

const getTextarea = (container: HTMLElement): HTMLTextAreaElement => {
  const textarea = container.querySelector(
    '[data-testid="composer-textarea"]',
  ) as HTMLTextAreaElement | null;
  if (!textarea) {
    throw new Error("未找到输入框");
  }
  return textarea;
};

const getPrimaryButton = (container: HTMLElement): HTMLButtonElement => {
  const button = container.querySelector(
    '[data-testid="composer-primary"]',
  ) as HTMLButtonElement | null;
  if (!button) {
    throw new Error("未找到主操作按钮");
  }
  return button;
};

describe("BaseComposer", () => {
  it("按 Enter 应发送消息", () => {
    const { container, onSend } = renderHarness({ initialText: "hello" });
    const textarea = getTextarea(container);

    act(() => {
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(
      expect.objectContaining({
        triggerSource: "enter",
        triggeredAt: expect.any(Number),
      }),
    );
  });

  it("启用延迟发送时按 Enter 应在下一帧发送消息", () => {
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_780_000_000_123);
    const rafCallbacks: Array<(timestamp: number) => void> = [];
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: (timestamp: number) => void) => {
        rafCallbacks.push(callback);
        return rafCallbacks.length;
      },
    );
    const { container, onSend } = renderHarness({
      initialText: "hello",
      deferSendOnEnter: true,
    });
    const textarea = getTextarea(container);

    act(() => {
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
    });

    expect(onSend).not.toHaveBeenCalled();
    act(() => {
      rafCallbacks.splice(0).forEach((callback) => callback(0));
    });
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith({
      triggeredAt: 1_780_000_000_123,
      triggerSource: "enter",
    });
    nowSpy.mockRestore();
  });

  it("点击主按钮应携带 button 触发时间", () => {
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_780_000_100_000);
    const { container, onSend } = renderHarness({ initialText: "hello" });
    const button = getPrimaryButton(container);

    act(() => {
      button.click();
    });

    expect(onSend).toHaveBeenCalledWith({
      triggeredAt: 1_780_000_100_000,
      triggerSource: "button",
    });
    nowSpy.mockRestore();
  });

  it("启用 pointerdown 发送时 click 不应重复触发", () => {
    const nowSpy = vi.spyOn(Date, "now");
    nowSpy.mockReturnValue(1_780_000_100_111);
    const { container, onSend } = renderHarness({
      initialText: "hello",
      sendOnPointerDown: true,
    });
    const button = getPrimaryButton(container);

    act(() => {
      button.dispatchEvent(new Event("pointerdown", { bubbles: true }));
      button.click();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith({
      triggeredAt: 1_780_000_100_111,
      triggerSource: "button",
    });
    nowSpy.mockRestore();
  });

  it("生成中按 Enter 不应触发发送", () => {
    const { container, onSend } = renderHarness({
      initialText: "hello",
      isLoading: true,
    });
    const textarea = getTextarea(container);

    act(() => {
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
    });

    expect(onSend).not.toHaveBeenCalled();
  });

  it("输入法合成阶段按 Enter 不应立即触发发送", () => {
    const { container, onSend } = renderHarness({ initialText: "你好" });
    const textarea = getTextarea(container);

    const imeEnterEvent = new KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
    });
    Object.defineProperty(imeEnterEvent, "isComposing", {
      value: true,
      configurable: true,
    });

    act(() => {
      textarea.dispatchEvent(imeEnterEvent);
    });

    expect(onSend).not.toHaveBeenCalled();
  });

  it("输入法用 Enter 确认合成后应自动发送", () => {
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: (timestamp: number) => void) => {
        callback(0);
        return 1;
      },
    );

    const { container, onSend } = renderHarness({ initialText: "你好" });
    const textarea = getTextarea(container);

    const imeEnterEvent = new KeyboardEvent("keydown", {
      key: "Enter",
      bubbles: true,
    });
    Object.defineProperty(imeEnterEvent, "isComposing", {
      value: true,
      configurable: true,
    });

    act(() => {
      textarea.dispatchEvent(imeEnterEvent);
    });

    expect(onSend).not.toHaveBeenCalled();

    act(() => {
      textarea.dispatchEvent(
        new CompositionEvent("compositionend", { bubbles: true }),
      );
    });

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(
      expect.objectContaining({
        triggerSource: "ime",
        triggeredAt: expect.any(Number),
      }),
    );
  });

  it("生成中点击主按钮应触发停止", () => {
    const { container, onSend, onStop } = renderHarness({
      initialText: "hello",
      isLoading: true,
    });
    const button = getPrimaryButton(container);

    act(() => {
      button.click();
    });

    expect(onStop).toHaveBeenCalledTimes(1);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("仅附件内容存在时应允许发送", () => {
    const { container, onSend } = renderHarness({
      initialText: "",
      hasAdditionalContent: true,
    });
    const button = getPrimaryButton(container);

    act(() => {
      button.click();
    });

    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("应为输入框提供稳定的 id 与 name，避免表单告警", () => {
    const { container } = renderHarness();
    const textarea = getTextarea(container);

    expect(textarea.getAttribute("id")).toBeTruthy();
    expect(textarea.getAttribute("name")).toBe("agent-chat-message");
  });

  it.each([
    ["Win32", "windows"],
    ["MacIntel", "macos"],
    ["Linux x86_64", "linux"],
  ])("%s DOM paste 应归一化 CRLF 并保留 shell sigil", (platform) => {
    Object.defineProperty(window.navigator, "platform", {
      configurable: true,
      value: platform,
    });
    const controller = new ComposerController();
    const { container } = renderHarness({ controller });
    const textarea = getTextarea(container);
    const pastedText = "%PATH%\r\n$VAR `echo hi` ^&|<>";
    const pasteEvent = new Event("paste", {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(pasteEvent, "clipboardData", {
      configurable: true,
      value: {
        getData: (format: string) => (format === "text" ? pastedText : ""),
      },
    });

    act(() => {
      textarea.dispatchEvent(pasteEvent);
    });

    expect(pasteEvent.defaultPrevented).toBe(true);
    expect(textarea.value).toBe("%PATH%\n$VAR `echo hi` ^&|<>");
    expect(controller.getDocument().text).toBe(textarea.value);
  });

  it("Windows DOM paste 的大文本使用占位符，提交 draft 时再展开", () => {
    Object.defineProperty(window.navigator, "platform", {
      configurable: true,
      value: "Win32",
    });
    const controller = new ComposerController();
    const { container } = renderHarness({ controller });
    const textarea = getTextarea(container);
    const pastedText = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 1);
    const pasteEvent = new Event("paste", {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(pasteEvent, "clipboardData", {
      configurable: true,
      value: { getData: () => pastedText },
    });

    act(() => {
      textarea.dispatchEvent(pasteEvent);
    });

    expect(textarea.value).toMatch(/^\[Pasted text 1:/);
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("accepted");
    if (receipt.kind === "accepted") {
      expect(receipt.draft.text).toBe(pastedText);
    }
  });

  it("Windows AltGr 字符不会被 Composer 快捷键拦截", () => {
    Object.defineProperty(window.navigator, "platform", {
      configurable: true,
      value: "Win32",
    });
    const { container, onSend } = renderHarness({ initialText: "" });
    const textarea = getTextarea(container);
    const altGrEvent = new KeyboardEvent("keydown", {
      altKey: true,
      bubbles: true,
      cancelable: true,
      ctrlKey: true,
      key: "@",
    });

    act(() => {
      textarea.dispatchEvent(altGrEvent);
    });

    expect(altGrEvent.defaultPrevented).toBe(false);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("上下键只在文本边界回放会话内历史", () => {
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: (timestamp: number) => void) => {
        callback(0);
        return 1;
      },
    );
    const controller = new ComposerController({ text: "第一条" });
    const firstReceipt = controller.submit("start");
    if (firstReceipt.kind === "accepted") {
      controller.commit(firstReceipt);
    }
    controller.setText("第二条");
    const secondReceipt = controller.submit("start");
    if (secondReceipt.kind === "accepted") {
      controller.commit(secondReceipt);
    }

    const { container } = renderHarness({ controller });
    const textarea = getTextarea(container);
    const up = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: "ArrowUp",
    });
    act(() => {
      textarea.dispatchEvent(up);
    });
    expect(up.defaultPrevented).toBe(true);
    expect(textarea.value).toBe("第二条");

    act(() => {
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", {
          bubbles: true,
          cancelable: true,
          key: "ArrowDown",
        }),
      );
    });
    expect(textarea.value).toBe("");
  });
});
