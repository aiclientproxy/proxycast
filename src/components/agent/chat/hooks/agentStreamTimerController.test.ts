import { describe, expect, it } from "vitest";
import {
  AGENT_STREAM_TEXT_DELTA_BACKLOG_FLUSH_CHARS,
  AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS,
  buildAgentStreamTextRenderTimerSchedulePlan,
  buildAgentStreamTimerClearPlan,
} from "./agentStreamTimerController";

describe("agentStreamTimerController", () => {
  it("应根据 timer 是否存在构造 clear 计划", () => {
    expect(buildAgentStreamTimerClearPlan({ hasTimer: false })).toEqual({
      shouldClearTimer: false,
      nextTimerId: null,
    });
    expect(buildAgentStreamTimerClearPlan({ hasTimer: true })).toEqual({
      shouldClearTimer: true,
      nextTimerId: null,
    });
  });

  it("首个可见文本应立即 flush，不等待低频 timer", () => {
    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: "你好",
        renderedContent: "",
        hasPendingTimer: false,
      }),
    ).toEqual({
      action: "flush_now",
      delayMs: null,
    });
  });

  it("已有 text render timer 时不应重复调度", () => {
    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: "你好，继续",
        renderedContent: "你好",
        hasPendingTimer: true,
      }),
    ).toEqual({
      action: "skip",
      delayMs: null,
    });
  });

  it("非首个可见文本且无 pending timer 时应调度低频 flush", () => {
    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: "你好，继续",
        renderedContent: "你好",
        hasPendingTimer: false,
      }),
    ).toEqual({
      action: "schedule_timer",
      delayMs: AGENT_STREAM_TEXT_DELTA_RENDER_FLUSH_MS,
    });
  });

  it("后续文本到达换行边界或积压阈值时应立即 flush", () => {
    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: "你好\n",
        renderedContent: "你好",
        hasPendingTimer: true,
      }),
    ).toEqual({
      action: "flush_now",
      delayMs: null,
    });

    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: `你好${"啊".repeat(
          AGENT_STREAM_TEXT_DELTA_BACKLOG_FLUSH_CHARS,
        )}`,
        renderedContent: "你好",
        hasPendingTimer: false,
      }),
    ).toEqual({
      action: "flush_now",
      delayMs: null,
    });
  });

  it("内容没有新增时不应调度 text render timer", () => {
    expect(
      buildAgentStreamTextRenderTimerSchedulePlan({
        accumulatedContent: "你好",
        renderedContent: "你好",
        hasPendingTimer: false,
      }),
    ).toEqual({
      action: "skip",
      delayMs: null,
    });
  });
});
