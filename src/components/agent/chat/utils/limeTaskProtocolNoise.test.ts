import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { changeLimeLocale } from "@/i18n/createI18n";

import {
  isLimeTaskProtocolFailure,
  resolveLimeTaskProtocolFailureDisplayText,
} from "./limeTaskProtocolNoise";

describe("limeTaskProtocolNoise", () => {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
  });
  afterEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("应把内容工作台任务协议错误映射为具体失败文案", () => {
    expect(
      isLimeTaskProtocolFailure({
        toolName: "video_generate",
        text: "-32603: -32002: video_generate",
      }),
    ).toBe(true);
    expect(
      resolveLimeTaskProtocolFailureDisplayText({
        toolName: "video_generate",
        text: "-32603: -32002: video_generate",
      }),
    ).toBe("视频生成失败");

    expect(
      resolveLimeTaskProtocolFailureDisplayText({
        toolName: "lime_create_audio_generation_task",
        text: "tool failed",
      }),
    ).toBe("配音生成失败");
  });

  it("应支持 direct 内容生成工具，并避免误判普通协议错误", () => {
    expect(
      resolveLimeTaskProtocolFailureDisplayText({
        toolName: "social_generate_cover_image",
        text: "-32603: -32002: social_generate_cover_image",
      }),
    ).toBe("封面图生成失败");

    expect(
      isLimeTaskProtocolFailure({
        toolName: "Bash",
        text: "-32603: -32002: command failed",
      }),
    ).toBe(false);
  });

  it("失败文案应跟随当前 locale", async () => {
    await changeLimeLocale("en-US");

    expect(
      resolveLimeTaskProtocolFailureDisplayText({
        toolName: "video_generate",
        text: "-32603: -32002: video_generate",
      }),
    ).toBe("Video generation failed");
  });
});
