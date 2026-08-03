import { describe, expect, it } from "vitest";

import {
  isToolResultSuccessful,
  normalizeHistoryImagePart,
  normalizeIncomingToolResult,
  normalizeToolResultImages,
} from "./agentChatToolResult";

describe("agentChatToolResult", () => {
  it("reported_success 为 true 时应覆盖非零 exit_code", () => {
    expect(
      isToolResultSuccessful({
        success: true,
        metadata: {
          exit_code: 1,
          reported_success: true,
        },
      }),
    ).toBe(true);
  });

  it("reported_success 为 false 时应优先判定失败", () => {
    expect(
      isToolResultSuccessful({
        success: true,
        metadata: {
          exit_code: 0,
          reported_success: false,
        },
      }),
    ).toBe(false);
  });

  it("应从 model-visible metadata 中归一化图片结果", () => {
    expect(
      normalizeToolResultImages(undefined, "", {
        model_visible_image: true,
        image_url: "data:image/png;base64,aW1hZ2U=",
        mime_type: "image/png",
      }),
    ).toEqual([
      {
        src: "data:image/png;base64,aW1hZ2U=",
        mimeType: "image/png",
        origin: "tool_payload",
      },
    ]);
  });

  it("归一化 incoming tool result 时应保留 metadata 图片预览", () => {
    const result = normalizeIncomingToolResult({
      success: true,
      output: "Viewed image: sample.png",
      metadata: {
        model_visible_image: true,
        image_url: "data:image/png;base64,c2FtcGxl",
        mime_type: "image/png",
      },
    });

    expect(result?.images).toEqual([
      {
        src: "data:image/png;base64,c2FtcGxl",
        mimeType: "image/png",
        origin: "tool_payload",
      },
    ]);
    expect(result?.output).not.toContain("base64");
  });

  it("应保留历史 data URL 图片附件的来源 URI", () => {
    expect(
      normalizeHistoryImagePart({
        type: "input_image",
        uri: "data:image/png;base64,aW1hZ2U=",
        metadata: {
          mediaType: "image/png",
          index: 2,
        },
      }),
    ).toEqual({
      data: "aW1hZ2U=",
      mediaType: "image/png",
      sourceUri: "data:image/png;base64,aW1hZ2U=",
      previewUrl: "data:image/png;base64,aW1hZ2U=",
      metadata: {
        mediaType: "image/png",
        index: 2,
      },
      index: 2,
    });
  });

  it("应保留历史本地图片附件的 sourcePath", () => {
    expect(
      normalizeHistoryImagePart({
        type: "image",
        uri: "/tmp/imported-local.png",
        metadata: {
          localPath: "/tmp/imported-local.png",
          index: 1,
        },
      }),
    ).toEqual({
      data: "",
      mediaType: "image/png",
      sourceUri: "/tmp/imported-local.png",
      sourcePath: "/tmp/imported-local.png",
      previewUrl: undefined,
      metadata: {
        localPath: "/tmp/imported-local.png",
        index: 1,
      },
      index: 1,
    });
  });

  it("data 字段为完整 data URL 时不应重复拼接协议头", () => {
    expect(
      normalizeHistoryImagePart({
        type: "image",
        mime_type: "image/jpeg",
        data: "data:image/jpeg;base64,aW1hZ2U=",
      }),
    ).toEqual({
      data: "aW1hZ2U=",
      mediaType: "image/jpeg",
      sourceUri: "data:image/jpeg;base64,aW1hZ2U=",
      previewUrl: "data:image/jpeg;base64,aW1hZ2U=",
      metadata: undefined,
      index: undefined,
    });
  });
});
