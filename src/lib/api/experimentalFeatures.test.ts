import { describe, expect, it, vi } from "vitest";
import {
  METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
  METHOD_EXPERIMENTAL_FEATURE_LIST,
} from "@limecloud/app-server-client";
import {
  getExperimentalConfig,
  saveExperimentalConfig,
} from "./experimentalFeatures";

function appServerClient(result: unknown) {
  return {
    request: vi.fn().mockResolvedValue({ result }),
  };
}

describe("experimentalFeatures API", () => {
  it("通过 App Server catalog 读取 WebMCP 状态", async () => {
    const client = appServerClient({
      data: [
        {
          name: "webmcp",
          stage: "underDevelopment",
          enabled: false,
          defaultEnabled: false,
        },
      ],
      nextCursor: null,
    });

    await expect(getExperimentalConfig(client)).resolves.toEqual({
      webmcp: { enabled: false },
    });
    expect(client.request).toHaveBeenCalledWith(
      METHOD_EXPERIMENTAL_FEATURE_LIST,
      {},
    );
  });

  it("通过 App Server enablement 更新 WebMCP 状态", async () => {
    const client = appServerClient({
      enablement: { webmcp: true },
    });

    await expect(
      saveExperimentalConfig({ webmcp: { enabled: true } }, client),
    ).resolves.toBeUndefined();
    expect(client.request).toHaveBeenCalledWith(
      METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
      { enablement: { webmcp: true } },
    );
  });

  it("缺少 WebMCP catalog entry 时 fail closed", async () => {
    const client = appServerClient({ data: [] });
    await expect(getExperimentalConfig(client)).rejects.toThrow(
      "did not return the webmcp feature",
    );
  });

  it("enablement 响应未确认目标值时 fail closed", async () => {
    const client = appServerClient({ enablement: { webmcp: false } });
    await expect(
      saveExperimentalConfig({ webmcp: { enabled: true } }, client),
    ).rejects.toThrow("did not apply webmcp");
  });
});
