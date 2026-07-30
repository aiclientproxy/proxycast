import { describe, expect, it, vi } from "vitest";

import { captureEvidenceScreenshot } from "./claw-chat-current-fixture-screenshot.mjs";

describe("claw chat current fixture evidence screenshot", () => {
  it("captures the full page when the primary attempt succeeds", async () => {
    const screenshot = vi.fn().mockResolvedValue(undefined);

    await expect(
      captureEvidenceScreenshot({
        page: { screenshot },
        path: "/tmp/gate-b.png",
      }),
    ).resolves.toEqual({
      path: "/tmp/gate-b.png",
      mode: "full-page",
      fallbackUsed: false,
      fullPageError: null,
    });
    expect(screenshot).toHaveBeenCalledOnce();
    expect(screenshot).toHaveBeenCalledWith({
      path: "/tmp/gate-b.png",
      fullPage: true,
      timeout: 15_000,
    });
  });

  it("falls back to a viewport capture when full-page capture times out", async () => {
    const screenshot = vi
      .fn()
      .mockRejectedValueOnce(new Error("full-page timeout"))
      .mockResolvedValueOnce(undefined);

    await expect(
      captureEvidenceScreenshot({
        page: { screenshot },
        path: "/tmp/gate-b.png",
      }),
    ).resolves.toEqual({
      path: "/tmp/gate-b.png",
      mode: "viewport",
      fallbackUsed: true,
      fullPageError: "Error: full-page timeout",
    });
    expect(screenshot).toHaveBeenNthCalledWith(2, {
      path: "/tmp/gate-b.png",
      fullPage: false,
      timeout: 15_000,
    });
  });
});
