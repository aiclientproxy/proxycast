import { describe, expect, it, vi } from "vitest";
import { readWindowsSandboxReadiness } from "./windowsSandbox";

describe("windowsSandbox", () => {
  it("通过 exact App Server method 读取真实 readiness", async () => {
    const request = vi.fn(async () => ({
      id: 1,
      result: { status: "updateRequired" as const },
      configWarnings: [],
    }));

    await expect(readWindowsSandboxReadiness({ request })).resolves.toBe(
      "updateRequired",
    );
    expect(request).toHaveBeenCalledWith("windowsSandbox/readiness", {});
  });

  it("未知状态应 fail closed", async () => {
    const request = vi.fn(async () => ({
      id: 1,
      result: { status: "installed" },
      configWarnings: [],
    }));

    await expect(readWindowsSandboxReadiness({ request })).rejects.toThrow(
      "returned invalid status",
    );
  });
});
