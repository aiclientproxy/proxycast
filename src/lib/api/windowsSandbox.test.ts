import { describe, expect, it, vi } from "vitest";
import {
  readWindowsSandboxNotification,
  readWindowsSandboxReadiness,
  startWindowsSandboxSetup,
} from "./windowsSandbox";

describe("windowsSandbox", () => {
  it("通过 exact App Server method 读取真实 readiness", async () => {
    const readReadiness = vi.fn(async () => ({
      id: 1,
      result: { status: "updateRequired" as const },
      configWarnings: [],
    }));

    await expect(
      readWindowsSandboxReadiness({
        readWindowsSandboxReadiness: readReadiness,
      }),
    ).resolves.toBe("updateRequired");
    expect(readReadiness).toHaveBeenCalledWith({});
  });

  it("未知状态应 fail closed", async () => {
    const readReadiness = vi.fn(async () => ({
      id: 1,
      result: { status: "installed" },
      configWarnings: [],
    }));

    await expect(
      readWindowsSandboxReadiness({
        readWindowsSandboxReadiness: readReadiness,
      }),
    ).rejects.toThrow("returned invalid status");
  });

  it("setup 只接受合法模式和绝对 cwd，并返回 typed response", async () => {
    const startSetup = vi.fn(async (params: unknown) => ({
      id: 1,
      result: { started: true },
      configWarnings: [],
      params,
    }));

    await expect(
      startWindowsSandboxSetup(
        { mode: "unelevated", cwd: "C:\\workspace" },
        { startWindowsSandboxSetup: startSetup },
      ),
    ).resolves.toEqual({ started: true });
    expect(startSetup).toHaveBeenCalledWith({
      mode: "unelevated",
      cwd: "C:\\workspace",
    });
    await expect(
      startWindowsSandboxSetup(
        { mode: "elevated", cwd: "relative" },
        { startWindowsSandboxSetup: startSetup },
      ),
    ).rejects.toThrow("cwd must be an absolute path");
  });

  it("只投影 schema 合法的 setup completion 和 world-writable warning", () => {
    expect(
      readWindowsSandboxNotification({
        method: "windowsSandbox/setupCompleted",
        params: { mode: "unelevated", success: false, error: "runner missing" },
      }),
    ).toEqual({
      method: "windowsSandbox/setupCompleted",
      params: { mode: "unelevated", success: false, error: "runner missing" },
    });
    expect(
      readWindowsSandboxNotification({
        method: "windows/worldWritableWarning",
        params: { samplePaths: ["C:\\tmp"], extraCount: 2, failedScan: false },
      }),
    ).toMatchObject({ method: "windows/worldWritableWarning" });
    expect(
      readWindowsSandboxNotification({
        method: "windows/worldWritableWarning",
        params: { samplePaths: ["C:\\tmp"], extraCount: -1, failedScan: false },
      }),
    ).toBeNull();
    expect(
      readWindowsSandboxNotification({
        method: "windows/worldWritableWarning",
        params: {
          samplePaths: Array.from(
            { length: 6 },
            (_, index) => `C:\\tmp\\${index}`,
          ),
          extraCount: 0,
          failedScan: false,
        },
      }),
    ).toBeNull();
    expect(
      readWindowsSandboxNotification({
        method: "windows/worldWritableWarning",
        params: {
          samplePaths: ["x".repeat(32_768)],
          extraCount: 0,
          failedScan: false,
        },
      }),
    ).toBeNull();
  });
});
