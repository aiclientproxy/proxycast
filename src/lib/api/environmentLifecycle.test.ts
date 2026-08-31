import { describe, expect, it, vi } from "vitest";
import { readEnvironmentRuntimeStatuses } from "./environmentLifecycle";

describe("readEnvironmentRuntimeStatuses", () => {
  it("通过 current environment/status 读取并投影状态", async () => {
    const readEnvironmentStatus = vi.fn(async ({ environmentId }) => ({
      result: {
        status:
          environmentId === "local"
            ? ("ready" as const)
            : ("disconnected" as const),
        ...(environmentId === "remote-a"
          ? { error: "exec-server unavailable" }
          : {}),
      },
    }));

    await expect(
      readEnvironmentRuntimeStatuses([" local ", "remote-a", "local"], {
        readEnvironmentStatus,
      } as never),
    ).resolves.toEqual([
      { environmentId: "local", status: "connected" },
      {
        environmentId: "remote-a",
        status: "disconnected",
        error: "exec-server unavailable",
      },
    ]);
    expect(readEnvironmentStatus).toHaveBeenCalledTimes(2);
    expect(readEnvironmentStatus).toHaveBeenCalledWith({
      environmentId: "remote-a",
    });
  });

  it("拒绝无效的 status 响应", async () => {
    await expect(
      readEnvironmentRuntimeStatuses(["remote-a"], {
        readEnvironmentStatus: vi.fn(async () => ({
          result: { status: "broken" },
        })),
      } as never),
    ).rejects.toThrow("environment/status returned an invalid status");
  });
});
