import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";

const desktopRuntimeMocks = vi.hoisted(() => ({
  hasDesktopHostInvokeCapability: vi.fn(),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostInvokeCapability:
    desktopRuntimeMocks.hasDesktopHostInvokeCapability,
}));

import {
  getDesktopHostDiagnostics,
  isDesktopHostDiagnostics,
} from "./desktopHostDiagnostics";

function diagnostics(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    stage: "ready",
    connected: true,
    connection_generation: 2,
    restart_pending: false,
    resume_recovery_pending: false,
    sidecar: {
      pid: 42,
      running: true,
      exit_code: null,
      signal: null,
      stderr_line_count: 1,
      stderr_tail: ["ready"],
    },
    last_failure: null,
    ...overrides,
  };
}

describe("desktopHostDiagnostics API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
  });

  it("桌面宿主不可用时不调用 Electron bridge", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(false);

    await expect(getDesktopHostDiagnostics()).resolves.toBeNull();
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("通过只读 Host command 返回严格诊断结果", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce(diagnostics());

    await expect(getDesktopHostDiagnostics()).resolves.toMatchObject({
      schema_version: 1,
      stage: "ready",
      connection_generation: 2,
    });
    expect(safeInvoke).toHaveBeenCalledWith("app_server_host_diagnostics");
  });

  it("拒绝不受约束的阶段、stderr 或连接代际", () => {
    expect(isDesktopHostDiagnostics(diagnostics())).toBe(true);
    expect(
      isDesktopHostDiagnostics(
        diagnostics({
          stage: "unknown",
        }),
      ),
    ).toBe(false);
    expect(
      isDesktopHostDiagnostics(
        diagnostics({
          connection_generation: -1,
        }),
      ),
    ).toBe(false);
    expect(
      isDesktopHostDiagnostics(
        diagnostics({
          sidecar: {
            ...diagnostics().sidecar,
            stderr_tail: Array.from({ length: 21 }, () => "line"),
          },
        }),
      ),
    ).toBe(false);
  });
});
