import { safeInvoke } from "@/lib/dev-bridge";
import { hasDesktopHostInvokeCapability } from "@/lib/desktop-runtime";

export type DesktopHostStage =
  | "idle"
  | "resolving"
  | "starting"
  | "initializing"
  | "ready"
  | "recovering"
  | "restarting"
  | "failed"
  | "stopping"
  | "stopped";

export interface DesktopHostFailure {
  stage: DesktopHostStage;
  message: string;
  occurred_at: string;
  exit_code: number | null;
  signal: string | null;
  stderr_tail: string[];
}

export interface DesktopHostSidecarDiagnostics {
  pid: number | null;
  running: boolean;
  exit_code: number | null;
  signal: string | null;
  stderr_line_count: number;
  stderr_tail: string[];
}

export interface DesktopHostDiagnostics {
  schema_version: 1;
  stage: DesktopHostStage;
  connected: boolean;
  connection_generation: number;
  restart_pending: boolean;
  resume_recovery_pending: boolean;
  sidecar: DesktopHostSidecarDiagnostics | null;
  last_failure: DesktopHostFailure | null;
}

const DESKTOP_HOST_DIAGNOSTIC_SCHEMA_VERSION = 1;
const DESKTOP_HOST_STDERR_TAIL_LIMIT = 20;
const DESKTOP_HOST_STDERR_LINE_LIMIT = 240;

function isDesktopHostStage(value: unknown): value is DesktopHostStage {
  return (
    value === "idle" ||
    value === "resolving" ||
    value === "starting" ||
    value === "initializing" ||
    value === "ready" ||
    value === "recovering" ||
    value === "restarting" ||
    value === "failed" ||
    value === "stopping" ||
    value === "stopped"
  );
}

function isNullableFiniteInteger(value: unknown): value is number | null {
  return (
    value === null || (typeof value === "number" && Number.isInteger(value))
  );
}

function isBoundedStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= DESKTOP_HOST_STDERR_TAIL_LIMIT &&
    value.every(
      (item) =>
        typeof item === "string" &&
        item.length <= DESKTOP_HOST_STDERR_LINE_LIMIT,
    )
  );
}

function isDesktopHostFailure(value: unknown): value is DesktopHostFailure {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const failure = value as Partial<DesktopHostFailure>;
  return (
    isDesktopHostStage(failure.stage) &&
    typeof failure.message === "string" &&
    failure.message.length <= 320 &&
    typeof failure.occurred_at === "string" &&
    isNullableFiniteInteger(failure.exit_code) &&
    (failure.signal === null || typeof failure.signal === "string") &&
    isBoundedStringArray(failure.stderr_tail)
  );
}

function isDesktopHostSidecarDiagnostics(
  value: unknown,
): value is DesktopHostSidecarDiagnostics {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const sidecar = value as Partial<DesktopHostSidecarDiagnostics>;
  return (
    isNullableFiniteInteger(sidecar.pid) &&
    typeof sidecar.running === "boolean" &&
    isNullableFiniteInteger(sidecar.exit_code) &&
    (sidecar.signal === null || typeof sidecar.signal === "string") &&
    typeof sidecar.stderr_line_count === "number" &&
    Number.isInteger(sidecar.stderr_line_count) &&
    sidecar.stderr_line_count >= 0 &&
    isBoundedStringArray(sidecar.stderr_tail)
  );
}

export function isDesktopHostDiagnostics(
  value: unknown,
): value is DesktopHostDiagnostics {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const diagnostics = value as Partial<DesktopHostDiagnostics>;
  return (
    diagnostics.schema_version === DESKTOP_HOST_DIAGNOSTIC_SCHEMA_VERSION &&
    isDesktopHostStage(diagnostics.stage) &&
    typeof diagnostics.connected === "boolean" &&
    typeof diagnostics.connection_generation === "number" &&
    Number.isInteger(diagnostics.connection_generation) &&
    diagnostics.connection_generation >= 0 &&
    typeof diagnostics.restart_pending === "boolean" &&
    typeof diagnostics.resume_recovery_pending === "boolean" &&
    (diagnostics.sidecar === null ||
      isDesktopHostSidecarDiagnostics(diagnostics.sidecar)) &&
    (diagnostics.last_failure === null ||
      isDesktopHostFailure(diagnostics.last_failure))
  );
}

export async function getDesktopHostDiagnostics(): Promise<DesktopHostDiagnostics | null> {
  if (!hasDesktopHostInvokeCapability()) {
    return null;
  }

  const result = await safeInvoke<unknown>("app_server_host_diagnostics");
  if (!isDesktopHostDiagnostics(result)) {
    throw new Error(
      "app_server_host_diagnostics 返回了无效的 Desktop Host 诊断结果",
    );
  }
  return result;
}
