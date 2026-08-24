import {
  METHOD_WINDOWS_SANDBOX_SETUP_COMPLETED,
  METHOD_WINDOWS_WORLD_WRITABLE_WARNING,
  type WindowsSandboxSetupCompletedNotification,
  type WindowsSandboxSetupMode,
  type WindowsSandboxSetupStartParams,
  type WindowsSandboxSetupStartResponse,
  type WindowsWorldWritableWarningNotification,
} from "@limecloud/app-server-client";
import { AppServerClient } from "@/lib/api/appServer";
import {
  subscribeAppServerNotifications,
  type AppServerEventBus,
} from "@/lib/api/appServerEventBus";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServerTypes";
import { type WindowsSandboxReadiness } from "@limecloud/app-server-client";

type WindowsSandboxAppServerClient = Pick<
  AppServerClient,
  "readWindowsSandboxReadiness" | "startWindowsSandboxSetup"
>;

const WINDOWS_SANDBOX_SETUP_MODES = new Set<WindowsSandboxSetupMode>([
  "elevated",
  "unelevated",
]);

export type WindowsSandboxNotification =
  | {
      method: typeof METHOD_WINDOWS_SANDBOX_SETUP_COMPLETED;
      params: WindowsSandboxSetupCompletedNotification;
    }
  | {
      method: typeof METHOD_WINDOWS_WORLD_WRITABLE_WARNING;
      params: WindowsWorldWritableWarningNotification;
    };

const WINDOWS_SANDBOX_READINESS_VALUES = new Set<WindowsSandboxReadiness>([
  "ready",
  "notConfigured",
  "updateRequired",
]);

export async function readWindowsSandboxReadiness(
  appServerClient: WindowsSandboxAppServerClient = new AppServerClient(),
): Promise<WindowsSandboxReadiness> {
  const response = await appServerClient.readWindowsSandboxReadiness({});
  const status = response.result?.status;
  if (
    typeof status !== "string" ||
    !WINDOWS_SANDBOX_READINESS_VALUES.has(status as WindowsSandboxReadiness)
  ) {
    throw new Error(
      "App Server windowsSandbox/readiness returned invalid status",
    );
  }
  return status as WindowsSandboxReadiness;
}

export async function startWindowsSandboxSetup(
  params: WindowsSandboxSetupStartParams,
  appServerClient: WindowsSandboxAppServerClient = new AppServerClient(),
): Promise<WindowsSandboxSetupStartResponse> {
  validateSetupParams(params);
  return (await appServerClient.startWindowsSandboxSetup(params)).result;
}

export function subscribeWindowsSandboxNotifications(
  handler: (notification: WindowsSandboxNotification) => void,
  options: { eventBus?: AppServerEventBus } = {},
): () => void {
  return subscribeAppServerNotifications(
    {
      onNotifications(notifications) {
        for (const notification of notifications) {
          const parsed = readWindowsSandboxNotification(notification);
          if (parsed) {
            handler(parsed);
          }
        }
      },
      getDrainOptions: () => ({
        activeIntervalMs: 25,
        intervalMs: 250,
        limit: 100,
      }),
    },
    options,
  );
}

export function readWindowsSandboxNotification(
  notification: AppServerJsonRpcNotification,
): WindowsSandboxNotification | null {
  if (notification.method === METHOD_WINDOWS_SANDBOX_SETUP_COMPLETED) {
    const params = readSetupCompletedParams(notification.params);
    return params
      ? { method: METHOD_WINDOWS_SANDBOX_SETUP_COMPLETED, params }
      : null;
  }
  if (notification.method === METHOD_WINDOWS_WORLD_WRITABLE_WARNING) {
    const params = readWorldWritableWarningParams(notification.params);
    return params
      ? { method: METHOD_WINDOWS_WORLD_WRITABLE_WARNING, params }
      : null;
  }
  return null;
}

function validateSetupParams(params: WindowsSandboxSetupStartParams): void {
  if (!WINDOWS_SANDBOX_SETUP_MODES.has(params.mode)) {
    throw new Error("windowsSandbox/setupStart mode is invalid");
  }
  if (
    params.cwd !== undefined &&
    params.cwd !== null &&
    !isAbsolutePath(params.cwd)
  ) {
    throw new Error("windowsSandbox/setupStart cwd must be an absolute path");
  }
}

function readSetupCompletedParams(
  value: unknown,
): WindowsSandboxSetupCompletedNotification | null {
  if (!isRecord(value)) {
    return null;
  }
  if (
    !WINDOWS_SANDBOX_SETUP_MODES.has(value.mode as WindowsSandboxSetupMode) ||
    typeof value.success !== "boolean" ||
    (value.error !== undefined &&
      value.error !== null &&
      typeof value.error !== "string")
  ) {
    return null;
  }
  return {
    mode: value.mode as WindowsSandboxSetupMode,
    success: value.success,
    error: (value.error as string | null | undefined) ?? null,
  };
}

function readWorldWritableWarningParams(
  value: unknown,
): WindowsWorldWritableWarningNotification | null {
  if (!isRecord(value) || !Array.isArray(value.samplePaths)) {
    return null;
  }
  if (
    !value.samplePaths.every((path) => typeof path === "string") ||
    typeof value.extraCount !== "number" ||
    !Number.isInteger(value.extraCount) ||
    value.extraCount < 0 ||
    typeof value.failedScan !== "boolean"
  ) {
    return null;
  }
  return {
    samplePaths: value.samplePaths,
    extraCount: value.extraCount,
    failedScan: value.failedScan,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isAbsolutePath(value: string): boolean {
  return value.startsWith("/") || /^[A-Za-z]:[\\/]/.test(value);
}

export type {
  WindowsSandboxReadiness,
  WindowsSandboxSetupCompletedNotification,
  WindowsSandboxSetupMode,
  WindowsSandboxSetupStartParams,
  WindowsSandboxSetupStartResponse,
  WindowsWorldWritableWarningNotification,
};
