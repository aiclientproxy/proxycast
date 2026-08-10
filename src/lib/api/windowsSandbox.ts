import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_WINDOWS_SANDBOX_READINESS,
  type WindowsSandboxReadiness,
  type WindowsSandboxReadinessResponse,
} from "@limecloud/app-server-client";

type WindowsSandboxAppServerClient = Pick<AppServerClient, "request">;

const WINDOWS_SANDBOX_READINESS_VALUES = new Set<WindowsSandboxReadiness>([
  "ready",
  "notConfigured",
  "updateRequired",
]);

export async function readWindowsSandboxReadiness(
  appServerClient: WindowsSandboxAppServerClient = new AppServerClient(),
): Promise<WindowsSandboxReadiness> {
  const response =
    await appServerClient.request<WindowsSandboxReadinessResponse>(
      METHOD_WINDOWS_SANDBOX_READINESS,
      {},
    );
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

export type { WindowsSandboxReadiness };
