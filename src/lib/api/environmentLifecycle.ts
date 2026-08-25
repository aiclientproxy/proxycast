import type { EnvironmentStatusKind } from "@limecloud/app-server-client";
import { AppServerClient } from "./appServer";

export type EnvironmentRuntimeConnectionStatus =
  | "connected"
  | "disconnected"
  | "pending";

export interface EnvironmentRuntimeConnectionState {
  environmentId: string;
  status: EnvironmentRuntimeConnectionStatus;
}

export type EnvironmentStatusAppServerClient = Pick<
  AppServerClient,
  "readEnvironmentStatus"
>;

function projectStatus(
  status: EnvironmentStatusKind,
): EnvironmentRuntimeConnectionStatus {
  switch (status) {
    case "ready":
      return "connected";
    case "pending":
      return "pending";
    case "disconnected":
    case "unknown":
      return "disconnected";
  }
}

export async function readEnvironmentRuntimeStatuses(
  environmentIds: readonly string[],
  appServerClient: EnvironmentStatusAppServerClient = new AppServerClient(),
): Promise<EnvironmentRuntimeConnectionState[]> {
  const normalizedIds = Array.from(
    new Set(environmentIds.map((id) => id.trim()).filter(Boolean)),
  );
  return await Promise.all(
    normalizedIds.map(async (environmentId) => {
      const response = await appServerClient.readEnvironmentStatus({
        environmentId,
      });
      const status = response.result?.status;
      if (
        status !== "ready" &&
        status !== "pending" &&
        status !== "disconnected" &&
        status !== "unknown"
      ) {
        throw new Error("environment/status returned an invalid status");
      }
      return { environmentId, status: projectStatus(status) };
    }),
  );
}
