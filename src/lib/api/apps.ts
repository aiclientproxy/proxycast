import { AppServerClient } from "./appServerClient";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusDrainOptions,
  type AppServerEventBusSubscription,
} from "./appServerEventBus";
import { appListUpdatedServerNotification } from "../../../packages/app-server-client/src/server-notifications";
import type { JsonRpcMessage } from "../../../packages/app-server-client/src/protocol";
import type {
  AppServerAppInfo,
  AppServerAppsInstalledParams,
  AppServerAppsInstalledResponse,
  AppServerAppsListParams,
  AppServerAppsListResponse,
  AppServerAppsReadParams,
  AppServerAppsReadResponse,
  AppServerInstalledApp,
} from "./appServerTypes";

export type AppsRequestClient = Pick<
  AppServerClient,
  "listApps" | "readApps" | "listInstalledApps"
>;

export interface AppsListUpdatedSubscription {
  onUpdate: (apps: AppServerAppInfo[]) => void;
  onError?: (error: unknown) => void;
}

export interface AppsReadiness {
  apps: AppServerAppInfo[];
  installed: AppServerInstalledApp[];
  ready: boolean;
}

export async function listApps(
  params: AppServerAppsListParams = {},
  deps: { appServerClient?: AppsRequestClient } = {},
): Promise<AppServerAppsListResponse> {
  const response = await (
    deps.appServerClient ?? new AppServerClient()
  ).listApps(params);
  assertAppsListResponse(response.result);
  return response.result;
}

export async function readApps(
  params: AppServerAppsReadParams,
  deps: { appServerClient?: AppsRequestClient } = {},
): Promise<AppServerAppsReadResponse> {
  const response = await (
    deps.appServerClient ?? new AppServerClient()
  ).readApps(params);
  assertAppsReadResponse(response.result);
  return response.result;
}

export async function listInstalledApps(
  params: AppServerAppsInstalledParams = {},
  deps: { appServerClient?: AppsRequestClient } = {},
): Promise<AppServerAppsInstalledResponse> {
  const response = await (
    deps.appServerClient ?? new AppServerClient()
  ).listInstalledApps(params);
  assertAppsInstalledResponse(response.result);
  return response.result;
}

export async function readAppsReadiness(
  params: AppServerAppsInstalledParams = {},
  deps: { appServerClient?: AppsRequestClient } = {},
): Promise<AppsReadiness> {
  const [catalog, installed] = await Promise.all([
    listApps({}, deps),
    listInstalledApps(params, deps),
  ]);
  return {
    apps: catalog.data,
    installed: installed.apps,
    // Enabled apps are ready only when the runtime exposes a model-callable tool.
    ready: installed.apps.every((app) => !app.enabled || app.callable),
  };
}

export function readAppListUpdatedNotification(
  message: unknown,
): AppServerAppInfo[] | null {
  if (!isRecord(message)) {
    return null;
  }
  const notification = appListUpdatedServerNotification(
    message as JsonRpcMessage,
  );
  return notification?.params.data ?? null;
}

export function subscribeAppsListUpdates(
  subscription: AppsListUpdatedSubscription,
  options: AppServerEventBusDrainOptions & {
    isBridgeAvailable?: () => boolean;
    subscribeNotifications?: (
      subscription: AppServerEventBusSubscription,
    ) => () => void;
  } = {},
): () => void {
  const subscribeNotifications =
    options.subscribeNotifications ?? subscribeAppServerNotifications;
  return subscribeNotifications({
    getDrainOptions: () => ({
      intervalMs: options.intervalMs,
      limit: options.limit,
    }),
    onError: subscription.onError,
    onNotifications: (notifications) => {
      for (const notification of notifications) {
        const apps = readAppListUpdatedNotification(notification);
        if (apps) {
          subscription.onUpdate(apps);
        }
      }
    },
    shouldDrain: options.isBridgeAvailable,
  });
}

function assertAppsListResponse(
  value: unknown,
): asserts value is AppServerAppsListResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.data) ||
    !value.data.every(isAppInfo) ||
    (value.nextCursor !== null &&
      value.nextCursor !== undefined &&
      typeof value.nextCursor !== "string")
  ) {
    throw new Error("App Server app/list 返回了无效 Apps catalog");
  }
}

function assertAppsReadResponse(
  value: unknown,
): asserts value is AppServerAppsReadResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.apps) ||
    !value.apps.every(isConnectorMetadata) ||
    !Array.isArray(value.missingAppIds) ||
    !value.missingAppIds.every((id) => typeof id === "string")
  ) {
    throw new Error("App Server app/read 返回了无效 Apps metadata");
  }
}

function assertAppsInstalledResponse(
  value: unknown,
): asserts value is AppServerAppsInstalledResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.apps) ||
    !value.apps.every(isInstalledApp)
  ) {
    throw new Error("App Server app/installed 返回了无效 Apps runtime state");
  }
}

function isAppInfo(value: unknown): value is AppServerAppInfo {
  const app = record(value);
  return Boolean(
    app &&
    typeof app.id === "string" &&
    app.id.trim().length > 0 &&
    typeof app.name === "string" &&
    app.name.trim().length > 0 &&
    typeof app.isAccessible === "boolean" &&
    typeof app.isEnabled === "boolean" &&
    Array.isArray(app.pluginDisplayNames) &&
    app.pluginDisplayNames.every((name) => typeof name === "string"),
  );
}

function isConnectorMetadata(value: unknown): boolean {
  const app = record(value);
  return Boolean(
    app &&
    typeof app.id === "string" &&
    app.id.trim().length > 0 &&
    typeof app.name === "string" &&
    app.name.trim().length > 0 &&
    Array.isArray(app.pluginDisplayNames) &&
    app.pluginDisplayNames.every((name) => typeof name === "string"),
  );
}

function isInstalledApp(value: unknown): value is AppServerInstalledApp {
  const app = record(value);
  return Boolean(
    app &&
    typeof app.id === "string" &&
    app.id.trim().length > 0 &&
    (app.runtimeName === null || typeof app.runtimeName === "string") &&
    typeof app.enabled === "boolean" &&
    typeof app.callable === "boolean",
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function record(value: unknown): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}
