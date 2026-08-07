import { describe, expect, it, vi } from "vitest";
import type {
  AppServerAppInfo,
  AppServerAppsInstalledResponse,
  AppServerAppsListResponse,
  AppServerAppsReadResponse,
  AppServerRequestResult,
} from "./appServerTypes";
import {
  listApps,
  listInstalledApps,
  readAppListUpdatedNotification,
  readApps,
  readAppsReadiness,
  subscribeAppsListUpdates,
  type AppsRequestClient,
} from "./apps";

const app: AppServerAppInfo = {
  id: "writer",
  name: "Writer",
  description: "Write documents",
  logoUrl: null,
  logoUrlDark: null,
  iconAssets: null,
  iconDarkAssets: null,
  distributionChannel: "repo",
  branding: null,
  appMetadata: null,
  labels: null,
  installUrl: null,
  isAccessible: true,
  isEnabled: true,
  pluginDisplayNames: ["writer-plugin"],
};

function requestResult<T>(result: T): AppServerRequestResult<T> {
  return {
    id: 1,
    result,
    response: { id: 1, result },
    notifications: [],
    configWarnings: [],
    messages: [],
  };
}

function client(overrides: Partial<AppsRequestClient> = {}): AppsRequestClient {
  return {
    listApps: vi.fn(async () =>
      requestResult<AppServerAppsListResponse>({
        data: [app],
        nextCursor: null,
      }),
    ),
    readApps: vi.fn(async () =>
      requestResult<AppServerAppsReadResponse>({
        apps: [
          {
            id: "writer",
            name: "Writer",
            description: "Write documents",
            iconUrl: null,
            iconUrlDark: null,
            distributionChannel: "repo",
            installUrl: null,
            pluginDisplayNames: ["writer-plugin"],
            toolSummaries: [],
          },
        ],
        missingAppIds: [],
      }),
    ),
    listInstalledApps: vi.fn(async () =>
      requestResult<AppServerAppsInstalledResponse>({
        apps: [
          {
            id: "writer",
            runtimeName: "Writer",
            enabled: true,
            callable: false,
          },
        ],
      }),
    ),
    ...overrides,
  };
}

describe("Apps App Server gateway", () => {
  it("通过 typed request client 读取 catalog、metadata 与 runtime state", async () => {
    const appServerClient = client();

    await expect(listApps({}, { appServerClient })).resolves.toEqual({
      data: [app],
      nextCursor: null,
    });
    await expect(
      readApps({ appIds: ["writer"], includeTools: true }, { appServerClient }),
    ).resolves.toMatchObject({ apps: [{ id: "writer" }] });
    await expect(
      listInstalledApps({}, { appServerClient }),
    ).resolves.toMatchObject({ apps: [{ callable: false }] });
  });

  it("将 enabled 但不可调用的本地 App 判定为未就绪", async () => {
    await expect(
      readAppsReadiness({}, { appServerClient: client() }),
    ).resolves.toMatchObject({
      apps: [app],
      installed: [{ id: "writer", callable: false }],
      ready: false,
    });
  });

  it("对畸形响应 fail closed", async () => {
    const appServerClient = client({
      listApps: vi.fn(async () => requestResult({ data: [{ id: "writer" }] })),
    } as Partial<AppsRequestClient>);

    await expect(listApps({}, { appServerClient })).rejects.toThrow(
      "无效 Apps catalog",
    );
  });

  it("只消费 typed app/list/updated 通知", () => {
    const valid = {
      jsonrpc: "2.0",
      method: "app/list/updated",
      params: { data: [app] },
    };
    expect(readAppListUpdatedNotification(valid)).toEqual([app]);
    expect(
      readAppListUpdatedNotification({
        ...valid,
        params: { data: [{ ...app, isEnabled: "yes" }] },
      }),
    ).toBeNull();

    let captured:
      | Parameters<
          NonNullable<
            Parameters<
              typeof subscribeAppsListUpdates
            >[1]["subscribeNotifications"]
          >
        >[0]
      | undefined;
    const onUpdate = vi.fn();
    const unsubscribe = vi.fn();
    subscribeAppsListUpdates(
      { onUpdate },
      {
        subscribeNotifications: (subscription) => {
          captured = subscription;
          return unsubscribe;
        },
      },
    );
    captured?.onNotifications?.([
      valid,
      {
        jsonrpc: "2.0",
        method: "app/list/updated",
        params: { data: [{ id: "broken" }] },
      },
    ]);
    expect(onUpdate).toHaveBeenCalledOnce();
    expect(onUpdate).toHaveBeenCalledWith([app]);
  });
});
