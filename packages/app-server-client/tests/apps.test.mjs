import assert from "node:assert/strict";
import { test } from "vitest";
import {
  AppServerClient,
  appListUpdatedServerNotification,
  isAppListUpdatedNotification,
} from "../dist/index.js";

const app = {
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

test("Apps request helpers use exact Codex methods", () => {
  const client = new AppServerClient({ initialRequestId: 41 });
  assert.deepEqual(client.listApps({ limit: 25 }), {
    id: 41,
    method: "app/list",
    params: { limit: 25 },
  });
  assert.deepEqual(client.readApps({ appIds: ["writer"] }), {
    id: 42,
    method: "app/read",
    params: { appIds: ["writer"] },
  });
  assert.deepEqual(client.listInstalledApps(), {
    id: 43,
    method: "app/installed",
    params: {},
  });
});

test("app/list/updated parser rejects malformed app payloads", () => {
  const notification = {
    method: "app/list/updated",
    params: { data: [app] },
  };
  assert.deepEqual(
    appListUpdatedServerNotification(notification),
    notification,
  );
  assert.equal(isAppListUpdatedNotification(notification), true);
  assert.equal(
    appListUpdatedServerNotification({
      ...notification,
      params: { data: [{ ...app, callable: "yes", isEnabled: "yes" }] },
    }),
    undefined,
  );
});
