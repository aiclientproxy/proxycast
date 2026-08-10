import fs from "node:fs";
import path from "node:path";
import {
  updateConfigFromPage,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";
import { assert, sanitizeJson } from "./claw-chat-current-fixture-utils.mjs";
import { normalizedString } from "./claw-image-live-smoke-common.mjs";

const AGNES_PROVIDER_NAME = "Agnes";

export async function bindImageDefaultsFromPage(page, options) {
  const result = await updateConfigFromPage(page, (currentConfig) => ({
        ...(currentConfig || {}),
        workspace_preferences: {
          ...(currentConfig?.workspace_preferences || {}),
          media_defaults: {
            ...(currentConfig?.workspace_preferences?.media_defaults || {}),
            image: {
              preferredProviderId: options.providerPreference,
              preferredModelId: options.modelPreference,
              allowFallback: false,
            },
          },
        },
      }), options.requestLog);
  await page.evaluate(() => {
      window.dispatchEvent(new Event("lime:app-config-changed"));
      window.dispatchEvent(
        new CustomEvent("provider-data-changed", {
          detail: {
            source: "live_image_smoke",
            timestamp: Date.now(),
          },
        }),
      );
  });
  return {
    providerId: options.providerPreference,
    modelId: options.modelPreference,
    imageDefaults: result.config.workspace_preferences?.media_defaults?.image ?? null,
  };
}

export function copyElectronConfigToAppServerConfig(runtimeEnv) {
  const electronConfigPath = path.join(
    runtimeEnv.electronUserDataDir,
    "config.yaml",
  );
  if (!fs.existsSync(electronConfigPath)) {
    return {
      copied: false,
      electronConfigPath,
      appServerConfigPath: runtimeEnv.configPath,
      reason: "electron-config-missing",
    };
  }
  fs.mkdirSync(path.dirname(runtimeEnv.configPath), { recursive: true });
  fs.copyFileSync(electronConfigPath, runtimeEnv.configPath);
  return {
    copied: true,
    electronConfigPath,
    appServerConfigPath: runtimeEnv.configPath,
  };
}

export async function ensureLiveAgnesProviderFromEnv(
  page,
  requestLog,
  options,
) {
  const apiKey = process.env[options.apiKeyEnv]?.trim();
  assert(apiKey, `${options.apiKeyEnv} 未配置`);
  const created = await invokeAppServerFromPage(
    page,
    "modelProvider/create",
    {
      name: AGNES_PROVIDER_NAME,
      providerType: "openai",
      apiHost: options.apiHost,
    },
    requestLog,
  );
  const provider = created.result?.provider;
  const providerId = normalizedString(provider?.id);
  assert(providerId, "modelProvider/create 未返回 provider id");
  await invokeAppServerFromPage(
    page,
    "modelProvider/update",
    {
      providerId,
      enabled: true,
      models: [{ id: options.modelPreference }],
      sortOrder: 1,
    },
    requestLog,
  );
  await invokeAppServerFromPage(
    page,
    "modelProviderKey/create",
    {
      providerId,
      apiKey,
      alias: "agnes-live-smoke",
      replaceExisting: true,
    },
    requestLog,
  );
  options.providerPreference = providerId;
  return sanitizeJson({
    providerId,
    providerName: provider?.name ?? AGNES_PROVIDER_NAME,
    apiHost: options.apiHost,
    modelId: options.modelPreference,
    apiKeyConfigured: true,
  });
}
