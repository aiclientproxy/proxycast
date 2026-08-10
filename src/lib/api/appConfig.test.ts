import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import {
  METHOD_CONFIG_BATCH_WRITE,
  METHOD_CONFIG_READ,
} from "@limecloud/app-server-client";
import {
  getConfig,
  getDefaultProvider,
  invalidateAppConfigCache,
  getEnvironmentPreview,
  saveConfig,
  updateConfig,
} from "./appConfig";

const { appServerRequest } = vi.hoisted(() => ({
  appServerRequest: vi.fn(),
}));

vi.mock("./appServer", () => ({
  AppServerClient: class {
    request = appServerRequest;
  },
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

function configReadResult(config: unknown, version = "version-1") {
  return {
    result: {
      config,
      origins: {},
      layers: [
        {
          name: {
            type: "user",
            file: "/tmp/lime/config.yaml",
            profile: null,
          },
          version,
          config,
        },
      ],
    },
  };
}

function configWriteResult(version = "version-2") {
  return {
    result: {
      status: "ok",
      version,
      filePath: "/tmp/lime/config.yaml",
      overriddenMetadata: null,
    },
  };
}

describe("appConfig API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    invalidateAppConfigCache();
  });

  it("配置读取走 App Server，宿主环境能力保持 Electron owner", async () => {
    appServerRequest.mockResolvedValueOnce(
      configReadResult({ default_provider: "claude" }),
    );
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({ entries: [] })
      .mockResolvedValueOnce("claude");

    await expect(getConfig()).resolves.toEqual(
      expect.objectContaining({ default_provider: "claude" }),
    );
    await expect(getEnvironmentPreview()).resolves.toEqual(
      expect.objectContaining({ entries: [] }),
    );
    await expect(getDefaultProvider()).resolves.toBe("claude");

    expect(appServerRequest).toHaveBeenCalledWith(METHOD_CONFIG_READ, {
      includeLayers: true,
    });
    expect(vi.mocked(safeInvoke)).toHaveBeenNthCalledWith(
      1,
      "get_environment_preview",
    );
    expect(vi.mocked(safeInvoke)).toHaveBeenNthCalledWith(
      2,
      "get_default_provider",
    );
  });

  it("环境预览应接收 Electron Host current 返回的局部 Shell 导入状态", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      shellImport: {
        enabled: false,
        status: "disabled",
        message: "Electron current 暂未接入 shell 环境导入预览。",
        importedCount: 0,
        durationMs: null,
      },
      entries: [],
    });

    await expect(getEnvironmentPreview()).resolves.toEqual({
      shellImport: {
        enabled: false,
        status: "disabled",
        message: "Electron current 暂未接入 shell 环境导入预览。",
        importedCount: 0,
        durationMs: null,
      },
      entries: [],
    });
  });

  it("环境预览遇到 degraded diagnostic facade 时应 fail closed", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      entries: [],
      diagnostic: {
        source: "electron-host-diagnostic",
        command: "get_environment_preview",
        status: "degraded",
      },
    });

    await expect(getEnvironmentPreview()).rejects.toThrow(
      "get_environment_preview 尚未接入真实环境预览 current 通道，收到 electron-host-diagnostic 诊断返回。",
    );
  });

  it("config/read 遇到无效配置或缺少单一用户层时应 fail closed", async () => {
    appServerRequest
      .mockResolvedValueOnce(configReadResult({ success: true }))
      .mockResolvedValueOnce(configReadResult({ default_provider: "" }))
      .mockResolvedValueOnce({
        result: {
          config: { default_provider: "claude" },
          origins: {},
          layers: [],
        },
      });

    await expect(getConfig()).rejects.toThrow("config/read 未返回有效配置");
    await expect(getConfig()).rejects.toThrow("config/read 未返回有效配置");
    await expect(getConfig()).rejects.toThrow(
      "config/read 未返回唯一 Desktop 用户配置层",
    );
  });

  it("默认 Provider 命令遇到 diagnostic facade 或无效形态时 fail closed", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        diagnostic: {
          source: "electron-host-diagnostic",
          command: "get_default_provider",
          status: "degraded",
        },
      })
      .mockResolvedValueOnce({ success: true })
      .mockResolvedValueOnce("");

    await expect(getDefaultProvider()).rejects.toThrow(
      "get_default_provider 尚未接入真实默认 Provider current 通道，收到 electron-host-diagnostic 诊断返回。",
    );
    await expect(getDefaultProvider()).rejects.toThrow(
      "get_default_provider 未返回有效默认 Provider",
    );
    await expect(getDefaultProvider()).rejects.toThrow(
      "get_default_provider 未返回有效默认 Provider",
    );
  });

  it("saveConfig 通过 batchWrite 写入变化字段并携带版本", async () => {
    appServerRequest
      .mockResolvedValueOnce(
        configReadResult({
          default_provider: "claude",
          language: "zh-CN",
          navigation: { schema_version: 3, enabled_items: [] },
        }),
      )
      .mockResolvedValueOnce(configWriteResult());

    await saveConfig({
      default_provider: "claude",
      language: "en-US",
      navigation: { schema_version: 3, enabled_items: ["companion"] },
    } as never);

    expect(appServerRequest).toHaveBeenNthCalledWith(
      2,
      METHOD_CONFIG_BATCH_WRITE,
      {
        edits: [
          {
            keyPath: "language",
            value: "en-US",
            mergeStrategy: "replace",
          },
        ],
        expectedVersion: "version-1",
        reloadUserConfig: true,
      },
    );
  });

  it("saveConfig 对无效写响应 fail closed，并失效旧缓存", async () => {
    appServerRequest
      .mockResolvedValueOnce(
        configReadResult({ default_provider: "claude", language: "zh-CN" }),
      )
      .mockResolvedValueOnce({ result: { status: "ok" } })
      .mockResolvedValueOnce(
        configReadResult(
          { default_provider: "claude", language: "zh-CN" },
          "version-2",
        ),
      );

    await getConfig();
    await expect(
      saveConfig({ default_provider: "claude", language: "en-US" } as never),
    ).rejects.toThrow("config/batchWrite 未返回有效写入结果");
    await getConfig();

    expect(
      appServerRequest.mock.calls.filter(
        ([method]) => method === METHOD_CONFIG_READ,
      ),
    ).toHaveLength(2);
  });

  it("getConfig 缓存结果并清理 navigation 旧入口", async () => {
    appServerRequest.mockResolvedValueOnce(
      configReadResult({
        default_provider: "claude",
        navigation: { schema_version: 2, enabled_items: ["companion"] },
      }),
    );

    const [first, second] = await Promise.all([getConfig(), getConfig()]);

    expect(appServerRequest).toHaveBeenCalledTimes(1);
    expect(first.navigation).toEqual({ schema_version: 3, enabled_items: [] });
    expect(second).toEqual(first);
    expect(first).not.toBe(second);
  });

  it("saveConfig 成功后更新版本化缓存", async () => {
    appServerRequest
      .mockResolvedValueOnce(
        configReadResult({ default_provider: "openai", language: "zh-CN" }),
      )
      .mockResolvedValueOnce(configWriteResult("version-2"));

    await getConfig();
    await saveConfig({
      default_provider: "openai",
      language: "ja-JP",
    } as never);
    await expect(getConfig()).resolves.toEqual(
      expect.objectContaining({ language: "ja-JP" }),
    );

    expect(appServerRequest).toHaveBeenCalledTimes(2);
  });

  it("updateConfig 串行合并连续 mutation，避免后写覆盖前一笔", async () => {
    let releaseFirstSave: (() => void) | undefined;
    let saveCount = 0;
    let secondUpdaterProvider: string | undefined;
    const writeParams: unknown[] = [];

    appServerRequest.mockImplementation(async (method, params) => {
      if (method === METHOD_CONFIG_READ) {
        return configReadResult({
          default_provider: "openai",
          workspace_preferences: {
            media_defaults: {
              image: {
                preferredProviderId: "old-provider",
                preferredModelId: "old-model",
                allowFallback: false,
              },
            },
          },
        });
      }
      if (method === METHOD_CONFIG_BATCH_WRITE) {
        writeParams.push(params);
        saveCount += 1;
        if (saveCount === 1) {
          await new Promise<void>((resolve) => {
            releaseFirstSave = resolve;
          });
        }
        return configWriteResult(`version-${saveCount + 1}`);
      }
      throw new Error(`unexpected method: ${method}`);
    });

    const providerUpdate = updateConfig((current) => ({
      ...current,
      workspace_preferences: {
        ...current.workspace_preferences,
        media_defaults: {
          ...current.workspace_preferences?.media_defaults,
          image: {
            preferredProviderId: "new-provider",
            allowFallback: false,
          },
        },
      },
    }));
    const modelUpdate = updateConfig((current) => {
      secondUpdaterProvider =
        current.workspace_preferences?.media_defaults?.image
          ?.preferredProviderId;
      return {
        ...current,
        workspace_preferences: {
          ...current.workspace_preferences,
          media_defaults: {
            ...current.workspace_preferences?.media_defaults,
            image: {
              ...current.workspace_preferences?.media_defaults?.image,
              preferredModelId: "new-model",
            },
          },
        },
      };
    });

    await vi.waitFor(() => expect(saveCount).toBe(1));
    expect(secondUpdaterProvider).toBeUndefined();
    releaseFirstSave?.();
    await Promise.all([providerUpdate, modelUpdate]);

    expect(secondUpdaterProvider).toBe("new-provider");
    expect(writeParams).toHaveLength(2);
    expect(writeParams[1]).toEqual(
      expect.objectContaining({ expectedVersion: "version-2" }),
    );
  });
});
