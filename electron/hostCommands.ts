/* global process */
import { app, shell } from "./electronRuntime";
import {
  METHOD_MODEL_LIST,
  METHOD_MODEL_PROVIDER_LIST,
  METHOD_WORKSPACE_BY_PATH_READ,
  METHOD_WORKSPACE_DEFAULT_ENSURE,
  METHOD_WORKSPACE_DEFAULT_READ,
  METHOD_WORKSPACE_ENSURE,
  METHOD_WORKSPACE_ENSURE_READY,
  METHOD_WORKSPACE_LIST,
  METHOD_WORKSPACE_PROJECTS_ROOT_READ,
  METHOD_WORKSPACE_PROJECT_PATH_RESOLVE,
  METHOD_WORKSPACE_READ,
  decodeModelRouteSelector,
  type Model,
  type ModelListResponse,
  type ModelProviderListResponse,
  type WorkspaceEnsureProjectResponse,
  type WorkspaceEnsureReadyResponse,
  type WorkspaceListResponse,
  type WorkspaceProjectPathResolveResponse,
  type WorkspaceProjectsRootReadResponse,
  type WorkspaceReadResponse,
} from "@limecloud/app-server-client";
import { mkdir, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import path from "node:path";
import type { ElectronAppServerHost } from "./appServerHost";
import { resolveCurrentDesktopStorageRoots } from "./appDataPaths";
import { AppConfigHost, buildDefaultConfig } from "./appConfigHost";
import {
  openProjectPathWithLocalTool,
  type ProjectPathOpenTool,
} from "./projectToolsHost";
import { showDesktopNotification } from "./desktopNotificationHost";
import { FileShellHost } from "./fileShellHost";
import { LayeredDesignProjectHost } from "./layeredDesignProjectHost";
import { openResourceManagerWindow } from "./resourceManagerWindowHost";
import { SystemUtilityHost } from "./systemUtilityHost";
import { VoiceModelHost } from "./voiceModelHost";

type HostArgs = Record<string, unknown> | null | undefined;
type AppServerParams = Record<string, unknown>;
type HostEventEmitter = (event: string, payload?: unknown) => void;
const OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_EVENT = "oem-cloud-oauth-callback";
const OEM_CLOUD_OAUTH_CALLBACK_PATH = "/oauth/callback";
const OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_TTL_MS = 10 * 60 * 1000;
const OEM_CLOUD_OAUTH_CALLBACK_HTML = `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Lime 登录回调</title>
    <script>
      (function () {
        if (window.location.hash && window.location.hash.length > 1) {
          var params = new URLSearchParams(window.location.hash.slice(1));
          var search = new URLSearchParams(window.location.search);
          params.forEach(function (value, key) {
            if (!search.has(key)) search.set(key, value);
          });
          window.location.replace(window.location.pathname + "?" + search.toString());
        }
      })();
    </script>
  </head>
  <body>
    <p>Lime 登录结果已返回，可以关闭此页面。</p>
  </body>
</html>`;

export class ElectronHostCommands {
  readonly #appServerHost: ElectronAppServerHost;
  readonly #userDataDir: string;
  readonly #emit: HostEventEmitter;
  readonly #fileShellHost = new FileShellHost();
  readonly #layeredDesignProjectHost = new LayeredDesignProjectHost();
  readonly #systemUtilityHost: SystemUtilityHost;
  readonly #voiceModelHost: VoiceModelHost;
  readonly #appConfigHost: AppConfigHost;
  #oauthCallbackBridgeServer: Server | null = null;
  #oauthCallbackBridgeTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    appServerHost: ElectronAppServerHost,
    userDataDir = app.getPath("userData"),
    emit: HostEventEmitter = () => undefined,
    appDataRoot = resolveCurrentDesktopStorageRoots(userDataDir).appDataRoot,
  ) {
    this.#appServerHost = appServerHost;
    this.#userDataDir = userDataDir;
    this.#emit = emit;
    this.#appConfigHost = new AppConfigHost(userDataDir);
    this.#systemUtilityHost = new SystemUtilityHost({
      appDataRoot,
      readConfig: () => this.#readConfig(),
    });
    this.#voiceModelHost = new VoiceModelHost(appDataRoot, emit);
  }

  async invoke(command: string, args?: HostArgs): Promise<unknown> {
    switch (command) {
      case "get_config":
        return await this.#readConfig();
      case "save_config":
        return await this.#saveConfig(args);
      case "get_experimental_config":
        return await this.#getExperimentalConfig();
      case "save_experimental_config":
        return await this.#saveExperimentalConfig(args);
      case "open_external_url":
        return await this.#systemUtilityHost.openExternalUrl(args);
      case "open_file_preview_window":
        return await this.#fileShellHost.openFilePreviewWindow(args);
      case "open_resource_manager_window":
        return openResourceManagerWindow(args);
      case "open_system_settings_url":
        return await this.#systemUtilityHost.openSystemSettingsUrl(args);
      case "show_desktop_notification":
        return showDesktopNotification(args);
      case "reveal_in_finder":
        return this.#fileShellHost.revealInFinder(args);
      case "open_with_default_app":
        return await this.#fileShellHost.openWithDefaultApp(args);
      case "open_project_path_with_tool":
        return await this.#openProjectPathWithTool(args);
      case "save_exported_document":
        return await this.#saveExportedDocument(args);
      case "save_layered_design_project_export":
        return await this.#layeredDesignProjectHost.saveExport(args);
      case "read_layered_design_project_export":
        return await this.#layeredDesignProjectHost.readExport(args);
      case "recognize_layered_design_text":
        return this.#layeredDesignProjectHost.recognizeText(args);
      case "analyze_layered_design_flat_image":
        return this.#layeredDesignProjectHost.analyzeFlatImage(args);
      case "get_home_dir":
        return this.#fileShellHost.getHomeDir();
      case "get_file_manager_locations":
        return await this.#fileShellHost.getFileManagerLocations();
      case "get_file_icon_data_url":
        return await this.#fileShellHost.getFileIconDataUrl(args);
      case "start_oem_cloud_oauth_callback_bridge":
        return await this.#startOemCloudOAuthCallbackBridge();
      case "get_runtime_provider_selection":
        return await this.#getRuntimeProviderSelection();
      case "get_default_provider":
        return await this.#getDefaultProvider();
      case "workspace_list":
        return await this.#listWorkspaces();
      case "workspace_get_default":
        return await this.#readDefaultWorkspace();
      case "get_or_create_default_project":
        return await this.#ensureDefaultWorkspace();
      case "workspace_get":
        return await this.#readWorkspace(args);
      case "workspace_get_by_path":
        return await this.#readWorkspaceByPath(args);
      case "workspace_ensure":
        return await this.#ensureWorkspace(args);
      case "workspace_get_projects_root":
        return await this.#readWorkspaceProjectsRoot();
      case "workspace_resolve_project_path":
        return await this.#resolveWorkspaceProjectPath(args);
      case "workspace_ensure_default_ready":
        return await this.#ensureDefaultWorkspaceReady();
      case "workspace_ensure_ready":
        return await this.#ensureWorkspaceReady(args);
      case "voice_models_list_catalog":
        return this.#voiceModelHost.listCatalog();
      case "voice_models_get_install_state":
        return await this.#voiceModelHost.getInstallState(args);
      case "voice_models_download":
        return await this.#voiceModelHost.download(args);
      case "voice_models_delete":
        return await this.#voiceModelHost.delete(args);
      case "get_environment_preview":
        return await this.#systemUtilityHost.getEnvironmentPreview();
      case "get_browser_connector_settings_cmd":
        return this.#systemUtilityHost.getBrowserConnectorSettings();
      case "get_browser_connector_install_status_cmd":
        return this.#systemUtilityHost.getBrowserConnectorInstallStatus();
      case "get_chrome_profile_sessions":
        return this.#systemUtilityHost.getChromeProfileSessions();
      case "get_chrome_bridge_endpoint_info":
        return this.#systemUtilityHost.getChromeBridgeEndpointInfo();
      case "get_chrome_bridge_status":
        return this.#systemUtilityHost.getChromeBridgeStatus();
      case "get_browser_backend_policy":
        return this.#systemUtilityHost.getBrowserBackendPolicy();
      case "get_browser_backends_status":
        return this.#systemUtilityHost.getBrowserBackendsStatus();
      case "report_frontend_debug_log":
        this.#reportFrontendDebugLog(args);
        return null;
      case "report_frontend_crash":
        this.#reportFrontendCrash(args);
        return { success: true };
      default:
        throw new Error(`Electron host command is not implemented: ${command}`);
    }
  }

  async #appServerRequest<T>(
    method: string,
    params: AppServerParams = {},
  ): Promise<T> {
    return await this.#appServerHost.request<T>(method, params);
  }

  async #openProjectPathWithTool(
    args: HostArgs,
  ): Promise<Record<string, never>> {
    const request = readRequest(args);
    const rootPath = readRequiredAbsolutePath(request, "rootPath");
    const tool = readProjectPathOpenTool(request);
    if (tool === "finder") {
      const errorMessage = await shell.openPath(rootPath);
      if (errorMessage) {
        throw new Error(errorMessage);
      }
      return {};
    }
    await openProjectPathWithLocalTool(rootPath, tool);
    return {};
  }

  async #saveExportedDocument(args: HostArgs): Promise<null> {
    const request = readRequest(args);
    const targetPath = readRequiredString(request, "filePath");
    const content = readRequiredRawString(request, "content");
    await mkdir(path.dirname(targetPath), { recursive: true });
    await writeFile(targetPath, content, "utf8");
    return null;
  }

  async #startOemCloudOAuthCallbackBridge(): Promise<{
    callbackUrl: string;
  }> {
    await this.#closeOemCloudOAuthCallbackBridge();

    const server = createServer((request, response) => {
      const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
      if (requestUrl.pathname !== OEM_CLOUD_OAUTH_CALLBACK_PATH) {
        response.writeHead(404, {
          "Content-Type": "text/plain; charset=utf-8",
        });
        response.end("Not found");
        return;
      }

      const payload = buildOemCloudOAuthCallbackPayload(requestUrl);
      if (shouldEmitOemCloudOAuthCallback(payload)) {
        this.#emit(OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_EVENT, payload);
        void this.#closeOemCloudOAuthCallbackBridge();
      }

      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(OEM_CLOUD_OAUTH_CALLBACK_HTML);
    });

    server.on("error", (error) => {
      console.warn(
        `[OAuthCallbackBridge] OAuth 本地回调桥运行失败: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });

    await new Promise<void>((resolve, reject) => {
      const onError = (error: Error) => {
        server.off("listening", onListening);
        reject(error);
      };
      const onListening = () => {
        server.off("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(0, "127.0.0.1");
    });

    const address = server.address();
    if (!address || typeof address === "string") {
      server.close();
      throw new Error("无法读取 OAuth 本地回调地址");
    }

    this.#oauthCallbackBridgeServer = server;
    this.#oauthCallbackBridgeTimer = setTimeout(() => {
      void this.#closeOemCloudOAuthCallbackBridge();
    }, OEM_CLOUD_OAUTH_CALLBACK_BRIDGE_TTL_MS);

    return {
      callbackUrl: `http://127.0.0.1:${address.port}${OEM_CLOUD_OAUTH_CALLBACK_PATH}`,
    };
  }

  async #closeOemCloudOAuthCallbackBridge(): Promise<void> {
    if (this.#oauthCallbackBridgeTimer) {
      clearTimeout(this.#oauthCallbackBridgeTimer);
      this.#oauthCallbackBridgeTimer = null;
    }
    const server = this.#oauthCallbackBridgeServer;
    this.#oauthCallbackBridgeServer = null;
    if (!server || !server.listening) {
      return;
    }
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
    });
  }

  async #getRuntimeProviderSelection(): Promise<{
    provider_configured: boolean;
    provider_name?: string;
    provider_selector?: string;
    model_name?: string;
  }> {
    const [config, providers, models] = await Promise.all([
      this.#readConfig().catch(() => buildDefaultConfig()),
      this.#listModelProviders().catch(() => []),
      this.#listModels().catch(() => []),
    ]);
    const defaultProvider = resolveCurrentDefaultProvider(
      config.default_provider,
      providers,
    );
    const configuredDefaultProvider = findProvider(providers, defaultProvider);
    const selectedProvider =
      (configuredDefaultProvider &&
      isConfiguredProvider(configuredDefaultProvider)
        ? configuredDefaultProvider
        : null) ??
      providers.find(isConfiguredProvider) ??
      configuredDefaultProvider;
    const selectedProviderId =
      readString(selectedProvider, "id")?.trim() ?? defaultProvider.trim();
    const normalizedSelectedProviderId =
      normalizeProviderIdentity(selectedProviderId);
    const selectedModel = selectedProviderId
      ? models.find((model) => {
          const providerId = normalizeProviderIdentity(
            decodeModelRouteSelector(model.id)?.providerId,
          );
          return providerId === normalizedSelectedProviderId;
        })
      : undefined;
    const modelName = selectedModel?.model;

    return {
      provider_configured: selectedProvider
        ? isConfiguredProvider(selectedProvider)
        : false,
      provider_name: readString(selectedProvider, "name") ?? selectedProviderId,
      provider_selector: selectedProviderId || undefined,
      model_name: modelName ?? undefined,
    };
  }

  async #getDefaultProvider(): Promise<string> {
    const [config, providers] = await Promise.all([
      this.#readConfig(),
      this.#listModelProviders(),
    ]);
    return resolveCurrentDefaultProvider(config.default_provider, providers);
  }

  async #getExperimentalConfig(): Promise<Record<string, unknown>> {
    const config = await this.#readConfig();
    return normalizeExperimentalConfig(config.experimental);
  }

  async #saveExperimentalConfig(args: HostArgs): Promise<null> {
    const request = readRequest(args);
    const experimentalConfig =
      readRecord(request, "experimentalConfig") ??
      readRecord(request, "experimental_config");
    if (!experimentalConfig) {
      throw new Error("save_experimental_config requires experimentalConfig");
    }
    const config = await this.#readConfig();
    return await this.#saveConfig({
      config: {
        ...config,
        experimental: normalizeExperimentalConfig(experimentalConfig),
      },
    });
  }

  async #listModelProviders(): Promise<unknown[]> {
    const response = await this.#appServerRequest<ModelProviderListResponse>(
      METHOD_MODEL_PROVIDER_LIST,
    );
    return response.providers ?? [];
  }

  async #listModels(
    params: AppServerParams = { includeHidden: true },
  ): Promise<Model[]> {
    const models: Model[] = [];
    const seenCursors = new Set<string>();
    let cursor: string | null = null;
    do {
      const response: ModelListResponse =
        await this.#appServerRequest<ModelListResponse>(METHOD_MODEL_LIST, {
          ...params,
          ...(cursor ? { cursor } : {}),
        });
      models.push(...response.data);
      cursor = response.nextCursor ?? null;
      if (cursor && seenCursors.has(cursor)) {
        throw new Error(`model/list repeated cursor: ${cursor}`);
      }
      if (cursor) {
        seenCursors.add(cursor);
      }
    } while (cursor);
    return models;
  }

  async #listWorkspaces(): Promise<unknown[]> {
    const response = await this.#appServerRequest<WorkspaceListResponse>(
      METHOD_WORKSPACE_LIST,
    );
    return response.workspaces;
  }

  async #readWorkspace(args: HostArgs): Promise<unknown | null> {
    const request = readRequest(args);
    const id = readString(request, "id") ?? readString(args, "id");
    if (!id) {
      return null;
    }
    const response = await this.#appServerRequest<WorkspaceReadResponse>(
      METHOD_WORKSPACE_READ,
      { id },
    );
    return response.workspace ?? null;
  }

  async #readWorkspaceByPath(args: HostArgs): Promise<unknown | null> {
    const request = readRequest(args);
    const rootPath =
      readString(request, "rootPath") ??
      readString(request, "root_path") ??
      readString(args, "rootPath") ??
      readString(args, "root_path");
    if (!rootPath) {
      return null;
    }
    const response = await this.#appServerRequest<WorkspaceReadResponse>(
      METHOD_WORKSPACE_BY_PATH_READ,
      { rootPath },
    );
    return response.workspace ?? null;
  }

  async #readDefaultWorkspace(): Promise<unknown | null> {
    const response = await this.#appServerRequest<WorkspaceReadResponse>(
      METHOD_WORKSPACE_DEFAULT_READ,
    );
    return response.workspace ?? null;
  }

  async #ensureDefaultWorkspace(): Promise<unknown> {
    const response = await this.#appServerRequest<WorkspaceReadResponse>(
      METHOD_WORKSPACE_DEFAULT_ENSURE,
    );
    if (!response.workspace) {
      throw new Error("workspace/default/ensure returned no workspace");
    }
    return response.workspace;
  }

  async #readWorkspaceProjectsRoot(): Promise<string> {
    const response =
      await this.#appServerRequest<WorkspaceProjectsRootReadResponse>(
        METHOD_WORKSPACE_PROJECTS_ROOT_READ,
      );
    return response.rootPath;
  }

  async #resolveWorkspaceProjectPath(args: HostArgs): Promise<string> {
    const request = readRequest(args);
    const name =
      readString(request, "name") ?? readString(args, "name") ?? "untitled";
    const parentRootPath =
      readString(request, "parentRootPath") ??
      readString(request, "parent_root_path") ??
      readString(args, "parentRootPath") ??
      readString(args, "parent_root_path");
    const response =
      await this.#appServerRequest<WorkspaceProjectPathResolveResponse>(
        METHOD_WORKSPACE_PROJECT_PATH_RESOLVE,
        {
          name,
          ...(parentRootPath ? { parentRootPath } : {}),
        },
      );
    return response.rootPath;
  }

  async #ensureWorkspace(args: HostArgs): Promise<unknown> {
    const request = readRequest(args);
    const name =
      readString(request, "name") ?? readString(args, "name") ?? "untitled";
    const rootPath =
      readString(request, "rootPath") ??
      readString(request, "root_path") ??
      readString(args, "rootPath") ??
      readString(args, "root_path");
    if (!rootPath) {
      throw new Error("workspace_ensure requires rootPath");
    }
    const workspaceType =
      readString(request, "workspaceType") ??
      readString(request, "workspace_type") ??
      readString(args, "workspaceType") ??
      readString(args, "workspace_type");
    const response =
      await this.#appServerRequest<WorkspaceEnsureProjectResponse>(
        METHOD_WORKSPACE_ENSURE,
        {
          name,
          rootPath,
          ...(workspaceType ? { workspaceType } : {}),
        },
      );
    return response.workspace;
  }

  async #ensureDefaultWorkspaceReady(): Promise<unknown | null> {
    const workspace = await this.#ensureDefaultWorkspace();
    const id = readString(workspace, "id");
    if (!id) {
      return null;
    }
    const response = await this.#appServerRequest<WorkspaceEnsureReadyResponse>(
      METHOD_WORKSPACE_ENSURE_READY,
      { id },
    );
    return response.result;
  }

  async #ensureWorkspaceReady(args: HostArgs): Promise<unknown> {
    const request = readRequest(args);
    const id = readString(request, "id") ?? readString(args, "id");
    if (!id) {
      throw new Error("workspace_ensure_ready requires id");
    }
    const response = await this.#appServerRequest<WorkspaceEnsureReadyResponse>(
      METHOD_WORKSPACE_ENSURE_READY,
      { id },
    );
    return response.result;
  }

  async #readConfig(): Promise<Record<string, unknown>> {
    return await this.#appConfigHost.readConfig();
  }

  async #saveConfig(args: HostArgs): Promise<null> {
    return await this.#appConfigHost.saveConfig(args);
  }

  #reportFrontendDebugLog(args: HostArgs): void {
    const report = readRecord(args, "report");
    const level = readString(report, "level") ?? "info";
    const message = readString(report, "message") ?? "";
    safeWriteElectronHostLog("log", `[electron-renderer:${level}] ${message}`);
  }

  #reportFrontendCrash(args: HostArgs): void {
    const report = readRecord(args, "report") ?? {};
    const message = readString(report, "message") ?? "renderer crash report";
    safeWriteElectronHostLog(
      "error",
      "[electron-renderer:crash]",
      message,
      report,
    );
  }

}

function readRecord(
  value: unknown,
  key: string,
): Record<string, unknown> | null {
  const record = toRecord(value);
  if (!record) {
    return null;
  }
  const next = record[key];
  return next && typeof next === "object" && !Array.isArray(next)
    ? (next as Record<string, unknown>)
    : null;
}

function readRequest(value: unknown): Record<string, unknown> {
  return readRecord(value, "request") ?? toRecord(value) ?? {};
}

function readString(value: unknown, key: string): string | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const next = (value as Record<string, unknown>)[key];
  return typeof next === "string" && next.trim() ? next.trim() : null;
}

function readRequiredString(value: unknown, key: string): string {
  const next = readString(value, key);
  if (!next) {
    throw new Error(`Missing required string field: ${key}`);
  }
  return next;
}

function readRequiredRawString(value: unknown, key: string): string {
  const record = toRecord(value);
  const next = record?.[key];
  if (typeof next !== "string") {
    throw new Error(`Missing required string field: ${key}`);
  }
  return next;
}

function readRequiredAbsolutePath(value: unknown, key: string): string {
  const next = readRequiredString(value, key);
  if (!path.isAbsolute(next)) {
    throw new Error(`${key} 必须是绝对路径`);
  }
  return next;
}

function readProjectPathOpenTool(
  value: Record<string, unknown>,
): ProjectPathOpenTool {
  const tool = readRequiredString(value, "tool");
  if (
    tool === "vscode" ||
    tool === "cursor" ||
    tool === "terminal" ||
    tool === "finder"
  ) {
    return tool;
  }
  throw new Error(`不支持的项目打开工具: ${tool}`);
}

function readBoolean(value: unknown, key: string): boolean | null {
  const record = toRecord(value);
  if (!record || typeof record[key] !== "boolean") {
    return null;
  }
  return record[key];
}

function findProvider(
  providers: unknown[],
  providerId: string,
): Record<string, unknown> | null {
  const normalizedProviderId = normalizeProviderIdentity(providerId);
  if (!normalizedProviderId) {
    return null;
  }
  return (
    (providers.find((provider) => {
      const record = toRecord(provider);
      const id = normalizeProviderIdentity(readString(record, "id"));
      const name = normalizeProviderIdentity(readString(record, "name"));
      return (
        record && (id === normalizedProviderId || name === normalizedProviderId)
      );
    }) as Record<string, unknown> | undefined) ?? null
  );
}

function normalizeProviderIdentity(value: string | null | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

function resolveCurrentDefaultProvider(
  value: unknown,
  providers: unknown[],
): string {
  if (typeof value !== "string") {
    return readString(providers.find(isConfiguredProvider), "id") ?? "";
  }
  const provider = value.trim();
  const configuredProvider = findProvider(providers, provider);
  if (configuredProvider && isConfiguredProvider(configuredProvider)) {
    return readString(configuredProvider, "id") ?? provider;
  }
  return readString(providers.find(isConfiguredProvider), "id") ?? "";
}

function isConfiguredProvider(
  provider: unknown,
): provider is Record<string, unknown> {
  const record = toRecord(provider);
  if (!record) {
    return false;
  }
  const enabled = record.enabled !== false;
  const apiKeyCount = record.api_key_count;
  return enabled && typeof apiKeyCount === "number" && apiKeyCount > 0;
}

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function normalizeExperimentalConfig(value: unknown): Record<string, unknown> {
  const record = toRecord(value) ?? {};
  const webmcp = toRecord(record.webmcp);
  return {
    ...record,
    webmcp: {
      ...webmcp,
      enabled: readBoolean(webmcp, "enabled") ?? false,
    },
  };
}

function normalizeCallbackBridgeValue(value: string | null): string | null {
  const normalized = value?.trim() ?? "";
  return normalized ? normalized : null;
}

function buildOemCloudOAuthCallbackPayload(
  requestUrl: URL,
): Record<string, string | null> {
  const params = requestUrl.searchParams;
  return {
    sourcePath: requestUrl.pathname,
    tenantId: normalizeCallbackBridgeValue(
      params.get("tenantId") ?? params.get("tenant_id"),
    ),
    token: normalizeCallbackBridgeValue(params.get("token")),
    next: normalizeCallbackBridgeValue(params.get("next")),
    error: normalizeCallbackBridgeValue(params.get("error")),
    deviceCode: normalizeCallbackBridgeValue(
      params.get("deviceCode") ?? params.get("device_code"),
    ),
    status: normalizeCallbackBridgeValue(params.get("status")),
  };
}

function shouldEmitOemCloudOAuthCallback(
  payload: Record<string, string | null>,
): boolean {
  return Boolean(
    payload.tenantId ||
    payload.token ||
    payload.error ||
    payload.deviceCode ||
    payload.status,
  );
}

type ElectronHostLogMethod = "log" | "error";
const CLOSED_OUTPUT_STREAM_ERROR_HANDLER_INSTALLED = Symbol.for(
  "lime.electron.hostCommands.closedOutputStreamErrorHandlerInstalled",
);

type ElectronHostOutputStream = {
  [CLOSED_OUTPUT_STREAM_ERROR_HANDLER_INSTALLED]?: boolean;
  on?: (event: "error", listener: (error: unknown) => void) => unknown;
};

function safeWriteElectronHostLog(
  method: ElectronHostLogMethod,
  ...args: unknown[]
): void {
  try {
    console[method](...args);
  } catch (error) {
    if (!isClosedOutputStreamError(error)) {
      throw error;
    }
  }
}

function installClosedOutputStreamErrorHandler(stream: unknown): void {
  if (!stream || typeof stream !== "object") {
    return;
  }

  const outputStream = stream as ElectronHostOutputStream;
  if (
    outputStream[CLOSED_OUTPUT_STREAM_ERROR_HANDLER_INSTALLED] ||
    typeof outputStream.on !== "function"
  ) {
    return;
  }

  Object.defineProperty(
    outputStream,
    CLOSED_OUTPUT_STREAM_ERROR_HANDLER_INSTALLED,
    {
      value: true,
    },
  );
  outputStream.on("error", (error: unknown) => {
    if (!isClosedOutputStreamError(error)) {
      throw error;
    }
  });
}

function isClosedOutputStreamError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const code = "code" in error ? String(error.code) : "";
  return (
    code === "EPIPE" ||
    code === "ERR_STREAM_DESTROYED" ||
    code === "ERR_STREAM_WRITE_AFTER_END"
  );
}

installClosedOutputStreamErrorHandler(process.stdout);
installClosedOutputStreamErrorHandler(process.stderr);
