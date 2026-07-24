import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import {
  PLUGIN_BRIDGE_PROTOCOL,
  PLUGIN_BRIDGE_VERSION,
} from "../runtime/hostBridge";
import contentFactoryFixture from "../testing/fixtures/content-factory-app.json";
import { buildInstalledPluginState } from "../install/installedAppState";
import { buildInstalledAppPreview } from "../install/installedAppPreview";
import { buildPluginLabResolvedSetupState } from "../install/labInstallFlow";
import { buildPackageIdentity } from "../install/packageIdentity";
import { buildWorkflowRuntimeCapabilityProfile } from "../testing/workflowRuntimeCapabilityProfile";
import type { AppManifest, InstalledPluginState } from "../types";
import { PluginRuntimePage } from "./PluginRuntimePage";

const LOCAL_APP_DIR = "/tmp/lime/content-factory-app";

const contentFactoryBaseManifest = contentFactoryFixture as AppManifest;

function buildContentFactoryUiRuntimeManifest(): AppManifest {
  return {
    ...contentFactoryBaseManifest,
    version: "0.3.0",
    runtimePackage: {
      ...contentFactoryBaseManifest.runtimePackage,
      ui: {
        path: "./dist/index.html",
      },
    },
    entries: [
      {
        key: "dashboard",
        kind: "page",
        title: "项目首页",
        route: "/dashboard",
        requiredCapabilities: ["lime.ui", "lime.agent", "lime.storage"],
      },
    ],
    knowledgeTemplates: [
      {
        key: "project_knowledge",
        type: "project",
        required: true,
      },
    ],
    artifacts: [
      ...(contentFactoryBaseManifest.artifacts ?? []).map((artifact) => ({
        ...artifact,
      })),
      {
        key: "content_table",
        title: "内容表",
        type: "content_table",
      },
    ],
    evals: [
      ...(contentFactoryBaseManifest.evals ?? []).map((evalRule) => ({
        ...evalRule,
      })),
      {
        key: "fact_grounding",
        kind: "fact_grounding",
      },
    ],
  };
}

export const contentFactoryManifest = buildContentFactoryUiRuntimeManifest();

const hoisted = vi.hoisted(() => ({
  apiMocks: {
    getPluginCloudCatalog: vi.fn(),
    listInstalledPlugins: vi.fn(),
    startPluginUiRuntime: vi.fn(),
  },
  appServerClientMocks: {
    createAppServerClient: vi.fn(),
    startSession: vi.fn(),
    startTurn: vi.fn(),
    steerTurn: vi.fn(),
    readSession: vi.fn(),
    readThread: vi.fn(),
    cancelTurn: vi.fn(),
    readWorkflow: vi.fn(),
  },
  toastMocks: {
    toast: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
  },
  t: (key: string, params?: Record<string, unknown>) => {
    const labels: Record<string, string> = {
      "plugin.apps.runtime.loading": "正在加载已安装 App...",
      "plugin.apps.runtime.empty":
        "还没有可使用的已安装 Plugin。请先到 Plugins 安装。",
      "plugin.apps.runtime.opening": `正在打开 ${String(params?.name)}...`,
      "plugin.apps.runtime.openFailed": "App 打开失败",
      "plugin.apps.runtime.retry": "重新打开",
      "plugin.apps.runtime.unavailable": "该 Plugin 暂不可用。",
      "plugin.apps.runtime.appInfo.toggle": "查看 App 信息",
      "plugin.apps.runtime.appInfo.version": `当前版本 v${String(params?.version)}`,
      "plugin.apps.runtime.appInfo.upgradeBadge": "可升级",
      "plugin.apps.runtime.appInfo.source": "来源",
      "plugin.apps.runtime.appInfo.source.cloud": "云端",
      "plugin.apps.runtime.appInfo.source.local": "本地",
      "plugin.apps.runtime.appInfo.latestVersion": "最新版本",
      "plugin.apps.runtime.appInfo.versionValue": `v${String(params?.version)}`,
      "plugin.apps.runtime.appInfo.entry": "入口",
      "plugin.apps.runtime.agentRun.aria": "Host AI 运行面板",
      "plugin.apps.runtime.agentRun.badge": `${String(params?.app)} 的 AI 同事`,
      "plugin.apps.runtime.agentRun.titleFallback": "Lime AI 运行现场",
      "plugin.apps.runtime.agentRun.subtitle":
        "思考、执行、Skill、模型、Token、费用和证据由 Lime Host 统一承载。",
      "plugin.apps.runtime.agentRun.close": "关闭 AI 运行面板",
      "plugin.apps.runtime.agentRun.expand": "查看运行现场",
      "plugin.apps.runtime.agentRun.collapse": "收起运行面板",
      "plugin.apps.runtime.agentRun.taskId": "任务 ID",
      "plugin.apps.runtime.agentRun.bridgeAction": "业务动作",
      "plugin.apps.runtime.agentRun.emptyValue": "待回写",
      "plugin.apps.runtime.agentRun.metric.model": "模型",
      "plugin.apps.runtime.agentRun.metric.tokens": "Token",
      "plugin.apps.runtime.agentRun.metric.cost": "费用",
      "plugin.apps.runtime.agentRun.metric.skills": "Skills",
      "plugin.apps.runtime.agentRun.facts.confirmations": "待确认",
      "plugin.apps.runtime.agentRun.facts.confirmations.empty": "暂无待确认项",
      "plugin.apps.runtime.agentRun.facts.confirmations.itemFallback":
        "待确认事项",
      "plugin.apps.runtime.agentRun.facts.artifacts": "交付物",
      "plugin.apps.runtime.agentRun.facts.artifacts.empty": "暂无交付物",
      "plugin.apps.runtime.agentRun.facts.artifacts.itemFallback":
        "交付物已生成",
      "plugin.apps.runtime.agentRun.facts.evidence": "证据",
      "plugin.apps.runtime.agentRun.facts.evidence.empty": "暂无证据",
      "plugin.apps.runtime.agentRun.facts.evidence.itemFallback": "证据已记录",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.completed.message": `宿主已用 ${String(params?.provider)} / ${String(params?.model)} 生成 ${String(params?.outputCount)} 个受控产物；正文保持 Generation Brief 边界。`,
      "plugin.apps.runtime.agentRun.hostManagedGeneration.completed.title":
        "宿主托管生成已完成",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.requested.message":
        "宿主正在为插件生成受控正文；过程说明可带 Soul，正式正文仍走 Generation Brief。",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.requested.title":
        "宿主托管生成中",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.skipped.message":
        "插件声明了托管生成入口，但本轮没有需要宿主生成的正文。",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.skipped.title":
        "宿主托管生成已跳过",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.unavailable.message":
        "宿主托管生成不可用，插件会按 fail-closed 边界处理，不伪造正文。",
      "plugin.apps.runtime.agentRun.hostManagedGeneration.unavailable.title":
        "宿主托管生成不可用",
      "plugin.apps.runtime.agentRun.timeline.running":
        "运行中，点击折叠或展开过程",
      "plugin.apps.runtime.agentRun.timeline.collapsed":
        "运行过程已折叠，点击查看完整现场",
      "plugin.apps.runtime.agentRun.timeline.event": "运行事件",
      "plugin.apps.runtime.agentRun.timeline.empty":
        "等待 AgentRuntime 回写运行过程。",
      "plugin.apps.runtime.agentRun.thinking": "思考过程",
      "plugin.apps.runtime.agentRun.execution": "执行过程",
      "plugin.apps.runtime.agentRun.output": "成稿流式输出",
      "plugin.apps.runtime.agentRun.action.approve": "确认",
      "plugin.apps.runtime.agentRun.action.reject": "拒绝",
      "plugin.apps.runtime.agentRun.action.answer": "补充说明",
      "plugin.apps.runtime.agentRun.action.edit": "编辑",
      "plugin.apps.runtime.agentRun.action.retry": "重试",
      "plugin.apps.runtime.agentRun.action.interrupt": "中断",
      "plugin.apps.runtime.agentRun.action.stop": "停止",
      "plugin.apps.toast.failed": "Plugin 操作失败",
    };
    return labels[key] ?? key;
  },
}));

export const apiMocks = hoisted.apiMocks;
export const appServerClientMocks = hoisted.appServerClientMocks;

vi.mock("@/lib/api/plugins", () => ({
  PLUGINS_CHANGED_EVENT: "lime:plugins-changed",
  getPluginCloudCatalog: hoisted.apiMocks.getPluginCloudCatalog,
  listInstalledPlugins: hoisted.apiMocks.listInstalledPlugins,
  startPluginUiRuntime: hoisted.apiMocks.startPluginUiRuntime,
}));

vi.mock("@/lib/api/appServer", () => ({
  createAppServerClient: hoisted.appServerClientMocks.createAppServerClient,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: hoisted.t,
  }),
}));

vi.mock("sonner", () => ({
  toast: Object.assign(hoisted.toastMocks.toast, {
    error: hoisted.toastMocks.error,
    success: hoisted.toastMocks.success,
  }),
}));

interface MountedPage {
  container: HTMLDivElement;
  root: Root;
}

const mountedPages: MountedPage[] = [];

export function buildReadyState(
  params: {
    setupResolved?: boolean;
    manifestPatch?: Partial<AppManifest>;
  } = {},
): InstalledPluginState {
  const manifest = {
    ...contentFactoryManifest,
    ...(params.manifestPatch ?? {}),
  } as AppManifest;
  const loadedAt = "2026-05-15T00:00:00.000Z";
  const identity = buildPackageIdentity({
    manifest,
    sourceKind: "local_folder",
    sourceUri: LOCAL_APP_DIR,
    loadedAt,
  });
  const profile = buildWorkflowRuntimeCapabilityProfile({
    realAdapterEnabled: true,
    uiRuntimeEnabled: true,
    workerRuntimeEnabled: true,
  });
  const setupPreview = buildInstalledAppPreview({
    fixture: manifest,
    identity,
    profile,
    loadedAt,
    checkedAt: loadedAt,
    generatedAt: loadedAt,
  });
  const setup =
    params.setupResolved === false
      ? undefined
      : buildPluginLabResolvedSetupState(setupPreview.projection);
  const preview = buildInstalledAppPreview({
    fixture: manifest,
    identity,
    setup,
    profile,
    loadedAt,
    checkedAt: loadedAt,
    generatedAt: loadedAt,
  });

  return buildInstalledPluginState({
    preview,
    setup,
    installedAt: loadedAt,
    updatedAt: loadedAt,
  });
}

export async function renderPage(state = buildReadyState()) {
  apiMocks.listInstalledPlugins.mockResolvedValue({
    states: [state],
    issues: [],
  });
  apiMocks.startPluginUiRuntime.mockResolvedValue({
    appId: state.appId,
    status: "running",
    baseUrl: "http://127.0.0.1:4199",
    entryUrl: "http://127.0.0.1:4199/dashboard",
    port: 4199,
    pid: 41990,
    entryKey: "dashboard",
    route: "/dashboard",
  });

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(
      <PluginRuntimePage
        pageParams={{
          appId: state.appId,
          entryKey: "dashboard",
          projectId: "workspace-1",
          launchRequestKey: 1,
        }}
      />,
    );
    await Promise.resolve();
    await Promise.resolve();
  });

  mountedPages.push({ container, root });
  return container;
}

export async function unmountLastRenderedPage() {
  const mounted = mountedPages.pop();
  if (!mounted) {
    throw new Error("mounted page not found");
  }
  await act(async () => {
    mounted.root.unmount();
  });
  mounted.container.remove();
}

export async function flush(times = 8) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

export function getRuntimeFrame(container: HTMLElement): HTMLIFrameElement {
  const frame = container.querySelector<HTMLIFrameElement>(
    '[data-testid="plugin-runtime-frame"]',
  );
  if (!frame) {
    throw new Error("runtime frame not found");
  }
  return frame;
}

function buildAppBridgeMessage(
  type: string,
  payload?: unknown,
  requestId = "req-1",
) {
  return {
    protocol: PLUGIN_BRIDGE_PROTOCOL,
    version: PLUGIN_BRIDGE_VERSION,
    type,
    requestId,
    appId: "content-factory-app",
    entryKey: "dashboard",
    payload,
  };
}

export async function dispatchBridgeMessage(
  frame: HTMLIFrameElement,
  type: string,
  payload?: unknown,
  requestId = "req-1",
  origin = "http://127.0.0.1:4199",
) {
  await act(async () => {
    window.dispatchEvent(
      new MessageEvent("message", {
        data: buildAppBridgeMessage(type, payload, requestId),
        origin,
        source: frame.contentWindow,
      }),
    );
    await Promise.resolve();
  });
}

export function usePluginRuntimePageTestLifecycle() {
  beforeEach(() => {
    document.documentElement.lang = "zh-CN";
    document.documentElement.dir = "ltr";
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    let runtimeTaskCounter = 0;
    appServerClientMocks.createAppServerClient.mockReturnValue({
      startSession: appServerClientMocks.startSession,
      startTurn: appServerClientMocks.startTurn,
      steerTurn: appServerClientMocks.steerTurn,
      readSession: appServerClientMocks.readSession,
      readThread: appServerClientMocks.readThread,
      cancelTurn: appServerClientMocks.cancelTurn,
      readWorkflow: appServerClientMocks.readWorkflow,
    });
    appServerClientMocks.startSession.mockImplementation(async (request) => ({
      id: 1,
      result: {
        approvalPolicy: null,
        approvalsReviewer: null,
        cwd: request.cwd ?? "",
        model: request.model ?? "unknown",
        modelProvider: request.modelProvider ?? "unknown",
        sandbox: null,
        thread: {
          cliVersion: "0.0.0-test",
          createdAt: 1_747_267_200,
          cwd: request.cwd ?? "",
          ephemeral: false,
          id: "plugin-thread-1",
          modelProvider: request.modelProvider ?? "unknown",
          preview: request.serviceName ?? "Plugin runtime task",
          sessionId: "plugin-session-1",
          source: request.threadSource ?? "plugin",
          status: { type: "idle" },
          threadSource: request.threadSource,
          turns: [],
          updatedAt: 1_747_267_200,
        },
      },
      response: { jsonrpc: "2.0", id: 1, result: {} },
      notifications: [],
      messages: [],
    }));
    appServerClientMocks.startTurn.mockImplementation(async (_request) => {
      runtimeTaskCounter += 1;
      return {
        id: 2,
        result: {
          turn: {
            id: `plugin-turn-${runtimeTaskCounter}`,
            items: [],
            itemsView: "full",
            status: "inProgress",
            startedAt: 1_747_267_200,
          },
        },
        response: { jsonrpc: "2.0", id: 2, result: {} },
        notifications: [],
        messages: [],
      };
    });
    appServerClientMocks.steerTurn.mockResolvedValue({
      id: 3,
      result: { turnId: "plugin-turn-1" },
      response: { jsonrpc: "2.0", id: 3, result: {} },
      notifications: [],
      messages: [],
    });
    appServerClientMocks.readSession.mockImplementation(async (request) => ({
      id: 3,
      result: {
        session: {
          sessionId: request.sessionId,
          threadId: "plugin-thread-1",
          appId: "content-factory-app",
          workspaceId: "workspace-1",
          status: "running",
          createdAt: "2026-05-15T00:00:00.000Z",
          updatedAt: "2026-05-15T00:00:01.000Z",
        },
        turns: [
          {
            turnId: "plugin-turn-1",
            sessionId: request.sessionId,
            threadId: "plugin-thread-1",
            status: "running",
          },
        ],
        detail: {
          thread_read: {
            session_id: request.sessionId,
            source: "app_server_runtime_client",
          },
        },
      },
      response: { jsonrpc: "2.0", id: 3, result: {} },
      notifications: [],
      messages: [],
    }));
    appServerClientMocks.readThread.mockImplementation(async (request) => ({
      id: 4,
      result: {
        thread: {
          cliVersion: "0.0.0-test",
          createdAt: 1_747_267_200,
          cwd: "",
          ephemeral: false,
          id: request.threadId,
          modelProvider: "unknown",
          preview: "Plugin runtime task",
          sessionId: "plugin-session-1",
          source: "plugin",
          status: { type: "active", activeFlags: [] },
          turns: [
            {
              id: "plugin-turn-1",
              items: [],
              itemsView: "full",
              status: "inProgress",
              startedAt: 1_747_267_200,
            },
          ],
          updatedAt: 1_747_267_201,
        },
      },
      response: { jsonrpc: "2.0", id: 4, result: {} },
      notifications: [],
      messages: [],
    }));
    appServerClientMocks.cancelTurn.mockResolvedValue({
      id: 5,
      result: {},
      response: { jsonrpc: "2.0", id: 5, result: {} },
      notifications: [],
      messages: [],
    });
    appServerClientMocks.readWorkflow.mockImplementation(async (request) => ({
      id: 7,
      result: {
        sessionId: request.sessionId,
        workflow: {
          activeWorkflowRunId: "plugin-workflow-run-1",
        },
        workflowRuns: [
          {
            workflowRunId: "plugin-workflow-run-1",
            status: "running",
          },
        ],
        workflowSteps: [
          {
            workflowRunId: "plugin-workflow-run-1",
            stepId: "draft",
            status: "running",
          },
        ],
      },
      response: { jsonrpc: "2.0", id: 7, result: {} },
      notifications: [],
      messages: [],
    }));
    apiMocks.getPluginCloudCatalog.mockResolvedValue({
      source: "seeded",
      payload: {
        schemaVersion: "plugin-cloud-bootstrap/v1",
        tenantId: "seeded",
        generatedAt: "2026-05-15T00:00:00.000Z",
        apps: [],
      },
    });
  });

  afterEach(() => {
    while (mountedPages.length > 0) {
      const mounted = mountedPages.pop();
      if (!mounted) {
        continue;
      }
      act(() => {
        mounted.root.unmount();
      });
      mounted.container.remove();
    }
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    delete window.__LIME_BOOTSTRAP__;
    delete window.__LIME_OEM_CLOUD__;
    delete window.__LIME_SESSION_TOKEN__;
    window.localStorage.clear();
    window.sessionStorage.clear();
    document.documentElement.removeAttribute("style");
  });
}
