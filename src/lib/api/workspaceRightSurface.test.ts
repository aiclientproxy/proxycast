import { describe, expect, it, vi } from "vitest";
import {
  consumeWorkspaceRightSurfacePending,
  dismissWorkspaceRightSurfacePending,
  drainWorkspaceRightSurfacePendingChangedNotifications,
  listWorkspaceRightSurfacePending,
  readWorkspaceRightSurfacePendingChangedNotification,
  requestWorkspaceRightSurface,
  type WorkspaceRightSurfaceAppServerClient,
} from "./workspaceRightSurface";

const pendingRequest = {
  requestId: "right-surface:req-1",
  workspaceId: "workspace-main",
  workspaceRoot: "/workspace/project",
  sessionId: "session-main",
  surfaceKind: "browser",
  origin: "agent",
  priority: "normal",
  status: "pending",
  reason: "browser workspace requested",
  candidateId: "browser:session-main",
  requestedAt: "2026-06-23T00:00:00.000Z",
  expiresAt: null,
  ttlMs: 30000,
  metadata: {
    source: "browser",
  },
};

function createAppServerClient(result: unknown): WorkspaceRightSurfaceAppServerClient {
  return {
    request: vi.fn().mockResolvedValue({ result }),
  };
}

describe("workspaceRightSurface API", () => {
  it("Right Surface 请求应走 App Server current method", async () => {
    const appServerClient = createAppServerClient({
      status: "queued",
      requestId: pendingRequest.requestId,
      pending: pendingRequest,
    });

    await expect(
      requestWorkspaceRightSurface(
        {
          workspaceId: "workspace-main",
          workspaceRoot: "/workspace/project",
          sessionId: "session-main",
          surfaceKind: "browser",
          origin: "agent",
          priority: "normal",
          reason: "browser workspace requested",
          candidateId: "browser:session-main",
        },
        { appServerClient },
      ),
    ).resolves.toMatchObject({
      requestId: pendingRequest.requestId,
      pending: {
        surfaceKind: "browser",
        origin: "agent",
      },
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "workspaceRightSurface/request",
      {
        workspaceId: "workspace-main",
        workspaceRoot: "/workspace/project",
        sessionId: "session-main",
        surfaceKind: "browser",
        origin: "agent",
        priority: "normal",
        reason: "browser workspace requested",
        candidateId: "browser:session-main",
      },
    );
  });

  it("pending 列表应走 App Server current method 并默认传空参数", async () => {
    const appServerClient = createAppServerClient({
      pending: [pendingRequest],
    });

    await expect(
      listWorkspaceRightSurfacePending(undefined, { appServerClient }),
    ).resolves.toEqual({
      pending: [pendingRequest],
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "workspaceRightSurface/pending/list",
      {},
    );
  });

  it("pending 消费应走 App Server current method", async () => {
    const appServerClient = createAppServerClient({
      status: "consumed",
      consumedRequestIds: ["right-surface:req-1"],
      missingRequestIds: ["right-surface:missing"],
    });

    await expect(
      consumeWorkspaceRightSurfacePending(
        {
          requestId: "right-surface:req-1",
          requestIds: ["right-surface:missing"],
        },
        { appServerClient },
      ),
    ).resolves.toEqual({
      status: "consumed",
      consumedRequestIds: ["right-surface:req-1"],
      missingRequestIds: ["right-surface:missing"],
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "workspaceRightSurface/pending/consume",
      {
        requestId: "right-surface:req-1",
        requestIds: ["right-surface:missing"],
      },
    );
  });

  it("pending 显式忽略应走 App Server current method", async () => {
    const appServerClient = createAppServerClient({
      status: "dismissed",
      dismissedRequestIds: ["right-surface:req-1"],
      missingRequestIds: ["right-surface:missing"],
    });

    await expect(
      dismissWorkspaceRightSurfacePending(
        {
          requestId: "right-surface:req-1",
          requestIds: ["right-surface:missing"],
          reason: "user_closed_surface",
        },
        { appServerClient },
      ),
    ).resolves.toEqual({
      status: "dismissed",
      dismissedRequestIds: ["right-surface:req-1"],
      missingRequestIds: ["right-surface:missing"],
    });

    expect(appServerClient.request).toHaveBeenCalledWith(
      "workspaceRightSurface/pending/dismiss",
      {
        requestId: "right-surface:req-1",
        requestIds: ["right-surface:missing"],
        reason: "user_closed_surface",
      },
    );
  });

  it("pending 列表返回半截数据时应 fail closed", async () => {
    const appServerClient = createAppServerClient({
      pending: [{ requestId: "right-surface:req-1" }],
    });

    await expect(
      listWorkspaceRightSurfacePending(
        { workspaceId: "workspace-main" },
        { appServerClient },
      ),
    ).rejects.toThrow(
      "App Server workspaceRightSurface/pending/list did not return valid pending requests",
    );
  });

  it("pending 消费返回半截数据时应 fail closed", async () => {
    const appServerClient = createAppServerClient({
      status: "consumed",
      consumedRequestIds: ["right-surface:req-1"],
    });

    await expect(
      consumeWorkspaceRightSurfacePending(
        { requestId: "right-surface:req-1" },
        { appServerClient },
      ),
    ).rejects.toThrow(
      "App Server workspaceRightSurface/pending/consume did not return consumed request ids",
    );
  });

  it("pending 显式忽略返回半截数据时应 fail closed", async () => {
    const appServerClient = createAppServerClient({
      status: "dismissed",
      dismissedRequestIds: ["right-surface:req-1"],
    });

    await expect(
      dismissWorkspaceRightSurfacePending(
        { requestId: "right-surface:req-1" },
        { appServerClient },
      ),
    ).rejects.toThrow(
      "App Server workspaceRightSurface/pending/dismiss did not return dismissed request ids",
    );
  });

  it("request 返回缺少 pending 时应 fail closed", async () => {
    const appServerClient = createAppServerClient({
      status: "queued",
      requestId: "right-surface:req-1",
    });

    await expect(
      requestWorkspaceRightSurface(
        {
          surfaceKind: "files",
          origin: "skill",
        },
        { appServerClient },
      ),
    ).rejects.toThrow(
      "App Server workspaceRightSurface/request did not return a valid pending request",
    );
  });

  it("pendingChanged notification parser 应只接受 Right Surface notification", () => {
    expect(
      readWorkspaceRightSurfacePendingChangedNotification({
        method: "workspaceRightSurface/pendingChanged",
        params: {
          changeType: "requested",
          requestIds: [pendingRequest.requestId],
          pending: [pendingRequest],
        },
      }),
    ).toMatchObject({
      method: "workspaceRightSurface/pendingChanged",
      params: {
        changeType: "requested",
        requestIds: [pendingRequest.requestId],
      },
    });

    expect(
      readWorkspaceRightSurfacePendingChangedNotification({
        method: "agentSession/event",
        params: {},
      }),
    ).toBeNull();
  });

  it("pendingChanged drain 应通过 App Server drainEvents 读取 notification params", async () => {
    const appServerClient = {
      drainEvents: vi.fn().mockResolvedValue([
        {
          method: "agentSession/event",
          params: {},
        },
        {
          method: "workspaceRightSurface/pendingChanged",
          params: {
            changeType: "dismissed",
            dismissedRequestIds: [pendingRequest.requestId],
          },
        },
      ]),
    };

    await expect(
      drainWorkspaceRightSurfacePendingChangedNotifications(5, {
        appServerClient,
      }),
    ).resolves.toEqual([
      {
        changeType: "dismissed",
        dismissedRequestIds: [pendingRequest.requestId],
      },
    ]);

    expect(appServerClient.drainEvents).toHaveBeenCalledWith(5);
  });
});
