import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import {
  APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  APP_SERVER_METHOD_MEDIA_READ,
  APP_SERVER_METHOD_CANCEL_REQUEST,
  APP_SERVER_METHOD_THREAD_RESUME,
  APP_SERVER_METHOD_TURN_INTERRUPT,
  APP_SERVER_METHOD_TURN_START,
  APP_SERVER_METHOD_TURN_STEER,
  APP_SERVER_METHOD_THREAD_ARCHIVE,
  APP_SERVER_METHOD_THREAD_COMPACT_START,
  APP_SERVER_METHOD_THREAD_LIST,
  APP_SERVER_METHOD_THREAD_NAME_SET,
  APP_SERVER_METHOD_THREAD_READ,
  APP_SERVER_METHOD_THREAD_SHELL_COMMAND,
  APP_SERVER_METHOD_THREAD_UNARCHIVE,
  APP_SERVER_METHOD_ARTIFACT_READ,
  APP_SERVER_METHOD_CAPABILITY_LIST,
  APP_SERVER_METHOD_EVIDENCE_EXPORT,
  APP_SERVER_METHOD_EXECUTION_PROCESS_DRAIN_OUTPUT,
  APP_SERVER_METHOD_EXECUTION_PROCESS_INTERRUPT,
  APP_SERVER_METHOD_EXECUTION_PROCESS_START,
  APP_SERVER_METHOD_FILE_SYSTEM_LIST_DIRECTORY,
  APP_SERVER_METHOD_FILE_SYSTEM_READ_FILE_PREVIEW,
  APP_SERVER_METHOD_PROJECT_GIT_DIFF,
  APP_SERVER_METHOD_GATEWAY_CHANNEL_START,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_CREATE,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_PROBE,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_RESTART,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_START,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_STATUS,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_STOP,
  APP_SERVER_METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL,
  APP_SERVER_METHOD_DIAGNOSTICS_LOG_STORAGE_READ,
  APP_SERVER_METHOD_DIAGNOSTICS_SERVER_READ,
  APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ,
  APP_SERVER_METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
  APP_SERVER_METHOD_INITIALIZED,
  APP_SERVER_METHOD_INITIALIZE,
  APP_SERVER_METHOD_LOG_CLEAR,
  APP_SERVER_METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR,
  APP_SERVER_METHOD_LOG_LIST,
  APP_SERVER_METHOD_LOG_PERSISTED_TAIL,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
  APP_SERVER_METHOD_WORKFLOW_READ,
  APP_SERVER_METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  APP_SERVER_PROTOCOL_VERSION,
  AppServerClient,
  AppServerRequestAbortedError,
  AppServerRpcError,
  createAppServerRequest,
  decodeAppServerMessage,
  encodeAppServerMessage,
  expectAppServerResponse,
} from "./appServer";

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

function line(value: unknown): string {
  return `${JSON.stringify(value)}\n`;
}

describe("App Server API", () => {
  beforeEach(() => {
    vi.mocked(safeInvoke).mockReset();
  });

  it("readThread 应通过 canonical thread/read typed method", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 1,
          result: {
            thread: {
              archived: false,
              createdAtMs: 100,
              sessionId: "session-1",
              status: { type: "active" },
              threadId: "thread-1",
              turns: [],
              turnsView: "full",
              updatedAtMs: 200,
            },
          },
        }),
      ],
    });

    const client = new AppServerClient();
    const result = await client.readThread({
      threadId: "thread-1",
      turnsView: "full",
    });

    expect(vi.mocked(safeInvoke)).toHaveBeenCalledWith(
      "app_server_handle_json_lines",
      expect.objectContaining({
        request: expect.objectContaining({
          lines: [
            expect.stringContaining(
              `"method":"${APP_SERVER_METHOD_THREAD_READ}"`,
            ),
          ],
        }),
      }),
    );
    expect(result.result.thread.threadId).toBe("thread-1");
  });

  it("thread settings control 应通过 v2 typed methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({ lines: [line({ id: 1, result: {} })] })
      .mockResolvedValueOnce({ lines: [line({ id: 2, result: {} })] });

    const client = new AppServerClient();
    await client.updateThreadSettings({
      threadId: "thread-1",
      collaborationMode: {
        mode: "plan",
        settings: {
          model: "gpt-5.4",
          reasoning_effort: "high",
          developer_instructions: null,
        },
      },
    });
    await client.setThreadMemoryMode({
      threadId: "thread-1",
      mode: "disabled",
    });

    expect(vi.mocked(safeInvoke).mock.calls).toEqual([
      [
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id: 1,
                method: "thread/settings/update",
                params: {
                  threadId: "thread-1",
                  collaborationMode: {
                    mode: "plan",
                    settings: {
                      model: "gpt-5.4",
                      reasoning_effort: "high",
                      developer_instructions: null,
                    },
                  },
                },
              }),
            ],
          },
        },
      ],
      [
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id: 2,
                method: "thread/memoryMode/set",
                params: { threadId: "thread-1", mode: "disabled" },
              }),
            ],
          },
        },
      ],
    ]);
  });

  it("thread rename 应通过 v2 thread/name/set", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [line({ id: 1, result: {} })],
    });

    await new AppServerClient().setThreadName({
      threadId: "thread-1",
      name: "新标题",
    });

    expect(vi.mocked(safeInvoke)).toHaveBeenCalledWith(
      "app_server_handle_json_lines",
      expect.objectContaining({
        request: expect.objectContaining({
          lines: [
            expect.stringContaining(
              `"method":"${APP_SERVER_METHOD_THREAD_NAME_SET}"`,
            ),
          ],
        }),
      }),
    );
  });

  it("listThreads 应通过 canonical thread/list typed method", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 1,
          result: {
            data: [
              {
                archived: false,
                createdAtMs: 100,
                parentThreadId: "thread-parent",
                sessionId: "session-child",
                status: { type: "idle" },
                threadId: "thread-child",
                turns: [],
                turnsView: "summary",
                updatedAtMs: 200,
              },
            ],
          },
        }),
      ],
    });

    const client = new AppServerClient();
    const result = await client.listThreads({
      limit: 100,
      turnsView: "summary",
    });

    expect(vi.mocked(safeInvoke)).toHaveBeenCalledWith(
      "app_server_handle_json_lines",
      expect.objectContaining({
        request: expect.objectContaining({
          lines: [
            expect.stringContaining(
              `"method":"${APP_SERVER_METHOD_THREAD_LIST}"`,
            ),
          ],
        }),
      }),
    );
    expect(result.result.data[0]?.parentThreadId).toBe("thread-parent");
  });

  it("initialize 应通过 App Server JSON-RPC 命令完成握手", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 1,
            result: {
              serverInfo: {
                name: "app-server",
                version: "1.58.0",
                protocolVersion: APP_SERVER_PROTOCOL_VERSION,
              },
              platform: {
                family: "desktop",
                os: "macos",
              },
              capabilities: {
                agentSession: true,
                capabilityDiscovery: true,
                artifact: true,
                evidence: true,
                workspace: true,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({ lines: [] });

    const client = new AppServerClient();
    const result = await client.initialize({
      clientInfo: {
        name: "content_studio",
        version: "0.1.0",
      },
      capabilities: {
        eventMethods: [APP_SERVER_METHOD_AGENT_SESSION_EVENT],
      },
    });

    expect(result.result.serverInfo.name).toBe("app-server");
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 1,
              method: APP_SERVER_METHOD_INITIALIZE,
              params: {
                clientInfo: {
                  name: "content_studio",
                  version: "0.1.0",
                },
                capabilities: {
                  eventMethods: [APP_SERVER_METHOD_AGENT_SESSION_EVENT],
                },
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              method: APP_SERVER_METHOD_INITIALIZED,
              params: {},
            }),
          ],
        },
      },
    );
  });

  it("listCapabilities 应透传 sessionId scope", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 3,
          result: {
            capabilities: [
              {
                id: "session.draft.write",
                title: "Session Draft Write",
                methods: [APP_SERVER_METHOD_TURN_START],
              },
            ],
            nextCursor: "1",
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 3 });
    const result = await client.listCapabilities({
      appId: "content-studio",
      workspaceId: "default",
      sessionId: "session-1",
      limit: 1,
    });

    expect(result.result.capabilities[0].id).toBe("session.draft.write");
    expect(result.result.nextCursor).toBe("1");
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 3,
              method: APP_SERVER_METHOD_CAPABILITY_LIST,
              params: {
                appId: "content-studio",
                workspaceId: "default",
                sessionId: "session-1",
                limit: 1,
              },
            }),
          ],
        },
      },
    );
  });

  it("workspaceRightSurface methods 应通过 App Server JSON-RPC 调度右侧 surface", async () => {
    const pending = {
      requestId: "right-surface:req-1",
      workspaceId: "workspace-1",
      workspaceRoot: "/workspace/project",
      sessionId: "session-1",
      surfaceKind: "objectCanvas",
      origin: "mcpTool",
      priority: "normal",
      status: "pending",
      requestedAt: "2026-06-23T00:00:00.000Z",
    };
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 4,
            result: {
              status: "pending",
              requestId: pending.requestId,
              pending,
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 5,
            result: {
              pending: [pending],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 6,
            result: {
              status: "consumed",
              consumedRequestIds: [pending.requestId],
              missingRequestIds: ["right-surface:missing"],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 7,
            result: {
              status: "dismissed",
              dismissedRequestIds: [pending.requestId],
              missingRequestIds: ["right-surface:gone"],
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 4 });
    const requestResult = await client.requestWorkspaceRightSurface({
      workspaceId: "workspace-1",
      workspaceRoot: "/workspace/project",
      sessionId: "session-1",
      surfaceKind: "objectCanvas",
      origin: "mcpTool",
      reason: "browser assist candidate",
    });
    const listResult = await client.listWorkspaceRightSurfacePending({
      workspaceId: "workspace-1",
      surfaceKind: "objectCanvas",
    });
    const consumeResult = await client.consumeWorkspaceRightSurfacePending({
      requestId: pending.requestId,
      requestIds: ["right-surface:missing"],
    });
    const dismissResult = await client.dismissWorkspaceRightSurfacePending({
      requestId: pending.requestId,
      requestIds: ["right-surface:gone"],
      reason: "user_closed_surface",
    });

    expect(requestResult.result.requestId).toBe("right-surface:req-1");
    expect(listResult.result.pending).toHaveLength(1);
    expect(consumeResult.result.consumedRequestIds).toEqual([
      "right-surface:req-1",
    ]);
    expect(consumeResult.result.missingRequestIds).toEqual([
      "right-surface:missing",
    ]);
    expect(dismissResult.result.dismissedRequestIds).toEqual([
      "right-surface:req-1",
    ]);
    expect(dismissResult.result.missingRequestIds).toEqual([
      "right-surface:gone",
    ]);
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 4,
              method: APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
              params: {
                workspaceId: "workspace-1",
                workspaceRoot: "/workspace/project",
                sessionId: "session-1",
                surfaceKind: "objectCanvas",
                origin: "mcpTool",
                reason: "browser assist candidate",
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 5,
              method: APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
              params: {
                workspaceId: "workspace-1",
                surfaceKind: "objectCanvas",
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      3,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 6,
              method: APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
              params: {
                requestId: pending.requestId,
                requestIds: ["right-surface:missing"],
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      4,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 7,
              method: APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,
              params: {
                requestId: pending.requestId,
                requestIds: ["right-surface:gone"],
                reason: "user_closed_surface",
              },
            }),
          ],
        },
      },
    );
  });

  it("listSessions 应通过 App Server JSON-RPC thread/list", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 5,
          result: {
            sessions: [
              {
                sessionId: "session-1",
                threadId: "thread-1",
                title: "Session 1",
                model: "gpt-5.4",
                createdAt: "2026-06-09T09:00:00.000Z",
                updatedAt: "2026-06-09T09:01:00.000Z",
                archivedAt: null,
                messagesCount: 2,
              },
            ],
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 5 });
    const result = await client.listSessions({
      includeArchived: true,
      workspaceId: "workspace-1",
      limit: 10,
    });

    expect(result.result.sessions[0].sessionId).toBe("session-1");
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 5,
            method: APP_SERVER_METHOD_THREAD_LIST,
            params: {
              includeArchived: true,
              workspaceId: "workspace-1",
              limit: 10,
            },
          }),
        ],
      },
    });
  });

  it("request 应兼容 Electron safeInvoke result 包络", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      result: {
        lines: [
          line({
            id: 3,
            result: {
              capabilities: [
                {
                  id: "session.read",
                  title: "Session Read",
                  methods: [],
                },
              ],
            },
          }),
        ],
      },
    });

    const client = new AppServerClient({ initialRequestId: 3 });
    const result = await client.listCapabilities();

    expect(result.result.capabilities[0].id).toBe("session.read");
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 3,
            method: APP_SERVER_METHOD_CAPABILITY_LIST,
            params: {},
          }),
        ],
      },
    });
  });

  it("request 收到 App Server bridge diagnostic facade 时应 fail closed", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      diagnostic: {
        command: "app_server_handle_json_lines",
        source: "electron-host",
        status: "not-supported",
      },
      result: {
        lines: [],
      },
    });

    const client = new AppServerClient({ initialRequestId: 3 });

    await expect(client.listCapabilities()).rejects.toThrow(
      "app_server_handle_json_lines 尚未接入真实 App Server bridge",
    );
  });

  it("request 收到 result 内层 diagnostic facade 时应 fail closed", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      result: {
        diagnostic: {
          command: "app_server_handle_json_lines",
          source: "electron-host",
          status: "degraded",
        },
        lines: [],
      },
    });

    const client = new AppServerClient({ initialRequestId: 3 });

    await expect(client.listCapabilities()).rejects.toThrow(
      "app_server_handle_json_lines 尚未接入真实 App Server bridge",
    );
  });

  it("readArtifacts 应通过 App Server JSON-RPC 读取 artifact summary/content", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 5,
          result: {
            artifacts: [
              {
                artifactRef: "artifact-report",
                eventId: "evt-artifact-1",
                sequence: 7,
                turnId: "turn-1",
                artifactId: "artifact-report",
                path: ".app-server/artifacts/report.md",
                title: "Report",
                kind: "markdown",
                status: "ready",
                content: "# Report",
                contentStatus: "available",
                metadata: {
                  version: 2,
                },
              },
            ],
            nextCursor: "1",
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 5 });
    const result = await client.readArtifacts({
      sessionId: "session-1",
      turnId: "turn-1",
      artifactRef: "artifact-report",
      includeContent: true,
      limit: 1,
    });

    expect(result.result.artifacts[0].artifactRef).toBe("artifact-report");
    expect(result.result.artifacts[0].content).toBe("# Report");
    expect(result.result.artifacts[0].contentStatus).toBe("available");
    expect(result.result.artifacts[0].metadata).toEqual({ version: 2 });
    expect(result.result.nextCursor).toBe("1");
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 5,
            method: APP_SERVER_METHOD_ARTIFACT_READ,
            params: {
              sessionId: "session-1",
              turnId: "turn-1",
              artifactRef: "artifact-report",
              includeContent: true,
              limit: 1,
            },
          }),
        ],
      },
    });
  });

  it("readMedia 应通过 App Server JSON-RPC 读取已知 sidecar media", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 6,
          result: {
            threadId: "thread-media",
            uri: "sidecar://media/demo",
            mimeType: "image/png",
            bytes: 4,
            totalBytes: 4,
            offset: 0,
            length: 4,
            contentRange: "bytes 0-3/4",
            hasMore: false,
            sha256: "sha256:demo",
            contentBase64: "iVBORw==",
            sidecarRef: {
              ref: "sidecar://media/demo",
              kind: "media",
              relativePath: "sessions/session-media/media/demo.png",
            },
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 6 });
    const result = await client.readMedia({
      threadId: "thread-media",
      uri: "sidecar://media/demo",
      maxBytes: 1024,
    });

    expect(result.result.contentBase64).toBe("iVBORw==");
    expect(result.result.contentRange).toBe("bytes 0-3/4");
    expect(result.result.hasMore).toBe(false);
    expect(result.result.sidecarRef).toMatchObject({
      ref: "sidecar://media/demo",
      kind: "media",
    });
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 6,
              method: APP_SERVER_METHOD_MEDIA_READ,
              params: {
                threadId: "thread-media",
                uri: "sidecar://media/demo",
                maxBytes: 1024,
              },
            }),
          ],
        },
      },
    );
  });

  it("readMedia abort 后应拒绝并忽略迟到 bridge 结果", async () => {
    let resolveSafeInvoke!: (value: unknown) => void;
    const pendingSafeInvoke = new Promise((resolve) => {
      resolveSafeInvoke = resolve;
    });
    vi.mocked(safeInvoke).mockReturnValueOnce(
      pendingSafeInvoke as ReturnType<typeof safeInvoke>,
    );
    const abortController = new AbortController();

    const client = new AppServerClient({ initialRequestId: 6 });
    const result = client.readMedia(
      {
        threadId: "thread-media",
        uri: "sidecar://media/demo",
        maxBytes: 1024,
      },
      { signal: abortController.signal },
    );

    abortController.abort("preview superseded");

    await expect(result).rejects.toMatchObject({
      name: "AppServerRequestAbortedError",
      method: APP_SERVER_METHOD_MEDIA_READ,
      requestId: 6,
      reason: "preview superseded",
    });
    await expect(result).rejects.toBeInstanceOf(AppServerRequestAbortedError);
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 6,
            method: APP_SERVER_METHOD_MEDIA_READ,
            params: {
              threadId: "thread-media",
              uri: "sidecar://media/demo",
              maxBytes: 1024,
            },
          }),
        ],
      },
    });
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              method: APP_SERVER_METHOD_CANCEL_REQUEST,
              params: { id: 6 },
            }),
          ],
        },
      },
    );

    resolveSafeInvoke({
      lines: [
        line({
          id: 6,
          result: {
            sessionId: "session-media",
            uri: "sidecar://media/demo",
            mimeType: "image/png",
            bytes: 4,
            totalBytes: 4,
            offset: 0,
            length: 4,
            contentRange: "bytes 0-3/4",
            hasMore: false,
            sha256: "sha256:demo",
            contentBase64: "iVBORw==",
            sidecarRef: {
              ref: "sidecar://media/demo",
              kind: "media",
              relativePath: "sessions/session-media/media/demo.png",
            },
          },
        }),
      ],
    });
    await Promise.resolve();
  });

  it("listDirectory/readFilePreview 应通过 App Server JSON-RPC 读取文件浏览数据", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 6,
            result: {
              path: "/workspace",
              parentPath: "/",
              entries: [
                {
                  name: "README.md",
                  path: "/workspace/README.md",
                  isDir: false,
                  size: 12,
                  modifiedAt: 1,
                  isHidden: false,
                  isSymlink: false,
                },
              ],
              error: null,
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 7,
            result: {
              path: "/workspace/README.md",
              content: "# Lime",
              isBinary: false,
              size: 6,
              error: null,
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 6 });
    const listing = await client.listDirectory({ path: "/workspace" });
    const preview = await client.readFilePreview({
      path: "/workspace/README.md",
      maxSize: 1024,
    });

    expect(listing.result.entries[0].name).toBe("README.md");
    expect(preview.result.content).toBe("# Lime");
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 6,
              method: APP_SERVER_METHOD_FILE_SYSTEM_LIST_DIRECTORY,
              params: {
                path: "/workspace",
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 7,
              method: APP_SERVER_METHOD_FILE_SYSTEM_READ_FILE_PREVIEW,
              params: {
                path: "/workspace/README.md",
                maxSize: 1024,
              },
            }),
          ],
        },
      },
    );
  });

  it("readProjectGitDiff 应通过 App Server JSON-RPC 读取 Git diff", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 8,
          result: {
            rootPath: "/workspace",
            repositoryRoot: "/workspace",
            hasGitRepository: true,
            currentRef: "main",
            comparisonBaseRef: "origin/main",
            patch: "diff --git a/README.md b/README.md\n+hello",
            uncommittedFileCount: 1,
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 8 });
    const diff = await client.readProjectGitDiff({
      rootPath: "/workspace",
      contextLines: 5,
      base: "branch",
    });

    expect(diff.result.patch).toContain("diff --git");
    expect(diff.result.currentRef).toBe("main");
    expect(diff.result.comparisonBaseRef).toBe("origin/main");
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 8,
            method: APP_SERVER_METHOD_PROJECT_GIT_DIFF,
            params: {
              rootPath: "/workspace",
              contextLines: 5,
              base: "branch",
            },
          }),
        ],
      },
    });
  });

  it("execution process 控制面应通过 App Server JSON-RPC current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 10,
            result: {
              snapshot: {
                processId: "process-1",
                toolId: "tool-1",
                toolName: "Bash",
                status: "running",
                exitCode: null,
                elapsedMs: 0,
                outputBytes: 0,
                outputOmittedBytes: 0,
                outputTruncated: false,
                retainedOutput: "",
                failure: null,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 11,
            result: {
              snapshot: {
                processId: "process-1",
                toolId: "tool-1",
                toolName: "Bash",
                status: "interrupted",
                exitCode: null,
                elapsedMs: 12,
                outputBytes: 0,
                outputOmittedBytes: 0,
                outputTruncated: false,
                retainedOutput: "",
                failure: null,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 12,
            result: {
              deltas: [
                {
                  processId: "process-1",
                  toolId: "tool-1",
                  sequence: 1,
                  kind: "stdout",
                  delta: "ok",
                  bytes: 2,
                  omittedBytes: 0,
                  truncated: false,
                },
              ],
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 10 });

    await expect(
      client.startExecutionProcess({
        processId: "process-1",
        toolId: "tool-1",
        toolName: "Bash",
        command: ["sh", "-c", "npm test"],
        workingDirectory: "/workspace",
        approvalPolicy: "never",
        sandboxPolicy: "danger-full-access",
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        result: expect.objectContaining({
          snapshot: expect.objectContaining({ processId: "process-1" }),
        }),
      }),
    );
    await expect(
      client.interruptExecutionProcess({ processId: "process-1" }),
    ).resolves.toEqual(
      expect.objectContaining({
        result: expect.objectContaining({
          snapshot: expect.objectContaining({ status: "interrupted" }),
        }),
      }),
    );
    await expect(
      client.drainExecutionProcessOutput({ processId: "process-1", limit: 16 }),
    ).resolves.toEqual(
      expect.objectContaining({
        result: expect.objectContaining({
          deltas: [expect.objectContaining({ delta: "ok" })],
        }),
      }),
    );

    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 10,
              method: APP_SERVER_METHOD_EXECUTION_PROCESS_START,
              params: {
                processId: "process-1",
                toolId: "tool-1",
                toolName: "Bash",
                command: ["sh", "-c", "npm test"],
                workingDirectory: "/workspace",
                approvalPolicy: "never",
                sandboxPolicy: "danger-full-access",
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 11,
              method: APP_SERVER_METHOD_EXECUTION_PROCESS_INTERRUPT,
              params: { processId: "process-1" },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      3,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 12,
              method: APP_SERVER_METHOD_EXECUTION_PROCESS_DRAIN_OUTPUT,
              params: { processId: "process-1", limit: 16 },
            }),
          ],
        },
      },
    );
  });

  it("exportEvidence 应通过 App Server JSON-RPC 导出 current evidence snapshot", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 6,
          result: {
            session: {
              sessionId: "session-1",
              threadId: "thread-1",
              appId: "content-studio",
              status: "running",
              createdAt: "2026-06-05T00:00:00.000Z",
              updatedAt: "2026-06-05T00:00:01.000Z",
            },
            turns: [
              {
                turnId: "turn-1",
                sessionId: "session-1",
                threadId: "thread-1",
                status: "accepted",
              },
            ],
            events: [
              {
                eventId: "evt-1",
                sequence: 1,
                sessionId: "session-1",
                threadId: "thread-1",
                turnId: "turn-1",
                type: "message.delta",
                timestamp: "2026-06-05T00:00:01Z",
                payload: {
                  text: "draft",
                },
              },
            ],
            artifacts: [
              {
                artifactRef: "artifact-report",
                eventId: "evt-2",
                sequence: 2,
                turnId: "turn-1",
                artifactId: "artifact-report",
                path: ".app-server/artifacts/report.md",
                contentStatus: "notRequested",
              },
            ],
            exportedAt: "2026-06-05T00:00:02.000Z",
            evidencePack: {
              packRelativeRoot: ".lime/harness/sessions/session-1/evidence",
              packAbsoluteRoot:
                "/workspace/.lime/harness/sessions/session-1/evidence",
              exportedAt: "2026-06-05T00:00:03.000Z",
              threadStatus: "running",
              latestTurnStatus: "accepted",
              turnCount: 1,
              itemCount: 3,
              pendingRequestCount: 0,
              queuedTurnCount: 0,
              recentArtifactCount: 1,
              knownGaps: ["gui_smoke_not_run"],
              observabilitySummary: {
                schema_version: "runtime-evidence-pack.v1",
              },
              completionAuditSummary: {
                decision: "in_progress",
              },
              artifacts: [
                {
                  kind: "summary",
                  title: "Evidence Summary",
                  relativePath:
                    ".lime/harness/sessions/session-1/evidence/summary.md",
                  bytes: 128,
                },
              ],
            },
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 6 });
    const result = await client.exportEvidence({
      sessionId: "session-1",
      turnId: "turn-1",
      includeEvents: true,
      includeArtifacts: true,
      includeEvidencePack: true,
    });

    expect(result.result.session.sessionId).toBe("session-1");
    expect(result.result.turns[0].turnId).toBe("turn-1");
    expect(result.result.events[0].type).toBe("message.delta");
    expect(result.result.artifacts[0].artifactRef).toBe("artifact-report");
    expect(result.result.artifacts[0].contentStatus).toBe("notRequested");
    expect(result.result.exportedAt).toBe("2026-06-05T00:00:02.000Z");
    expect(
      (result.result as { threadStatus?: string }).threadStatus,
    ).toBeUndefined();
    expect(result.result.evidencePack?.threadStatus).toBe("running");
    expect(
      (
        result.result.evidencePack?.completionAuditSummary as {
          decision?: string;
        }
      )?.decision,
    ).toBe("in_progress");
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 6,
            method: APP_SERVER_METHOD_EVIDENCE_EXPORT,
            params: {
              sessionId: "session-1",
              turnId: "turn-1",
              includeEvents: true,
              includeArtifacts: true,
              includeEvidencePack: true,
            },
          }),
        ],
      },
    });
  });

  it("startTurn 应返回同步 result 并保留同批 notification", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 7,
          result: {
            turn: {
              id: "turn-1",
              items: [],
              itemsView: "notLoaded",
              status: "inProgress",
            },
          },
        }),
        line({
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-1",
              sequence: 1,
              sessionId: "session-1",
              threadId: "thread-1",
              turnId: "turn-1",
              type: "turn.accepted",
              timestamp: "2026-06-04T00:00:00Z",
              payload: {},
            },
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 7 });
    const result = await client.startTurn({
      threadId: "thread-1",
      input: [{ type: "text", text: "hello" }],
    });

    expect(result.result.turn.id).toBe("turn-1");
    expect(result.notifications).toHaveLength(1);
    expect(result.notifications[0].method).toBe(
      APP_SERVER_METHOD_AGENT_SESSION_EVENT,
    );
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 7,
            method: APP_SERVER_METHOD_TURN_START,
            params: {
              threadId: "thread-1",
              input: [{ type: "text", text: "hello" }],
            },
          }),
        ],
      },
    });
  });

  it("readWorkflow 应走 App Server current workflow/read 方法", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 11,
          result: {
            sessionId: "session-1",
            workflow: {
              threadId: "thread-1",
              workflowRuns: [],
              workflowSteps: [],
              actions: [],
              updatedAt: "2026-07-04T00:00:00.000Z",
            },
          },
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 11 });
    const result = await client.readWorkflow({ sessionId: "session-1" });

    expect(result.result.workflow.workflowRuns).toEqual([]);
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 11,
            method: APP_SERVER_METHOD_WORKFLOW_READ,
            params: {
              sessionId: "session-1",
            },
          }),
        ],
      },
    });
  });

  it("thread control 应通过 App Server JSON-RPC current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 10,
            result: {},
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 11,
            result: {
              thread: { id: "thread-1", sessionId: "session-1", turns: [] },
              model: "gpt-5.4",
              modelProvider: "openai",
              cwd: "/tmp/workspace",
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 10 });

    await expect(
      client.startThreadCompaction({
        threadId: "thread-1",
      }),
    ).resolves.toMatchObject({ result: {} });
    await expect(
      client.resumeThread({ threadId: "thread-1", excludeTurns: true }),
    ).resolves.toMatchObject({
      result: { thread: { id: "thread-1", sessionId: "session-1" } },
    });
    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 10,
              method: APP_SERVER_METHOD_THREAD_COMPACT_START,
              params: {
                threadId: "thread-1",
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 11,
              method: APP_SERVER_METHOD_THREAD_RESUME,
              params: { threadId: "thread-1", excludeTurns: true },
            }),
          ],
        },
      },
    );
  });

  it("cancelTurn 应通过 App Server JSON-RPC 取消指定 turn", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 8,
          result: {},
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 8 });
    const result = await client.cancelTurn({
      threadId: "thread-1",
      turnId: "turn-1",
    });

    expect(result.result).toEqual({});
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 8,
            method: APP_SERVER_METHOD_TURN_INTERRUPT,
            params: {
              threadId: "thread-1",
              turnId: "turn-1",
            },
          }),
        ],
      },
    });
  });

  it("steerTurn 应通过 v2 turn/steer 追加当前 turn 输入", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [line({ id: 9, result: { turnId: "turn-1" } })],
    });

    const client = new AppServerClient({ initialRequestId: 9 });
    const result = await client.steerTurn({
      threadId: "thread-1",
      expectedTurnId: "turn-1",
      input: [{ type: "text", text: "follow up" }],
    });

    expect(result.result).toEqual({ turnId: "turn-1" });
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 9,
            method: APP_SERVER_METHOD_TURN_STEER,
            params: {
              threadId: "thread-1",
              expectedTurnId: "turn-1",
              input: [{ type: "text", text: "follow up" }],
            },
          }),
        ],
      },
    });
  });

  it("archiveThread / unarchiveThread 应通过 v2 thread 生命周期方法", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [line({ id: 9, result: {} })],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 10,
            result: {
              thread: {
                id: "thread-1",
                sessionId: "session-1",
                createdAt: 1780704000,
                updatedAt: 1780704000,
                status: { type: "idle" },
                turns: [],
              },
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 9 });
    await client.archiveThread({ threadId: "thread-1" });
    const restored = await client.unarchiveThread({ threadId: "thread-1" });

    expect(restored.result.thread.id).toBe("thread-1");
    expect(vi.mocked(safeInvoke).mock.calls).toEqual([
      [
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id: 9,
                method: APP_SERVER_METHOD_THREAD_ARCHIVE,
                params: { threadId: "thread-1" },
              }),
            ],
          },
        },
      ],
      [
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id: 10,
                method: APP_SERVER_METHOD_THREAD_UNARCHIVE,
                params: { threadId: "thread-1" },
              }),
            ],
          },
        },
      ],
    ]);
  });

  it("runThreadShellCommand 应通过 typed thread method 提交用户命令", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [line({ id: 9, result: {} })],
    });

    const client = new AppServerClient({ initialRequestId: 9 });
    await client.runThreadShellCommand({
      threadId: "thread-1",
      command: "printf ready",
    });

    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 9,
            method: APP_SERVER_METHOD_THREAD_SHELL_COMMAND,
            params: {
              threadId: "thread-1",
              command: "printf ready",
            },
          }),
        ],
      },
    });
  });

  it("respondAction 应通过 App Server JSON-RPC 响应 action.required", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          id: 9,
          result: {},
        }),
      ],
    });

    const client = new AppServerClient({ initialRequestId: 9 });
    const result = await client.respondAction({
      sessionId: "session-1",
      requestId: "req-confirm-1",
      actionType: "tool_confirmation",
      confirmed: true,
      response: "allow",
      userData: {
        reason: "approved",
      },
      eventName: "agentSession/event/session-1",
      actionScope: {
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
      },
    });

    expect(result.result).toEqual({});
    expect(safeInvoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: {
        lines: [
          line({
            id: 9,
            method: APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
            params: {
              sessionId: "session-1",
              requestId: "req-confirm-1",
              actionType: "tool_confirmation",
              confirmed: true,
              response: "allow",
              userData: {
                reason: "approved",
              },
              eventName: "agentSession/event/session-1",
              actionScope: {
                sessionId: "session-1",
                threadId: "thread-1",
                turnId: "turn-1",
              },
            },
          }),
        ],
      },
    });
  });

  it("logs helpers 应通过 App Server JSON-RPC current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 9,
            result: {
              entries: [{ timestamp: "t", level: "info", message: "m" }],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 10,
            result: {
              entries: [{ timestamp: "t2", level: "warn", message: "m2" }],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 11,
            result: { cleared: true },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 12,
            result: { cleared: true },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 13,
            result: {
              currentLogExists: true,
              inMemoryLogCount: 0,
              relatedLogFiles: [],
              rawResponseFiles: [],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 14,
            result: {
              bundlePath: "/tmp/Lime-Support.zip",
              outputDirectory: "/tmp",
              generatedAt: "2026-06-09T00:00:00Z",
              platform: "darwin",
              includedSections: ["meta/manifest.json"],
              omittedSections: [],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 15,
            result: {
              generatedAt: "2026-06-09T00:00:00Z",
              running: true,
              host: "127.0.0.1",
              port: 0,
              telemetrySummary: {
                totalRequests: 0,
                successfulRequests: 0,
                failedRequests: 0,
                timeoutRequests: 0,
                successRate: 0,
                avgLatencyMs: 0,
                totalInputTokens: 0,
                totalOutputTokens: 0,
                totalTokens: 0,
              },
              capabilityRouting: {
                filterEvalTotal: 0,
                filterExcludedTotal: 0,
                filterExcludedToolsTotal: 0,
                filterExcludedVisionTotal: 0,
                filterExcludedContextTotal: 0,
                providerFallbackTotal: 0,
                modelFallbackTotal: 0,
                allCandidatesExcludedTotal: 0,
              },
              responseCache: {
                config: {
                  enabled: false,
                  ttlSecs: 0,
                  cacheableStatusCodes: [],
                },
                stats: {},
                hitRatePercent: 0,
              },
              requestDedup: {
                config: { enabled: false, ttlSecs: 0 },
                stats: {},
                replayRatePercent: 0,
              },
              idempotency: {
                config: { enabled: false, ttlSecs: 0 },
                stats: {},
                replayRatePercent: 0,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 16,
            result: {
              platform: "darwin",
              checks: [],
              hasBlockingIssues: false,
              hasWarnings: false,
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 17,
            result: {
              available: true,
              traces: [
                {
                  sessionId: "session-a",
                  traceId: "trace-a",
                  path: "sessions/session_session-a/trace_trace-a.jsonl",
                  sizeBytes: 128,
                  eventCount: 2,
                },
              ],
              redaction: {
                mode: "summary_only",
                rawAgentEventPayload: false,
                promptText: false,
                providerPayload: false,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 18,
            result: {
              available: true,
              trace: {
                sessionId: "session-a",
                traceId: "trace-a",
                path: "sessions/session_session-a/trace_trace-a.jsonl",
                sizeBytes: 128,
                eventCount: 2,
              },
              events: [
                {
                  schemaVersion: 1,
                  seq: 1,
                  wallTimeUnixMs: 1_780_000_000_000,
                  traceId: "trace-a",
                  sessionId: "session-a",
                  eventId: "evt-a",
                  eventSequence: 1,
                  eventType: "message.delta",
                  checkpoint: "app_server.message_delta.emitted",
                  metrics: { text_chars: 4 },
                  redaction: {
                    mode: "summary_only",
                    rawAgentEventPayload: false,
                    promptText: false,
                    providerPayload: false,
                  },
                },
              ],
              redaction: {
                mode: "summary_only",
                rawAgentEventPayload: false,
                promptText: false,
                providerPayload: false,
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 19,
            result: {
              available: true,
              exported: true,
              trace: {
                sessionId: "session-a",
                traceId: "trace-a",
                path: "sessions/session_session-a/trace_trace-a.jsonl",
                sizeBytes: 128,
                eventCount: 2,
              },
              bundlePath: "/tmp/claw-trace-session-a-trace-a.zip",
              outputDirectory: "/tmp",
              generatedAt: "2026-06-27T00:00:00.000Z",
              includedSections: ["trace/events.jsonl"],
              omittedSections: ["prompt text"],
              redaction: {
                mode: "summary_only",
                rawAgentEventPayload: false,
                promptText: false,
                providerPayload: false,
              },
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 9 });

    await expect(client.listLogs()).resolves.toMatchObject({
      result: {
        entries: [expect.objectContaining({ level: "info" })],
      },
    });
    await expect(
      client.readPersistedLogTail({ lines: 250 }),
    ).resolves.toMatchObject({
      result: {
        entries: [expect.objectContaining({ level: "warn" })],
      },
    });
    await expect(client.clearLogs()).resolves.toMatchObject({
      result: { cleared: true },
    });
    await expect(client.clearDiagnosticLogHistory()).resolves.toMatchObject({
      result: { cleared: true },
    });
    await expect(client.readLogStorageDiagnostics()).resolves.toMatchObject({
      result: { currentLogExists: true },
    });
    await expect(client.exportSupportBundle()).resolves.toMatchObject({
      result: { bundlePath: "/tmp/Lime-Support.zip" },
    });
    await expect(client.readServerDiagnostics()).resolves.toMatchObject({
      result: { running: true },
    });
    await expect(client.readWindowsStartupDiagnostics()).resolves.toMatchObject(
      {
        result: { platform: "darwin" },
      },
    );
    await expect(
      client.listDiagnosticsTraces({ sessionId: "session-a", limit: 5 }),
    ).resolves.toMatchObject({
      result: {
        traces: [expect.objectContaining({ traceId: "trace-a" })],
      },
    });
    await expect(
      client.readDiagnosticsTrace({
        sessionId: "session-a",
        traceId: "trace-a",
        maxEvents: 20,
      }),
    ).resolves.toMatchObject({
      result: {
        events: [
          expect.objectContaining({
            checkpoint: "app_server.message_delta.emitted",
          }),
        ],
      },
    });
    await expect(
      client.exportDiagnosticsTrace({
        sessionId: "session-a",
        traceId: "trace-a",
      }),
    ).resolves.toMatchObject({
      result: {
        exported: true,
        bundlePath: "/tmp/claw-trace-session-a-trace-a.zip",
      },
    });

    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 9,
              method: APP_SERVER_METHOD_LOG_LIST,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 10,
              method: APP_SERVER_METHOD_LOG_PERSISTED_TAIL,
              params: { lines: 250 },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      3,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 11,
              method: APP_SERVER_METHOD_LOG_CLEAR,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      4,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 12,
              method: APP_SERVER_METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      5,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 13,
              method: APP_SERVER_METHOD_DIAGNOSTICS_LOG_STORAGE_READ,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      6,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 14,
              method: APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      7,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 15,
              method: APP_SERVER_METHOD_DIAGNOSTICS_SERVER_READ,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      8,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 16,
              method: APP_SERVER_METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
              params: {},
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      9,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 17,
              method: APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST,
              params: { sessionId: "session-a", limit: 5 },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      10,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 18,
              method: APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ,
              params: {
                sessionId: "session-a",
                traceId: "trace-a",
                maxEvents: 20,
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      11,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 19,
              method: APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT,
              params: {
                sessionId: "session-a",
                traceId: "trace-a",
              },
            }),
          ],
        },
      },
    );
  });

  it("media task artifact helpers 应通过 App Server JSON-RPC current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 13,
            result: {
              success: true,
              task_id: "task-image-1",
              task_type: "image_generate",
              task_family: "image",
              status: "pending_submit",
              normalized_status: "pending",
              path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              artifact_path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 14,
            result: {
              success: true,
              task_id: "task-audio-1",
              task_type: "audio_generate",
              task_family: "audio",
              status: "pending_submit",
              normalized_status: "pending",
              path: ".lime/tasks/audio_generate/task-audio-1.json",
              absolute_path:
                "/workspace/.lime/tasks/audio_generate/task-audio-1.json",
              artifact_path: ".lime/tasks/audio_generate/task-audio-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/audio_generate/task-audio-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 15,
            result: {
              success: true,
              task_id: "task-audio-1",
              task_type: "audio_generate",
              task_family: "audio",
              status: "succeeded",
              normalized_status: "succeeded",
              path: ".lime/tasks/audio_generate/task-audio-1.json",
              absolute_path:
                "/workspace/.lime/tasks/audio_generate/task-audio-1.json",
              artifact_path: ".lime/tasks/audio_generate/task-audio-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/audio_generate/task-audio-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 16,
            result: {
              success: true,
              task_id: "task-image-1",
              task_type: "image_generate",
              task_family: "image",
              status: "succeeded",
              normalized_status: "succeeded",
              path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              artifact_path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 17,
            result: {
              success: true,
              task_id: "task-image-1",
              task_type: "image_generate",
              task_family: "image",
              status: "pending_submit",
              normalized_status: "pending",
              path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              artifact_path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 18,
            result: {
              success: true,
              workspace_root: "/workspace",
              artifact_root: "/workspace/.lime/tasks",
              filters: { task_family: "image", limit: 10 },
              total: 1,
              modality_runtime_contracts: {},
              tasks: [],
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 19,
            result: {
              success: true,
              task_id: "task-image-1",
              task_type: "image_generate",
              task_family: "image",
              status: "cancelled",
              normalized_status: "cancelled",
              path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              artifact_path: ".lime/tasks/image_generate/task-image-1.json",
              absolute_artifact_path:
                "/workspace/.lime/tasks/image_generate/task-image-1.json",
              reused_existing: false,
              record: {},
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 13 });
    const imageRequest = {
      projectRootPath: "/workspace",
      prompt: "未来感青柠实验室",
      mode: "generate",
    };
    const audioCreateRequest = {
      projectRootPath: "/workspace",
      sourceText: "请生成温暖旁白",
    };
    const audioCompleteRequest = {
      projectRootPath: "/workspace",
      taskRef: "task-audio-1",
      audioPath: ".lime/runtime/audio/task-audio-1.mp3",
    };
    const imageCompleteRequest = {
      projectRootPath: "/workspace",
      taskRef: "task-image-1",
      images: [
        {
          url: "file:///workspace/.lime/runtime/images/task-image-1.png",
          revisedPrompt: "未来感青柠实验室",
        },
      ],
    };
    const lookupRequest = {
      projectRootPath: "/workspace",
      taskRef: "task-image-1",
    };
    const listRequest = {
      projectRootPath: "/workspace",
      taskFamily: "image",
      limit: 10,
    };

    await expect(
      client.createImageMediaTaskArtifact(imageRequest),
    ).resolves.toMatchObject({
      result: { task_id: "task-image-1" },
    });
    await expect(
      client.createAudioMediaTaskArtifact(audioCreateRequest),
    ).resolves.toMatchObject({
      result: { task_id: "task-audio-1" },
    });
    await expect(
      client.completeAudioMediaTaskArtifact(audioCompleteRequest),
    ).resolves.toMatchObject({
      result: { normalized_status: "succeeded" },
    });
    await expect(
      client.completeImageMediaTaskArtifact(imageCompleteRequest),
    ).resolves.toMatchObject({
      result: { normalized_status: "succeeded" },
    });
    await expect(
      client.getMediaTaskArtifact(lookupRequest),
    ).resolves.toMatchObject({
      result: { task_id: "task-image-1" },
    });
    await expect(
      client.listMediaTaskArtifacts(listRequest),
    ).resolves.toMatchObject({
      result: { total: 1 },
    });
    await expect(
      client.cancelMediaTaskArtifact(lookupRequest),
    ).resolves.toMatchObject({
      result: { normalized_status: "cancelled" },
    });

    const expectedCalls = [
      [13, APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE, imageRequest],
      [
        14,
        APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
        audioCreateRequest,
      ],
      [
        15,
        APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
        audioCompleteRequest,
      ],
      [
        16,
        APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        imageCompleteRequest,
      ],
      [17, APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET, lookupRequest],
      [18, APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST, listRequest],
      [19, APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_CANCEL, lookupRequest],
    ] as const;

    expectedCalls.forEach(([id, method, params], index) => {
      expect(safeInvoke).toHaveBeenNthCalledWith(
        index + 1,
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id,
                method,
                params,
              }),
            ],
          },
        },
      );
    });
  });

  it("request 遇到 JSON-RPC error 时抛出 AppServerRpcError", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-error-1",
              sequence: 1,
              sessionId: "session-1",
              turnId: "turn-1",
              type: "turn.failed",
              timestamp: "2026-06-06T00:00:00.000Z",
              payload: {
                message: "session not found",
              },
            },
          },
        }),
        line({
          id: 1,
          error: {
            code: -32010,
            message: "session not found",
            data: {
              sessionId: "missing",
            },
          },
        }),
      ],
    });

    const client = new AppServerClient();
    await expect(
      client.readSession({ sessionId: "missing" }),
    ).rejects.toMatchObject({
      name: "AppServerRpcError",
      code: -32010,
      data: {
        sessionId: "missing",
      },
      response: {
        id: 1,
        error: {
          code: -32010,
          message: "session not found",
        },
      },
      notifications: [
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        },
      ],
      messages: [
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        },
        {
          id: 1,
          error: {
            message: "session not found",
          },
        },
      ],
    });
  });

  it("drainEvents 应走 app_server_drain_events 并解码 notification", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-2",
              sequence: 2,
              sessionId: "session-1",
              type: "message.delta",
              timestamp: "2026-06-04T00:00:00Z",
              payload: {
                text: "hi",
              },
            },
          },
        }),
      ],
    });

    const client = new AppServerClient();
    const messages = await client.drainEvents(5);

    expect(messages).toEqual([
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: expect.objectContaining({
            eventId: "evt-2",
            type: "message.delta",
          }),
        },
      },
    ]);
    expect(safeInvoke).toHaveBeenCalledWith("app_server_drain_events", {
      request: { limit: 5 },
    });
  });

  it("drainEvents 应透传 includeRecent 读取最近镜像 notification", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      lines: [
        line({
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              eventId: "evt-recent-1",
              sequence: 4,
              sessionId: "session-1",
              type: "runtime.status",
              timestamp: "2026-06-04T00:00:02Z",
              payload: {
                phase: "reading",
              },
            },
          },
        }),
      ],
    });

    const client = new AppServerClient();
    const messages = await client.drainEvents({
      includeRecent: true,
      limit: 20,
    });

    expect(messages).toEqual([
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: expect.objectContaining({
            eventId: "evt-recent-1",
            type: "runtime.status",
          }),
        },
      },
    ]);
    expect(safeInvoke).toHaveBeenCalledWith("app_server_drain_events", {
      request: { includeRecent: true, limit: 20 },
    });
  });

  it("drainEvents 应兼容 Electron safeInvoke result 包络", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      result: {
        lines: [
          line({
            method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
            params: {
              event: {
                eventId: "evt-envelope-1",
                sequence: 3,
                sessionId: "session-1",
                type: "turn.completed",
                timestamp: "2026-06-04T00:00:01Z",
                payload: {},
              },
            },
          }),
        ],
      },
    });

    const client = new AppServerClient();
    const messages = await client.drainEvents(1);

    expect(messages).toEqual([
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: expect.objectContaining({
            eventId: "evt-envelope-1",
            type: "turn.completed",
          }),
        },
      },
    ]);
    expect(safeInvoke).toHaveBeenCalledWith("app_server_drain_events", {
      request: { limit: 1 },
    });
  });

  it("drainEvents 收到 App Server bridge diagnostic facade 时应 fail closed", async () => {
    vi.mocked(safeInvoke).mockResolvedValueOnce({
      diagnostic: {
        command: "app_server_drain_events",
        source: "electron-host",
        status: "not-supported",
      },
      lines: [],
    });

    const client = new AppServerClient();

    await expect(client.drainEvents(1)).rejects.toThrow(
      "app_server_drain_events 尚未接入真实 App Server bridge",
    );
  });

  it("JSON-RPC 编解码与 response matcher 保持稳定", () => {
    const request = createAppServerRequest(3, "thread/read", {
      sessionId: "session-1",
    });

    expect(decodeAppServerMessage(encodeAppServerMessage(request))).toEqual(
      request,
    );
    expect(
      expectAppServerResponse<{ ok: boolean }>(
        [
          {
            id: 3,
            result: {
              ok: true,
            },
          },
        ],
        3,
        "thread/read",
      ).result.ok,
    ).toBe(true);
    expect(() =>
      expectAppServerResponse(
        [
          {
            id: 4,
            error: {
              code: -32601,
              message: "method not found",
            },
          },
        ],
        4,
        "missing",
      ),
    ).toThrow(AppServerRpcError);
  });

  it("Channels side-effect 应通过 App Server current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 1,
            result: {
              channel: "telegram",
              status: { running_accounts: 1, accounts: [] },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 2,
            result: { runtimeModel: "openai/gpt-5.4" },
          }),
        ],
      });

    const client = new AppServerClient();
    await expect(
      client.startGatewayChannel({
        channel: "telegram",
        accountId: "default",
        pollTimeoutSecs: 25,
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        result: expect.objectContaining({ channel: "telegram" }),
      }),
    );
    await expect(
      client.setWechatChannelRuntimeModel({
        providerId: "openai",
        modelId: "gpt-5.4",
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        result: { runtimeModel: "openai/gpt-5.4" },
      }),
    );

    expect(safeInvoke).toHaveBeenNthCalledWith(
      1,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 1,
              method: APP_SERVER_METHOD_GATEWAY_CHANNEL_START,
              params: {
                channel: "telegram",
                accountId: "default",
                pollTimeoutSecs: 25,
              },
            }),
          ],
        },
      },
    );
    expect(safeInvoke).toHaveBeenNthCalledWith(
      2,
      "app_server_handle_json_lines",
      {
        request: {
          lines: [
            line({
              id: 2,
              method: APP_SERVER_METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET,
              params: {
                providerId: "openai",
                modelId: "gpt-5.4",
              },
            }),
          ],
        },
      },
    );
  });

  it("Gateway Tunnel 应通过 App Server current methods", async () => {
    vi.mocked(safeInvoke)
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 30,
            result: {
              ok: true,
              provider: "cloudflare",
              mode: "managed",
              binary: "cloudflared",
              configReady: true,
              message: "ready",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 31,
            result: {
              installed: true,
              binary: "cloudflared",
              platform: "macos",
              installSupported: true,
              requiresPrivilege: false,
              message: "installed",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 32,
            result: {
              ok: true,
              attempted: true,
              platform: "macos",
              installed: true,
              stdout: "",
              stderr: "",
              message: "installed",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 33,
            result: {
              result: {
                ok: true,
                tunnelName: "lime",
                message: "created",
              },
              status: {
                running: true,
                provider: "cloudflare",
                mode: "managed",
                binary: "cloudflared",
                localUrl: "http://127.0.0.1:3000",
              },
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 34,
            result: {
              running: true,
              provider: "cloudflare",
              mode: "managed",
              binary: "cloudflared",
              localUrl: "http://127.0.0.1:3000",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 35,
            result: {
              running: false,
              provider: "cloudflare",
              mode: "managed",
              binary: "cloudflared",
              localUrl: "http://127.0.0.1:3000",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 36,
            result: {
              running: true,
              provider: "cloudflare",
              mode: "managed",
              binary: "cloudflared",
              localUrl: "http://127.0.0.1:3000",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 37,
            result: {
              running: true,
              provider: "cloudflare",
              mode: "managed",
              binary: "cloudflared",
              localUrl: "http://127.0.0.1:3000",
            },
          }),
        ],
      })
      .mockResolvedValueOnce({
        lines: [
          line({
            id: 38,
            result: {
              channel: "feishu",
              webhookPath: "/feishu/default",
              publicBaseUrl: "https://lime.example.com",
              webhookUrl: "https://lime.example.com/feishu/default",
              persisted: true,
            },
          }),
        ],
      });

    const client = new AppServerClient({ initialRequestId: 30 });
    await client.probeGatewayTunnel();
    await client.detectGatewayTunnelCloudflared();
    await client.installGatewayTunnelCloudflared({ confirm: true });
    await client.createGatewayTunnel({
      tunnelName: "lime",
      dnsName: "bot.example.com",
      persist: true,
    });
    await client.startGatewayTunnel();
    await client.stopGatewayTunnel();
    await client.restartGatewayTunnel();
    await client.readGatewayTunnelStatus();
    await client.syncGatewayTunnelWebhookUrl({
      channel: "feishu",
      accountId: "default",
      webhookPath: "/feishu/default",
      persist: true,
    });

    const expectedCalls = [
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_PROBE, {}],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT, {}],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL, { confirm: true }],
      [
        APP_SERVER_METHOD_GATEWAY_TUNNEL_CREATE,
        {
          tunnelName: "lime",
          dnsName: "bot.example.com",
          persist: true,
        },
      ],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_START, {}],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_STOP, {}],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_RESTART, {}],
      [APP_SERVER_METHOD_GATEWAY_TUNNEL_STATUS, {}],
      [
        APP_SERVER_METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL,
        {
          channel: "feishu",
          accountId: "default",
          webhookPath: "/feishu/default",
          persist: true,
        },
      ],
    ] as const;

    expectedCalls.forEach(([method, params], index) => {
      expect(safeInvoke).toHaveBeenNthCalledWith(
        index + 1,
        "app_server_handle_json_lines",
        {
          request: {
            lines: [
              line({
                id: 30 + index,
                method,
                params,
              }),
            ],
          },
        },
      );
    });
    expect(safeInvoke).not.toHaveBeenCalledWith(
      expect.stringMatching(/^gateway_tunnel_/),
      expect.anything(),
    );
  });
});
