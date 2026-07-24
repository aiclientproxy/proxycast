import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  buildReadyState,
  dispatchBridgeMessage,
  flush,
  getRuntimeFrame,
  renderPage,
  unmountLastRenderedPage,
  usePluginRuntimePageTestLifecycle,
} from "./PluginRuntimePage.testFixtures";

describe("PluginRuntimePage Host AI run surface", () => {
  usePluginRuntimePageTestLifecycle();

  it("App 可通过 lime.ui 打开、更新并关闭 Host 级 AI 运行面板", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.agent",
        method: "startTask",
        input: {
          taskId: "plugin-task-1",
          title: "生成内容批次",
          prompt: "基于项目知识生成内容批次",
          taskKind: "content.production",
          idempotencyKey: "dashboard:production",
          input: { projectId: "project-1" },
          humanReview: true,
        },
      },
      "agent-run-task-start",
    );

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-1",
          bridgeAction: "content_factory.production",
          title: "生成内容批次",
          mode: "drawer",
          runtimeProcess: {
            model: {
              provider: "anthropic",
              model: "claude-sonnet-4-5",
              label: "Claude Sonnet 4.5",
            },
            usage: {
              inputTokens: 120,
              outputTokens: 80,
              totalTokens: 200,
            },
            cost: {
              estimatedTotalCost: 0.0123,
              currency: "USD",
            },
            skillNames: ["content_factory_writer"],
            invokedSkillNames: ["content_factory_writer"],
            terminal: false,
            timeline: [
              {
                kind: "routing",
                title: "模型路由",
                message: "AgentRuntime 已选择内容生成模型。",
                statusText: "decided",
                meta: "routing",
              },
              {
                kind: "skill",
                title: "Skill · content_factory_writer",
                message: "正在调用内容工厂写作 Skill。",
                statusText: "running",
                meta: "skill-1",
              },
              {
                kind: "tool",
                title: "Tool · browser_snapshot",
                message: "正在读取业务页面上下文。",
                statusText: "running",
                meta: "tool-1",
              },
              {
                kind: "tool",
                title: "Tool · browser_snapshot",
                message: "页面截图已读取。",
                statusText: "已完成",
                meta: "tool-2",
              },
              {
                kind: "execution",
                title: "正在规划内容结构",
                message: "AgentRuntime 已开始读取项目上下文。",
                statusText: "running",
              },
              {
                kind: "output",
                title: "成稿流式输出",
                message: "第一段流式文案。",
                statusText: "streaming",
                collapseKey: "stream:assistant_text:main",
              },
              {
                kind: "output",
                title: "成稿流式输出",
                message: "第二段流式文案。",
                statusText: "streaming",
                collapseKey: "stream:assistant_text:main",
              },
            ],
            thinkingText: "先确认项目资料，再拆内容主题。",
            executionText: "### 执行过程\n调用内容工厂 Skill。",
            streamText: "## 首批文案\n- 正在生成第一批文案。",
          },
          events: [
            {
              eventType: "task:reviewRequested",
              requestId: "review-content-batch",
              message: "请确认首批内容选题。",
            },
            {
              eventType: "artifact:created",
              artifactRef: ".lime/artifacts/content-batch.json",
              payload: {
                artifact: {
                  title: "内容批次 JSON",
                },
              },
            },
            {
              eventType: "evidence:recorded",
              evidenceRef: "evidence:content-batch",
              message: "内容批次 evidence 已记录。",
            },
          ],
        },
      },
      "agent-run-open",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "agent-run-open",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            opened: true,
            surface: "host_agent_run",
            mode: "drawer",
            taskId: "plugin-task-1",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).toBeNull();
    const dock = container.querySelector(
      '[data-testid="plugin-host-agent-run-dock"]',
    ) as HTMLButtonElement;
    expect(dock).not.toBeNull();
    expect(dock.className).toContain("top-3");
    expect(dock.className).not.toContain("bottom-4");
    expect(dock.className).not.toContain("min(320px");
    expect(dock.className).toContain("max-w-[180px]");
    expect(container.textContent).toContain("查看运行现场");

    await act(async () => {
      dock.click();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="agent-run-process-panel"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-agent-run-renderer="host-shared"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("生成内容批次");
    expect(container.textContent).toContain("Claude Sonnet 4.5");
    expect(container.textContent).toContain("200");
    expect(container.textContent).toContain("USD 0.0123");
    expect(container.textContent).toContain("content_factory_writer");
    expect(container.textContent).toContain("模型路由");
    expect(container.textContent).toContain("Skill · content_factory_writer");
    expect(container.textContent).toContain(
      "先执行技能 content_factory_writer",
    );
    expect(container.textContent).toContain("正在调用内容工厂写作 Skill");
    expect(container.textContent).toContain("Tool · 页面截图");
    expect(container.textContent).toContain("先抓取页面状态");
    expect(container.textContent).toContain("正在读取业务页面上下文");
    expect(container.textContent).toContain("已拿到页面快照");
    expect(container.textContent).toContain("页面截图已读取");
    expect(
      container.querySelector("[data-testid='agent-run-standard-projection']"),
    ).not.toBeNull();
    expect(container.querySelector(".agent-ui-projection")).not.toBeNull();
    expect(container.querySelector(".agent-ui-main")).not.toBeNull();
    expect(container.querySelector(".agent-ui-sidecar")).not.toBeNull();
    expect(
      container.querySelector('[data-agent-run-timeline-kind="skill"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-agent-run-timeline-kind="tool"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inline-tool-process-step"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-agent-run-timeline-group="collapse:stream:assistant_text:main"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain("正在规划内容结构");
    expect(container.textContent).toContain("第一段流式文案。");
    expect(container.textContent).toContain("第二段流式文案。");
    expect(
      container.querySelector('[data-testid="agent-run-markdown-output"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="agent-run-markdown-execution"]'),
    ).toBeNull();
    expect(container.textContent).toContain("执行过程");
    expect(container.textContent).toContain("首批文案");
    expect(
      container.querySelector('[data-testid="thinking-block"]'),
    ).toBeNull();
    expect(container.textContent).toContain("思考过程");
    expect(container.textContent).toContain("先确认项目资料");
    expect(container.textContent).toContain("待确认");
    expect(container.textContent).toContain("请确认首批内容选题。");
    expect(
      container.querySelector('[data-testid="agent-run-projection-panel"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '.agent-action-required-list [data-action-id="review-content-batch"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('.agent-process-entry[data-entry-kind="tool"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '.agent-message-part[data-part-type="artifact-card"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("交付物");
    expect(container.textContent).toContain("内容批次 JSON");
    expect(container.textContent).toContain("证据");
    expect(container.textContent).toContain("内容批次 evidence 已记录。");
    const rejectButton = container.querySelector<HTMLButtonElement>(
      '.agent-action-required-list .agent-event-action[data-action-decision="reject"]',
    );
    expect(rejectButton?.textContent).toBe("拒绝");

    await act(async () => {
      rejectButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    await flush();

    expect(
      container.querySelector(
        '[data-action-id="review-content-batch"][data-action-resolved="true"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '.agent-action-required-list [data-action-id="review-content-batch"]',
      ),
    ).not.toBeNull();

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "updateAgentRun",
        input: {
          taskId: "plugin-task-1",
          runtimeProcess: {
            model: { label: "Claude Sonnet 4.5" },
            usage: { inputTokens: 120, outputTokens: 180, totalTokens: 300 },
            terminal: true,
            collapsedByDefault: true,
            timeline: [
              {
                kind: "completed",
                title: "内容批次已写回",
                message: "Host 保留完整运行过程。",
                statusText: "completed",
              },
            ],
            streamText: "第一批文案已完成。",
          },
        },
      },
      "agent-run-update",
    );

    expect(container.textContent).toContain("300");
    expect(container.textContent).toContain("正在规划内容结构");
    expect(container.textContent).toContain("内容批次已写回");
    expect(
      container
        .querySelector('[data-testid="agent-run-projection-panel"]')
        ?.getAttribute("data-agent-run-projection-terminal"),
    ).toBe("true");
    expect(
      container.querySelector('[data-testid="agent-run-markdown-output"]'),
    ).toBeNull();

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "closeAgentRun",
        input: { taskId: "plugin-task-1" },
      },
      "agent-run-close",
    );

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "agent-run-close",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            closed: true,
            surface: "host_agent_run",
            taskId: "plugin-task-1",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).toBeNull();
  });

  it("用户关闭 Host 级 AI 运行面板后，同一任务轮询更新不会自动重开", async () => {
    const container = await renderPage();
    await flush();
    const frame = getRuntimeFrame(container);
    const postMessage = vi.spyOn(frame.contentWindow!, "postMessage");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-dismissed",
          bridgeAction: "content_factory.production",
          title: "生成内容批次",
          mode: "drawer",
          runtimeProcess: {
            terminal: false,
            timeline: [
              {
                kind: "execution",
                title: "正在生成内容",
                message: "内容工厂正在轮询运行状态。",
                statusText: "running",
              },
            ],
          },
        },
      },
      "agent-run-open-dismissible",
    );

    const dock = container.querySelector<HTMLButtonElement>(
      '[data-testid="plugin-host-agent-run-dock"]',
    );
    expect(dock).not.toBeNull();

    await act(async () => {
      dock?.click();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).not.toBeNull();
    const closeButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="plugin-host-agent-run-close"]',
    );
    expect(closeButton).not.toBeNull();

    await act(async () => {
      closeButton?.click();
    });
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).toBeNull();

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "updateAgentRun",
        input: {
          taskId: "plugin-task-dismissed",
          bridgeAction: "content_factory.production",
          runtimeProcess: {
            terminal: false,
            timeline: [
              {
                kind: "execution",
                title: "轮询更新",
                message: "这条更新不应重新打开右侧面板。",
                statusText: "running",
              },
            ],
          },
        },
      },
      "agent-run-update-after-user-close",
    );
    await flush();

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "agent-run-update-after-user-close",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            updated: true,
            surface: "host_agent_run",
            taskId: "plugin-task-dismissed",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-drawer"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("轮询更新");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-dismissed",
          bridgeAction: "content_factory.production",
          title: "重新查看运行",
          mode: "drawer",
        },
      },
      "agent-run-open-after-user-close",
    );
    await flush();

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "host:response",
        requestId: "agent-run-open-after-user-close",
        payload: expect.objectContaining({
          ok: true,
          result: expect.objectContaining({
            opened: true,
            surface: "host_agent_run",
            taskId: "plugin-task-dismissed",
          }),
        }),
      }),
      "http://127.0.0.1:4199",
    );
    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("重新查看运行");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-same-action",
          bridgeAction: "content_factory.production",
          title: "同一业务动作的新任务不应出现",
          mode: "drawer",
        },
      },
      "agent-run-open-same-action",
    );
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("同一业务动作的新任务不应出现");

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-next",
          bridgeAction: "content_factory.review",
          title: "下一次运行",
          mode: "drawer",
        },
      },
      "agent-run-open-next-task",
    );
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("下一次运行");
  });

  it("Host 级 AI 运行面板在 runtime surface 重挂载后仍保留折叠入口", async () => {
    const state = buildReadyState();
    const container = await renderPage(state);
    await flush();
    const frame = getRuntimeFrame(container);

    await dispatchBridgeMessage(
      frame,
      "capability:invoke",
      {
        capability: "lime.ui",
        method: "openAgentRun",
        input: {
          taskId: "plugin-task-persisted",
          bridgeAction: "content_factory.production",
          title: "生成本轮内容",
          mode: "drawer",
          runtimeProcess: {
            terminal: true,
            collapsedByDefault: true,
            timeline: [
              {
                kind: "completed",
                title: "本轮内容已写回",
                message: "运行过程需要在完成后继续保留。",
                statusText: "completed",
              },
            ],
            streamText: "已生成 20 条草稿。",
          },
        },
      },
      "agent-run-open-persisted",
    );
    await flush();

    expect(
      container.querySelector('[data-testid="plugin-host-agent-run-dock"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("生成本轮内容");

    await unmountLastRenderedPage();

    const restoredContainer = await renderPage(state);
    await flush();

    expect(
      restoredContainer.querySelector(
        '[data-testid="plugin-host-agent-run-dock"]',
      ),
    ).not.toBeNull();
    expect(restoredContainer.textContent).toContain("生成本轮内容");
    expect(restoredContainer.textContent).toContain(
      "运行过程已折叠，点击查看完整现场",
    );
  });
});
