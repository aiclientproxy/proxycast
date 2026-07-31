import { describe, expect, it } from "vitest";
import {
  buildModelEffectiveEvent,
  buildRuntimeLifecycleEvents,
  buildRuntimeStatusEvents,
  buildTaskProfileResolvedEvent,
  buildTurnStartedEvent,
} from "./runtimeLifecycleProjection";

const baseContext = {
  sessionId: "session-1",
  threadId: "thread-1",
  runId: "agent_turn_stream:session-1",
  timestamp: "2026-05-09T00:00:00.000Z",
};

describe("runtimeLifecycleProjection", () => {
  it("应由 lifecycle owner 统一分发 thread_started", () => {
    const events = buildRuntimeLifecycleEvents(
      {
        type: "thread_started",
        thread_id: "thread-1",
      },
      baseContext,
    );

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: "session.opened",
      sourceType: "thread_started",
      sessionId: "session-1",
      threadId: "thread-1",
      owner: "session",
      scope: "thread",
      phase: "accepted",
      surface: "session_tabs",
      persistence: "snapshot",
    });
  });

  it("应把 turn_started 映射为 run.started", () => {
    const event = buildTurnStartedEvent(
      {
        type: "turn_started",
        turn: {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "整理今天的国际新闻",
          status: "running",
          started_at: "2026-05-09T00:00:00.000Z",
          created_at: "2026-05-09T00:00:00.000Z",
          updated_at: "2026-05-09T00:00:00.000Z",
        },
      },
      baseContext,
    );

    expect(event).toMatchObject({
      type: "run.started",
      sourceType: "turn_started",
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-1",
      owner: "runtime",
      scope: "turn",
      phase: "accepted",
      surface: "runtime_status",
      persistence: "snapshot",
      payload: {
        status: "running",
        promptLength: 9,
      },
    });
  });

  it("应由 lifecycle owner 统一分发 run terminal 事件", () => {
    const turn = {
      id: "turn-1",
      thread_id: "thread-1",
      prompt_text: "继续",
      status: "completed",
      started_at: "2026-05-09T00:00:00.000Z",
      completed_at: "2026-05-09T00:00:05.000Z",
      created_at: "2026-05-09T00:00:00.000Z",
      updated_at: "2026-05-09T00:00:05.000Z",
    };

    expect(
      buildRuntimeLifecycleEvents(
        {
          type: "turn_completed",
          turn,
          text: "完成",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.finished",
      sourceType: "turn_completed",
      owner: "runtime",
      scope: "run",
      phase: "completed",
      surface: "runtime_status",
      persistence: "archive",
    });

    expect(
      buildRuntimeLifecycleEvents(
        {
          type: "turn_failed",
          turn: {
            ...turn,
            status: "failed",
            error_message: "模型返回失败",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.failed",
      sourceType: "turn_failed",
      owner: "runtime",
      scope: "run",
      phase: "failed",
      surface: "runtime_status",
      persistence: "archive",
      payload: {
        errorPreview: "模型返回失败",
      },
    });
  });

  it("typed error 不应提前生成 run.failed，终态只由 turn_failed 负责", () => {
    expect(
      buildRuntimeLifecycleEvents(
        {
          message: "provider stream disconnected",
          protocol_method: "error",
          type: "error",
          will_retry: false,
        },
        baseContext,
      ),
    ).toEqual([]);

    expect(
      buildRuntimeLifecycleEvents(
        {
          message: "legacy synthetic error",
          type: "error",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      phase: "failed",
      sourceType: "error",
      type: "run.failed",
    });
  });

  it("应把 runtime_status 映射为 run.status、permission 与 team projection", () => {
    const events = buildRuntimeStatusEvents(
      {
        type: "runtime_status",
        status: {
          phase: "permission_review",
          title: "等待权限确认",
          detail: "需要批准 profile",
          checkpoints: ["plan", "permission"],
          metadata: {
            permission_status: "requires_confirmation",
            required_profile_keys: ["read_files"],
            ask_profile_keys: ["read_files"],
            blocking_profile_keys: [],
            decision_source: "runtime",
            decision_scope: "turn",
            confirmation_status: "not_requested",
            confirmation_request_id: "approval-1",
            confirmation_source: "policy",
            team_phase: "waiting",
            team_parallel_budget: 3,
            team_active_count: 1,
            team_queued_count: 2,
            provider_concurrency_group: "default",
            provider_parallel_budget: 4,
            queue_reason: "provider_limit",
            retryable_overload: true,
          },
        },
      },
      baseContext,
    );

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "run.status",
      sourceType: "runtime_status",
      owner: "runtime",
      scope: "run",
      phase: "waiting",
      surface: "runtime_status",
      runtimeStatus: "waiting",
      latestTurnStatus: "waiting",
      teamPhase: "waiting",
      teamParallelBudget: 3,
      teamQueuedCount: 2,
      payload: {
        title: "等待权限确认",
        checkpointCount: 2,
        providerConcurrencyGroup: "default",
      },
    });
    expect(events[1]).toMatchObject({
      type: "permission.changed",
      sourceType: "runtime_status",
      owner: "policy",
      scope: "run",
      phase: "waiting",
      payload: {
        permissionStatus: "requires_confirmation",
        confirmationRequestId: "approval-1",
      },
    });
  });

  it("应把 model_effective 映射为模型生效状态", () => {
    const event = buildModelEffectiveEvent(
      {
        type: "model_effective",
        modelRef: {
          providerId: "openai",
          modelId: "gpt-codex",
        },
        modelName: "gpt-codex",
        serviceModelSlot: "coding",
      },
      baseContext,
    );

    expect(event).toMatchObject({
      type: "run.status",
      sourceType: "model_effective",
      owner: "runtime",
      scope: "run",
      phase: "routing",
      surface: "runtime_status",
      payload: {
        model: "gpt-codex",
        mode: "coding",
      },
    });
  });

  it("应由 lifecycle owner 统一分发 model_change", () => {
    const events = buildRuntimeLifecycleEvents(
      {
        type: "model_change",
        model: "gpt-codex",
        mode: "fast",
      },
      baseContext,
    );

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: "run.status",
      sourceType: "model_change",
      owner: "runtime",
      scope: "run",
      phase: "routing",
      surface: "runtime_status",
      persistence: "snapshot",
      payload: {
        model: "gpt-codex",
        mode: "fast",
      },
    });
  });

  it("应把 task_profile_resolved 映射为 task.changed", () => {
    const event = buildTaskProfileResolvedEvent(
      {
        type: "task_profile_resolved",
        task_profile: {
          kind: "research",
          source: "runtime",
          traits: ["web"],
          routingSlot: "deep-search",
          permissionProfileKeys: ["read_files"],
        },
      },
      baseContext,
    );

    expect(event).toMatchObject({
      type: "task.changed",
      sourceType: "task_profile_resolved",
      owner: "task",
      scope: "run",
      phase: "routing",
      surface: "task_capsule",
      payload: {
        kind: "research",
        source: "runtime",
        traits: ["web"],
        routingSlot: "deep-search",
        permissionProfileKeys: ["read_files"],
      },
    });
  });
});
