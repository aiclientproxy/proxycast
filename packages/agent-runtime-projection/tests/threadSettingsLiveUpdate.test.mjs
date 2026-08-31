import test from "node:test";
import assert from "node:assert/strict";

import {
  buildThreadSettingsLiveUpdateProjectionEvent,
  extractThreadSettingsLiveUpdateSnapshot,
} from "../dist/index.js";

function settingsNotification(threadId, threadSettings) {
  return {
    method: "thread/settings/updated",
    params: {
      threadId,
      threadSettings,
    },
  };
}

function baseInput(overrides = {}) {
  return {
    threadId: "thread-settings",
    settingsUpdateRequest: {
      threadId: "thread-settings",
      model: "gpt-5.4",
      serviceTier: "fast",
      cwd: "/repo/next",
      collaborationMode: {
        mode: "default",
        settings: {
          model: "gpt-5.4",
        },
      },
    },
    settingsNotification: [
      settingsNotification("thread-settings", {
        model: "gpt-5.4",
        serviceTier: "fast",
        cwd: "/repo/next",
        collaborationMode: {
          mode: "default",
          settings: {
            model: "gpt-5.4",
          },
        },
      }),
    ],
    modelRequestsDuringSettingsUpdate: [],
    activeTurnBefore: {
      id: "turn-active",
      model: "mock-model",
      serviceTier: "standard",
      cwd: "/repo/current",
    },
    activeTurnAfter: {
      id: "turn-active",
      model: "mock-model",
      serviceTier: "standard",
      cwd: "/repo/current",
    },
    futureTurnRequired: true,
    futureTurnRequest: {
      model: "gpt-5.4",
      service_tier: "fast",
      environment_context:
        "<environment_context><cwd>/repo/next</cwd></environment_context>",
    },
    transcriptItems: [],
    readModelItems: [],
    cachedSessionBeforeAck: {
      model: "mock-model",
      serviceTier: "standard",
      cwd: "/repo/current",
    },
    cachedSessionAfterAck: {
      model: "mock-model",
      serviceTier: "standard",
      cwd: "/repo/current",
    },
    cachedSessionAfterNotification: {
      model: "gpt-5.4",
      serviceTier: "fast",
      cwd: "/repo/next",
    },
    ...overrides,
  };
}

test("thread settings live update uses notification facts and future turns without transcript pollution", () => {
  const event = buildThreadSettingsLiveUpdateProjectionEvent(baseInput(), {
    sequence: 211,
    sessionId: "session-thread",
    turnId: "turn-settings",
    timestamp: "2026-07-09T00:00:00.000Z",
  });

  assert.deepEqual(
    {
      type: event.type,
      sourceType: event.sourceType,
      sequence: event.sequence,
      sessionId: event.sessionId,
      threadId: event.threadId,
      turnId: event.turnId,
      owner: event.owner,
      scope: event.scope,
      phase: event.phase,
      surface: event.surface,
      persistence: event.persistence,
      control: event.control,
      runtimeStatus: event.runtimeStatus,
    },
    {
      type: "context.changed",
      sourceType: "thread_settings_live_update_projection",
      sequence: 211,
      sessionId: "session-thread",
      threadId: "thread-settings",
      turnId: "turn-settings",
      owner: "context",
      scope: "thread",
      phase: "completed",
      surface: "runtime_status",
      persistence: "snapshot",
      control: "none",
      runtimeStatus: "completed",
    },
  );
  assert.equal(event.payload.settingsNotificationSeen, true);
  assert.equal(event.payload.notificationMatchesExpected, true);
  assert.equal(event.payload.settingsOnlyModelRequestCount, 0);
  assert.equal(event.payload.activeTurnSettingsStable, true);
  assert.equal(event.payload.futureTurnUsesUpdatedModel, true);
  assert.equal(event.payload.futureTurnUsesUpdatedServiceTier, true);
  assert.equal(event.payload.futureTurnUsesUpdatedCwd, true);
  assert.equal(event.payload.transcriptClean, true);
  assert.deepEqual(event.payload.validationIssues, []);
});

test("thread settings live update rejects settings-only model requests and transcript/read-model leaks", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      modelRequestsDuringSettingsUpdate: [{ model: "gpt-5.4" }],
      transcriptItems: [
        {
          type: "thread_settings_updated",
          role: "assistant",
          surface: "conversation",
        },
      ],
      readModelItems: [
        {
          type: "thread_settings",
          model: "gpt-5.4",
        },
      ],
    }),
  );

  assert.equal(snapshot.transcriptClean, false);
  assert.equal(snapshot.readModelClean, false);
  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    [
      "settings_only_started_model_request",
      "settings_rendered_as_transcript_item",
      "settings_persisted_as_read_model_item",
    ],
  );
});

test("thread settings live update keeps active turn settings stable and applies changes to future turns", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      activeTurnAfter: {
        id: "turn-active",
        model: "gpt-5.4",
        serviceTier: "fast",
        cwd: "/repo/next",
      },
      futureTurnRequest: {
        model: "mock-model",
        service_tier: "standard",
        environment_context:
          "<environment_context><cwd>/repo/current</cwd></environment_context>",
      },
    }),
  );

  assert.equal(snapshot.activeTurnSettingsStable, false);
  assert.equal(snapshot.futureTurnUsesUpdatedModel, false);
  assert.equal(snapshot.futureTurnUsesUpdatedServiceTier, false);
  assert.equal(snapshot.futureTurnUsesUpdatedCwd, false);
  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    [
      "active_turn_settings_polluted",
      "future_turn_missing_updated_model",
      "future_turn_missing_updated_service_tier",
      "future_turn_missing_updated_cwd",
    ],
  );
});

test("thread settings live update updates cached session only from notification", () => {
  const ackMutated = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      cachedSessionAfterAck: {
        model: "gpt-5.4",
        serviceTier: "fast",
        cwd: "/repo/next",
      },
    }),
  );
  assert.deepEqual(
    ackMutated.validationIssues.map((item) => item.code),
    ["ack_updated_cached_session_without_notification"],
  );

  const notificationIgnored = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      cachedSessionAfterNotification: {
        model: "mock-model",
        serviceTier: "standard",
        cwd: "/repo/current",
      },
    }),
  );
  assert.deepEqual(
    notificationIgnored.validationIssues.map((item) => item.code),
    ["cached_session_not_updated_from_notification"],
  );
});

test("thread settings live update handles service tier clear as future request omission", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      settingsUpdateRequest: {
        threadId: "thread-settings",
        model: "gpt-5.4",
        serviceTier: null,
      },
      settingsNotification: [
        settingsNotification("thread-settings", {
          model: "gpt-5.4",
          serviceTier: "auto",
        }),
      ],
      futureTurnRequest: {
        model: "gpt-5.4",
      },
    }),
  );

  assert.equal(snapshot.futureTurnUsesUpdatedServiceTier, true);
  assert.deepEqual(snapshot.validationIssues, []);
});

test("thread settings live update reads permission provenance from the active profile", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      settingsUpdateRequest: {
        threadId: "thread-settings",
        permissions: ":workspace",
      },
      settingsNotification: [
        settingsNotification("thread-settings", {
          activePermissionProfile: { id: ":workspace" },
          sandboxPolicy: "workspace-write",
        }),
      ],
      expectedSettings: {
        permissions: ":workspace",
      },
    }),
  );

  assert.equal(snapshot.notificationMatchesExpected, true);
  assert.deepEqual(snapshot.validationIssues, []);
});

test("thread settings live update rejects sandboxPolicy combined with permissions without current error", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      settingsUpdateRequest: {
        threadId: "thread-settings",
        sandboxPolicy: "danger-full-access",
        permissions: ":workspace",
      },
      expectedSettings: {},
      invalidSettingsUpdateError: undefined,
    }),
  );

  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    ["sandbox_policy_combined_with_permissions"],
  );

  const rejected = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      settingsUpdateRequest: {
        threadId: "thread-settings",
        sandboxPolicy: "danger-full-access",
        permissions: ":workspace",
      },
      expectedSettings: {},
      invalidSettingsUpdateError: {
        status: "invalid_request",
      },
    }),
  );
  assert.deepEqual(rejected.validationIssues, []);
});

test("thread settings live update requires scoped settings notification", () => {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(
    baseInput({
      settingsNotification: [
        settingsNotification("other-thread", { model: "gpt-5.4" }),
      ],
    }),
  );

  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    ["missing_settings_notification", "settings_notification_wrong_thread"],
  );

  const turnOverride = extractThreadSettingsLiveUpdateSnapshot({
    threadId: "thread-settings",
    turnStartOverride: {
      threadId: "thread-settings",
      model: "gpt-5.4",
    },
  });
  assert.deepEqual(
    turnOverride.validationIssues.map((item) => item.code),
    [
      "missing_settings_notification",
      "turn_override_missing_settings_notification",
    ],
  );
});
