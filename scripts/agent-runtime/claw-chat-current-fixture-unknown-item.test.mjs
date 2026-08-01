import { describe, expect, it } from "vitest";

import {
  buildUnknownItemScenarioAssertions,
  readUnknownItemRecoveryEvidence,
  renderUnknownItemBackendEventsExpression,
  UNKNOWN_ITEM_SAFE_FIELD_NAMES,
  UNKNOWN_ITEM_SCENARIO,
  UNKNOWN_ITEM_UPSTREAM_TYPE,
} from "./claw-chat-current-fixture-unknown-item.mjs";

function positiveInput() {
  return {
    appServerRequestMethods: ["turn/start", "thread/read"],
    backendLedger: [
      { kind: "turnStart", inputText: "整理今天的国际新闻" },
      {
        kind: "backendEmit",
        eventTypes: [
          "item.started",
          "item.completed",
          "provider.request.started",
          "turn.completed",
        ],
      },
    ],
    summary: {
      unknownItem: {
        gui: {
          visible: true,
          upstreamTypeVisible: true,
          safeFieldNamesVisible: true,
          internalTypeHidden: true,
          rawValuesHidden: true,
        },
        readModel: {
          present: true,
          upstreamType: UNKNOWN_ITEM_UPSTREAM_TYPE,
          status: "completed",
          fieldNames: UNKNOWN_ITEM_SAFE_FIELD_NAMES,
          rawValuesHidden: true,
        },
      },
    },
  };
}

describe("unknown Item current fixture", () => {
  it("renders one future Item lifecycle without adding a production fallback", () => {
    const script = renderUnknownItemBackendEventsExpression();

    expect(UNKNOWN_ITEM_SCENARIO).toBe("unknown-item");
    expect(script).toContain('type: "item.started"');
    expect(script).toContain('type: "item.completed"');
    expect(script).toContain(`type: "${UNKNOWN_ITEM_UPSTREAM_TYPE}"`);
    expect(script).toContain(
      'secretToken: "UNKNOWN_ITEM_SECRET_MUST_NOT_LEAK"',
    );
  });

  it("requires live, recovery and redaction evidence together", () => {
    const assertions = buildUnknownItemScenarioAssertions(positiveInput());

    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });

  it("reads the typed v2 unknown item from its parent thread and turn", () => {
    expect(
      readUnknownItemRecoveryEvidence({
        thread: {
          id: "thread-1",
          turns: [
            {
              id: "turn-1",
              status: "completed",
              items: [
                {
                  id: "unknown-item-1",
                  type: "unknownItem",
                  upstreamType: UNKNOWN_ITEM_UPSTREAM_TYPE,
                  fieldNames: UNKNOWN_ITEM_SAFE_FIELD_NAMES,
                },
              ],
            },
          ],
        },
      }),
    ).toEqual({
      id: "unknown-item-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      status: "completed",
      upstream_type: UNKNOWN_ITEM_UPSTREAM_TYPE,
      field_names: UNKNOWN_ITEM_SAFE_FIELD_NAMES,
    });
  });

  it("fails when the read model keeps raw values", () => {
    const input = positiveInput();
    input.summary.unknownItem.readModel.rawValuesHidden = false;

    expect(
      buildUnknownItemScenarioAssertions(input)
        .unknownItemReadModelFieldsSanitized,
    ).toBe(false);
  });
});
