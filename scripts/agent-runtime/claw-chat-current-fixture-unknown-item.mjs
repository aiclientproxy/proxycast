import {
  ASSISTANT_DONE_TEXT,
  NEWS_PROMPT,
} from "./claw-chat-current-fixture-constants.mjs";
import { collectReadModelItems } from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  readArray,
  readRecord,
  readString,
} from "./claw-chat-current-fixture-utils.mjs";

export const UNKNOWN_ITEM_SCENARIO = "unknown-item";
export const UNKNOWN_ITEM_PROMPT = NEWS_PROMPT;
export const UNKNOWN_ITEM_DONE_TEXT = ASSISTANT_DONE_TEXT;
export const UNKNOWN_ITEM_UPSTREAM_TYPE = "futureCapability";
export const UNKNOWN_ITEM_SECRET_MARKER = "UNKNOWN_ITEM_SECRET_MUST_NOT_LEAK";
export const UNKNOWN_ITEM_SAFE_FIELD_NAMES = [
  "[redacted]",
  "label",
  "opaquePayload",
  "status",
];

export function renderUnknownItemBackendEventsExpression() {
  return `
    ...(process.env.CLAW_CHAT_FIXTURE_SCENARIO === "${UNKNOWN_ITEM_SCENARIO}"
      ? [
          {
            type: "item.started",
            payload: {
              item: {
                id: "unknown-item-" + currentTurnIdForItem,
                threadId: currentThreadId(),
                turnId: currentTurnId(),
                type: "${UNKNOWN_ITEM_UPSTREAM_TYPE}",
                label: "future capability",
                opaquePayload: "opaque-value-must-not-render",
                secretToken: "${UNKNOWN_ITEM_SECRET_MARKER}",
                status: "inProgress"
              }
            }
          },
          {
            type: "item.completed",
            payload: {
              item: {
                id: "unknown-item-" + currentTurnIdForItem,
                threadId: currentThreadId(),
                turnId: currentTurnId(),
                type: "${UNKNOWN_ITEM_UPSTREAM_TYPE}",
                label: "future capability",
                opaquePayload: "opaque-value-must-not-render",
                secretToken: "${UNKNOWN_ITEM_SECRET_MARKER}",
                status: "completed"
              }
            }
          }
        ]
      : []),`;
}

export function readUnknownItemRecoveryEvidence(value) {
  const projectedItem = collectReadModelItems(value).find(
    (item) =>
      item?.type === "unknown_item" &&
      item?.upstream_type === UNKNOWN_ITEM_UPSTREAM_TYPE,
  );
  if (projectedItem) {
    return projectedItem;
  }

  const thread = readRecord(value?.thread);
  const threadId = readString(thread, "id", "threadId", "thread_id");
  for (const candidateTurn of readArray(thread, "turns")) {
    const turn = readRecord(candidateTurn);
    const item = readArray(turn, "items")
      .map((candidate) => readRecord(candidate))
      .find(
        (candidate) =>
          candidate?.type === "unknownItem" &&
          candidate?.upstreamType === UNKNOWN_ITEM_UPSTREAM_TYPE,
      );
    if (!item) continue;

    return {
      id: readString(item, "id"),
      thread_id: threadId,
      turn_id: readString(turn, "id", "turnId", "turn_id"),
      status: readString(turn, "status"),
      upstream_type: readString(item, "upstreamType"),
      field_names: readArray(item, "fieldNames").filter(
        (fieldName) => typeof fieldName === "string",
      ),
    };
  }
  return null;
}

export async function collectUnknownItemScenarioEvidence({ page, readModel }) {
  const unsupportedItem = page.locator(
    '[data-testid="timeline-unsupported-item"]',
    { hasText: UNKNOWN_ITEM_UPSTREAM_TYPE },
  );
  await unsupportedItem.waitFor({ state: "visible" });
  const gui = await page.evaluate(
    ({ upstreamType, secretMarker, safeFieldNames }) => {
      const rows = Array.from(
        document.querySelectorAll('[data-testid="timeline-unsupported-item"]'),
      );
      const target = rows.find((row) =>
        (row.textContent || "").includes(upstreamType),
      );
      const text = target?.textContent || "";
      const bodyText = document.body?.textContent || "";
      return {
        count: rows.length,
        visible: Boolean(target),
        upstreamTypeVisible: text.includes(upstreamType),
        safeFieldNamesVisible: safeFieldNames.every((name) =>
          text.includes(name),
        ),
        internalTypeHidden: !text.includes("unknown_item"),
        rawValuesHidden:
          !bodyText.includes(secretMarker) &&
          !bodyText.includes("opaque-value-must-not-render") &&
          !bodyText.includes("future capability"),
      };
    },
    {
      upstreamType: UNKNOWN_ITEM_UPSTREAM_TYPE,
      secretMarker: UNKNOWN_ITEM_SECRET_MARKER,
      safeFieldNames: UNKNOWN_ITEM_SAFE_FIELD_NAMES,
    },
  );
  const item = readUnknownItemRecoveryEvidence(readModel);
  const serializedReadModel = JSON.stringify(readModel || {});
  return {
    gui,
    readModel: {
      present: Boolean(item),
      itemId: item?.id ?? null,
      threadId: item?.thread_id ?? null,
      turnId: item?.turn_id ?? null,
      status: item?.status ?? null,
      upstreamType: item?.upstream_type ?? null,
      fieldNames: Array.isArray(item?.field_names) ? item.field_names : [],
      rawValuesHidden:
        !serializedReadModel.includes(UNKNOWN_ITEM_SECRET_MARKER) &&
        !serializedReadModel.includes("opaque-value-must-not-render") &&
        !serializedReadModel.includes("future capability"),
    },
  };
}

function backendEventTypesForPrompt(backendLedger) {
  const startIndex = backendLedger.findIndex(
    (entry) =>
      entry?.kind === "turnStart" && entry?.inputText === UNKNOWN_ITEM_PROMPT,
  );
  if (startIndex < 0) return [];
  const eventTypes = [];
  for (const entry of backendLedger.slice(startIndex + 1)) {
    if (entry?.kind === "turnStart") break;
    if (entry?.kind === "backendEmit" && Array.isArray(entry.eventTypes)) {
      eventTypes.push(...entry.eventTypes.filter(Boolean));
    }
  }
  return eventTypes;
}

function includesOrdered(eventTypes, expected) {
  let cursor = 0;
  for (const eventType of eventTypes) {
    if (eventType === expected[cursor]) cursor += 1;
    if (cursor === expected.length) return true;
  }
  return false;
}

export function buildUnknownItemScenarioAssertions({
  appServerRequestMethods,
  backendLedger,
  summary,
}) {
  const evidence = summary.unknownItem ?? {};
  const eventTypes = backendEventTypesForPrompt(backendLedger);
  return {
    unknownItemBackendLifecycleObserved: includesOrdered(eventTypes, [
      "item.started",
      "item.completed",
      "turn.completed",
    ]),
    unknownItemUsesCurrentAppServerRead:
      appServerRequestMethods.includes("turn/start") &&
      appServerRequestMethods.includes("thread/read"),
    unknownItemGuiFailVisible:
      evidence.gui?.visible === true &&
      evidence.gui?.upstreamTypeVisible === true,
    unknownItemGuiFieldsSanitized:
      evidence.gui?.safeFieldNamesVisible === true &&
      evidence.gui?.internalTypeHidden === true &&
      evidence.gui?.rawValuesHidden === true,
    unknownItemReadModelRecovered:
      evidence.readModel?.present === true &&
      evidence.readModel?.upstreamType === UNKNOWN_ITEM_UPSTREAM_TYPE &&
      evidence.readModel?.status === "completed",
    unknownItemReadModelFieldsSanitized:
      UNKNOWN_ITEM_SAFE_FIELD_NAMES.every((name) =>
        evidence.readModel?.fieldNames?.includes(name),
      ) && evidence.readModel?.rawValuesHidden === true,
  };
}
