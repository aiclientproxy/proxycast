import { writeFile } from "node:fs/promises";

export async function writeTerminalExternalBackend(
  backendPath,
  { completedText, command, scenario = "complete" },
) {
  await writeFile(
    backendPath,
    `#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";

const input = JSON.parse(readFileSync(0, "utf8"));
const ledgerPath = process.argv[2];
const session = input.request.session ?? {};
const turn = input.request.turn ?? {};
const toolItem = (
  status,
  output,
  { callId = "terminal-command", name = "Bash" } = {},
) => ({
  sessionId: session.sessionId,
  threadId: session.threadId,
  turnId: turn.turnId,
  itemId: "item_" + callId,
  sequence: 1,
  ordinal: 1,
  createdAtMs: Date.now(),
  updatedAtMs: Date.now(),
  ...(status === "completed" ? { completedAtMs: Date.now() } : {}),
  kind: "tool",
  status,
  payload: {
    type: "tool",
    call_id: callId,
    name,
    arguments: [{ name: "command", value: ${JSON.stringify(command)} }],
    ...(output
      ? {
          output: {
            text: output.text,
            truncated: false,
          },
        }
      : {}),
  },
  metadata: {},
});
let events = [];
if (input.kind === "turnStart") {
  if (${JSON.stringify(scenario)} === "approval") {
    events = [
      {
        type: "item.started",
        payload: { item: toolItem("inProgress") },
      },
      {
        type: "action.required",
        payload: {
          actionType: "tool_confirmation",
          actionKind: "tool_execution_policy",
          requestId: "terminal-approval",
          actionId: "terminal-approval",
          toolCallId: "terminal-command",
          toolName: "Bash",
          toolFamily: "shell_command",
          runtime_contract: {
            contract_key: "shell_command",
            tool_family: "shell_command",
            session_cache_supported: false,
          },
          approvalScope: {
            contractKey: "shell_command",
            toolFamily: "shell_command",
            riskClass: "shell_command_requires_approval",
            workingDirHash: "sha256:test",
          },
          prompt: "Allow terminal command?",
          arguments: { command: ${JSON.stringify(command)} },
          cwd: "/tmp",
          availableDecisions: ["allow_once", "decline", "cancel"],
        },
      },
    ];
  } else if (${JSON.stringify(scenario)} === "user-input") {
    events = [
      {
        type: "item.started",
        payload: {
          item: toolItem("inProgress", undefined, {
            callId: "terminal-question",
            name: "request_user_input",
          }),
        },
      },
      {
        type: "action.required",
        payload: {
          actionType: "ask_user",
          requestId: "terminal-user-input",
          toolCallId: "terminal-question",
          questions: [
            {
              id: "mode",
              header: "Mode",
              question: "Choose a mode",
              options: [
                { value: "fast", label: "Fast", description: "Continue quickly" },
                { value: "safe", label: "Safe", description: "Review every step" },
              ],
            },
          ],
        },
      },
    ];
  } else if (
    ${JSON.stringify(scenario)} === "interrupt" ||
    ${JSON.stringify(scenario)} === "queue-edit"
  ) {
    events = [
      {
        type: "message.delta",
        payload: {
          itemId: "terminal-assistant",
          text:
            ${JSON.stringify(scenario)} === "queue-edit"
              ? "QUEUE_EDIT_READY"
              : "INTERRUPT_READY",
        },
      },
    ];
  } else if (${JSON.stringify(scenario)} === "failure") {
    events = [
      { type: "runtime.error", payload: { message: "fixture backend failure", willRetry: false } },
      { type: "turn.failed", payload: { status: "failed", error: { message: "fixture backend failure" } } },
    ];
  } else {
    events = [
      {
        type: "message.delta",
        payload: {
          itemId: "terminal-assistant",
          role: "assistant",
          text: ${JSON.stringify(completedText)},
        },
      },
      {
        type: "item.started",
        payload: { item: toolItem("inProgress") },
      },
      {
        type: "item.completed",
        payload: {
          item: toolItem("completed", { text: "terminal-gate-b" }),
        },
      },
      { type: "turn.completed", payload: { status: "completed" } },
    ];
  }
} else if (input.kind === "actionRespond") {
  const decision = input.request.decision ?? null;
  const canceled = decision === "cancel";
  const isAskUser = String(input.request.actionType ?? "").toLowerCase().includes("ask");
  const toolCallId = isAskUser
    ? "terminal-question"
    : "terminal-command";
  events = [
    {
      type: canceled ? "action.canceled" : "action.resolved",
      payload: {
        requestId: input.request.requestId,
        actionId: input.request.requestId,
        actionType: input.request.actionType,
        toolCallId,
        decision,
        confirmed: !canceled,
        scope: input.request.actionScope ?? null,
      },
    },
    { type: "message.delta", payload: { itemId: "terminal-assistant", text: ${JSON.stringify(completedText)} } },
    {
      type: "item.completed",
      payload: {
        item: toolItem(
          "completed",
          { text: "terminal-gate-b" },
          isAskUser
            ? { callId: "terminal-question", name: "request_user_input" }
            : undefined,
        ),
      },
    },
    { type: "turn.completed", payload: { status: "completed" } },
  ];
} else if (input.kind === "turnCancel") {
  events = [{ type: "turn.canceled", payload: { status: "canceled" } }];
}
appendFileSync(
  ledgerPath,
  JSON.stringify({
    kind: input.kind,
    inputText: (input.request.input?.parts ?? [])
      .map((part) => part?.Text?.text ?? "")
      .join(""),
    threadId: input.request.session?.threadId ?? null,
    turnId: input.request.turn?.turnId ?? null,
    requestId: input.request.requestId ?? null,
    decision: input.request.decision ?? null,
    scenario: ${JSON.stringify(scenario)},
    eventTypes: events.map((event) => event.type),
  }) + "\\n",
);
console.log(JSON.stringify({ events }));
if (
  input.kind === "turnStart" &&
  (${JSON.stringify(scenario)} === "interrupt" ||
    ${JSON.stringify(scenario)} === "queue-edit")
) {
  // An unresolved Promise alone does not keep Node's event loop alive. Keep
  // one active handle so the App Server can deliver a real turn/interrupt.
  setInterval(() => {}, 1000);
  await new Promise(() => {});
}
`,
  );
}
