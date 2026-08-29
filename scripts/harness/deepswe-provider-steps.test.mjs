import { describe, expect, it } from "vitest";

import {
  providerStepExhaustion,
  providerStepsFromCurrentFacts,
} from "./deepswe-adapter-core.mjs";

describe("DeepSWE provider step evidence", () => {
  it("summarizes output and enforces comparable token accounting", () => {
    const summary = providerStepsFromCurrentFacts(
      {
        threadRead: {
          providerSteps: [
            {
              sequence: 10,
              timestamp: "2026-07-16T00:00:00Z",
              attempt: 1,
              completed: true,
              finish_reason: "tool_call",
              toolNames: ["Read", "apply_patch", "Read"],
              text_output_chars: 7,
              reasoning_output_chars: 40,
              tool_call_count: 1,
              usage: {
                input_tokens: 100,
                output_tokens: 20,
                cached_input_tokens: 40,
              },
            },
            {
              sequence: 20,
              attempt: 2,
              completed: true,
              finish_reason: "stop",
              toolNames: ["exec_command", "apply_patch"],
              text_output_chars: 12,
              reasoning_output_chars: 60,
              tool_call_count: 0,
              usage: {
                input_tokens: 200,
                output_tokens: 30,
                cached_input_tokens: 50,
              },
            },
          ],
        },
      },
      { maxProviderSteps: 2, tokenBudget: 250 },
    );

    expect(summary).toMatchObject({
      stepCount: 2,
      usageStatus: "complete",
      usage: {
        inputTokens: 300,
        outputTokens: 50,
        cachedInputTokens: 90,
        budgetTokens: 260,
      },
      toolCatalog: {
        status: "complete",
        requestCount: 2,
        requestsWithTools: 2,
        uniqueToolNames: ["Read", "apply_patch", "exec_command"],
        applyPatchAvailableOnEveryRequest: true,
      },
      budgets: {
        exhausted: true,
        reasons: ["provider_steps", "token_budget"],
        remainingProviderSteps: 0,
        remainingTokens: 0,
      },
    });
    expect(summary.steps[0].output).toEqual({
      textChars: 7,
      reasoningChars: 40,
      toolCalls: 1,
    });
    expect(summary.steps[0].toolNames).toEqual(["Read", "apply_patch"]);
    expect(summary.steps[1].toolNames).toEqual(["apply_patch", "exec_command"]);
  });

  it("does not classify a natural final answer at the step limit as exhausted", () => {
    expect(
      providerStepExhaustion({
        stepCount: 2,
        budgets: { reasons: ["provider_steps"] },
        usage: { budgetTokens: 200 },
        steps: [{ finishReason: "tool_call" }, { finishReason: "stop" }],
      }),
    ).toBeNull();
  });

  it("prefers canonical runtime events when v2 thread/read omits audit facts", () => {
    const summary = providerStepsFromCurrentFacts({
      runtimeEvents: [
        {
          sequence: 10,
          timestamp: "2026-08-26T00:00:00Z",
          type: "provider.request.started",
          payload: { attempt: 1, tool_names: ["Read", "apply_patch"] },
        },
        {
          sequence: 20,
          timestamp: "2026-08-26T00:00:01Z",
          type: "provider.step",
          payload: {
            attempt: 1,
            completed: true,
            finish_reason: "tool_call",
            text_output_chars: 7,
            reasoning_output_chars: 8,
            tool_call_count: 1,
            usage: {
              input_tokens: 100,
              output_tokens: 20,
              cached_input_tokens: 40,
            },
          },
        },
      ],
      threadRead: { thread: { turns: [] } },
    });

    expect(summary).toMatchObject({
      stepCount: 1,
      usageStatus: "complete",
      usage: { budgetTokens: 80 },
      toolCatalog: {
        status: "complete",
        uniqueToolNames: ["Read", "apply_patch"],
        applyPatchAvailableOnEveryRequest: true,
      },
    });
    expect(summary.steps[0]).toMatchObject({
      sequence: 20,
      attempt: 1,
      toolNames: ["Read", "apply_patch"],
    });
  });
});
