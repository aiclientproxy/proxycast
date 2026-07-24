import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  projectAppServerParamsForEvidence,
  readWorkflowJsonlEvents,
  summarizeWorkflowResumeLifecycle,
  workflowResumeBindingsFromTrace,
  workflowResumeEventBinding,
} from "./content-factory-production-workflow-evidence.mjs";

function writeJsonl(filePath, records) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    `${records.map((record) => JSON.stringify(record)).join("\n")}\n`,
    "utf8",
  );
}

describe("content factory production workflow evidence", () => {
  it("缺少 workflow JSONL 文件时返回空事件，collector 会 fail closed", () => {
    expect(
      readWorkflowJsonlEvents(
        path.join(os.tmpdir(), "missing-content-factory-workflow.jsonl"),
      ),
    ).toEqual([]);
  });

  it("从 workflow/respond trace 和 workflow JSONL 匹配 resume lifecycle", () => {
    const workflowJsonl = path.join(
      fs.mkdtempSync(path.join(os.tmpdir(), "content-factory-workflow-")),
      "workflow-events.jsonl",
    );
    writeJsonl(workflowJsonl, [
      {
        eventType: "workflow.step.resuming",
        payload: {
          actionId: "article-draft-review",
          decision: "approved",
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_prod:content-article",
        },
      },
      {
        event: {
          eventType: "workflow.run.resuming",
          payload: {
            action_id: "article-draft-review",
            decision: "approved",
            step_id: "draft",
            workflow_key: "content_article_workflow",
            workflow_run_id: "turn_prod:content-article",
          },
        },
      },
    ]);

    const traceBindings = workflowResumeBindingsFromTrace([
      {
        appServerRequests: [
          {
            method: "workflow/respond",
            params: {
              confirmed: true,
              request_id: "article-draft-review",
              metadata: {
                workflowResume: {
                  stepId: "draft",
                  workflowKey: "content_article_workflow",
                  workflowRunId: "turn_prod:content-article",
                },
              },
            },
          },
        ],
      },
    ]);
    const eventBindings = readWorkflowJsonlEvents(workflowJsonl)
      .map(workflowResumeEventBinding)
      .filter(Boolean);

    expect(
      summarizeWorkflowResumeLifecycle(traceBindings, eventBindings),
    ).toEqual(
      expect.objectContaining({
        actionId: "article-draft-review",
        auditEventsPresent: true,
        actionMetadataPresent: true,
        decision: "approved",
        stepId: "draft",
        workflowKey: "content_article_workflow",
        workflowRunId: "turn_prod:content-article",
      }),
    );
  });

  it("拒绝已退役的 agentSession/action/respond trace", () => {
    const traceBindings = workflowResumeBindingsFromTrace([
      {
        appServerRequests: [
          {
            method: "agentSession/action/respond",
            params: {
              confirmed: true,
              requestId: "article-draft-review",
              metadata: {
                workflowResume: {
                  stepId: "draft",
                  workflowKey: "content_article_workflow",
                  workflowRunId: "turn_prod:content-article",
                },
              },
            },
          },
        ],
      },
    ]);

    expect(traceBindings).toEqual([]);
  });

  it("拒绝非 canonical workflowResume metadata", () => {
    expect(
      workflowResumeBindingsFromTrace([
        {
          appServerRequests: [
            {
              method: "workflow/respond",
              params: {
                confirmed: true,
                requestId: "article-draft-review",
                metadata: {
                  stepId: "draft",
                  workflowKey: "content_article_workflow",
                  workflowRunId: "turn_prod:content-article",
                },
              },
            },
          ],
        },
      ]),
    ).toEqual([]);
  });

  it("App Server params evidence projection drops prompts, raw text, and secrets", () => {
    const projected = projectAppServerParamsForEvidence({
      authorization: "Bearer live-production-token-value",
      documentText: "完整正文不应写入 trace evidence",
      message: "用户原文不应写入 trace evidence",
      prompt: "帮我写一篇生产文章",
      sessionId: "sess_prod",
      threadId: "thread_prod",
      request_id: "article-draft-review",
      confirmed: true,
      metadata: {
        workflowResume: {
          stepId: "draft",
          workflowKey: "content_article_workflow",
          workflowRunId: "turn_prod:content-article",
        },
      },
    });

    expect(projected).toEqual({
      actionId: null,
      approved: undefined,
      confirmed: true,
      decision: null,
      requestId: "article-draft-review",
      sessionId: "sess_prod",
      threadId: "thread_prod",
      turnId: null,
      workflowResume: {
        stepId: "draft",
        workflowKey: "content_article_workflow",
        workflowRunId: "turn_prod:content-article",
      },
    });
    expect(JSON.stringify(projected)).not.toContain("帮我写");
    expect(JSON.stringify(projected)).not.toContain("完整正文");
    expect(JSON.stringify(projected)).not.toContain("Bearer");
  });
});
