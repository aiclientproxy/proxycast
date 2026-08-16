import React, { useState } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getMediaTaskArtifact,
  listMediaTaskArtifacts,
} from "@/lib/api/mediaTasks";
import type {
  ListMediaTaskArtifactsOutput,
  MediaTaskArtifactOutput,
} from "@/lib/api/agentRuntime/mediaTaskTypes";
import { changeLimeLocale } from "@/i18n/createI18n";
import { readFilePreview } from "@/lib/api/fileBrowser";
import { safeListen } from "@/lib/dev-bridge";
import type { Message } from "../types";
import { useWorkspaceTranscriptionTaskPreviewRuntime } from "./useWorkspaceTranscriptionTaskPreviewRuntime";

vi.mock("@/lib/api/mediaTasks", () => ({
  getMediaTaskArtifact: vi.fn(),
  listMediaTaskArtifacts: vi.fn(),
}));

vi.mock("@/lib/api/fileBrowser", () => ({
  readFilePreview: vi.fn(),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeListen: vi.fn(),
}));

type HookProps = Parameters<
  typeof useWorkspaceTranscriptionTaskPreviewRuntime
>[0];

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function buildRunningTranscriptionMessage(): Message {
  return {
    id: "assistant-transcription-1",
    role: "assistant",
    content: "转写任务已创建。",
    timestamp: new Date("2026-04-30T00:00:00.000Z"),
    taskPreview: {
      kind: "transcription_generate",
      taskId: "task-transcription-1",
      taskType: "transcription_generate",
      prompt: "请转写访谈音频",
      title: "内容转写任务",
      status: "running",
      artifactPath:
        ".lime/runtime/transcription-generate/task-transcription-1.md",
      taskFilePath:
        ".lime/tasks/transcription_generate/task-transcription-1.json",
      sourcePath: "materials/interview.wav",
      sourceUrl: null,
      transcriptPath: null,
      language: "zh-CN",
      outputFormat: "txt",
    },
  };
}

function buildCompletedTranscriptionArtifact(): MediaTaskArtifactOutput {
  return {
    success: true,
    task_id: "task-transcription-1",
    task_type: "transcription_generate",
    task_family: "audio",
    status: "succeeded",
    normalized_status: "succeeded",
    current_attempt_id: "attempt-1",
    path: ".lime/tasks/transcription_generate/task-transcription-1.json",
    absolute_path:
      "/workspace/.lime/tasks/transcription_generate/task-transcription-1.json",
    artifact_path:
      ".lime/tasks/transcription_generate/task-transcription-1.json",
    absolute_artifact_path:
      "/workspace/.lime/tasks/transcription_generate/task-transcription-1.json",
    reused_existing: false,
    record: {
      task_id: "task-transcription-1",
      task_type: "transcription_generate",
      task_family: "audio",
      payload: {
        prompt: "请转写访谈音频",
        source_path: "materials/interview.wav",
        provider_id: "openai-asr",
        model: "gpt-4o-transcribe",
        modality_contract_key: "audio_transcription",
        transcript: {
          kind: "transcript",
          status: "completed",
          transcript_path: ".lime/runtime/transcripts/task-transcription-1.txt",
          source_path: "materials/interview.wav",
          language: "zh-CN",
          output_format: "txt",
          provider_id: "openai-asr",
          model: "gpt-4o-transcribe",
        },
      },
      status: "succeeded",
      normalized_status: "succeeded",
      created_at: "2026-04-30T00:00:00.000Z",
      result: {
        kind: "transcription_result",
        status: "completed",
        transcript_path: ".lime/runtime/transcripts/task-transcription-1.txt",
        transcript: {
          kind: "transcript",
          status: "completed",
          transcript_path: ".lime/runtime/transcripts/task-transcription-1.txt",
        },
      },
    },
  };
}

function buildFailedTranscriptionArtifact(): MediaTaskArtifactOutput {
  return {
    success: true,
    task_id: "task-transcription-1",
    task_type: "transcription_generate",
    task_family: "audio",
    status: "failed",
    normalized_status: "failed",
    current_attempt_id: "attempt-1",
    path: ".lime/tasks/transcription_generate/task-transcription-1.json",
    absolute_path:
      "/workspace/.lime/tasks/transcription_generate/task-transcription-1.json",
    artifact_path:
      ".lime/tasks/transcription_generate/task-transcription-1.json",
    absolute_artifact_path:
      "/workspace/.lime/tasks/transcription_generate/task-transcription-1.json",
    reused_existing: false,
    record: {
      task_id: "task-transcription-1",
      task_type: "transcription_generate",
      task_family: "audio",
      payload: {
        prompt: "请转写访谈音频",
        source_path: "materials/interview.wav",
        provider_id: "missing-provider",
        model: "gpt-4o-transcribe",
        modality_contract_key: "audio_transcription",
        transcript: {
          kind: "transcript",
          status: "failed",
          source_path: "materials/interview.wav",
          provider_id: "missing-provider",
          model: "gpt-4o-transcribe",
          error_code: "transcription_provider_unconfigured",
          error_message:
            "未找到可用的 audio_transcription provider/API Key: missing-provider。",
          retryable: true,
          stage: "provider_config",
        },
      },
      status: "failed",
      normalized_status: "failed",
      created_at: "2026-04-30T00:00:00.000Z",
      last_error: {
        code: "transcription_provider_unconfigured",
        message:
          "未找到可用的 audio_transcription provider/API Key: missing-provider。",
        retryable: true,
        stage: "provider_config",
      },
      progress: {
        phase: "failed",
        message:
          "未找到可用的 audio_transcription provider/API Key: missing-provider。",
      },
    },
  };
}

function buildEmptyTranscriptionTaskIndex(): ListMediaTaskArtifactsOutput {
  return {
    success: true,
    workspace_root: "/workspace",
    artifact_root: "/workspace/.lime/tasks",
    filters: {
      task_family: "audio",
      task_type: "transcription_generate",
      modality_contract_key: "audio_transcription",
      limit: 24,
    },
    total: 0,
    modality_runtime_contracts: {
      snapshot_count: 0,
      contract_keys: [],
      execution_profile_keys: [],
      executor_adapter_keys: [],
      limecore_policy_refs: [],
      limecore_policy_snapshot_count: 0,
      limecore_policy_snapshot_statuses: [],
      limecore_policy_decisions: [],
      blocked_count: 0,
      routing_outcomes: [],
      model_registry_assessment_count: 0,
      audio_output_count: 0,
      audio_output_statuses: [],
      audio_output_error_codes: [],
      transcript_count: 0,
      transcript_statuses: [],
      transcript_error_codes: [],
      snapshots: [],
    },
    tasks: [],
  };
}

function buildCompletedTranscriptionTaskIndex(): ListMediaTaskArtifactsOutput {
  return {
    ...buildEmptyTranscriptionTaskIndex(),
    total: 1,
    modality_runtime_contracts: {
      snapshot_count: 1,
      contract_keys: ["audio_transcription"],
      execution_profile_keys: ["audio_transcription_profile"],
      executor_adapter_keys: ["app_server:mediaTaskArtifact/transcription/create"],
      limecore_policy_refs: [
        "model_catalog",
        "provider_offer",
        "tenant_feature_flags",
      ],
      limecore_policy_snapshot_count: 1,
      limecore_policy_snapshot_statuses: [
        { status: "local_defaults_evaluated", count: 1 },
      ],
      limecore_policy_decisions: ["allow"],
      blocked_count: 0,
      routing_outcomes: [{ outcome: "accepted", count: 1 }],
      model_registry_assessment_count: 0,
      audio_output_count: 0,
      audio_output_statuses: [],
      audio_output_error_codes: [],
      transcript_count: 1,
      transcript_statuses: [{ status: "completed", count: 1 }],
      transcript_error_codes: [],
      snapshots: [
        {
          task_id: "task-transcription-1",
          task_type: "transcription_generate",
          normalized_status: "succeeded",
          contract_key: "audio_transcription",
          routing_slot: "audio_transcription_model",
          provider_id: "openai-asr",
          model: "gpt-4o-transcribe",
          execution_profile_key: "audio_transcription_profile",
          executor_adapter_key: "app_server:mediaTaskArtifact/transcription/create",
          limecore_policy_refs: [
            "model_catalog",
            "provider_offer",
            "tenant_feature_flags",
          ],
          limecore_policy_snapshot_status: "local_defaults_evaluated",
          limecore_policy_decision: "allow",
          routing_event: "task_created",
          routing_outcome: "accepted",
          failure_code: null,
          transcript_status: "completed",
          transcript_path: ".lime/runtime/transcripts/task-transcription-1.txt",
          transcript_source_path: "materials/interview.wav",
          transcript_source_url: null,
          transcript_language: "zh-CN",
          transcript_output_format: "txt",
          transcript_error_code: null,
          transcript_retryable: null,
        },
      ],
    },
  };
}

function buildFailedTranscriptionTaskIndex(): ListMediaTaskArtifactsOutput {
  return {
    ...buildEmptyTranscriptionTaskIndex(),
    total: 1,
    modality_runtime_contracts: {
      snapshot_count: 1,
      contract_keys: ["audio_transcription"],
      execution_profile_keys: ["audio_transcription_profile"],
      executor_adapter_keys: ["app_server:mediaTaskArtifact/transcription/create"],
      limecore_policy_refs: [
        "model_catalog",
        "provider_offer",
        "tenant_feature_flags",
      ],
      limecore_policy_snapshot_count: 1,
      limecore_policy_snapshot_statuses: [
        { status: "local_defaults_evaluated", count: 1 },
      ],
      limecore_policy_decisions: ["allow"],
      blocked_count: 0,
      routing_outcomes: [{ outcome: "failed", count: 1 }],
      model_registry_assessment_count: 0,
      audio_output_count: 0,
      audio_output_statuses: [],
      audio_output_error_codes: [],
      transcript_count: 1,
      transcript_statuses: [{ status: "failed", count: 1 }],
      transcript_error_codes: ["transcription_provider_unconfigured"],
      snapshots: [
        {
          task_id: "task-transcription-1",
          task_type: "transcription_generate",
          normalized_status: "failed",
          contract_key: "audio_transcription",
          routing_slot: "audio_transcription_model",
          provider_id: "missing-provider",
          model: "gpt-4o-transcribe",
          execution_profile_key: "audio_transcription_profile",
          executor_adapter_key: "app_server:mediaTaskArtifact/transcription/create",
          limecore_policy_refs: [
            "model_catalog",
            "provider_offer",
            "tenant_feature_flags",
          ],
          limecore_policy_snapshot_status: "local_defaults_evaluated",
          limecore_policy_decision: "allow",
          routing_event: "task_created",
          routing_outcome: "failed",
          failure_code: "transcription_provider_unconfigured",
          transcript_status: "failed",
          transcript_path: null,
          transcript_source_path: "materials/interview.wav",
          transcript_source_url: null,
          transcript_language: "zh-CN",
          transcript_output_format: "txt",
          transcript_error_code: "transcription_provider_unconfigured",
          transcript_retryable: true,
        },
      ],
    },
  };
}

function buildFailedTranscriptionTaskIndexWithPolicyDeny(): ListMediaTaskArtifactsOutput {
  const output = buildFailedTranscriptionTaskIndex();
  const blockingRefs = ["model_catalog"];
  output.modality_runtime_contracts = {
    ...output.modality_runtime_contracts,
    blocked_count: 1,
    routing_outcomes: [{ outcome: "blocked", count: 1 }],
    limecore_policy_evaluation_statuses: [{ status: "evaluated", count: 1 }],
    limecore_policy_evaluation_decisions: ["deny"],
    limecore_policy_evaluation_decision_sources: ["policy_input_evaluator"],
    limecore_policy_evaluation_blocking_refs: blockingRefs,
    limecore_policy_evaluation_ask_refs: [],
    limecore_policy_evaluation_pending_refs: [],
    snapshots: output.modality_runtime_contracts.snapshots.map((snapshot) => ({
      ...snapshot,
      routing_outcome: "blocked",
      failure_code: "limecore_policy_denied",
      limecore_policy_snapshot_status: "policy_inputs_evaluated",
      limecore_policy_decision: "deny",
      limecore_policy_decision_source: "policy_input_evaluator",
      limecore_policy_decision_scope: "resolved_policy_inputs",
      limecore_policy_decision_reason: "model_catalog_capability_gap",
      limecore_policy_evaluation_status: "evaluated",
      limecore_policy_evaluation_decision: "deny",
      limecore_policy_evaluation_decision_source: "policy_input_evaluator",
      limecore_policy_evaluation_decision_scope: "resolved_policy_inputs",
      limecore_policy_evaluation_decision_reason:
        "model_catalog_capability_gap",
      limecore_policy_evaluation_blocking_refs: blockingRefs,
      limecore_policy_evaluation_ask_refs: [],
      limecore_policy_evaluation_pending_refs: [],
    })),
  };
  return output;
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestMessages: Message[] = props?.messages ?? [
    buildRunningTranscriptionMessage(),
  ];

  function Probe(currentProps: Partial<HookProps>) {
    const [messages, setMessages] = useState<Message[]>(
      currentProps.messages ?? [buildRunningTranscriptionMessage()],
    );
    latestMessages = messages;
    useWorkspaceTranscriptionTaskPreviewRuntime({
      projectRootPath: "/workspace",
      ...currentProps,
      messages,
      setChatMessages: setMessages,
    });
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ container, root });
  return {
    render,
    getMessages: () => latestMessages,
  };
}

beforeEach(async () => {
  await changeLimeLocale("zh-CN");
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(getMediaTaskArtifact).mockReset();
  vi.mocked(listMediaTaskArtifacts).mockReset();
  vi.mocked(readFilePreview).mockReset();
  vi.mocked(listMediaTaskArtifacts).mockResolvedValue(
    buildEmptyTranscriptionTaskIndex(),
  );
  vi.mocked(readFilePreview).mockResolvedValue({
    path: "/workspace/.lime/runtime/transcripts/task-transcription-1.txt",
    content: null,
    isBinary: false,
    size: 0,
    error: "not configured",
  });
  vi.mocked(safeListen).mockReset();
  vi.mocked(safeListen).mockResolvedValue(vi.fn());
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.restoreAllMocks();
});

describe("useWorkspaceTranscriptionTaskPreviewRuntime", () => {
  it("普通文本对话没有转写任务时不应启动转写任务旁路", async () => {
    const setIntervalSpy = vi.spyOn(window, "setInterval");
    const { render } = renderHook({
      messages: [
        {
          id: "assistant-news-1",
          role: "assistant",
          content: "今天国际新闻主要集中在能源、气候和地区安全。",
          timestamp: new Date("2026-04-30T00:00:00.000Z"),
        },
      ],
    });

    await render();
    await act(async () => {
      await Promise.resolve();
    });

    expect(listMediaTaskArtifacts).not.toHaveBeenCalled();
    expect(getMediaTaskArtifact).not.toHaveBeenCalled();
    expect(readFilePreview).not.toHaveBeenCalled();
    expect(safeListen).not.toHaveBeenCalled();
    expect(setIntervalSpy).not.toHaveBeenCalled();
  });

  it("应优先从统一媒体任务索引恢复 transcript 完成态而不读取隐藏 task JSON", async () => {
    vi.mocked(listMediaTaskArtifacts).mockResolvedValueOnce(
      buildCompletedTranscriptionTaskIndex(),
    );
    vi.mocked(readFilePreview).mockResolvedValueOnce({
      path: "/workspace/.lime/runtime/transcripts/task-transcription-1.txt",
      content: "欢迎来到 Lime 访谈节目。\n今天我们讨论多模态工作流。",
      isBinary: false,
      size: 31,
      error: null,
    });
    const { render, getMessages } = renderHook();

    await render();

    await vi.waitFor(() => {
      expect(listMediaTaskArtifacts).toHaveBeenCalledWith({
        projectRootPath: "/workspace",
        taskFamily: "audio",
        taskType: "transcription_generate",
        modalityContractKey: "audio_transcription",
        limit: 24,
      });
    });
    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "complete",
        transcriptPath: ".lime/runtime/transcripts/task-transcription-1.txt",
        sourcePath: "materials/interview.wav",
        language: "zh-CN",
        outputFormat: "txt",
        providerId: "openai-asr",
        model: "gpt-4o-transcribe",
        transcriptText: "欢迎来到 Lime 访谈节目。\n今天我们讨论多模态工作流。",
        statusMessage:
          "转写结果已同步，工作区已载入 transcript 文本，可直接复制校对。",
      });
    });
    expect(readFilePreview).toHaveBeenCalledWith(
      "/workspace/.lime/runtime/transcripts/task-transcription-1.txt",
      256 * 1024,
    );
    expect(getMediaTaskArtifact).not.toHaveBeenCalled();
  });

  it("应从 transcript 文件解析时间轴和说话人并写入 viewer 文档", async () => {
    vi.mocked(listMediaTaskArtifacts).mockResolvedValueOnce(
      buildCompletedTranscriptionTaskIndex(),
    );
    vi.mocked(readFilePreview).mockResolvedValueOnce({
      path: "/workspace/.lime/runtime/transcripts/task-transcription-1.vtt",
      content: `WEBVTT

00:00:01.000 --> 00:00:03.500
<v 主持人>欢迎来到 Lime 访谈。</v>

00:00:04.000 --> 00:00:06.000
嘉宾: 这次我们讲转写 viewer。
`,
      isBinary: false,
      size: 120,
      error: null,
    });
    const { render, getMessages } = renderHook();

    await render();

    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "complete",
        transcriptSegments: [
          {
            index: 1,
            startMs: 1000,
            endMs: 3500,
            speaker: "主持人",
            text: "欢迎来到 Lime 访谈。",
          },
          {
            index: 2,
            startMs: 4000,
            endMs: 6000,
            speaker: "嘉宾",
            text: "这次我们讲转写 viewer。",
          },
        ],
      });
    });
    expect(getMessages()[0]?.artifacts?.[0]).toMatchObject({
      meta: {
        artifactDocument: {
          blocks: expect.arrayContaining([
            expect.objectContaining({
              id: "transcript-segments",
              type: "table",
              title: "转写时间轴（可逐段编辑校对）",
              columns: ["时间", "说话人", "内容"],
              rows: [
                ["00:01 - 00:03", "主持人", "欢迎来到 Lime 访谈。"],
                ["00:04 - 00:06", "嘉宾", "这次我们讲转写 viewer。"],
              ],
            }),
          ]),
          metadata: {
            transcriptSegments: expect.arrayContaining([
              expect.objectContaining({
                speaker: "主持人",
                text: "欢迎来到 Lime 访谈。",
              }),
            ]),
          },
        },
      },
    });
    expect(getMediaTaskArtifact).not.toHaveBeenCalled();
  });

  it("应优先从统一媒体任务索引恢复 provider 失败且不保留旧 transcript_path", async () => {
    vi.mocked(listMediaTaskArtifacts).mockResolvedValueOnce(
      buildFailedTranscriptionTaskIndex(),
    );
    const runningWithStaleTranscriptPath = buildRunningTranscriptionMessage();
    if (
      runningWithStaleTranscriptPath.taskPreview?.kind ===
      "transcription_generate"
    ) {
      runningWithStaleTranscriptPath.taskPreview.transcriptPath =
        ".lime/runtime/transcripts/stale.txt";
    }
    const { render, getMessages } = renderHook({
      messages: [runningWithStaleTranscriptPath],
    });

    await render();

    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "failed",
        transcriptPath: null,
        providerId: "missing-provider",
        model: "gpt-4o-transcribe",
        errorCode: "transcription_provider_unconfigured",
        retryable: true,
        statusMessage:
          "转写 Provider 未配置，请先在转写设置中选择可用 Provider；任务保留在 transcription_generate，不会回退 frontend ASR。",
      });
    });
    expect(getMediaTaskArtifact).not.toHaveBeenCalled();
  });

  it("应从统一媒体任务索引把 LimeCore policy deny 恢复到任务卡 meta", async () => {
    vi.mocked(listMediaTaskArtifacts).mockResolvedValueOnce(
      buildFailedTranscriptionTaskIndexWithPolicyDeny(),
    );
    const { render, getMessages } = renderHook();

    await render();

    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "failed",
        metaItems: expect.arrayContaining([
          "LimeCore 策略输入阻断: model_catalog",
        ]),
      });
    });
    expect(getMediaTaskArtifact).not.toHaveBeenCalled();
  });

  it("应从完成态 transcription_generate task artifact 恢复 transcript viewer", async () => {
    vi.mocked(getMediaTaskArtifact).mockResolvedValueOnce(
      buildCompletedTranscriptionArtifact(),
    );
    vi.mocked(readFilePreview).mockResolvedValueOnce({
      path: "/workspace/.lime/runtime/transcripts/task-transcription-1.txt",
      content: "欢迎来到 Lime 访谈节目。\n今天我们讨论多模态工作流。",
      isBinary: false,
      size: 31,
      error: null,
    });
    const { render, getMessages } = renderHook();

    await render();

    await vi.waitFor(() => {
      expect(getMediaTaskArtifact).toHaveBeenCalledWith({
        projectRootPath: "/workspace",
        taskRef: ".lime/tasks/transcription_generate/task-transcription-1.json",
      });
    });
    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "complete",
        transcriptPath: ".lime/runtime/transcripts/task-transcription-1.txt",
        providerId: "openai-asr",
        model: "gpt-4o-transcribe",
        transcriptText: "欢迎来到 Lime 访谈节目。\n今天我们讨论多模态工作流。",
        statusMessage:
          "转写结果已同步，工作区已载入 transcript 文本，可直接复制校对。",
      });
    });
    expect(getMessages()[0]?.artifacts?.[0]).toMatchObject({
      title: "task-transcription-1.md",
      status: "complete",
      meta: {
        taskId: "task-transcription-1",
        taskType: "transcription_generate",
        transcriptPath: ".lime/runtime/transcripts/task-transcription-1.txt",
        artifactDocument: {
          status: "ready",
          metadata: {
            transcriptPath:
              ".lime/runtime/transcripts/task-transcription-1.txt",
            transcriptText:
              "欢迎来到 Lime 访谈节目。\n今天我们讨论多模态工作流。",
          },
        },
      },
    });
    expect(
      JSON.stringify(getMessages()[0]?.artifacts?.[0]?.meta.artifactDocument),
    ).toContain("转写文本（可编辑校对）");
  });

  it("应从失败态 transcription_generate task artifact 回流 provider 错误且不伪造 transcript_path", async () => {
    vi.mocked(getMediaTaskArtifact).mockResolvedValueOnce(
      buildFailedTranscriptionArtifact(),
    );
    const { render, getMessages } = renderHook();

    await render();

    await vi.waitFor(() => {
      expect(getMessages()[0]?.taskPreview).toMatchObject({
        kind: "transcription_generate",
        status: "failed",
        transcriptPath: null,
        errorCode: "transcription_provider_unconfigured",
        errorMessage:
          "未找到可用的 audio_transcription provider/API Key: missing-provider。",
        retryable: true,
        statusMessage:
          "转写 Provider 未配置，请先在转写设置中选择可用 Provider；任务保留在 transcription_generate，不会回退 frontend ASR。",
      });
    });
    expect(getMessages()[0]?.artifacts?.[0]).toMatchObject({
      title: "task-transcription-1.md",
      status: "error",
      meta: {
        taskId: "task-transcription-1",
        taskType: "transcription_generate",
        transcriptPath: null,
        errorCode: "transcription_provider_unconfigured",
        artifactDocument: {
          status: "failed",
          metadata: {
            transcriptPath: null,
            errorCode: "transcription_provider_unconfigured",
          },
        },
      },
    });
    expect(
      JSON.stringify(getMessages()[0]?.artifacts?.[0]?.meta.artifactDocument),
    ).toContain("不会回退 frontend ASR");
  });
});
