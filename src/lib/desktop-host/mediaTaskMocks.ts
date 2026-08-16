type MockMediaTaskProtocol = {
  taskType: "image_generate" | "audio_generate" | "transcription_generate";
  taskFamily: "image" | "audio";
  defaultTaskId: string;
  contractKey: "image_generation" | "voice_generation" | "audio_transcription";
  modality: "image" | "audio";
  requiredCapabilities: string[];
  routingSlot:
    | "image_generation_model"
    | "voice_generation_model"
    | "audio_transcription_model";
  runtimeContract: Record<string, unknown>;
};

type MockMediaTaskOutput = ReturnType<typeof buildMockMediaTaskOutput>;

function normalizeMockMediaTaskId(
  taskRef?: string,
  fallbackTaskId = "task-image-mock-1",
): string {
  const raw = (taskRef || fallbackTaskId).trim();
  if (!raw) {
    return fallbackTaskId;
  }

  const normalizedPath = raw.replace(/\\/g, "/");
  const lastSegment =
    normalizedPath.split("/").filter(Boolean).pop()?.trim() || normalizedPath;
  const baseName = lastSegment.replace(/\.json$/i, "").trim() || lastSegment;
  const normalized = baseName.replace(/[^a-zA-Z0-9_-]+/g, "-");
  return normalized || fallbackTaskId;
}

function buildMockLimeCorePolicySnapshot(refs: string[]) {
  return {
    status: "local_defaults_evaluated",
    decision: "allow",
    source: "modality_runtime_contract",
    decision_source: "local_default_policy",
    decision_scope: "local_defaults_only",
    decision_reason: "declared_policy_refs_with_no_local_deny_rule",
    refs,
    evaluated_refs: [],
    unresolved_refs: [...refs],
    missing_inputs: [...refs],
    policy_inputs: refs.map((refKey) => ({
      ref_key: refKey,
      status: "declared_only",
      source: "modality_runtime_contract",
      value_source: "limecore_pending",
    })),
    pending_hit_refs: [...refs],
    policy_value_hits: [],
    policy_value_hit_count: 0,
    policy_evaluation: {
      status: "input_gap",
      decision: "ask",
      decision_source: "policy_input_evaluator",
      decision_scope: "pending_policy_inputs",
      decision_reason: "declared_policy_refs_missing_inputs",
      blocking_refs: [],
      ask_refs: [...refs],
      pending_refs: [...refs],
    },
  };
}

function resolveMockMediaTaskProtocol(
  request: Record<string, any>,
  overrides?: Partial<Record<string, unknown>>,
): MockMediaTaskProtocol {
  const requestedTaskType =
    typeof overrides?.task_type === "string"
      ? overrides.task_type
      : (request.taskType ?? request.task_type);
  const requestedTaskFamily = request.taskFamily ?? request.task_family;
  const requestedContractKey =
    request.modalityContractKey ?? request.modality_contract_key;
  const requestedModality = request.modality;
  const requestedTaskRef =
    typeof request.taskRef === "string" ? request.taskRef.toLowerCase() : "";
  const isTranscription =
    requestedTaskType === "transcription_generate" ||
    requestedContractKey === "audio_transcription" ||
    requestedTaskRef.includes("transcription_generate");
  const isAudio =
    requestedTaskType === "audio_generate" ||
    requestedTaskFamily === "audio" ||
    requestedContractKey === "voice_generation" ||
    requestedModality === "audio" ||
    requestedTaskRef.includes("audio_generate");

  if (isTranscription) {
    const limecorePolicyRefs = [
      "model_catalog",
      "provider_offer",
      "tenant_feature_flags",
    ];
    return {
      taskType: "transcription_generate",
      taskFamily: "audio",
      defaultTaskId: "task-transcription-mock-1",
      contractKey: "audio_transcription",
      modality: "audio",
      requiredCapabilities: ["text_generation", "audio_transcription"],
      routingSlot: "audio_transcription_model",
      runtimeContract: {
        contract_key: "audio_transcription",
        modality: "audio",
        required_capabilities: ["text_generation", "audio_transcription"],
        routing_slot: "audio_transcription_model",
        executor_binding: {
          executor_kind: "skill",
          binding_key: "transcription_generate",
        },
        execution_profile: {
          profile_key: "audio_transcription_profile",
        },
        executor_adapter: {
          adapter_key: "skill:transcription_generate",
        },
        limecore_policy_refs: limecorePolicyRefs,
        limecore_policy_snapshot:
          buildMockLimeCorePolicySnapshot(limecorePolicyRefs),
      },
    };
  }

  if (isAudio) {
    const limecorePolicyRefs = [
      "client_scenes",
      "tenant_feature_flags",
      "provider_offer",
    ];
    return {
      taskType: "audio_generate",
      taskFamily: "audio",
      defaultTaskId: "task-audio-mock-1",
      contractKey: "voice_generation",
      modality: "audio",
      requiredCapabilities: ["text_generation", "voice_generation"],
      routingSlot: "voice_generation_model",
      runtimeContract: {
        contract_key: "voice_generation",
        modality: "audio",
        required_capabilities: ["text_generation", "voice_generation"],
        routing_slot: "voice_generation_model",
        route_execution_status: "executable",
        executor_binding: {
          executor_kind: "app_server",
          binding_key: "mediaTaskArtifact/audio/create",
        },
        execution_profile: {
          profile_key: "voice_generation_profile",
        },
        limecore_policy_refs: limecorePolicyRefs,
        limecore_policy_snapshot:
          buildMockLimeCorePolicySnapshot(limecorePolicyRefs),
      },
    };
  }

  const limecorePolicyRefs = [
    "model_catalog",
    "provider_offer",
    "tenant_feature_flags",
  ];
  return {
    taskType: "image_generate",
    taskFamily: "image",
    defaultTaskId: "task-image-mock-1",
    contractKey: "image_generation",
    modality: "image",
    requiredCapabilities: [
      "text_generation",
      "image_generation",
      "vision_input",
    ],
    routingSlot: "image_generation_model",
    runtimeContract: {
      contract_key: "image_generation",
      modality: "image",
      required_capabilities: [
        "text_generation",
        "image_generation",
        "vision_input",
      ],
      routing_slot: "image_generation_model",
      executor_binding: {
        executor_kind: "skill",
        binding_key: "image_generate",
      },
      execution_profile: {
        profile_key: "image_generation_profile",
      },
      executor_adapter: {
        adapter_key: "skill:image_generate",
      },
      limecore_policy_refs: limecorePolicyRefs,
      limecore_policy_snapshot:
        buildMockLimeCorePolicySnapshot(limecorePolicyRefs),
    },
  };
}

function buildMockMediaTaskOutput(
  args: any,
  overrides?: Partial<Record<string, unknown>>,
) {
  const request = args?.request ?? args ?? {};
  const protocol = resolveMockMediaTaskProtocol(request, overrides);
  const taskId = normalizeMockMediaTaskId(
    typeof request.taskRef === "string" ? request.taskRef : undefined,
    protocol.defaultTaskId,
  );
  const projectRootPath =
    typeof request.projectRootPath === "string" &&
    request.projectRootPath.trim()
      ? request.projectRootPath.trim()
      : "/mock/workspace";
  const promptCandidates = [
    request.prompt,
    request.sourceText,
    request.source_text,
    request.text,
  ];
  const prompt =
    promptCandidates
      .find((value) => typeof value === "string" && value.trim())
      ?.trim() ??
    (protocol.taskFamily === "audio" ? "mock audio task" : "mock image task");
  const status =
    typeof overrides?.status === "string" ? overrides.status : "pending_submit";
  const normalizedStatus =
    typeof overrides?.normalized_status === "string"
      ? overrides.normalized_status
      : status === "cancelled"
        ? "cancelled"
        : status === "failed"
          ? "failed"
          : "pending";
  const attemptCount =
    typeof overrides?.attempt_count === "number" ? overrides.attempt_count : 1;
  const currentAttemptId =
    typeof overrides?.current_attempt_id === "string"
      ? overrides.current_attempt_id
      : `attempt-${attemptCount}`;
  const createdAt = "2026-04-04T00:00:00.000Z";
  const path = `.lime/tasks/${protocol.taskType}/${taskId}.json`;
  const runtimeContract =
    request.runtimeContract ??
    request.runtime_contract ??
    protocol.runtimeContract;
  const costState = request.costState ?? request.cost_state ?? null;
  const limitState = request.limitState ?? request.limit_state ?? null;
  const limitEvent = request.limitEvent ?? request.limit_event ?? null;
  const runtimeSummary =
    request.runtimeSummary ?? request.runtime_summary ?? null;
  const taskProfile = request.taskProfile ?? request.task_profile ?? null;
  const sessionId = request.sessionId ?? request.session_id ?? null;
  const threadId = request.threadId ?? request.thread_id ?? null;
  const turnId = request.turnId ?? request.turn_id ?? null;
  const projectId = request.projectId ?? request.project_id ?? null;
  const contentId = request.contentId ?? request.content_id ?? null;
  const payload =
    protocol.taskType === "transcription_generate"
      ? {
          prompt,
          raw_text: request.rawText ?? request.raw_text ?? null,
          source_url: request.sourceUrl ?? request.source_url ?? null,
          source_path: request.sourcePath ?? request.source_path ?? null,
          language: request.language ?? null,
          output_format: request.outputFormat ?? request.output_format ?? "txt",
          speaker_labels:
            request.speakerLabels ?? request.speaker_labels ?? null,
          timestamps: request.timestamps ?? null,
          provider_id: request.providerId ?? request.provider_id ?? null,
          model: request.model ?? null,
          session_id: sessionId,
          thread_id: threadId,
          turn_id: turnId,
          project_id: projectId,
          content_id: contentId,
          entry_source:
            request.entrySource ??
            request.entry_source ??
            "at_transcription_command",
          modality_contract_key:
            request.modalityContractKey ??
            request.modality_contract_key ??
            protocol.contractKey,
          modality: request.modality ?? protocol.modality,
          required_capabilities:
            request.requiredCapabilities ??
            request.required_capabilities ??
            protocol.requiredCapabilities,
          routing_slot:
            request.routingSlot ?? request.routing_slot ?? protocol.routingSlot,
          runtime_contract: runtimeContract,
          requested_target:
            request.requestedTarget ?? request.requested_target ?? "transcript",
          transcript: {
            kind: "transcript",
            status: request.transcriptStatus ?? "pending",
            transcript_path:
              request.transcriptPath ?? request.transcript_path ?? null,
            source_url: request.sourceUrl ?? request.source_url ?? null,
            source_path: request.sourcePath ?? request.source_path ?? null,
            language: request.language ?? null,
            output_format:
              request.outputFormat ?? request.output_format ?? "txt",
            provider_id: request.providerId ?? request.provider_id ?? null,
            model: request.model ?? null,
            modality_contract_key: protocol.contractKey,
            modality: protocol.modality,
            routing_slot: protocol.routingSlot,
          },
        }
      : protocol.taskFamily === "audio"
        ? {
            prompt,
            source_text: prompt,
            raw_text: request.rawText ?? request.raw_text ?? null,
            voice: request.voice ?? null,
            voice_style: request.voiceStyle ?? request.voice_style ?? null,
            target_language:
              request.targetLanguage ?? request.target_language ?? null,
            provider_id: request.providerId ?? request.provider_id ?? null,
            model: request.model ?? null,
            session_id: sessionId,
            thread_id: threadId,
            turn_id: turnId,
            project_id: projectId,
            content_id: contentId,
            entry_source:
              request.entrySource ?? request.entry_source ?? "at_voice_command",
            modality_contract_key:
              request.modalityContractKey ??
              request.modality_contract_key ??
              protocol.contractKey,
            modality: request.modality ?? protocol.modality,
            required_capabilities:
              request.requiredCapabilities ??
              request.required_capabilities ??
              protocol.requiredCapabilities,
            routing_slot:
              request.routingSlot ??
              request.routing_slot ??
              protocol.routingSlot,
            runtime_contract: runtimeContract,
            audio_output: {
              kind: "audio_output",
              status: "pending",
              audio_path: request.audioPath ?? request.audio_path ?? null,
              mime_type: request.mimeType ?? request.mime_type ?? "audio/mpeg",
              duration_ms: request.durationMs ?? request.duration_ms ?? null,
              source_text: prompt,
              voice: request.voice ?? null,
              voice_style: request.voiceStyle ?? request.voice_style ?? null,
              target_language:
                request.targetLanguage ?? request.target_language ?? null,
            },
          }
        : {
            prompt,
            mode: request.mode ?? "generate",
            size: request.size ?? "1024x1024",
            count: request.count ?? 1,
            persona_context:
              request.personaContext ?? request.persona_context ?? null,
            presentation: request.presentation ?? null,
            taste_context:
              request.tasteContext ?? request.taste_context ?? null,
            provider_id: request.providerId ?? request.provider_id ?? null,
            model: request.model ?? null,
            session_id: sessionId,
            thread_id: threadId,
            turn_id: turnId,
            project_id: projectId,
            content_id: contentId,
            entry_source:
              request.entrySource ?? request.entry_source ?? "at_image_command",
            modality_contract_key:
              request.modalityContractKey ??
              request.modality_contract_key ??
              protocol.contractKey,
            modality: request.modality ?? protocol.modality,
            required_capabilities:
              request.requiredCapabilities ??
              request.required_capabilities ??
              protocol.requiredCapabilities,
            routing_slot:
              request.routingSlot ??
              request.routing_slot ??
              protocol.routingSlot,
            runtime_contract: runtimeContract,
            model_capability_assessment:
              request.modelCapabilityAssessment ??
              request.model_capability_assessment ??
              null,
          };
  for (const [key, value] of Object.entries({
    cost_state: costState,
    limit_state: limitState,
    limit_event: limitEvent,
    runtime_summary: runtimeSummary,
    task_profile: taskProfile,
  })) {
    if (value !== null && value !== undefined) {
      (payload as Record<string, unknown>)[key] = value;
    }
  }
  const record = {
    task_id: taskId,
    task_type: protocol.taskType,
    task_family: protocol.taskFamily,
    title: request.title ?? null,
    summary: "mock media task",
    payload,
    status,
    normalized_status: normalizedStatus,
    created_at: createdAt,
    current_attempt_id: currentAttemptId,
    retry_count: Math.max(attemptCount - 1, 0),
    last_error: overrides?.last_error ?? null,
    attempts: Array.from({ length: attemptCount }, (_, index) => ({
      attempt_id: `attempt-${index + 1}`,
      attempt_index: index + 1,
      status: index === attemptCount - 1 ? status : "cancelled",
      input_snapshot: {
        prompt,
      },
    })),
  };

  return {
    success: true,
    task_id: taskId,
    task_type: protocol.taskType,
    task_family: protocol.taskFamily,
    status,
    normalized_status: normalizedStatus,
    current_attempt_id: currentAttemptId,
    attempt_count: attemptCount,
    path,
    absolute_path: `${projectRootPath}/${path}`,
    artifact_path: path,
    absolute_artifact_path: `${projectRootPath}/${path}`,
    reused_existing: false,
    record,
    ...overrides,
  };
}

function buildMockCompletedAudioTaskOutput(args: any) {
  const request = args?.request ?? args ?? {};
  const output = buildMockMediaTaskOutput(args, {
    task_type: "audio_generate",
    status: "succeeded",
    normalized_status: "succeeded",
  });
  const record = output.record as Record<string, any>;
  const payload = record.payload as Record<string, any>;
  const audioPath =
    request.audioPath ??
    request.audio_path ??
    request.audioUrl ??
    request.audio_url ??
    payload.audio_path ??
    "/mock/workspace/.lime/runtime/audio/task-audio-mock-1.mp3";
  const mimeType =
    request.mimeType ?? request.mime_type ?? payload.mime_type ?? "audio/mpeg";
  const durationMs =
    request.durationMs ??
    request.duration_ms ??
    payload.duration_ms ??
    payload.audio_output?.duration_ms ??
    1200;
  const providerId =
    request.providerId ?? request.provider_id ?? payload.provider_id ?? null;
  const model = request.model ?? payload.model ?? null;
  const audioOutput = {
    ...(payload.audio_output ?? {}),
    kind: "audio_output",
    status: "completed",
    audio_path: audioPath,
    mime_type: mimeType,
    duration_ms: durationMs,
    provider_id: providerId,
    model,
    modality_contract_key: "voice_generation",
    modality: "audio",
    routing_slot: "voice_generation_model",
  };

  payload.audio_path = audioPath;
  payload.mime_type = mimeType;
  payload.duration_ms = durationMs;
  payload.provider_id = providerId;
  payload.model = model;
  payload.audio_output = audioOutput;
  record.result = {
    kind: "audio_generation_result",
    status: "completed",
    audio_output: audioOutput,
    outputs: [audioOutput],
    audio_path: audioPath,
    mime_type: mimeType,
    duration_ms: durationMs,
  };
  record.progress = {
    phase: "succeeded",
    percent: 100,
    message: "音频任务已完成，audio_output 已回写。",
    preview_slots: [],
  };

  return output;
}

function readPayloadString(...values: unknown[]) {
  const value = values.find(
    (candidate) => typeof candidate === "string" && candidate.trim().length > 0,
  );
  return typeof value === "string" ? value.trim() : null;
}

function countByStatus<T extends Record<string, unknown>>(
  snapshots: T[],
  key: keyof T,
) {
  return snapshots.reduce(
    (items, snapshot) => {
      const status = snapshot[key];
      if (typeof status !== "string" || !status) {
        return items;
      }
      const existing = items.find((item) => item.status === status);
      if (existing) {
        existing.count += 1;
      } else {
        items.push({ status, count: 1 });
      }
      return items;
    },
    [] as Array<{ status: string; count: number }>,
  );
}

function buildMediaTaskContractSnapshot(item: MockMediaTaskOutput) {
  const payload = item.record?.payload as Record<string, any> | undefined;
  const lastError = item.record?.last_error as
    | { code?: string }
    | null
    | undefined;
  const assessment = payload?.model_capability_assessment as
    | Record<string, any>
    | null
    | undefined;
  const audioOutput = payload?.audio_output as
    | Record<string, any>
    | null
    | undefined;
  const transcript = payload?.transcript as
    | Record<string, any>
    | null
    | undefined;
  const taskProfile = (payload?.task_profile ?? payload?.taskProfile) as
    | Record<string, any>
    | null
    | undefined;
  const runtimeSummary = (payload?.runtime_summary ??
    payload?.runtimeSummary) as Record<string, any> | null | undefined;
  const costStatePayload = (payload?.cost_state ??
    payload?.costState ??
    taskProfile?.cost_state ??
    taskProfile?.costState) as Record<string, any> | string | null | undefined;
  const limitStatePayload = (payload?.limit_state ??
    payload?.limitState ??
    taskProfile?.limit_state ??
    taskProfile?.limitState) as Record<string, any> | string | null | undefined;
  const limitEventPayload = (payload?.limit_event ??
    payload?.limitEvent ??
    (typeof limitStatePayload === "object"
      ? (limitStatePayload?.limit_event ??
        limitStatePayload?.limitEvent ??
        limitStatePayload?.event)
      : null)) as Record<string, any> | null | undefined;
  const entryKey = readPayloadString(
    payload?.entry_key,
    payload?.entryKey,
    payload?.entry_source,
    payload?.entrySource,
  );
  const threadId = readPayloadString(payload?.thread_id, payload?.threadId);
  const turnId = readPayloadString(payload?.turn_id, payload?.turnId);
  const contentId = readPayloadString(payload?.content_id, payload?.contentId);
  const modality = readPayloadString(payload?.modality);
  const modelId = readPayloadString(
    payload?.model_id,
    payload?.modelId,
    payload?.model,
  );
  const costState = readPayloadString(
    typeof costStatePayload === "string" ? costStatePayload : null,
    typeof costStatePayload === "object" ? costStatePayload?.status : null,
    typeof costStatePayload === "object" ? costStatePayload?.state : null,
    runtimeSummary?.cost_status,
    runtimeSummary?.costStatus,
  );
  const estimatedCostClass = readPayloadString(
    typeof costStatePayload === "object"
      ? costStatePayload?.estimated_cost_class
      : null,
    typeof costStatePayload === "object"
      ? costStatePayload?.estimatedCostClass
      : null,
    typeof costStatePayload === "object" ? costStatePayload?.cost_class : null,
    typeof costStatePayload === "object" ? costStatePayload?.costClass : null,
    runtimeSummary?.estimated_cost_class,
    runtimeSummary?.estimatedCostClass,
  );
  const limitState = readPayloadString(
    typeof limitStatePayload === "string" ? limitStatePayload : null,
    typeof limitStatePayload === "object" ? limitStatePayload?.status : null,
    typeof limitStatePayload === "object" ? limitStatePayload?.state : null,
    typeof limitStatePayload === "object"
      ? limitStatePayload?.limit_status
      : null,
    typeof limitStatePayload === "object"
      ? limitStatePayload?.limitStatus
      : null,
    runtimeSummary?.limit_status,
    runtimeSummary?.limitStatus,
  );
  const limitEventKind = readPayloadString(
    limitEventPayload?.event_kind,
    limitEventPayload?.eventKind,
    limitEventPayload?.kind,
    runtimeSummary?.limit_event_kind,
    runtimeSummary?.limitEventKind,
  );
  const quotaLow =
    typeof limitEventPayload?.quota_low === "boolean"
      ? limitEventPayload.quota_low
      : typeof limitEventPayload?.quotaLow === "boolean"
        ? limitEventPayload.quotaLow
        : typeof runtimeSummary?.quota_low === "boolean"
          ? runtimeSummary.quota_low
          : typeof runtimeSummary?.quotaLow === "boolean"
            ? runtimeSummary.quotaLow
            : limitEventKind === "quota_low"
              ? true
              : null;
  const runtimeContract = (payload?.runtime_contract ??
    payload?.runtimeContract) as Record<string, any> | null | undefined;
  const executionProfile = (runtimeContract?.execution_profile ??
    runtimeContract?.executionProfile) as
    | Record<string, any>
    | null
    | undefined;
  const executorAdapter = (runtimeContract?.executor_adapter ??
    runtimeContract?.executorAdapter) as Record<string, any> | null | undefined;
  const executorBinding = (runtimeContract?.executor_binding ??
    runtimeContract?.executorBinding) as Record<string, any> | null | undefined;
  const executorKind = readPayloadString(
    payload?.executor_kind,
    payload?.executorKind,
    executorBinding?.executor_kind,
    executorBinding?.executorKind,
  );
  const executorBindingKey = readPayloadString(
    payload?.executor_binding_key,
    payload?.executorBindingKey,
    executorBinding?.binding_key,
    executorBinding?.bindingKey,
  );
  const skillId =
    readPayloadString(
      payload?.skill_id,
      payload?.skillId,
      payload?.service_skill_id,
      payload?.serviceSkillId,
    ) ??
    (executorKind === "skill" || executorKind === "service_skill"
      ? executorBindingKey
      : null);
  const limecorePolicySnapshot = (runtimeContract?.limecore_policy_snapshot ??
    runtimeContract?.limecorePolicySnapshot) as
    | Record<string, any>
    | null
    | undefined;
  const limecorePolicyRefs = [
    ...new Set(
      [
        ...(Array.isArray(payload?.limecore_policy_refs)
          ? payload.limecore_policy_refs
          : []),
        ...(Array.isArray(payload?.limecorePolicyRefs)
          ? payload.limecorePolicyRefs
          : []),
        ...(Array.isArray(runtimeContract?.limecore_policy_refs)
          ? runtimeContract.limecore_policy_refs
          : []),
        ...(Array.isArray(runtimeContract?.limecorePolicyRefs)
          ? runtimeContract.limecorePolicyRefs
          : []),
        ...(Array.isArray(limecorePolicySnapshot?.refs)
          ? limecorePolicySnapshot.refs
          : []),
      ].filter(
        (candidate): candidate is string => typeof candidate === "string",
      ),
    ),
  ];
  const limecorePolicyValueHits = Array.isArray(
    limecorePolicySnapshot?.policy_value_hits,
  )
    ? limecorePolicySnapshot.policy_value_hits
    : Array.isArray(limecorePolicySnapshot?.policyValueHits)
      ? limecorePolicySnapshot.policyValueHits
      : [];
  const resolvedPolicyHitRefs = new Set(
    limecorePolicyValueHits
      .filter((hit) => hit?.status === "resolved")
      .map((hit) => hit.ref_key ?? hit.refKey ?? hit.ref)
      .filter((refKey): refKey is string => typeof refKey === "string"),
  );
  const pendingPolicyRefs = limecorePolicyRefs.filter(
    (refKey) => !resolvedPolicyHitRefs.has(refKey),
  );
  const limecorePolicyMissingInputs = Array.isArray(
    limecorePolicySnapshot?.missing_inputs,
  )
    ? limecorePolicySnapshot.missing_inputs
    : Array.isArray(limecorePolicySnapshot?.missingInputs)
      ? limecorePolicySnapshot.missingInputs
      : Array.isArray(limecorePolicySnapshot?.unresolved_refs)
        ? limecorePolicySnapshot.unresolved_refs
        : Array.isArray(limecorePolicySnapshot?.unresolvedRefs)
          ? limecorePolicySnapshot.unresolvedRefs
          : pendingPolicyRefs;
  const limecorePolicyPendingHitRefs = Array.isArray(
    limecorePolicySnapshot?.pending_hit_refs,
  )
    ? limecorePolicySnapshot.pending_hit_refs
    : Array.isArray(limecorePolicySnapshot?.pendingHitRefs)
      ? limecorePolicySnapshot.pendingHitRefs
      : limecorePolicyMissingInputs;
  const limecorePolicyValueHitCount =
    typeof limecorePolicySnapshot?.policy_value_hit_count === "number"
      ? limecorePolicySnapshot.policy_value_hit_count
      : typeof limecorePolicySnapshot?.policyValueHitCount === "number"
        ? limecorePolicySnapshot.policyValueHitCount
        : limecorePolicyValueHits.length;
  const limecorePolicyEvaluation = (limecorePolicySnapshot?.policy_evaluation ??
    limecorePolicySnapshot?.policyEvaluation) as
    | Record<string, any>
    | null
    | undefined;
  const limecorePolicyEvaluationBlockingRefs = Array.isArray(
    limecorePolicyEvaluation?.blocking_refs,
  )
    ? limecorePolicyEvaluation.blocking_refs
    : Array.isArray(limecorePolicyEvaluation?.blockingRefs)
      ? limecorePolicyEvaluation.blockingRefs
      : [];
  const limecorePolicyEvaluationAskRefs = Array.isArray(
    limecorePolicyEvaluation?.ask_refs,
  )
    ? limecorePolicyEvaluation.ask_refs
    : Array.isArray(limecorePolicyEvaluation?.askRefs)
      ? limecorePolicyEvaluation.askRefs
      : limecorePolicyPendingHitRefs;
  const limecorePolicyEvaluationPendingRefs = Array.isArray(
    limecorePolicyEvaluation?.pending_refs,
  )
    ? limecorePolicyEvaluation.pending_refs
    : Array.isArray(limecorePolicyEvaluation?.pendingRefs)
      ? limecorePolicyEvaluation.pendingRefs
      : limecorePolicyPendingHitRefs;

  return {
    task_id: item.task_id,
    task_type: item.task_type,
    normalized_status: item.normalized_status,
    contract_key: payload?.modality_contract_key ?? null,
    entry_key: entryKey,
    thread_id: threadId,
    turn_id: turnId,
    content_id: contentId,
    modality,
    skill_id: skillId,
    model_id: modelId,
    cost_state: costState,
    limit_state: limitState,
    estimated_cost_class: estimatedCostClass,
    limit_event_kind: limitEventKind,
    quota_low: quotaLow,
    routing_slot: payload?.routing_slot ?? null,
    provider_id: payload?.provider_id ?? null,
    model: payload?.model ?? null,
    execution_profile_key:
      payload?.execution_profile_key ??
      payload?.executionProfileKey ??
      executionProfile?.profile_key ??
      executionProfile?.profileKey ??
      null,
    executor_adapter_key:
      payload?.executor_adapter_key ??
      payload?.executorAdapterKey ??
      executorAdapter?.adapter_key ??
      executorAdapter?.adapterKey ??
      null,
    executor_kind: executorKind,
    executor_binding_key: executorBindingKey,
    limecore_policy_refs: limecorePolicyRefs,
    limecore_policy_snapshot_status:
      limecorePolicySnapshot?.status ??
      (limecorePolicyRefs.length > 0 ? "local_defaults_evaluated" : null),
    limecore_policy_decision:
      limecorePolicySnapshot?.decision ??
      (limecorePolicyRefs.length > 0 ? "allow" : null),
    limecore_policy_decision_source:
      limecorePolicySnapshot?.decision_source ??
      limecorePolicySnapshot?.decisionSource ??
      (limecorePolicyRefs.length > 0 ? "local_default_policy" : null),
    limecore_policy_decision_scope:
      limecorePolicySnapshot?.decision_scope ??
      limecorePolicySnapshot?.decisionScope ??
      (limecorePolicyRefs.length > 0 ? "local_defaults_only" : null),
    limecore_policy_decision_reason:
      limecorePolicySnapshot?.decision_reason ??
      limecorePolicySnapshot?.decisionReason ??
      (limecorePolicyRefs.length > 0
        ? "declared_policy_refs_with_no_local_deny_rule"
        : null),
    limecore_policy_unresolved_refs: Array.isArray(
      limecorePolicySnapshot?.unresolved_refs,
    )
      ? limecorePolicySnapshot.unresolved_refs
      : Array.isArray(limecorePolicySnapshot?.unresolvedRefs)
        ? limecorePolicySnapshot.unresolvedRefs
        : pendingPolicyRefs,
    limecore_policy_missing_inputs: limecorePolicyMissingInputs,
    limecore_policy_pending_hit_refs: limecorePolicyPendingHitRefs,
    limecore_policy_value_hits: limecorePolicyValueHits,
    limecore_policy_value_hit_count: limecorePolicyValueHitCount,
    limecore_policy_evaluation_status:
      limecorePolicyEvaluation?.status ??
      (limecorePolicyPendingHitRefs.length > 0 ? "input_gap" : null),
    limecore_policy_evaluation_decision:
      limecorePolicyEvaluation?.decision ??
      (limecorePolicyPendingHitRefs.length > 0 ? "ask" : null),
    limecore_policy_evaluation_decision_source:
      limecorePolicyEvaluation?.decision_source ??
      limecorePolicyEvaluation?.decisionSource ??
      (limecorePolicyPendingHitRefs.length > 0
        ? "policy_input_evaluator"
        : null),
    limecore_policy_evaluation_decision_scope:
      limecorePolicyEvaluation?.decision_scope ??
      limecorePolicyEvaluation?.decisionScope ??
      (limecorePolicyPendingHitRefs.length > 0
        ? "pending_policy_inputs"
        : null),
    limecore_policy_evaluation_decision_reason:
      limecorePolicyEvaluation?.decision_reason ??
      limecorePolicyEvaluation?.decisionReason ??
      (limecorePolicyPendingHitRefs.length > 0
        ? "declared_policy_refs_missing_inputs"
        : null),
    limecore_policy_evaluation_blocking_refs:
      limecorePolicyEvaluationBlockingRefs,
    limecore_policy_evaluation_ask_refs: limecorePolicyEvaluationAskRefs,
    limecore_policy_evaluation_pending_refs:
      limecorePolicyEvaluationPendingRefs,
    routing_event:
      payload?.modality_contract_key === "voice_generation" ||
      payload?.modality_contract_key === "audio_transcription"
        ? "executor_invoked"
        : "model_routing_decision",
    routing_outcome:
      item.normalized_status === "failed" ? "failed" : "accepted",
    failure_code: lastError?.code ?? null,
    model_capability_assessment_source: assessment?.source ?? null,
    model_supports_image_generation:
      payload?.modality_contract_key === "image_generation"
        ? (assessment?.supports_image_generation ?? null)
        : null,
    audio_output_status: audioOutput?.status ?? null,
    audio_output_path: audioOutput?.audio_path ?? null,
    audio_output_mime_type: audioOutput?.mime_type ?? null,
    audio_output_duration_ms: audioOutput?.duration_ms ?? null,
    audio_output_error_code: audioOutput?.error_code ?? null,
    audio_output_retryable: audioOutput?.retryable ?? null,
    transcript_status: transcript?.status ?? null,
    transcript_path: transcript?.transcript_path ?? null,
    transcript_source_url: transcript?.source_url ?? null,
    transcript_source_path: transcript?.source_path ?? null,
    transcript_language: transcript?.language ?? null,
    transcript_output_format: transcript?.output_format ?? null,
    transcript_error_code: transcript?.error_code ?? null,
    transcript_retryable: transcript?.retryable ?? null,
  };
}

function listMockMediaTaskArtifacts(args: any) {
  const request = args?.request ?? args ?? {};
  const task = buildMockMediaTaskOutput(args);
  const contractKey =
    task.record?.payload?.modality_contract_key ?? "image_generation";
  const routingOutcome =
    task.normalized_status === "failed" ? "failed" : "accepted";
  const requestedContractKey =
    request.modalityContractKey ?? request.modality_contract_key;
  const requestedRoutingOutcome =
    request.routingOutcome ?? request.routing_outcome;
  const requestedTaskFamily = request.taskFamily ?? request.task_family;
  const requestedTaskType = request.taskType ?? request.task_type;
  const matchesContract =
    !requestedContractKey || requestedContractKey === contractKey;
  const matchesRouting =
    !requestedRoutingOutcome || requestedRoutingOutcome === routingOutcome;
  const matchesTaskFamily =
    !requestedTaskFamily || requestedTaskFamily === task.task_family;
  const matchesTaskType =
    !requestedTaskType || requestedTaskType === task.task_type;
  const tasks =
    matchesContract && matchesRouting && matchesTaskFamily && matchesTaskType
      ? [task]
      : [];
  const snapshots = tasks.map(buildMediaTaskContractSnapshot);
  const audioOutputStatuses = countByStatus(snapshots, "audio_output_status");
  const transcriptStatuses = countByStatus(snapshots, "transcript_status");
  const limecorePolicySnapshotStatuses = countByStatus(
    snapshots,
    "limecore_policy_snapshot_status",
  );
  const limecorePolicyEvaluationStatuses = countByStatus(
    snapshots,
    "limecore_policy_evaluation_status",
  );

  return {
    success: true,
    workspace_root: request.projectRootPath ?? "/mock/workspace",
    artifact_root: `${request.projectRootPath ?? "/mock/workspace"}/.lime/tasks`,
    filters: {
      status: request.status ?? null,
      task_family: requestedTaskFamily ?? null,
      task_type: requestedTaskType ?? null,
      modality_contract_key: requestedContractKey ?? null,
      routing_outcome: requestedRoutingOutcome ?? null,
      limit: request.limit ?? null,
    },
    total: tasks.length,
    modality_runtime_contracts: {
      snapshot_count: snapshots.length,
      contract_keys: [
        ...new Set(snapshots.map((item) => item.contract_key).filter(Boolean)),
      ],
      entry_keys: [
        ...new Set(snapshots.map((item) => item.entry_key).filter(Boolean)),
      ],
      thread_ids: [
        ...new Set(snapshots.map((item) => item.thread_id).filter(Boolean)),
      ],
      turn_ids: [
        ...new Set(snapshots.map((item) => item.turn_id).filter(Boolean)),
      ],
      content_ids: [
        ...new Set(snapshots.map((item) => item.content_id).filter(Boolean)),
      ],
      modalities: [
        ...new Set(snapshots.map((item) => item.modality).filter(Boolean)),
      ],
      skill_ids: [
        ...new Set(snapshots.map((item) => item.skill_id).filter(Boolean)),
      ],
      model_ids: [
        ...new Set(snapshots.map((item) => item.model_id).filter(Boolean)),
      ],
      cost_states: [
        ...new Set(snapshots.map((item) => item.cost_state).filter(Boolean)),
      ],
      limit_states: [
        ...new Set(snapshots.map((item) => item.limit_state).filter(Boolean)),
      ],
      estimated_cost_classes: [
        ...new Set(
          snapshots.map((item) => item.estimated_cost_class).filter(Boolean),
        ),
      ],
      limit_event_kinds: [
        ...new Set(
          snapshots.map((item) => item.limit_event_kind).filter(Boolean),
        ),
      ],
      quota_low_count: snapshots.filter((item) => item.quota_low === true)
        .length,
      execution_profile_keys: [
        ...new Set(
          snapshots.map((item) => item.execution_profile_key).filter(Boolean),
        ),
      ],
      executor_adapter_keys: [
        ...new Set(
          snapshots.map((item) => item.executor_adapter_key).filter(Boolean),
        ),
      ],
      executor_kinds: [
        ...new Set(snapshots.map((item) => item.executor_kind).filter(Boolean)),
      ],
      executor_binding_keys: [
        ...new Set(
          snapshots.map((item) => item.executor_binding_key).filter(Boolean),
        ),
      ],
      limecore_policy_refs: [
        ...new Set(snapshots.flatMap((item) => item.limecore_policy_refs)),
      ],
      limecore_policy_snapshot_count: snapshots.filter(
        (item) => item.limecore_policy_refs.length > 0,
      ).length,
      limecore_policy_snapshot_statuses: limecorePolicySnapshotStatuses,
      limecore_policy_decisions: [
        ...new Set(
          snapshots
            .map((item) => item.limecore_policy_decision)
            .filter(Boolean),
        ),
      ],
      limecore_policy_decision_sources: [
        ...new Set(
          snapshots
            .map((item) => item.limecore_policy_decision_source)
            .filter(Boolean),
        ),
      ],
      limecore_policy_evaluation_statuses: limecorePolicyEvaluationStatuses,
      limecore_policy_evaluation_decisions: [
        ...new Set(
          snapshots
            .map((item) => item.limecore_policy_evaluation_decision)
            .filter(Boolean),
        ),
      ],
      limecore_policy_evaluation_decision_sources: [
        ...new Set(
          snapshots
            .map((item) => item.limecore_policy_evaluation_decision_source)
            .filter(Boolean),
        ),
      ],
      limecore_policy_evaluation_blocking_refs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_evaluation_blocking_refs ?? [],
          ),
        ),
      ],
      limecore_policy_evaluation_ask_refs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_evaluation_ask_refs ?? [],
          ),
        ),
      ],
      limecore_policy_evaluation_pending_refs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_evaluation_pending_refs ?? [],
          ),
        ),
      ],
      limecore_policy_unresolved_refs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_unresolved_refs ?? [],
          ),
        ),
      ],
      limecore_policy_missing_inputs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_missing_inputs ?? [],
          ),
        ),
      ],
      limecore_policy_pending_hit_refs: [
        ...new Set(
          snapshots.flatMap(
            (item) => item.limecore_policy_pending_hit_refs ?? [],
          ),
        ),
      ],
      limecore_policy_value_hit_count: snapshots.reduce(
        (count, item) => count + (item.limecore_policy_value_hit_count ?? 0),
        0,
      ),
      blocked_count: snapshots.filter(
        (item) => item.routing_outcome === "blocked",
      ).length,
      routing_outcomes: snapshots.length
        ? [{ outcome: snapshots[0].routing_outcome, count: snapshots.length }]
        : [],
      model_registry_assessment_count: snapshots.filter(
        (item) => item.model_capability_assessment_source === "model_registry",
      ).length,
      audio_output_count: snapshots.filter((item) => item.audio_output_status)
        .length,
      audio_output_statuses: audioOutputStatuses,
      audio_output_error_codes: [
        ...new Set(
          snapshots.map((item) => item.audio_output_error_code).filter(Boolean),
        ),
      ],
      transcript_count: snapshots.filter((item) => item.transcript_status)
        .length,
      transcript_statuses: transcriptStatuses,
      transcript_error_codes: [
        ...new Set(
          snapshots.map((item) => item.transcript_error_code).filter(Boolean),
        ),
      ],
      snapshots,
    },
    tasks,
  };
}

export function createMediaTaskMockHandlers(): Record<
  string,
  (args: any) => any
> {
  return {
    create_image_generation_task_artifact: (args: any) =>
      buildMockMediaTaskOutput(args),
    create_audio_generation_task_artifact: (args: any) =>
      buildMockMediaTaskOutput(args, {
        task_type: "audio_generate",
      }),
    complete_audio_generation_task_artifact: (args: any) =>
      buildMockCompletedAudioTaskOutput(args),
    get_media_task_artifact: (args: any) => buildMockMediaTaskOutput(args),
    list_media_task_artifacts: (args: any) => listMockMediaTaskArtifacts(args),
    cancel_media_task_artifact: (args: any) =>
      buildMockMediaTaskOutput(args, {
        status: "cancelled",
        normalized_status: "cancelled",
      }),
  };
}

export const mediaTaskMocks: Record<string, (args: any) => any> = {};
