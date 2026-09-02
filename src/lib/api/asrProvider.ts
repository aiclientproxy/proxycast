/**
 * ASR Provider 类型定义
 *
 * 定义语音识别服务相关的类型，与 Rust 后端保持一致。
 */

import {
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_CREATE,
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET,
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DELETE,
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_LIST,
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_TEST,
  APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_UPDATE,
  APP_SERVER_METHOD_VOICE_INSTRUCTION_DELETE,
  APP_SERVER_METHOD_VOICE_INSTRUCTION_LIST,
  APP_SERVER_METHOD_VOICE_INSTRUCTION_SAVE,
  APP_SERVER_METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT,
  APP_SERVER_METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO,
  createAppServerClient,
  type AppServerVoiceAsrCredential,
  type AppServerVoiceAsrCredentialCreateParams,
  type AppServerVoiceAsrProviderType,
  type AppServerVoiceTranscriptionTranscribeAudioResponse,
} from "./appServer";
import { getConfig, saveConfig, type Config } from "./appConfig";

// ============ ASR Provider 类型 ============

/** ASR Provider 类型 */
export type AsrProviderType =
  | "whisper_local"
  | "sensevoice_local"
  | "xunfei"
  | "baidu"
  | "openai";

/** Whisper 模型大小 */
export type WhisperModelSize = "tiny" | "base" | "small" | "medium";

/** Whisper 本地配置 */
export interface WhisperLocalConfig {
  model: WhisperModelSize;
  model_path?: string;
}

/** SenseVoice 本地配置 */
export interface SenseVoiceLocalConfig {
  model_id: string;
  model_dir?: string;
  use_itn: boolean;
  num_threads: number;
  vad_model_id?: string;
}

/** 讯飞配置 */
export interface XunfeiConfig {
  app_id: string;
  api_key: string;
  api_secret: string;
}

/** 百度配置 */
export interface BaiduConfig {
  api_key: string;
  secret_key: string;
}

/** OpenAI ASR 配置 */
export interface OpenAIAsrConfig {
  api_key: string;
  base_url?: string;
  proxy_url?: string;
}

/** ASR 凭证条目 */
export interface AsrCredentialEntry {
  id: string;
  provider: AsrProviderType;
  name?: string;
  is_default: boolean;
  disabled: boolean;
  language: string;
  whisper_config?: WhisperLocalConfig;
  sensevoice_config?: SenseVoiceLocalConfig;
  xunfei_config?: XunfeiConfig;
  baidu_config?: BaiduConfig;
  openai_config?: OpenAIAsrConfig;
}

// ============ 语音输入配置类型 ============

/** 语音输出模式 */
export type VoiceOutputMode = "type" | "clipboard" | "both";

/** 语音处理配置 */
export interface VoiceProcessorConfig {
  polish_enabled: boolean;
  polish_provider?: string;
  polish_model?: string;
  default_instruction_id: string;
}

/** 语音输出配置 */
export interface VoiceOutputConfig {
  mode: VoiceOutputMode;
  type_delay_ms: number;
}

/** 语音处理指令 */
export interface VoiceInstruction {
  id: string;
  name: string;
  description?: string;
  prompt: string;
  shortcut?: string;
  is_preset: boolean;
  icon?: string;
}

/** 语音输入功能配置 */
export interface VoiceInputConfig {
  enabled: boolean;
  shortcut: string;
  processor: VoiceProcessorConfig;
  output: VoiceOutputConfig;
  instructions: VoiceInstruction[];
  /** 选择的麦克风设备 ID（为空时使用系统默认设备） */
  selected_device_id?: string;
  /** 是否启用交互音效 */
  sound_enabled: boolean;
  /** 翻译模式使用的指令 ID */
  translate_instruction_id: string;
}

type ConfigWithVoiceInput = Config & {
  experimental?: Config["experimental"] & {
    voice_input?: (Partial<VoiceInputConfig> & Record<string, unknown>) | null;
  };
};

const DEFAULT_VOICE_INSTRUCTIONS: VoiceInstruction[] = [
  {
    id: "default",
    name: "默认润色",
    prompt: "{{text}}",
    is_preset: true,
  },
  {
    id: "translate_en",
    name: "翻译为英文",
    prompt: "{{text}}",
    is_preset: true,
  },
  {
    id: "raw",
    name: "原始输出",
    prompt: "{{text}}",
    is_preset: true,
  },
];

const DEFAULT_VOICE_INPUT_CONFIG: VoiceInputConfig = {
  enabled: false,
  shortcut: "CommandOrControl+Shift+V",
  processor: {
    polish_enabled: true,
    polish_provider: "openai",
    polish_model: "gpt-4.1-mini",
    default_instruction_id: "default",
  },
  output: {
    mode: "type",
    type_delay_ms: 10,
  },
  instructions: DEFAULT_VOICE_INSTRUCTIONS,
  selected_device_id: undefined,
  sound_enabled: true,
  translate_instruction_id: "translate_en",
};

const VOICE_REALTIME_CURRENT_BLOCKED_MESSAGE =
  "旧实时语音转写、润色、输出与录音控制入口已退役，请使用 App Server current 语音转写通道";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function cloneVoiceInputConfig(config: VoiceInputConfig): VoiceInputConfig {
  return {
    ...config,
    processor: { ...config.processor },
    output: { ...config.output },
    instructions: config.instructions.map((instruction) => ({
      ...instruction,
    })),
  };
}

function normalizeVoiceInputConfig(
  value: Partial<VoiceInputConfig> | null | undefined,
): VoiceInputConfig {
  return {
    ...cloneVoiceInputConfig(DEFAULT_VOICE_INPUT_CONFIG),
    ...(value ?? {}),
    processor: {
      ...DEFAULT_VOICE_INPUT_CONFIG.processor,
      ...(value?.processor ?? {}),
    },
    output: {
      ...DEFAULT_VOICE_INPUT_CONFIG.output,
      ...(value?.output ?? {}),
    },
    instructions:
      Array.isArray(value?.instructions) && value.instructions.length > 0
        ? value.instructions.map((instruction) => ({ ...instruction }))
        : DEFAULT_VOICE_INSTRUCTIONS.map((instruction) => ({ ...instruction })),
    selected_device_id: value?.selected_device_id,
  };
}

function mergeVoiceInputConfig(
  appConfig: Config,
  voiceInputConfig: VoiceInputConfig,
): Config {
  const currentVoiceInput = (appConfig as ConfigWithVoiceInput).experimental
    ?.voice_input;

  return {
    ...appConfig,
    experimental: {
      ...appConfig.experimental,
      webmcp: appConfig.experimental?.webmcp ?? { enabled: false },
      voice_input: {
        ...(currentVoiceInput ?? {}),
        ...cloneVoiceInputConfig(voiceInputConfig),
      },
    },
  } as Config;
}

// ============ Desktop Host / App Server 命令封装 ============

function failClosedRetiredVoiceInputCommand(): never {
  throw new Error(
    `${VOICE_REALTIME_CURRENT_BLOCKED_MESSAGE}，旧进程内桌面命令已退役。`,
  );
}

function assertVoidResult(command: string, value: unknown): void {
  if (
    value !== undefined &&
    value !== null &&
    (!isRecord(value) || Object.keys(value).length > 0)
  ) {
    throw new Error(`${command} did not return an empty result`);
  }
}

function assertTestAsrCredentialResult(
  command: string,
  value: unknown,
): { success: boolean; message: string } {
  if (
    !isRecord(value) ||
    typeof value.success !== "boolean" ||
    typeof value.message !== "string"
  ) {
    throw new Error(`${command} did not return a test result`);
  }
  return value as { success: boolean; message: string };
}

function asrProviderFromAppServer(
  provider: AppServerVoiceAsrProviderType,
): AsrProviderType {
  if (provider === "sense_voice_local") {
    return "sensevoice_local";
  }
  return provider;
}

function asrProviderToAppServer(
  provider: AsrProviderType,
): AppServerVoiceAsrProviderType {
  if (provider === "sensevoice_local") {
    return "sense_voice_local";
  }
  return provider;
}

function asrCredentialFromAppServer(
  credential: AppServerVoiceAsrCredential,
): AsrCredentialEntry {
  return {
    ...credential,
    provider: asrProviderFromAppServer(credential.provider),
  };
}

function asrCredentialToAppServer(
  credential: AsrCredentialEntry,
): AppServerVoiceAsrCredential {
  return {
    ...credential,
    provider: asrProviderToAppServer(credential.provider),
  };
}

function asrCredentialCreateToAppServer(
  credential: Omit<AsrCredentialEntry, "id">,
): AppServerVoiceAsrCredentialCreateParams {
  return {
    ...credential,
    provider: asrProviderToAppServer(credential.provider),
  };
}

/** 获取 ASR 凭证列表 */
export async function getAsrCredentials(): Promise<AsrCredentialEntry[]> {
  const response = await createAppServerClient().listVoiceAsrCredentials();
  const result = response.result;
  if (!isRecord(result) || !Array.isArray(result.credentials)) {
    throw new Error(
      `${APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_LIST} did not return ASR credentials`,
    );
  }
  return result.credentials.map(asrCredentialFromAppServer);
}

/** 添加 ASR 凭证 */
export async function addAsrCredential(
  entry: Omit<AsrCredentialEntry, "id">,
): Promise<AsrCredentialEntry> {
  const response = await createAppServerClient().createVoiceAsrCredential(
    asrCredentialCreateToAppServer(entry),
  );
  if (!isRecord(response.result) || !isRecord(response.result.credential)) {
    throw new Error(
      `${APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_CREATE} did not return an ASR credential`,
    );
  }
  return asrCredentialFromAppServer(
    response.result.credential as AppServerVoiceAsrCredential,
  );
}

/** 更新 ASR 凭证 */
export async function updateAsrCredential(
  entry: AsrCredentialEntry,
): Promise<void> {
  const response = await createAppServerClient().updateVoiceAsrCredential({
    credential: asrCredentialToAppServer(entry),
  });
  assertVoidResult(
    APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_UPDATE,
    response.result,
  );
}

/** 删除 ASR 凭证 */
export async function deleteAsrCredential(id: string): Promise<void> {
  const response = await createAppServerClient().deleteVoiceAsrCredential({
    id,
  });
  assertVoidResult(
    APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DELETE,
    response.result,
  );
}

/** 设置默认 ASR 凭证 */
export async function setDefaultAsrCredential(id: string): Promise<void> {
  const response = await createAppServerClient().setDefaultVoiceAsrCredential({
    id,
  });
  assertVoidResult(
    APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET,
    response.result,
  );
}

/** 测试 ASR 凭证连通性 */
export async function testAsrCredential(
  id: string,
): Promise<{ success: boolean; message: string }> {
  const response = await createAppServerClient().testVoiceAsrCredential({ id });
  return assertTestAsrCredentialResult(
    APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_TEST,
    response.result,
  );
}

// ============ 语音输入配置命令 ============

/** 获取语音输入配置 */
export async function getVoiceInputConfig(): Promise<VoiceInputConfig> {
  const appConfig = (await getConfig()) as ConfigWithVoiceInput;
  return normalizeVoiceInputConfig(appConfig.experimental?.voice_input);
}

/** 保存语音输入配置 */
export async function saveVoiceInputConfig(
  config: VoiceInputConfig,
): Promise<void> {
  const appConfig = await getConfig({ forceRefresh: true });
  await saveConfig(mergeVoiceInputConfig(appConfig, config));
}

/** 获取指令列表 */
export async function getVoiceInstructions(): Promise<VoiceInstruction[]> {
  const response = await createAppServerClient().listVoiceInstructions();
  const result = response.result;
  if (!isRecord(result) || !Array.isArray(result.instructions)) {
    throw new Error(
      `${APP_SERVER_METHOD_VOICE_INSTRUCTION_LIST} did not return voice instructions`,
    );
  }
  return result.instructions as VoiceInstruction[];
}

/** 保存指令 */
export async function saveVoiceInstruction(
  instruction: VoiceInstruction,
): Promise<void> {
  const response = await createAppServerClient().saveVoiceInstruction({
    instruction,
  });
  assertVoidResult(APP_SERVER_METHOD_VOICE_INSTRUCTION_SAVE, response.result);
}

/** 删除指令 */
export async function deleteVoiceInstruction(id: string): Promise<void> {
  const response = await createAppServerClient().deleteVoiceInstruction({ id });
  assertVoidResult(APP_SERVER_METHOD_VOICE_INSTRUCTION_DELETE, response.result);
}

// ============ 语音识别和润色命令 ============

/** 语音识别结果 */
export interface TranscribeResult {
  text: string;
  provider: string;
}

export interface VoiceInputTranscriptionRequest {
  audioBase64: string;
  mimeType: string;
  credentialId?: string;
}

export interface VoiceInputTranscriptionResult {
  text: string;
  provider: AsrProviderType;
  durationSecs: number;
  sampleRate: number;
  language?: string;
}

export interface VoiceInputPolishRequest {
  text: string;
  instructionId?: string;
}

export interface VoiceInputPolishResult {
  text: string;
  instructionName: string;
  polished: boolean;
}

/** 润色结果 */
export interface PolishResult {
  text: string;
  instruction_name: string;
}

/** 执行语音识别 */
export async function transcribeAudio(
  audioData: Uint8Array,
  sampleRate: number,
  credentialId?: string,
): Promise<TranscribeResult> {
  void audioData;
  void sampleRate;
  void credentialId;
  failClosedRetiredVoiceInputCommand();
}

export async function transcribeVoiceInputAudio({
  audioBase64,
  mimeType,
  credentialId,
}: VoiceInputTranscriptionRequest): Promise<VoiceInputTranscriptionResult> {
  const params: {
    audio_base64: string;
    mime_type: string;
    credential_id?: string;
  } = {
    audio_base64: audioBase64,
    mime_type: mimeType,
  };
  if (credentialId) {
    params.credential_id = credentialId;
  }
  const response = await createAppServerClient().transcribeVoiceAudio(params);
  const result = response.result;
  if (
    !isRecord(result) ||
    typeof result.text !== "string" ||
    typeof result.provider !== "string" ||
    typeof result.duration_secs !== "number" ||
    typeof result.sample_rate !== "number"
  ) {
    throw new Error(
      `${APP_SERVER_METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO} did not return a transcription result`,
    );
  }
  const typedResult =
    result as AppServerVoiceTranscriptionTranscribeAudioResponse;
  return {
    text: typedResult.text,
    provider: asrProviderFromAppServer(typedResult.provider),
    durationSecs: typedResult.duration_secs,
    sampleRate: typedResult.sample_rate,
    language: typedResult.language ?? undefined,
  };
}

export async function polishVoiceInputText({
  text,
  instructionId,
}: VoiceInputPolishRequest): Promise<VoiceInputPolishResult> {
  const params: {
    text: string;
    instruction_id?: string;
  } = { text };
  if (instructionId) {
    params.instruction_id = instructionId;
  }
  const response = await createAppServerClient().polishVoiceText(params);
  const result = response.result;
  if (
    !isRecord(result) ||
    typeof result.text !== "string" ||
    typeof result.instruction_name !== "string" ||
    typeof result.polished !== "boolean"
  ) {
    throw new Error(
      `${APP_SERVER_METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT} did not return a polish result`,
    );
  }
  return {
    text: result.text,
    instructionName: result.instruction_name,
    polished: result.polished,
  };
}

/** 润色文本 */
export async function polishVoiceText(
  text: string,
  instructionId?: string,
): Promise<PolishResult> {
  void text;
  void instructionId;
  failClosedRetiredVoiceInputCommand();
}

/** 输出文本到系统 */
export async function outputVoiceText(
  text: string,
  mode?: "type" | "clipboard" | "both",
): Promise<void> {
  void text;
  void mode;
  failClosedRetiredVoiceInputCommand();
}

// ============ 录音控制命令 ============

/** 录音状态 */
export interface RecordingStatus {
  /** 是否正在录音 */
  is_recording: boolean;
  /** 当前音量级别（0-100） */
  volume: number;
  /** 录音时长（秒） */
  duration: number;
}

/** 停止录音结果 */
export interface StopRecordingResult {
  /** 音频数据（i16 样本的字节数组，小端序） */
  audio_data: number[];
  /** 采样率 */
  sample_rate: number;
  /** 录音时长（秒） */
  duration: number;
}

/** 录音快照结果 */
export interface RecordingSnapshotResult {
  /** 音频数据（i16 样本的字节数组，小端序） */
  audio_data: number[];
  /** 采样率 */
  sample_rate: number;
  /** 录音时长（秒） */
  duration: number;
}

/** 录音片段结果 */
export interface RecordingSegmentResult extends RecordingSnapshotResult {
  /** 片段起始 sample offset */
  start_sample: number;
  /** 片段结束 sample offset */
  end_sample: number;
  /** 当前录音总 sample 数 */
  total_samples: number;
}

/** 开始录音 */
export async function startRecording(deviceId?: string): Promise<void> {
  void deviceId;
  failClosedRetiredVoiceInputCommand();
}

/** 停止录音并返回音频数据 */
export async function stopRecording(): Promise<StopRecordingResult> {
  failClosedRetiredVoiceInputCommand();
}

/** 获取当前录音快照，不停止录音 */
export async function getRecordingSnapshot(): Promise<RecordingSnapshotResult> {
  failClosedRetiredVoiceInputCommand();
}

/** 获取当前录音片段，不停止录音 */
export async function getRecordingSegment(
  startSample: number,
  maxDurationSecs?: number,
): Promise<RecordingSegmentResult> {
  void startSample;
  void maxDurationSecs;
  failClosedRetiredVoiceInputCommand();
}

/** 取消录音 */
export async function cancelRecording(): Promise<void> {
  failClosedRetiredVoiceInputCommand();
}

/** 获取录音状态 */
export async function getRecordingStatus(): Promise<RecordingStatus> {
  failClosedRetiredVoiceInputCommand();
}
