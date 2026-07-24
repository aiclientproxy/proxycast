export interface MemoryProfileConfig {
  current_status?: string;
  strengths?: string[];
  explanation_style?: string[];
  challenge_preference?: string[];
}

export type MemorySoulImportSource = "manual" | "soul_md";

export type MemorySoulArtifactVoiceSource = "creator_voice" | "brand_voice";

export interface MemorySoulArtifactVoiceConfig {
  enabled?: boolean;
  voice_source?: MemorySoulArtifactVoiceSource | null;
  creator_voice_id?: string | null;
  brand_voice_id?: string | null;
  evidence_pack_id?: string | null;
  evidence_refs?: string[];
}

export type MemorySoulStyleProfileId = string;

export interface MemorySoulConfig {
  enabled?: boolean;
  name?: string | null;
  summary?: string | null;
  style_profile_id?: MemorySoulStyleProfileId | null;
  tone?: string[];
  communication_style?: string[];
  explanation_depth?: string | null;
  challenge_style?: string | null;
  avoid?: string[];
  artifact_voice?: MemorySoulArtifactVoiceConfig;
  imported_from?: MemorySoulImportSource | null;
  updated_at?: string | null;
}

export interface MemorySourcesConfig {
  managed_policy_path?: string | null;
  project_memory_paths?: string[];
  project_rule_dirs?: string[];
  user_memory_path?: string | null;
}

export interface MemoryAutoConfig {
  enabled?: boolean;
  entrypoint?: string;
  max_loaded_lines?: number;
  root_dir?: string | null;
}

export interface MemoryResolveConfig {
  additional_dirs?: string[];
  follow_imports?: boolean;
  import_max_depth?: number;
  load_additional_dirs_memory?: boolean;
}

export type MemoryEmbeddingProvider =
  | "auto"
  | "local_onnx"
  | "builtin"
  | "openai_api"
  | "provider"
  | "disabled";

export interface MemoryEmbeddingConfig {
  provider?: MemoryEmbeddingProvider;
  provider_id?: string | null;
  model?: string | null;
}

export interface MemoryConfig {
  enabled: boolean;
  max_entries?: number;
  retention_days?: number;
  auto_cleanup?: boolean;
  profile?: MemoryProfileConfig;
  soul?: MemorySoulConfig;
  sources?: MemorySourcesConfig;
  auto?: MemoryAutoConfig;
  resolve?: MemoryResolveConfig;
  embedding?: MemoryEmbeddingConfig;
}
