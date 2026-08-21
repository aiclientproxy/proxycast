import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  ChevronDown,
  FileText,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { getConfig, saveConfig, type Config } from "@/lib/api/appConfig";
import type {
  MemoryAutoConfig,
  MemoryConfig,
  MemoryEmbeddingConfig,
  MemoryEmbeddingProvider,
  MemoryProfileConfig,
  MemoryResolveConfig,
  MemorySoulConfig,
  MemorySourcesConfig,
} from "@/lib/api/memoryConfigTypes";
import {
  buildSoulMarkdown,
  hasSoulContent,
  normalizeSoulConfig,
  parseSoulMarkdown,
  type SoulImportResult,
  type SoulImportWarningCode,
} from "@/lib/soul/soulConfig";
import type { SoulStyleProfileId } from "@/lib/soul/style-profiles";
import { MemoryStoreStatusPanel } from "./MemoryStoreStatusPanel";
import { StyleProfileSection } from "./soul/StyleProfileSection";

type EmbeddingProviderChoice =
  | "auto"
  | "local_onnx"
  | "ollama"
  | "openai_api"
  | "disabled";
type MemorySettingsTab = "memory" | "soul" | "advanced";
type SoulTemplateId = "balanced" | "direct" | "creator";

interface SoulTemplateDefinition {
  id: SoulTemplateId;
  titleKey: string;
  descriptionKey: string;
  summaryKey: string;
  communicationKeys: string[];
  avoidKeys: string[];
}

interface ProviderChoiceDefinition {
  value: EmbeddingProviderChoice;
  labelKey: string;
  descriptionKey: string;
  runtimeProvider: MemoryEmbeddingProvider;
  providerId?: string;
  model: string;
}

const PROVIDER_CHOICES: ProviderChoiceDefinition[] = [
  {
    value: "auto",
    labelKey: "settings.memory.embedding.provider.auto.label",
    descriptionKey: "settings.memory.embedding.provider.auto.description",
    runtimeProvider: "auto",
    model: "all-MiniLM-L6-v2",
  },
  {
    value: "local_onnx",
    labelKey: "settings.memory.embedding.provider.localOnnx.label",
    descriptionKey: "settings.memory.embedding.provider.localOnnx.description",
    runtimeProvider: "local_onnx",
    model: "all-MiniLM-L6-v2",
  },
  {
    value: "ollama",
    labelKey: "settings.memory.embedding.provider.ollama.label",
    descriptionKey: "settings.memory.embedding.provider.ollama.description",
    runtimeProvider: "provider",
    providerId: "ollama",
    model: "nomic-embed-text",
  },
  {
    value: "openai_api",
    labelKey: "settings.memory.embedding.provider.openaiApi.label",
    descriptionKey: "settings.memory.embedding.provider.openaiApi.description",
    runtimeProvider: "openai_api",
    model: "text-embedding-3-small",
  },
  {
    value: "disabled",
    labelKey: "settings.memory.embedding.provider.disabled.label",
    descriptionKey: "settings.memory.embedding.provider.disabled.description",
    runtimeProvider: "disabled",
    model: "all-MiniLM-L6-v2",
  },
];
const SOUL_WARNING_KEYS: Record<SoulImportWarningCode, string> = {
  empty: "settings.memory.soul.import.warning.empty",
  local_path: "settings.memory.soul.import.warning.localPath",
  project_rules: "settings.memory.soul.import.warning.projectRules",
  secret_like: "settings.memory.soul.import.warning.secretLike",
  too_long: "settings.memory.soul.import.warning.tooLong",
};

const SOUL_TEMPLATE_DEFINITIONS: SoulTemplateDefinition[] = [
  {
    id: "balanced",
    titleKey: "settings.memory.soul.template.balanced.title",
    descriptionKey: "settings.memory.soul.template.balanced.description",
    summaryKey: "settings.memory.soul.template.balanced.summary",
    communicationKeys: [
      "settings.memory.soul.template.balanced.communication.1",
      "settings.memory.soul.template.balanced.communication.2",
      "settings.memory.soul.template.balanced.communication.3",
    ],
    avoidKeys: [
      "settings.memory.soul.template.balanced.avoid.1",
      "settings.memory.soul.template.balanced.avoid.2",
    ],
  },
  {
    id: "direct",
    titleKey: "settings.memory.soul.template.direct.title",
    descriptionKey: "settings.memory.soul.template.direct.description",
    summaryKey: "settings.memory.soul.template.direct.summary",
    communicationKeys: [
      "settings.memory.soul.template.direct.communication.1",
      "settings.memory.soul.template.direct.communication.2",
      "settings.memory.soul.template.direct.communication.3",
    ],
    avoidKeys: [
      "settings.memory.soul.template.direct.avoid.1",
      "settings.memory.soul.template.direct.avoid.2",
    ],
  },
  {
    id: "creator",
    titleKey: "settings.memory.soul.template.creator.title",
    descriptionKey: "settings.memory.soul.template.creator.description",
    summaryKey: "settings.memory.soul.template.creator.summary",
    communicationKeys: [
      "settings.memory.soul.template.creator.communication.1",
      "settings.memory.soul.template.creator.communication.2",
      "settings.memory.soul.template.creator.communication.3",
    ],
    avoidKeys: [
      "settings.memory.soul.template.creator.avoid.1",
      "settings.memory.soul.template.creator.avoid.2",
    ],
  },
];

function memoryT(
  t: TFunction<"settings">,
  key: string,
  values: Record<string, string | number | boolean> = {},
): string {
  const translate = t as unknown as (
    key: string,
    values?: Record<string, string | number | boolean>,
  ) => string;
  return String(translate(key, values));
}

function normalizeProfile(profile?: MemoryProfileConfig): MemoryProfileConfig {
  return {
    current_status: profile?.current_status || undefined,
    strengths: profile?.strengths || [],
    explanation_style: profile?.explanation_style || [],
    challenge_preference: profile?.challenge_preference || [],
  };
}

function normalizeSources(sources?: MemorySourcesConfig): MemorySourcesConfig {
  return {
    managed_policy_path: sources?.managed_policy_path ?? undefined,
    project_memory_paths: sources?.project_memory_paths?.filter(
      (item) => item.trim().length > 0,
    ).length
      ? sources.project_memory_paths
      : [".lime/AGENTS.md"],
    project_rule_dirs: sources?.project_rule_dirs?.filter(
      (item) => item.trim().length > 0,
    ).length
      ? sources.project_rule_dirs
      : [".agents/rules"],
    user_memory_path: sources?.user_memory_path ?? undefined,
  };
}

function normalizeAuto(auto?: MemoryAutoConfig): MemoryAutoConfig {
  return {
    enabled: auto?.enabled ?? true,
    entrypoint: auto?.entrypoint || "MEMORY.md",
    max_loaded_lines: auto?.max_loaded_lines ?? 200,
    root_dir: auto?.root_dir ?? undefined,
  };
}

function normalizeResolve(resolve?: MemoryResolveConfig): MemoryResolveConfig {
  return {
    additional_dirs: resolve?.additional_dirs || [],
    follow_imports: resolve?.follow_imports ?? true,
    import_max_depth: resolve?.import_max_depth ?? 5,
    load_additional_dirs_memory: resolve?.load_additional_dirs_memory ?? false,
  };
}

function normalizeEmbedding(
  embedding?: MemoryEmbeddingConfig,
): MemoryEmbeddingConfig {
  const provider = embedding?.provider ?? "auto";
  const defaultModel =
    provider === "local_onnx" || provider === "auto"
      ? "all-MiniLM-L6-v2"
      : provider === "provider" && embedding?.provider_id === "ollama"
        ? "nomic-embed-text"
        : "text-embedding-3-small";

  return {
    provider,
    provider_id: embedding?.provider_id ?? undefined,
    model: embedding?.model?.trim() || defaultModel,
  };
}

function normalizeMemoryConfig(memory?: MemoryConfig): MemoryConfig {
  return {
    enabled: memory?.enabled ?? true,
    max_entries: memory?.max_entries ?? 1000,
    retention_days: memory?.retention_days ?? 30,
    auto_cleanup: memory?.auto_cleanup ?? true,
    profile: normalizeProfile(memory?.profile),
    soul: normalizeSoulConfig(memory?.soul),
    sources: normalizeSources(memory?.sources),
    auto: normalizeAuto(memory?.auto),
    resolve: normalizeResolve(memory?.resolve),
    embedding: normalizeEmbedding(memory?.embedding),
  };
}

function resolveProviderChoice(
  embedding?: MemoryEmbeddingConfig,
): EmbeddingProviderChoice {
  const normalized = normalizeEmbedding(embedding);
  if (
    normalized.provider === "provider" &&
    normalized.provider_id === "ollama"
  ) {
    return "ollama";
  }
  if (
    normalized.provider === "auto" ||
    normalized.provider === "local_onnx" ||
    normalized.provider === "openai_api" ||
    normalized.provider === "disabled"
  ) {
    return normalized.provider;
  }
  return "auto";
}

function buildEmbeddingConfig(
  choice: EmbeddingProviderChoice,
): MemoryEmbeddingConfig {
  const definition =
    PROVIDER_CHOICES.find((item) => item.value === choice) ??
    PROVIDER_CHOICES[0];

  return {
    provider: definition.runtimeProvider,
    provider_id: definition.providerId,
    model: definition.model,
  };
}

function buildSoulDraftPatch(
  current: MemorySoulConfig | undefined,
  patch: Partial<MemorySoulConfig>,
): MemorySoulConfig {
  return normalizeSoulConfig({
    ...normalizeSoulConfig(current),
    ...patch,
    imported_from: patch.imported_from ?? "manual",
    updated_at: new Date().toISOString(),
  });
}

function readSoulTemplateList(
  t: TFunction<"settings">,
  keys: string[],
): string[] {
  return keys
    .map((key) => memoryT(t, key))
    .filter((item) => item.trim().length > 0);
}

function buildSoulTemplatePatch(
  t: TFunction<"settings">,
  template: SoulTemplateDefinition,
): Partial<MemorySoulConfig> {
  return {
    enabled: true,
    name: memoryT(t, template.titleKey),
    summary: memoryT(t, template.summaryKey),
    communication_style: readSoulTemplateList(t, template.communicationKeys),
    avoid: readSoulTemplateList(t, template.avoidKeys),
    imported_from: "manual",
  };
}

function LoadingSkeleton() {
  return (
    <div
      className="mx-auto max-w-[820px] space-y-5 pb-8"
      data-testid="settings-memory-page"
    >
      <div className="h-[88px] animate-pulse rounded-md border border-slate-200 bg-white" />
      <div className="h-[160px] animate-pulse rounded-md border border-slate-200 bg-white" />
      <div className="h-[220px] animate-pulse rounded-md border border-slate-200 bg-white" />
    </div>
  );
}

export function MemorySettings() {
  const { t } = useTranslation("settings");
  const [config, setConfig] = useState<Config | null>(null);
  const [draft, setDraft] = useState<MemoryConfig>(() =>
    normalizeMemoryConfig(),
  );
  const [snapshot, setSnapshot] = useState<MemoryConfig>(() =>
    normalizeMemoryConfig(),
  );
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const messageTimeoutRef = useRef<number | null>(null);
  const [soulImportText, setSoulImportText] = useState("");
  const [soulImportPreview, setSoulImportPreview] =
    useState<SoulImportResult | null>(null);
  const [activeTab, setActiveTab] = useState<MemorySettingsTab>("memory");

  const showMessage = useCallback((nextMessage: string) => {
    if (messageTimeoutRef.current !== null) {
      window.clearTimeout(messageTimeoutRef.current);
    }
    setMessage(nextMessage);
    messageTimeoutRef.current = window.setTimeout(() => {
      messageTimeoutRef.current = null;
      setMessage(null);
    }, 2500);
  }, []);

  useEffect(
    () => () => {
      if (messageTimeoutRef.current !== null) {
        window.clearTimeout(messageTimeoutRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    let disposed = false;

    async function load() {
      setLoading(true);
      try {
        const nextConfig = await getConfig();
        if (disposed) {
          return;
        }
        const nextMemory = normalizeMemoryConfig(nextConfig.memory);
        setConfig(nextConfig);
        setDraft(nextMemory);
        setSnapshot(nextMemory);
      } catch (error) {
        console.error("加载记忆设置失败:", error);
      } finally {
        if (!disposed) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      disposed = true;
    };
  }, []);

  const dirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(snapshot),
    [draft, snapshot],
  );
  const providerChoice = resolveProviderChoice(draft.embedding);
  const activeProvider =
    PROVIDER_CHOICES.find((item) => item.value === providerChoice) ??
    PROVIDER_CHOICES[0];
  const embeddingConfig = normalizeEmbedding(draft.embedding);
  const vectorSearchEnabled =
    draft.enabled && embeddingConfig.provider !== "disabled";
  const memoryStatusDescriptionKey = vectorSearchEnabled
    ? "settings.memory.embedding.status.configuredDescription"
    : "settings.memory.embedding.status.fullTextDescription";
  const soul = normalizeSoulConfig(draft.soul);
  const soulExportMarkdown = buildSoulMarkdown(soul);
  const soulEnabledWithContent = soul.enabled && hasSoulContent(soul);
  const tabs: Array<{
    id: MemorySettingsTab;
    labelKey: string;
    descriptionKey: string;
  }> = [
    {
      id: "memory",
      labelKey: "settings.memory.tabs.memory",
      descriptionKey: "settings.memory.tabs.memory.description",
    },
    {
      id: "soul",
      labelKey: "settings.memory.tabs.soul",
      descriptionKey: "settings.memory.tabs.soul.description",
    },
    {
      id: "advanced",
      labelKey: "settings.memory.tabs.advanced",
      descriptionKey: "settings.memory.tabs.advanced.description",
    },
  ];
  const soulPreviewItems = [
    {
      id: "summary",
      labelKey: "settings.memory.soul.current.summary",
      values: soul.summary ? [soul.summary] : [],
    },
    {
      id: "communication",
      labelKey: "settings.memory.soul.current.communication",
      values: soul.communication_style ?? [],
    },
    {
      id: "avoid",
      labelKey: "settings.memory.soul.current.avoid",
      values: soul.avoid ?? [],
    },
    {
      id: "depth",
      labelKey: "settings.memory.soul.current.depth",
      values: soul.explanation_depth ? [soul.explanation_depth] : [],
    },
    {
      id: "challenge",
      labelKey: "settings.memory.soul.current.challenge",
      values: soul.challenge_style ? [soul.challenge_style] : [],
    },
  ].filter((item) => item.values.length > 0);

  const handleProviderChange = (choice: EmbeddingProviderChoice) => {
    setDraft((previous) => ({
      ...previous,
      embedding: buildEmbeddingConfig(choice),
    }));
  };

  const handleSoulTemplateApply = (template: SoulTemplateDefinition) => {
    setDraft((previous) => ({
      ...previous,
      soul: buildSoulDraftPatch(
        previous.soul,
        buildSoulTemplatePatch(t, template),
      ),
    }));
    showMessage(memoryT(t, "settings.memory.soul.message.templateApplied"));
  };

  const handleSoulStyleProfileChange = (styleProfileId: SoulStyleProfileId) => {
    setDraft((previous) => ({
      ...previous,
      soul: buildSoulDraftPatch(previous.soul, {
        enabled: true,
        style_profile_id: styleProfileId,
      }),
    }));
    showMessage(
      memoryT(t, "settings.memory.soul.message.styleProfileUpdated"),
    );
  };

  const handleSoulReset = () => {
    setDraft((previous) => {
      const previousSoul = normalizeSoulConfig(previous.soul);
      return {
        ...previous,
        soul: normalizeSoulConfig({
          enabled: false,
          name: undefined,
          summary: undefined,
          style_profile_id: undefined,
          tone: [],
          communication_style: [],
          explanation_depth: undefined,
          challenge_style: undefined,
          avoid: [],
          artifact_voice: previousSoul.artifact_voice,
          imported_from: "manual",
          updated_at: new Date().toISOString(),
        }),
      };
    });
    setSoulImportPreview(null);
    setSoulImportText("");
    showMessage(memoryT(t, "settings.memory.soul.message.reset"));
  };

  const handleSoulImportPreview = () => {
    const result = parseSoulMarkdown(soulImportText);
    setSoulImportPreview(result);
    showMessage(
      memoryT(
        t,
        result.canImport
          ? "settings.memory.soul.message.importPreviewReady"
          : "settings.memory.soul.message.importEmpty",
      ),
    );
  };

  const handleSoulImportApply = () => {
    if (!soulImportPreview?.canImport) {
      return;
    }
    setDraft((previous) => {
      const previousSoul = normalizeSoulConfig(previous.soul);
      return {
        ...previous,
        soul: normalizeSoulConfig({
          ...soulImportPreview.draft,
          artifact_voice: previousSoul.artifact_voice,
          style_profile_id: previousSoul.style_profile_id,
        }),
      };
    });
    showMessage(memoryT(t, "settings.memory.soul.message.importApplied"));
  };

  const handleSoulExportCopy = async () => {
    if (!soulExportMarkdown) {
      showMessage(memoryT(t, "settings.memory.soul.message.exportEmpty"));
      return;
    }

    try {
      await navigator.clipboard.writeText(soulExportMarkdown);
      showMessage(memoryT(t, "settings.memory.soul.message.exportCopied"));
    } catch (error) {
      console.error("复制 SOUL.md 失败:", error);
      showMessage(
        memoryT(t, "settings.memory.soul.message.exportCopyFailed"),
      );
    }
  };

  const handleCancel = () => {
    setDraft(snapshot);
    showMessage(memoryT(t, "settings.memory.message.restored"));
  };

  const handleSave = async () => {
    if (!config) {
      return;
    }
    setSaving(true);
    try {
      const updatedConfig: Config = {
        ...config,
        memory: normalizeMemoryConfig(draft),
      };
      await saveConfig(updatedConfig);
      setConfig(updatedConfig);
      setSnapshot(updatedConfig.memory ?? normalizeMemoryConfig());
      setDraft(updatedConfig.memory ?? normalizeMemoryConfig());
      showMessage(memoryT(t, "settings.memory.message.saved"));
    } catch (error) {
      console.error("保存记忆设置失败:", error);
      showMessage(memoryT(t, "settings.memory.message.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <LoadingSkeleton />;
  }

  const messageIsError = Boolean(
    message &&
    /失败|失敗|실패|failed|error|cannot|can't/u.test(message.toLowerCase()),
  );

  return (
    <div
      className="mx-auto max-w-[820px] space-y-5 pb-8"
      data-testid="settings-memory-page"
    >
      {message ? (
        <div
          className={cn(
            "flex items-center gap-3 rounded-md border px-4 py-3 text-sm shadow-sm shadow-slate-950/5",
            messageIsError
              ? "border-rose-200 bg-rose-50 text-rose-700"
              : "border-emerald-200 bg-emerald-50 text-emerald-700",
          )}
        >
          {messageIsError ? (
            <AlertCircle className="h-4 w-4 shrink-0" />
          ) : (
            <CheckCircle2 className="h-4 w-4 shrink-0" />
          )}
          <span>{message}</span>
        </div>
      ) : null}

      <section className="rounded-md border border-slate-200/90 bg-white p-5 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1.5">
            <h1 className="text-[24px] font-semibold text-slate-950">
              {memoryT(t, "settings.memory.title")}
            </h1>
            <p className="text-sm leading-6 text-slate-500">
              {memoryT(t, "settings.memory.hero.description")}
            </p>
          </div>
          <div className="flex items-center gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
            <span className="text-xs font-medium text-slate-600">
              {draft.enabled
                ? memoryT(t, "settings.memory.status.enabled")
                : memoryT(t, "settings.memory.status.disabled")}
            </span>
            <Switch
              aria-label={memoryT(t, "settings.memory.toggle.aria")}
              data-testid="settings-memory-toggle"
              checked={draft.enabled}
              onCheckedChange={(checked) =>
                setDraft((previous) => ({
                  ...previous,
                  enabled: checked,
                }))
              }
            />
          </div>
        </div>
      </section>

      <section className="rounded-md border border-slate-200/90 bg-white p-2 shadow-sm shadow-slate-950/5">
        <div
          className="grid gap-2 md:grid-cols-3"
          role="tablist"
          aria-label={memoryT(t, "settings.memory.tabs.aria")}
        >
          {tabs.map((tab) => {
            const selected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={selected}
                data-testid={`settings-memory-tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "rounded-md px-3 py-3 text-left transition",
                  selected
                    ? "bg-emerald-50 text-emerald-900 ring-1 ring-emerald-200"
                    : "text-slate-500 hover:bg-slate-50 hover:text-slate-900",
                )}
              >
                <span className="block text-sm font-semibold">
                  {memoryT(t, tab.labelKey)}
                </span>
                <span className="mt-1 block text-xs leading-5">
                  {memoryT(t, tab.descriptionKey)}
                </span>
              </button>
            );
          })}
        </div>
      </section>

      {activeTab === "soul" ? (
        <section
          className="rounded-md border border-slate-200/90 bg-white p-5 shadow-sm shadow-slate-950/5"
          data-testid="settings-memory-soul-panel"
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex min-w-0 items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-sky-200 bg-sky-50 text-sky-700">
                <Brain className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-slate-950">
                  {memoryT(t, "settings.memory.soul.title")}
                </p>
                <p className="mt-1 text-sm leading-6 text-slate-500">
                  {memoryT(t, "settings.memory.soul.description")}
                </p>
              </div>
            </div>
            <div
              className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600"
              data-testid="settings-memory-soul-current-status"
              data-state={soulEnabledWithContent ? "enabled" : "disabled"}
            >
              {soulEnabledWithContent
                ? memoryT(t, "settings.memory.soul.current.enabled")
                : memoryT(t, "settings.memory.soul.current.disabled")}
            </div>
          </div>

          <div className="mt-5 rounded-md border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-950">
                  {soulEnabledWithContent
                    ? (soul.name ??
                      memoryT(t, "settings.memory.soul.current.title"))
                    : memoryT(t, "settings.memory.soul.current.emptyTitle")}
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {soulEnabledWithContent
                    ? memoryT(t, "settings.memory.soul.current.description")
                    : memoryT(
                        t,
                        "settings.memory.soul.current.emptyDescription",
                      )}
                </p>
              </div>
            </div>
            {soulPreviewItems.length > 0 ? (
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {soulPreviewItems.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-md bg-white p-3 ring-1 ring-slate-200"
                  >
                    <p className="text-xs font-semibold text-slate-500">
                      {memoryT(t, item.labelKey)}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {item.values.map((value) => (
                        <span
                          key={value}
                          className="rounded-md bg-slate-100 px-2 py-1 text-xs leading-5 text-slate-700"
                        >
                          {value}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
          <StyleProfileSection
            value={soul.style_profile_id}
            onChange={handleSoulStyleProfileChange}
            showMessage={showMessage}
            t={t}
          />

          <div className="mt-5">
            <p className="text-sm font-semibold text-slate-950">
              {memoryT(t, "settings.memory.soul.template.title")}
            </p>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {memoryT(t, "settings.memory.soul.template.description")}
            </p>
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              {SOUL_TEMPLATE_DEFINITIONS.map((template) => {
                return (
                  <div
                    key={template.id}
                    className="flex min-h-[168px] flex-col justify-between rounded-md border border-slate-200 bg-white p-4"
                  >
                    <div>
                      <p className="text-sm font-semibold text-slate-950">
                        {memoryT(t, template.titleKey)}
                      </p>
                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        {memoryT(t, template.descriptionKey)}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleSoulTemplateApply(template)}
                      data-testid={`settings-memory-soul-template-${template.id}`}
                      className="mt-4 rounded-md border border-slate-950 bg-slate-950 px-3 py-2 text-xs font-medium text-white transition hover:bg-slate-800"
                    >
                      {memoryT(t, "settings.memory.soul.template.apply")}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 bg-slate-50 p-3">
            <div>
              <p className="text-xs font-semibold text-slate-700">
                {memoryT(t, "settings.memory.soul.boundary.title")}
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                {memoryT(t, "settings.memory.soul.boundary.description")}
              </p>
            </div>
            <button
              type="button"
              onClick={handleSoulReset}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-950"
            >
              {memoryT(t, "settings.memory.soul.action.reset")}
            </button>
          </div>
        </section>
      ) : null}

      {activeTab === "advanced" ? (
        <section
          className="rounded-md border border-slate-200/90 bg-white p-5 shadow-sm shadow-slate-950/5"
          data-testid="settings-memory-advanced-panel"
        >
          <label
            className="text-sm font-semibold text-slate-950"
            htmlFor="memory-embedding-provider"
          >
            {memoryT(t, "settings.memory.embedding.providerSelect.title")}
          </label>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            {memoryT(t, "settings.memory.embedding.providerSelect.description")}
          </p>

          <div className="relative mt-4">
            <select
              id="memory-embedding-provider"
              value={providerChoice}
              onChange={(event) =>
                handleProviderChange(
                  event.target.value as EmbeddingProviderChoice,
                )
              }
              className="w-full appearance-none rounded-md border border-slate-200 bg-slate-50 px-4 py-3 pr-10 text-sm font-medium text-slate-800 outline-none transition focus:border-slate-400 focus:bg-white"
              aria-label={memoryT(
                t,
                "settings.memory.embedding.providerSelect.aria",
              )}
            >
              {PROVIDER_CHOICES.map((item) => (
                <option key={item.value} value={item.value}>
                  {memoryT(t, item.labelKey)}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          </div>

          <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm font-semibold text-slate-900">
              {memoryT(t, activeProvider.labelKey)}
            </p>
            <p className="mt-1 text-sm leading-6 text-slate-500">
              {memoryT(t, activeProvider.descriptionKey)}
            </p>
          </div>

          <div className="mt-5 rounded-md border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex min-w-0 items-start gap-3">
                <FileText className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-900">
                    {memoryT(t, "settings.memory.soul.export.title")}
                  </p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    {memoryT(t, "settings.memory.soul.export.description")}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleSoulExportCopy}
                disabled={!soulExportMarkdown}
                data-testid="settings-memory-soul-copy-export"
                className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
              >
                {memoryT(t, "settings.memory.soul.action.copyExport")}
              </button>
            </div>
            <pre className="mt-3 max-h-[220px] overflow-auto whitespace-pre-wrap rounded-md bg-white p-3 text-xs leading-5 text-slate-600 ring-1 ring-slate-200">
              {soulExportMarkdown ||
                memoryT(t, "settings.memory.soul.export.empty")}
            </pre>
          </div>

          <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-start gap-3">
              <FileText className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-slate-900">
                  {memoryT(t, "settings.memory.soul.import.title")}
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {memoryT(t, "settings.memory.soul.import.description")}
                </p>
              </div>
            </div>
            <textarea
              value={soulImportText}
              onChange={(event) => setSoulImportText(event.target.value)}
              rows={5}
              data-testid="settings-memory-soul-import-textarea"
              placeholder={memoryT(
                t,
                "settings.memory.soul.import.placeholder",
              )}
              className="mt-3 min-h-[128px] w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm leading-6 text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-slate-400"
            />
            {soulImportPreview ? (
              <div className="mt-3 space-y-2 rounded-md border border-slate-200 bg-white p-3">
                <p className="text-xs font-semibold text-slate-600">
                  {memoryT(t, "settings.memory.soul.import.preview")}
                </p>
                {soulImportPreview.warnings.length > 0 ? (
                  <ul className="space-y-1 text-xs leading-5 text-amber-700">
                    {soulImportPreview.warnings.map((warning) => (
                      <li key={warning}>
                        {memoryT(t, SOUL_WARNING_KEYS[warning])}
                      </li>
                    ))}
                  </ul>
                ) : null}
                <pre className="max-h-[160px] overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-50">
                  {soulImportPreview.preview ||
                    memoryT(t, "settings.memory.soul.import.emptyPreview")}
                </pre>
              </div>
            ) : null}
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleSoulImportPreview}
                className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
              >
                {memoryT(t, "settings.memory.soul.action.previewImport")}
              </button>
              <button
                type="button"
                onClick={handleSoulImportApply}
                disabled={!soulImportPreview?.canImport}
                className="rounded-md border border-slate-950 bg-slate-950 px-3 py-2 text-xs font-medium text-white transition hover:bg-slate-800 disabled:opacity-60"
              >
                {memoryT(t, "settings.memory.soul.action.applyImport")}
              </button>
            </div>
          </div>
        </section>
      ) : null}

      {activeTab === "memory" ? (
        <MemoryStoreStatusPanel
          vectorSearchEnabled={vectorSearchEnabled}
          memoryStatusDescriptionKey={memoryStatusDescriptionKey}
          showMessage={showMessage}
        />
      ) : null}

      <div className="flex flex-col gap-3 rounded-md border border-slate-200/90 bg-white p-4 shadow-sm shadow-slate-950/5 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs leading-5 text-slate-500">
          {dirty
            ? memoryT(t, "settings.memory.toggle.unsavedHint")
            : memoryT(t, "settings.memory.toggle.syncedHint")}
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:justify-end">
          <button
            type="button"
            onClick={handleCancel}
            disabled={!dirty || saving}
            data-testid="settings-memory-cancel"
            className="rounded-md border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-900 disabled:opacity-60"
          >
            {memoryT(t, "settings.memory.action.cancel")}
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={!dirty || saving}
            data-testid="settings-memory-save"
            className="rounded-md border border-slate-950 bg-slate-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:opacity-60"
          >
            {saving
              ? memoryT(t, "settings.memory.action.saving")
              : memoryT(t, "settings.memory.action.save")}
          </button>
        </div>
      </div>
    </div>
  );
}
