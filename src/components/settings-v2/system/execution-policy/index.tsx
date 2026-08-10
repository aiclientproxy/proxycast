import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import {
  AlertCircle,
  Plus,
  RefreshCw,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  TerminalSquare,
  Trash2,
  Workflow,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  getConfig,
  saveConfig,
  type Config,
  type NativeAgentConfig,
  type ToolExecutionCommandRiskLevelConfig,
  type ToolExecutionCommandRuleConfig,
  type ToolExecutionNetworkRuleConfig,
  type WorkspaceSandboxConfig,
} from "@/lib/api/appConfig";
import type { ExecutionPolicyFocusContext } from "@/types/page";
import { resolveExecutionPolicyNetworkRuleFocus } from "./executionPolicyFocus";
import { ExecutionPolicyNetworkFocusPanel } from "./ExecutionPolicyNetworkFocusPanel";
import { WindowsSandboxReadinessStatus } from "./WindowsSandboxReadinessStatus";

type RuleKind = "shell" | "network";
type MatchType = NonNullable<ToolExecutionCommandRuleConfig["match_type"]>;
type NetworkTarget = NonNullable<ToolExecutionNetworkRuleConfig["target"]>;
type WarningPolicy = "none" | "shell_command_risk";
type PolicySourceLayerId =
  | "default"
  | "persisted"
  | "organization"
  | "user"
  | "runtime"
  | "request";

interface PolicySourceLayerView {
  id: PolicySourceLayerId;
  order: number;
  editable: boolean;
  active: boolean;
  detail: string;
}

const DEFAULT_WORKSPACE_SANDBOX: WorkspaceSandboxConfig = {
  enabled: false,
  strict: false,
  notify_on_fallback: true,
};

const DEFAULT_AGENT_CONFIG: NativeAgentConfig = {
  use_default_system_prompt: true,
  default_model: "claude-sonnet-4-20250514",
  temperature: 0.7,
  max_tokens: 4096,
  workspace_sandbox: DEFAULT_WORKSPACE_SANDBOX,
  tool_execution: {
    tool_overrides: {},
    shell_command_rules: [],
    network_rules: [],
  },
};

const FIELD_CLASS_NAME =
  "w-full rounded-[16px] border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-slate-300 focus:ring-2 focus:ring-slate-200";
const SELECT_CLASS_NAME =
  "w-full rounded-[16px] border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-700 outline-none transition focus:border-slate-300 focus:ring-2 focus:ring-slate-200";
const PRIMARY_BUTTON_CLASS_NAME =
  "inline-flex items-center gap-2 rounded-full border border-slate-950 bg-slate-950 px-4 py-2 text-sm font-medium text-white shadow-sm shadow-slate-950/15 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";
const SECONDARY_BUTTON_CLASS_NAME =
  "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50";
const DANGER_BUTTON_CLASS_NAME =
  "inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-700 transition hover:border-rose-300 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50";

function normalizeAgentConfig(config?: Config | null): NativeAgentConfig {
  const agent = config?.agent ?? {};
  const toolExecution = agent.tool_execution ?? {};
  const toolOverrides = toolExecution.tool_overrides ?? {};

  return {
    ...DEFAULT_AGENT_CONFIG,
    ...agent,
    workspace_sandbox: {
      ...DEFAULT_WORKSPACE_SANDBOX,
      ...(agent.workspace_sandbox ?? {}),
    },
    tool_execution: {
      tool_overrides: {
        ...toolOverrides,
      },
      shell_command_rules: [...(toolExecution.shell_command_rules ?? [])].map(
        normalizeShellRule,
      ),
      network_rules: [...(toolExecution.network_rules ?? [])].map(
        normalizeNetworkRule,
      ),
    },
  };
}

function normalizeShellRule(
  rule: ToolExecutionCommandRuleConfig,
): ToolExecutionCommandRuleConfig {
  return {
    rule_id: rule.rule_id || "",
    match_type: rule.match_type ?? "regex",
    pattern: rule.pattern || "",
    risk_level: rule.risk_level ?? "medium",
    reason_code: rule.reason_code || "",
    reason: rule.reason || "",
  };
}

function normalizeNetworkRule(
  rule: ToolExecutionNetworkRuleConfig,
): ToolExecutionNetworkRuleConfig {
  return {
    rule_id: rule.rule_id || "",
    match_type: rule.match_type ?? "regex",
    target: rule.target ?? "url",
    pattern: rule.pattern || "",
    risk_level: rule.risk_level ?? "medium",
    reason_code: rule.reason_code || "",
    reason: rule.reason || "",
  };
}

function createShellRule(): ToolExecutionCommandRuleConfig {
  return {
    rule_id: "",
    match_type: "prefix",
    pattern: "",
    risk_level: "high",
    reason_code: "",
    reason: "",
  };
}

function createNetworkRule(): ToolExecutionNetworkRuleConfig {
  return {
    rule_id: "",
    match_type: "prefix",
    target: "host",
    pattern: "",
    risk_level: "high",
    reason_code: "",
    reason: "",
  };
}

function statusPillClassName(active: boolean) {
  return active
    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
    : "border-slate-200 bg-slate-100 text-slate-500";
}

function sourceLayerPillClassName(params: {
  active: boolean;
  editable: boolean;
}) {
  if (params.editable) {
    return "border-sky-200 bg-sky-50 text-sky-700";
  }
  if (params.active) {
    return "border-amber-200 bg-amber-50 text-amber-700";
  }
  return "border-slate-200 bg-slate-100 text-slate-500";
}

function hasToolExecutionPolicy(policy: NativeAgentConfig["tool_execution"]) {
  return Boolean(
    policy &&
    ((policy.shell_command_rules?.length ?? 0) > 0 ||
      (policy.network_rules?.length ?? 0) > 0 ||
      Object.keys(policy.tool_overrides ?? {}).length > 0),
  );
}

function buildPolicySourceLayers(params: {
  hasCustomPolicy: boolean;
  shellRuleCount: number;
  networkRuleCount: number;
  toolOverrideCount: number;
  t: TFunction<"settings", undefined>;
}): PolicySourceLayerView[] {
  return [
    {
      id: "default",
      order: 1,
      editable: false,
      active: true,
      detail: params.t("settings.executionPolicy.sources.default.detail"),
    },
    {
      id: "persisted",
      order: 2,
      editable: true,
      active: params.hasCustomPolicy,
      detail: params.t("settings.executionPolicy.sources.persisted.detail", {
        shellRules: params.shellRuleCount,
        networkRules: params.networkRuleCount,
        toolOverrides: params.toolOverrideCount,
      }),
    },
    {
      id: "organization",
      order: 3,
      editable: false,
      active: false,
      detail: params.t("settings.executionPolicy.sources.organization.detail"),
    },
    {
      id: "user",
      order: 4,
      editable: false,
      active: false,
      detail: params.t("settings.executionPolicy.sources.user.detail"),
    },
    {
      id: "runtime",
      order: 5,
      editable: false,
      active: false,
      detail: params.t("settings.executionPolicy.sources.runtime.detail"),
    },
    {
      id: "request",
      order: 6,
      editable: false,
      active: false,
      detail: params.t("settings.executionPolicy.sources.request.detail"),
    },
  ];
}

function InlineMessage({
  type,
  text,
}: {
  type: "success" | "error";
  text: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-[18px] border px-4 py-3 text-sm shadow-sm shadow-slate-950/5",
        type === "success"
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-rose-200 bg-rose-50 text-rose-700",
      )}
    >
      <AlertCircle className="h-4 w-4" />
      <span>{text}</span>
    </div>
  );
}

function ToggleRow({
  title,
  description,
  checked,
  disabled,
  ariaLabel,
  onCheckedChange,
}: {
  title: string;
  description: string;
  checked: boolean;
  disabled?: boolean;
  ariaLabel: string;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-4 rounded-[22px] border border-slate-200/80 bg-slate-50 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="space-y-1">
        <p className="text-sm font-semibold text-slate-900">{title}</p>
        <p className="text-sm leading-6 text-slate-500">{description}</p>
      </div>
      <Switch
        aria-label={ariaLabel}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
      />
    </div>
  );
}

function PolicySourceLayerPanel({
  layers,
}: {
  layers: PolicySourceLayerView[];
}) {
  const { t } = useTranslation("settings");

  return (
    <section
      className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5"
      data-testid="execution-policy-source-layers"
    >
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
        <Workflow className="h-4 w-4 text-sky-600" />
        {t("settings.executionPolicy.sources.title")}
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-500">
        {t("settings.executionPolicy.sources.description")}
      </p>
      <div className="mt-4 space-y-2">
        {layers.map((layer) => (
          <div
            key={layer.id}
            className="rounded-[18px] border border-slate-200 bg-slate-50 px-3 py-3"
            data-testid={`execution-policy-source-${layer.id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold text-slate-400">
                    {layer.order}
                  </span>
                  <span className="text-sm font-semibold text-slate-900">
                    {t(`settings.executionPolicy.sources.${layer.id}.title`)}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {layer.detail}
                </p>
              </div>
              <span
                className={cn(
                  "shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-medium",
                  sourceLayerPillClassName({
                    active: layer.active,
                    editable: layer.editable,
                  }),
                )}
              >
                {layer.editable
                  ? t("settings.executionPolicy.sources.badge.editable")
                  : layer.active
                    ? t("settings.executionPolicy.sources.badge.active")
                    : t("settings.executionPolicy.sources.badge.readonly")}
              </span>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 rounded-[18px] border border-amber-200 bg-amber-50 px-3 py-3 text-xs leading-5 text-amber-900">
        {t("settings.executionPolicy.sources.note")}
      </p>
    </section>
  );
}

function RuleEditor({
  kind,
  rules,
  disabled,
  focusedIndex,
  onAdd,
  onRemove,
  onUpdate,
}: {
  kind: RuleKind;
  rules: Array<ToolExecutionCommandRuleConfig | ToolExecutionNetworkRuleConfig>;
  disabled?: boolean;
  focusedIndex?: number | null;
  onAdd: () => void;
  onRemove: (index: number) => void;
  onUpdate: (
    index: number,
    nextRule: ToolExecutionCommandRuleConfig | ToolExecutionNetworkRuleConfig,
  ) => void;
}) {
  const { t } = useTranslation("settings");
  const isNetwork = kind === "network";

  return (
    <div className="space-y-3">
      {rules.length === 0 ? (
        <div className="rounded-[20px] border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-sm leading-6 text-slate-500">
          {t(`settings.executionPolicy.rules.${kind}.empty`)}
        </div>
      ) : (
        rules.map((rule, index) => (
          <div
            key={`${kind}-${index}`}
            data-testid={`${kind}-rule-${index}`}
            data-focused={focusedIndex === index ? "true" : undefined}
            className={cn(
              "grid gap-3 rounded-[22px] border bg-slate-50 p-4 transition xl:grid-cols-[1.1fr_0.8fr_0.8fr_1.1fr_auto]",
              focusedIndex === index
                ? "border-amber-300 bg-amber-50/70 ring-2 ring-amber-200"
                : "border-slate-200",
            )}
          >
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-slate-600">
                {t("settings.executionPolicy.rules.field.ruleId")}
              </span>
              <input
                className={FIELD_CLASS_NAME}
                value={rule.rule_id}
                disabled={disabled}
                placeholder={t(
                  "settings.executionPolicy.rules.field.ruleIdPlaceholder",
                )}
                onChange={(event) =>
                  onUpdate(index, {
                    ...rule,
                    rule_id: event.target.value,
                  })
                }
              />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-slate-600">
                {t("settings.executionPolicy.rules.field.matchType")}
              </span>
              <select
                className={SELECT_CLASS_NAME}
                value={rule.match_type ?? "regex"}
                disabled={disabled}
                onChange={(event) =>
                  onUpdate(index, {
                    ...rule,
                    match_type: event.target.value as MatchType,
                  })
                }
              >
                <option value="regex">
                  {t("settings.executionPolicy.matchType.regex")}
                </option>
                <option value="prefix">
                  {t("settings.executionPolicy.matchType.prefix")}
                </option>
                <option value="exact">
                  {t("settings.executionPolicy.matchType.exact")}
                </option>
              </select>
            </label>
            {isNetwork ? (
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-slate-600">
                  {t("settings.executionPolicy.rules.field.target")}
                </span>
                <select
                  className={SELECT_CLASS_NAME}
                  value={
                    (rule as ToolExecutionNetworkRuleConfig).target ?? "url"
                  }
                  disabled={disabled}
                  onChange={(event) =>
                    onUpdate(index, {
                      ...rule,
                      target: event.target.value as NetworkTarget,
                    } as ToolExecutionNetworkRuleConfig)
                  }
                >
                  <option value="url">
                    {t("settings.executionPolicy.networkTarget.url")}
                  </option>
                  <option value="host">
                    {t("settings.executionPolicy.networkTarget.host")}
                  </option>
                </select>
              </label>
            ) : (
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-slate-600">
                  {t("settings.executionPolicy.rules.field.risk")}
                </span>
                <select
                  className={SELECT_CLASS_NAME}
                  value={rule.risk_level ?? "medium"}
                  disabled={disabled}
                  onChange={(event) =>
                    onUpdate(index, {
                      ...rule,
                      risk_level: event.target
                        .value as ToolExecutionCommandRiskLevelConfig,
                    })
                  }
                >
                  <option value="low">
                    {t("settings.executionPolicy.risk.low")}
                  </option>
                  <option value="medium">
                    {t("settings.executionPolicy.risk.medium")}
                  </option>
                  <option value="high">
                    {t("settings.executionPolicy.risk.high")}
                  </option>
                </select>
              </label>
            )}
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-slate-600">
                {t("settings.executionPolicy.rules.field.pattern")}
              </span>
              <input
                className={FIELD_CLASS_NAME}
                value={rule.pattern}
                disabled={disabled}
                placeholder={t(
                  `settings.executionPolicy.rules.${kind}.patternPlaceholder`,
                )}
                onChange={(event) =>
                  onUpdate(index, {
                    ...rule,
                    pattern: event.target.value,
                  })
                }
              />
            </label>
            <button
              type="button"
              className={cn(DANGER_BUTTON_CLASS_NAME, "self-end")}
              disabled={disabled}
              onClick={() => onRemove(index)}
            >
              <Trash2 className="h-4 w-4" />
              {t("settings.executionPolicy.rules.action.remove")}
            </button>
            <label className="space-y-1.5 xl:col-span-2">
              <span className="text-xs font-medium text-slate-600">
                {t("settings.executionPolicy.rules.field.reasonCode")}
              </span>
              <input
                className={FIELD_CLASS_NAME}
                value={rule.reason_code ?? ""}
                disabled={disabled}
                placeholder={t(
                  "settings.executionPolicy.rules.field.reasonCodePlaceholder",
                )}
                onChange={(event) =>
                  onUpdate(index, {
                    ...rule,
                    reason_code: event.target.value,
                  })
                }
              />
            </label>
            <label className="space-y-1.5 xl:col-span-3">
              <span className="text-xs font-medium text-slate-600">
                {t("settings.executionPolicy.rules.field.reason")}
              </span>
              <input
                className={FIELD_CLASS_NAME}
                value={rule.reason ?? ""}
                disabled={disabled}
                placeholder={t(
                  "settings.executionPolicy.rules.field.reasonPlaceholder",
                )}
                onChange={(event) =>
                  onUpdate(index, {
                    ...rule,
                    reason: event.target.value,
                  })
                }
              />
            </label>
            {isNetwork ? (
              <label className="space-y-1.5 xl:col-span-2">
                <span className="text-xs font-medium text-slate-600">
                  {t("settings.executionPolicy.rules.field.risk")}
                </span>
                <select
                  className={SELECT_CLASS_NAME}
                  value={rule.risk_level ?? "medium"}
                  disabled={disabled}
                  onChange={(event) =>
                    onUpdate(index, {
                      ...rule,
                      risk_level: event.target
                        .value as ToolExecutionCommandRiskLevelConfig,
                    })
                  }
                >
                  <option value="low">
                    {t("settings.executionPolicy.risk.low")}
                  </option>
                  <option value="medium">
                    {t("settings.executionPolicy.risk.medium")}
                  </option>
                  <option value="high">
                    {t("settings.executionPolicy.risk.high")}
                  </option>
                </select>
              </label>
            ) : null}
          </div>
        ))
      )}
      <button
        type="button"
        className={SECONDARY_BUTTON_CLASS_NAME}
        disabled={disabled}
        onClick={onAdd}
      >
        <Plus className="h-4 w-4" />
        {t(`settings.executionPolicy.rules.${kind}.add`)}
      </button>
    </div>
  );
}

export function ExecutionPolicySettings({
  focus,
}: {
  focus?: ExecutionPolicyFocusContext | null;
} = {}) {
  const { t } = useTranslation("settings");
  const focusedNetworkRuleRef = useRef<HTMLDivElement | null>(null);
  const [fullConfig, setFullConfig] = useState<Config | null>(null);
  const [agentConfig, setAgentConfig] =
    useState<NativeAgentConfig>(DEFAULT_AGENT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const workspaceSandbox =
    agentConfig.workspace_sandbox ?? DEFAULT_WORKSPACE_SANDBOX;
  const toolExecution = useMemo(
    () => agentConfig.tool_execution ?? {},
    [agentConfig.tool_execution],
  );
  const shellRules = useMemo(
    () => toolExecution.shell_command_rules ?? [],
    [toolExecution.shell_command_rules],
  );
  const networkRules = useMemo(
    () => toolExecution.network_rules ?? [],
    [toolExecution.network_rules],
  );
  const networkRuleFocus = useMemo(
    () =>
      resolveExecutionPolicyNetworkRuleFocus(
        networkRules as ToolExecutionNetworkRuleConfig[],
        focus,
      ),
    [focus, networkRules],
  );
  const bashWarningPolicy: WarningPolicy =
    toolExecution.tool_overrides?.bash?.warning_policy ??
    toolExecution.tool_overrides?.Bash?.warning_policy ??
    "shell_command_risk";
  const toolOverrideCount = useMemo(
    () =>
      new Set(
        Object.keys(toolExecution.tool_overrides ?? {}).map((key) =>
          key.toLowerCase(),
        ),
      ).size,
    [toolExecution.tool_overrides],
  );
  const hasCustomPolicy = useMemo(
    () =>
      workspaceSandbox.enabled ||
      workspaceSandbox.strict ||
      bashWarningPolicy !== "shell_command_risk" ||
      hasToolExecutionPolicy(toolExecution),
    [
      bashWarningPolicy,
      toolExecution,
      workspaceSandbox.enabled,
      workspaceSandbox.strict,
    ],
  );
  const policySourceLayers = useMemo(
    () =>
      buildPolicySourceLayers({
        hasCustomPolicy,
        shellRuleCount: shellRules.length,
        networkRuleCount: networkRules.length,
        toolOverrideCount,
        t,
      }),
    [
      hasCustomPolicy,
      networkRules.length,
      shellRules.length,
      t,
      toolOverrideCount,
    ],
  );

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setMessage(null);
    try {
      const config = await getConfig({ forceRefresh: true });
      setFullConfig(config);
      setAgentConfig(normalizeAgentConfig(config));
    } catch (error) {
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.executionPolicy.message.loadFailed"),
      });
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    if (!networkRuleFocus) {
      return;
    }
    focusedNetworkRuleRef.current?.scrollIntoView?.({
      block: "center",
      behavior: "smooth",
    });
  }, [networkRuleFocus]);

  const updateWorkspaceSandbox = useCallback(
    (patch: Partial<WorkspaceSandboxConfig>) => {
      setAgentConfig((current) => ({
        ...current,
        workspace_sandbox: {
          ...DEFAULT_WORKSPACE_SANDBOX,
          ...(current.workspace_sandbox ?? {}),
          ...patch,
        },
      }));
    },
    [],
  );

  const updateBashWarningPolicy = useCallback((policy: WarningPolicy) => {
    setAgentConfig((current) => {
      const currentToolExecution = current.tool_execution ?? {};
      const currentOverrides = currentToolExecution.tool_overrides ?? {};
      const { Bash: legacyBashOverride, ...otherOverrides } = currentOverrides;
      return {
        ...current,
        tool_execution: {
          ...currentToolExecution,
          tool_overrides: {
            ...otherOverrides,
            bash: {
              ...(currentOverrides.bash ?? legacyBashOverride ?? {}),
              warning_policy: policy,
              restriction_profile: "workspace_shell_command",
              sandbox_profile: "workspace_command",
            },
          },
        },
      };
    });
  }, []);

  const addRule = useCallback((kind: RuleKind) => {
    setAgentConfig((current) => {
      const currentToolExecution = current.tool_execution ?? {};
      return {
        ...current,
        tool_execution: {
          ...currentToolExecution,
          shell_command_rules:
            kind === "shell"
              ? [
                  ...(currentToolExecution.shell_command_rules ?? []),
                  createShellRule(),
                ]
              : (currentToolExecution.shell_command_rules ?? []),
          network_rules:
            kind === "network"
              ? [
                  ...(currentToolExecution.network_rules ?? []),
                  createNetworkRule(),
                ]
              : (currentToolExecution.network_rules ?? []),
        },
      };
    });
  }, []);

  const addNetworkRule = useCallback((rule: ToolExecutionNetworkRuleConfig) => {
    setAgentConfig((current) => {
      const currentToolExecution = current.tool_execution ?? {};
      return {
        ...current,
        tool_execution: {
          ...currentToolExecution,
          network_rules: [
            ...(currentToolExecution.network_rules ?? []),
            normalizeNetworkRule(rule),
          ],
        },
      };
    });
  }, []);

  const removeRule = useCallback((kind: RuleKind, index: number) => {
    setAgentConfig((current) => {
      const currentToolExecution = current.tool_execution ?? {};
      const shell = [...(currentToolExecution.shell_command_rules ?? [])];
      const network = [...(currentToolExecution.network_rules ?? [])];
      if (kind === "shell") {
        shell.splice(index, 1);
      } else {
        network.splice(index, 1);
      }
      return {
        ...current,
        tool_execution: {
          ...currentToolExecution,
          shell_command_rules: shell,
          network_rules: network,
        },
      };
    });
  }, []);

  const updateRule = useCallback(
    (
      kind: RuleKind,
      index: number,
      nextRule: ToolExecutionCommandRuleConfig | ToolExecutionNetworkRuleConfig,
    ) => {
      setAgentConfig((current) => {
        const currentToolExecution = current.tool_execution ?? {};
        const shell = [...(currentToolExecution.shell_command_rules ?? [])];
        const network = [...(currentToolExecution.network_rules ?? [])];
        if (kind === "shell") {
          shell[index] = normalizeShellRule(
            nextRule as ToolExecutionCommandRuleConfig,
          );
        } else {
          network[index] = normalizeNetworkRule(
            nextRule as ToolExecutionNetworkRuleConfig,
          );
        }
        return {
          ...current,
          tool_execution: {
            ...currentToolExecution,
            shell_command_rules: shell,
            network_rules: network,
          },
        };
      });
    },
    [],
  );

  const handleSave = useCallback(async () => {
    if (!fullConfig || saving) {
      return;
    }

    setSaving(true);
    setMessage(null);
    try {
      await saveConfig({
        ...fullConfig,
        agent: {
          ...(fullConfig.agent ?? {}),
          ...agentConfig,
        },
      });
      setFullConfig((current) =>
        current
          ? {
              ...current,
              agent: {
                ...(current.agent ?? {}),
                ...agentConfig,
              },
            }
          : current,
      );
      setMessage({
        type: "success",
        text: t("settings.executionPolicy.message.saved"),
      });
    } catch (error) {
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.executionPolicy.message.saveFailed"),
      });
    } finally {
      setSaving(false);
    }
  }, [agentConfig, fullConfig, saving, t]);

  if (loading) {
    return (
      <div className="space-y-6 pb-8" data-testid="execution-policy-settings">
        <div className="h-[132px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <div className="h-[220px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
            <div className="h-[260px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
          </div>
          <div className="h-[260px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-8" data-testid="execution-policy-settings">
      <section className="rounded-[26px] border border-slate-200/80 bg-white px-5 py-4 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-sky-600" />
              <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">
                {t("settings.executionPolicy.title")}
              </h1>
              <WorkbenchInfoTip
                ariaLabel={t("settings.executionPolicy.hero.tipAria")}
                content={t("settings.executionPolicy.hero.tip")}
                tone="slate"
              />
            </div>
            <p className="text-sm leading-6 text-slate-500">
              {t("settings.executionPolicy.description")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            <span
              className={cn(
                "rounded-full border px-3 py-1.5 text-xs font-medium",
                statusPillClassName(hasCustomPolicy),
              )}
            >
              {hasCustomPolicy
                ? t("settings.executionPolicy.status.custom")
                : t("settings.executionPolicy.status.default")}
            </span>
            <button
              type="button"
              className={SECONDARY_BUTTON_CLASS_NAME}
              disabled={saving}
              onClick={() => void loadConfig()}
            >
              <RefreshCw className="h-4 w-4" />
              {t("settings.executionPolicy.action.reload")}
            </button>
            <button
              type="button"
              className={PRIMARY_BUTTON_CLASS_NAME}
              disabled={saving || !fullConfig}
              onClick={() => void handleSave()}
            >
              <Save className="h-4 w-4" />
              {saving
                ? t("settings.executionPolicy.action.saving")
                : t("settings.executionPolicy.action.save")}
            </button>
          </div>
        </div>
      </section>

      {message ? (
        <InlineMessage type={message.type} text={message.text} />
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
          <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <ShieldCheck className="h-4 w-4 text-sky-600" />
              {t("settings.executionPolicy.workspace.title")}
              <WorkbenchInfoTip
                ariaLabel={t("settings.executionPolicy.workspace.tipAria")}
                content={t("settings.executionPolicy.workspace.description")}
                tone="slate"
              />
            </div>
            <div className="mt-5 space-y-3">
              <ToggleRow
                title={t("settings.executionPolicy.workspace.enabled.title")}
                description={t(
                  "settings.executionPolicy.workspace.enabled.description",
                )}
                ariaLabel={t("settings.executionPolicy.workspace.enabled.aria")}
                checked={workspaceSandbox.enabled}
                disabled={saving}
                onCheckedChange={(checked) =>
                  updateWorkspaceSandbox({ enabled: checked })
                }
              />
              <ToggleRow
                title={t("settings.executionPolicy.workspace.strict.title")}
                description={t(
                  "settings.executionPolicy.workspace.strict.description",
                )}
                ariaLabel={t("settings.executionPolicy.workspace.strict.aria")}
                checked={workspaceSandbox.strict}
                disabled={saving || !workspaceSandbox.enabled}
                onCheckedChange={(checked) =>
                  updateWorkspaceSandbox({ strict: checked })
                }
              />
              <ToggleRow
                title={t("settings.executionPolicy.workspace.notify.title")}
                description={t(
                  "settings.executionPolicy.workspace.notify.description",
                )}
                ariaLabel={t("settings.executionPolicy.workspace.notify.aria")}
                checked={workspaceSandbox.notify_on_fallback}
                disabled={saving || !workspaceSandbox.enabled}
                onCheckedChange={(checked) =>
                  updateWorkspaceSandbox({ notify_on_fallback: checked })
                }
              />
              <WindowsSandboxReadinessStatus />
            </div>
          </section>

          <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <TerminalSquare className="h-4 w-4 text-sky-600" />
                  {t("settings.executionPolicy.shell.title")}
                  <WorkbenchInfoTip
                    ariaLabel={t("settings.executionPolicy.shell.tipAria")}
                    content={t("settings.executionPolicy.shell.description")}
                    tone="slate"
                  />
                </div>
              </div>
              <label className="min-w-[240px] space-y-1.5">
                <span className="text-xs font-medium text-slate-600">
                  {t("settings.executionPolicy.shell.warningPolicy")}
                </span>
                <select
                  className={SELECT_CLASS_NAME}
                  value={bashWarningPolicy}
                  disabled={saving}
                  onChange={(event) =>
                    updateBashWarningPolicy(event.target.value as WarningPolicy)
                  }
                >
                  <option value="shell_command_risk">
                    {t(
                      "settings.executionPolicy.shell.warningPolicy.shellCommandRisk",
                    )}
                  </option>
                  <option value="none">
                    {t("settings.executionPolicy.shell.warningPolicy.none")}
                  </option>
                </select>
              </label>
            </div>
            <div className="mt-5">
              <RuleEditor
                kind="shell"
                rules={shellRules}
                disabled={saving}
                onAdd={() => addRule("shell")}
                onRemove={(index) => removeRule("shell", index)}
                onUpdate={(index, rule) => updateRule("shell", index, rule)}
              />
            </div>
          </section>

          <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <SlidersHorizontal className="h-4 w-4 text-sky-600" />
              {t("settings.executionPolicy.network.title")}
              <WorkbenchInfoTip
                ariaLabel={t("settings.executionPolicy.network.tipAria")}
                content={t("settings.executionPolicy.network.description")}
                tone="slate"
              />
            </div>
            <div className="mt-5">
              <ExecutionPolicyNetworkFocusPanel
                disabled={saving}
                focus={focus}
                focusedNetworkRuleRef={focusedNetworkRuleRef}
                networkRuleFocus={networkRuleFocus}
                onAddSuggestedRule={addNetworkRule}
              />
              <RuleEditor
                kind="network"
                rules={networkRules}
                disabled={saving}
                focusedIndex={networkRuleFocus?.index ?? null}
                onAdd={() => addRule("network")}
                onRemove={(index) => removeRule("network", index)}
                onUpdate={(index, rule) => updateRule("network", index, rule)}
              />
            </div>
          </section>
        </div>

        <aside className="space-y-4">
          <PolicySourceLayerPanel layers={policySourceLayers} />
          <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
            <div className="text-sm font-semibold text-slate-900">
              {t("settings.executionPolicy.summary.title")}
            </div>
            <dl className="mt-4 space-y-3 text-sm">
              <div className="flex items-center justify-between gap-3">
                <dt className="text-slate-500">
                  {t("settings.executionPolicy.summary.workspaceSandbox")}
                </dt>
                <dd className="font-medium text-slate-900">
                  {workspaceSandbox.enabled
                    ? t("settings.executionPolicy.summary.enabled")
                    : t("settings.executionPolicy.summary.disabled")}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-3">
                <dt className="text-slate-500">
                  {t("settings.executionPolicy.summary.strict")}
                </dt>
                <dd className="font-medium text-slate-900">
                  {workspaceSandbox.strict
                    ? t("settings.executionPolicy.summary.enabled")
                    : t("settings.executionPolicy.summary.disabled")}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-3">
                <dt className="text-slate-500">
                  {t("settings.executionPolicy.summary.shellRules")}
                </dt>
                <dd className="font-medium text-slate-900">
                  {shellRules.length}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-3">
                <dt className="text-slate-500">
                  {t("settings.executionPolicy.summary.networkRules")}
                </dt>
                <dd className="font-medium text-slate-900">
                  {networkRules.length}
                </dd>
              </div>
            </dl>
          </section>
          <section className="rounded-[26px] border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-900">
            {t("settings.executionPolicy.recoveryNote")}
          </section>
        </aside>
      </div>
    </div>
  );
}
