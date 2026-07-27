import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  MessageSquare,
  ScanQrCode,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import QRCode from "qrcode";
import { Modal, ModalBody, ModalFooter, ModalHeader } from "@/components/Modal";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useConfiguredProviders } from "@/hooks/useConfiguredProviders";
import { getConfig, saveConfig, type Config } from "@/lib/api/appConfig";
import {
  feishuChannelProbe,
  gatewayChannelStart,
  telegramChannelProbe,
  wechatChannelListAccounts,
  wechatChannelLoginStart,
  wechatChannelLoginWait,
  wechatChannelProbe,
  type ChannelsConfig,
  type FeishuBotConfig,
  type TelegramBotConfig,
  type WechatBotConfig,
  type WechatConfiguredAccount,
  type WechatLoginWaitResult,
} from "@/lib/api/channelsRuntime";
import {
  filterProviderModelsByCompatibility,
  getProviderModelCompatibilityIssue,
} from "@/components/agent/chat/utils/providerModelCompatibility";
import { cn } from "@/lib/utils";

const ChannelsDebugWorkbench = lazy(() =>
  import("@/components/settings-v2/system/channels/ChannelsDebugWorkbench").then(
    (module) => ({
      default: module.ChannelsDebugWorkbench,
    }),
  ),
);

type FlashMessage = {
  type: "success" | "error";
  text: string;
} | null;

type ManagedChannel = "telegram" | "feishu" | "wechat";

const DEFAULT_WECHAT_BASE_URL = "https://ilinkai.weixin.qq.com";
const DEFAULT_WECHAT_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c";
const DEFAULT_WECHAT_BOT_TYPE = "3";

const DEFAULT_CHANNELS: ChannelsConfig = {
  telegram: {
    enabled: false,
    bot_token: "",
    allowed_user_ids: [],
    default_model: undefined,
  },
  discord: {
    enabled: false,
    bot_token: "",
    allowed_server_ids: [],
    default_model: undefined,
    default_account: "default",
    accounts: {},
    dm_policy: "pairing",
    allow_from: [],
    group_policy: "allowlist",
    group_allow_from: [],
    streaming: "partial",
    reply_to_mode: "off",
  },
  feishu: {
    enabled: false,
    app_id: "",
    app_secret: "",
    default_model: undefined,
    dm_policy: "open",
    allow_from: ["*"],
    group_policy: "allowlist",
    group_allow_from: [],
  },
  wechat: {
    enabled: false,
    bot_token: "",
    base_url: "",
    cdn_base_url: "",
    default_model: undefined,
    default_account: "default",
    accounts: {},
    dm_policy: "pairing",
    allow_from: [],
    group_policy: "allowlist",
    group_allow_from: [],
    streaming: "off",
    reply_to_mode: "off",
  },
};

function normalizeChannelsConfig(
  input?: Partial<ChannelsConfig>,
): ChannelsConfig {
  return {
    telegram: {
      ...DEFAULT_CHANNELS.telegram,
      ...(input?.telegram ?? {}),
      allowed_user_ids:
        input?.telegram?.allowed_user_ids ??
        DEFAULT_CHANNELS.telegram.allowed_user_ids,
    },
    discord: {
      ...DEFAULT_CHANNELS.discord,
      ...(input?.discord ?? {}),
      accounts: input?.discord?.accounts ?? DEFAULT_CHANNELS.discord.accounts,
      allow_from:
        input?.discord?.allow_from ?? DEFAULT_CHANNELS.discord.allow_from,
      group_allow_from:
        input?.discord?.group_allow_from ??
        DEFAULT_CHANNELS.discord.group_allow_from,
    },
    feishu: {
      ...DEFAULT_CHANNELS.feishu,
      ...(input?.feishu ?? {}),
      allow_from:
        input?.feishu?.allow_from ?? DEFAULT_CHANNELS.feishu.allow_from,
      group_allow_from:
        input?.feishu?.group_allow_from ??
        DEFAULT_CHANNELS.feishu.group_allow_from,
      groups: input?.feishu?.groups ?? DEFAULT_CHANNELS.feishu.groups,
    },
    wechat: {
      ...DEFAULT_CHANNELS.wechat,
      ...(input?.wechat ?? {}),
      accounts: input?.wechat?.accounts ?? DEFAULT_CHANNELS.wechat.accounts,
      allow_from:
        input?.wechat?.allow_from ?? DEFAULT_CHANNELS.wechat.allow_from,
      group_allow_from:
        input?.wechat?.group_allow_from ??
        DEFAULT_CHANNELS.wechat.group_allow_from,
      groups: input?.wechat?.groups ?? DEFAULT_CHANNELS.wechat.groups,
    },
  };
}

function describeDmPolicy(policy?: string): string {
  switch ((policy || "").trim().toLowerCase()) {
    case "open":
      return "开放模式";
    case "allowlist":
      return "白名单";
    case "pairing":
      return "配对模式";
    case "disabled":
      return "已关闭";
    default:
      return "未设置";
  }
}

function describeTelegramScope(config: TelegramBotConfig): string {
  return config.allowed_user_ids.length === 0
    ? "所有用户可发起"
    : `已限制 ${config.allowed_user_ids.length} 个用户`;
}

function describeWechatStatus(config: WechatBotConfig): string {
  const accountCount = Object.keys(config.accounts ?? {}).length;
  if (!config.enabled) {
    return "未启用";
  }
  if (accountCount === 0) {
    return "等待扫码";
  }
  return `已接入 ${accountCount} 个账号`;
}

function describeWechatAccountModel(account: {
  default_model?: string;
}): string {
  return account.default_model?.trim() || "跟随全局模型";
}

function describeWechatAccountRoute(
  account: {
    base_url?: string;
  },
  fallbackBaseUrl?: string,
): string {
  const accountBaseUrl = account.base_url?.trim();
  const normalizedFallback = fallbackBaseUrl?.trim();
  if (!accountBaseUrl || accountBaseUrl === normalizedFallback) {
    return "沿用默认地址";
  }
  return "自定义地址";
}

function normalizeConnectionMode(value?: string): "webhook" | "websocket" {
  return (value || "").trim().toLowerCase() === "websocket"
    ? "websocket"
    : "webhook";
}

function normalizeStreamingMode(value?: string): "partial" | "block" | "off" {
  const normalized = (value || "").trim().toLowerCase();
  if (normalized === "block" || normalized === "off") {
    return normalized;
  }
  return "partial";
}

function normalizeReplyToMode(value?: string): "off" | "first" | "all" {
  const normalized = (value || "").trim().toLowerCase();
  if (normalized === "first" || normalized === "all") {
    return normalized;
  }
  return "off";
}

function normalizeWechatGroupPolicy(
  value?: string,
): "allowlist" | "open" | "disabled" {
  const normalized = (value || "").trim().toLowerCase();
  if (normalized === "open" || normalized === "disabled") {
    return normalized;
  }
  return "allowlist";
}

function maskMiddle(value: string, keep = 4): string {
  const trimmed = value.trim();
  if (trimmed.length <= keep * 2) {
    return trimmed || "未填写";
  }
  return `${trimmed.slice(0, keep)}...${trimmed.slice(-keep)}`;
}

function buildNextConfig(
  currentConfig: Config | null,
  channels: ChannelsConfig,
): Config {
  return {
    ...(currentConfig ?? ({} as Config)),
    channels,
  } as Config;
}

function SectionMessage({ message }: { message: FlashMessage }) {
  if (!message) {
    return null;
  }

  return (
    <div
      className={cn(
        "rounded-2xl border px-4 py-3 text-sm",
        message.type === "success"
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-rose-200 bg-rose-50 text-rose-700",
      )}
    >
      {message.text}
    </div>
  );
}

function StatusBadge({
  tone,
  children,
}: {
  tone: "success" | "warning" | "neutral";
  children: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium",
        tone === "success" &&
          "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200",
        tone === "warning" &&
          "bg-amber-100 text-amber-700 ring-1 ring-amber-200",
        tone === "neutral" &&
          "bg-slate-100 text-slate-600 ring-1 ring-slate-200",
      )}
    >
      {children}
    </span>
  );
}

function ActionButton({
  kind = "secondary",
  onClick,
  children,
  disabled,
  type = "button",
  ariaLabel,
  dataTestId,
}: {
  kind?: "primary" | "secondary";
  onClick?: () => void;
  children: string;
  disabled?: boolean;
  type?: "button" | "submit";
  ariaLabel?: string;
  dataTestId?: string;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      data-testid={dataTestId}
      className={cn(
        "inline-flex items-center justify-center rounded-xl px-3.5 py-2 text-sm font-medium transition",
        kind === "primary"
          ? "border border-emerald-200 bg-[image:var(--lime-primary-gradient)] text-white shadow-sm shadow-emerald-950/15 hover:opacity-95"
          : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      {children}
    </button>
  );
}

function SwitchField({
  checked,
  onChange,
  title,
  description,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="space-y-1">
        <div className="text-sm font-medium text-slate-900">{title}</div>
        <p className="text-xs leading-5 text-slate-500">{description}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
          checked ? "bg-emerald-500" : "bg-slate-300",
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 rounded-full bg-white transition-transform",
            checked ? "translate-x-6" : "translate-x-1",
          )}
        />
      </button>
    </div>
  );
}

function FieldLabel({
  label,
  optional,
}: {
  label: string;
  optional?: boolean;
}) {
  return (
    <label className="mb-1.5 block text-sm font-medium text-slate-700">
      {label}
      {optional ? (
        <span className="ml-1 text-xs font-normal text-slate-400">可选</span>
      ) : null}
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
  optional,
  mono,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  optional?: boolean;
  mono?: boolean;
  type?: "text" | "number";
}) {
  return (
    <div>
      <FieldLabel label={label} optional={optional} />
      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className={cn(
          "h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200",
          mono && "font-mono",
        )}
      />
    </div>
  );
}

function SecretField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(false);
  }, [label]);

  return (
    <div>
      <FieldLabel label={label} />
      <div className="flex items-center gap-2">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
        />
        <ActionButton onClick={() => setVisible((current) => !current)}>
          {visible ? "隐藏" : "显示"}
        </ActionButton>
      </div>
    </div>
  );
}

function StringListEditor({
  label,
  values,
  onChange,
  placeholder,
}: {
  label: string;
  values: string[];
  onChange: (values: string[]) => void;
  placeholder: string;
}) {
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    setInputValue("");
  }, [label]);

  const addItems = (raw: string) => {
    const candidates = raw
      .split(/[\n,]/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (candidates.length === 0) {
      return;
    }
    const nextValues = Array.from(new Set([...values, ...candidates]));
    onChange(nextValues);
    setInputValue("");
  };

  const removeValue = (target: string) => {
    onChange(values.filter((item) => item !== target));
  };

  return (
    <div>
      <FieldLabel label={label} />
      <div className="rounded-2xl border border-slate-200 bg-white p-3">
        {values.length > 0 ? (
          <div className="mb-3 flex flex-wrap gap-2">
            {values.map((value) => (
              <span
                key={value}
                className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700"
              >
                {value}
                <button
                  type="button"
                  onClick={() => removeValue(value)}
                  className="text-slate-400 hover:text-slate-700"
                  aria-label={`移除 ${value}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        ) : null}
        <div className="flex items-center gap-2">
          <input
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === ",") {
                event.preventDefault();
                addItems(inputValue);
              }
            }}
            placeholder={placeholder}
            className="h-10 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
          />
          <ActionButton onClick={() => addItems(inputValue)}>添加</ActionButton>
        </div>
      </div>
    </div>
  );
}

function ModelSelect({
  value,
  onChange,
}: {
  value?: string;
  onChange: (value: string | undefined) => void;
}) {
  const { providers, loading } = useConfiguredProviders();

  useEffect(() => {
    if (!value || loading) {
      return;
    }

    const separatorIndex = value.indexOf("/");
    if (separatorIndex < 0) {
      return;
    }

    const providerKey = value.slice(0, separatorIndex);
    const modelName = value.slice(separatorIndex + 1);
    const provider = providers.find((item) => item.key === providerKey);
    if (!provider) {
      return;
    }

    const issue = getProviderModelCompatibilityIssue({
      providerType: provider.type,
      configuredProviderType: provider.type,
      model: modelName,
    });
    if (issue) {
      onChange(undefined);
    }
  }, [loading, onChange, providers, value]);

  return (
    <div>
      <FieldLabel label="默认模型" optional />
      <select
        value={value || ""}
        onChange={(event) => onChange(event.target.value || undefined)}
        className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
      >
        <option value="">跟随全局默认</option>
        {loading ? <option disabled>加载中...</option> : null}
        {providers.map((provider) => {
          const models = provider.models ?? [];
          const { compatibleModels } = filterProviderModelsByCompatibility(
            {
              providerType: provider.type,
              configuredProviderType: provider.type,
            },
            models,
          );

          if (compatibleModels.length === 0) {
            return null;
          }

          return (
            <optgroup key={provider.key} label={provider.label}>
              {compatibleModels.map((model) => (
                <option
                  key={`${provider.key}/${model}`}
                  value={`${provider.key}/${model}`}
                >
                  {model}
                </option>
              ))}
            </optgroup>
          );
        })}
      </select>
    </div>
  );
}

function InlineMessage({ message }: { message: FlashMessage }) {
  if (!message) {
    return null;
  }

  return (
    <div
      className={cn(
        "rounded-2xl px-4 py-3 text-sm",
        message.type === "success"
          ? "bg-emerald-50 text-emerald-700"
          : "bg-rose-50 text-rose-700",
      )}
    >
      {message.text}
    </div>
  );
}

function ProbePanel({
  description,
  message,
  busy,
  onProbe,
  onOpenAdvanced,
}: {
  description: string;
  message: FlashMessage;
  busy: boolean;
  onProbe: () => void | Promise<void>;
  onOpenAdvanced?: () => void;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="text-sm font-medium text-slate-900">接入检查</div>
          <p className="text-xs leading-5 text-slate-500">{description}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <ActionButton onClick={() => void onProbe()} disabled={busy}>
            {busy ? "检测中..." : "检测连接"}
          </ActionButton>
          {onOpenAdvanced ? (
            <ActionButton onClick={onOpenAdvanced}>查看日志与检查</ActionButton>
          ) : null}
        </div>
      </div>
      {message ? (
        <div className="mt-3">
          <InlineMessage message={message} />
        </div>
      ) : null}
    </div>
  );
}

function QrCodePreview({ value }: { value: string }) {
  const [dataUrl, setDataUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const trimmed = value.trim();

    if (!trimmed) {
      setDataUrl("");
      setError(null);
      return;
    }

    setError(null);
    void QRCode.toDataURL(trimmed, {
      width: 320,
      margin: 1,
      errorCorrectionLevel: "M",
    })
      .then((nextValue) => {
        if (!cancelled) {
          setDataUrl(nextValue);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setDataUrl("");
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [value]);

  if (error) {
    return (
      <div className="flex h-[260px] items-center justify-center rounded-[24px] border border-rose-200 bg-rose-50 px-6 text-center text-sm text-rose-700">
        二维码生成失败：{error}
      </div>
    );
  }

  if (!dataUrl) {
    return (
      <div className="flex h-[260px] items-center justify-center rounded-[24px] border border-slate-200 bg-slate-50 text-sm text-slate-500">
        正在生成二维码...
      </div>
    );
  }

  return (
    <div className="rounded-[24px] border border-slate-200 bg-white p-4">
      <img
        src={dataUrl}
        alt="微信扫码二维码"
        className="mx-auto h-[228px] w-[228px]"
      />
    </div>
  );
}

function ChannelCard({
  icon,
  title,
  subtitle,
  status,
  statusTone,
  details,
  actions,
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  status: string;
  statusTone: "success" | "warning" | "neutral";
  details: Array<{ label: string; value: string }>;
  actions: ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm shadow-slate-950/5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-4">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
            {icon}
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-semibold tracking-tight text-slate-900">
                {title}
              </h2>
              <StatusBadge tone={statusTone}>{status}</StatusBadge>
            </div>
            <p className="text-sm leading-6 text-slate-500">{subtitle}</p>
          </div>
        </div>
      </div>
      <dl className="mt-6 space-y-3">
        {details.map((detail) => (
          <div
            key={`${title}-${detail.label}`}
            className="flex items-center justify-between gap-4 border-b border-slate-100 pb-3 text-sm last:border-b-0 last:pb-0"
          >
            <dt className="text-slate-500">{detail.label}</dt>
            <dd className="text-right font-medium text-slate-800">
              {detail.value}
            </dd>
          </div>
        ))}
      </dl>
      <div className="mt-6 flex flex-wrap gap-3">{actions}</div>
    </section>
  );
}

interface TelegramConfigDialogProps {
  isOpen: boolean;
  config: TelegramBotConfig;
  onClose: () => void;
  onSave: (config: TelegramBotConfig, successText: string) => Promise<void>;
  onOpenAdvanced?: () => void;
}

function TelegramConfigDialog({
  isOpen,
  config,
  onClose,
  onSave,
  onOpenAdvanced,
}: TelegramConfigDialogProps) {
  const [draft, setDraft] = useState(config);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<FlashMessage>(null);

  useEffect(() => {
    if (isOpen) {
      setDraft(config);
      setBusy(false);
      setMessage(null);
    }
  }, [config, isOpen]);

  const handleProbe = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await telegramChannelProbe();
      setMessage({
        type: result.ok ? "success" : "error",
        text:
          result.message ||
          (result.ok ? "Telegram 连接正常" : "Telegram 连接失败"),
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    if (draft.enabled && !draft.bot_token.trim()) {
      setMessage({
        type: "error",
        text: "启用 Telegram 前请先填写机器人 Token",
      });
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      await onSave(
        {
          ...draft,
          bot_token: draft.bot_token.trim(),
          allowed_user_ids: draft.allowed_user_ids
            .map((item) => item.trim())
            .filter(Boolean),
        },
        "Telegram 配置已保存",
      );
      onClose();
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="max-w-2xl"
      className="lime-workbench-theme-scope border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text)]"
    >
      <ModalHeader>Telegram 配置</ModalHeader>
      <ModalBody className="space-y-5">
        <SwitchField
          checked={draft.enabled}
          onChange={(checked) =>
            setDraft((current) => ({ ...current, enabled: checked }))
          }
          title="启用 Telegram"
          description="机器人 Token、用户范围和接入检查都在这里。"
        />
        <SecretField
          label="机器人 Token"
          value={draft.bot_token}
          onChange={(value) =>
            setDraft((current) => ({ ...current, bot_token: value }))
          }
          placeholder="123456:ABC-DEF..."
        />
        <StringListEditor
          label="允许用户"
          values={draft.allowed_user_ids}
          onChange={(values) =>
            setDraft((current) => ({ ...current, allowed_user_ids: values }))
          }
          placeholder="输入 Telegram 用户 ID"
        />
        <ModelSelect
          value={draft.default_model}
          onChange={(value) =>
            setDraft((current) => ({ ...current, default_model: value }))
          }
        />
        <ProbePanel
          description="填好 Token 后直接检测，不通再看日志。"
          message={message}
          busy={busy}
          onProbe={handleProbe}
          onOpenAdvanced={onOpenAdvanced}
        />
      </ModalBody>
      <ModalFooter>
        <ActionButton onClick={onClose}>取消</ActionButton>
        <ActionButton
          kind="primary"
          onClick={() => void handleSave()}
          disabled={busy}
        >
          保存
        </ActionButton>
      </ModalFooter>
    </Modal>
  );
}

interface FeishuConfigDialogProps {
  isOpen: boolean;
  config: FeishuBotConfig;
  onClose: () => void;
  onSave: (config: FeishuBotConfig, successText: string) => Promise<void>;
  onOpenAdvanced?: () => void;
}

function FeishuConfigDialog({
  isOpen,
  config,
  onClose,
  onSave,
  onOpenAdvanced,
}: FeishuConfigDialogProps) {
  const [draft, setDraft] = useState(config);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<FlashMessage>(null);
  const [credentialsOpen, setCredentialsOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const connectionMode = normalizeConnectionMode(draft.connection_mode);
  const streamingMode = normalizeStreamingMode(draft.streaming);
  const replyToMode = normalizeReplyToMode(draft.reply_to_mode);
  const dmPolicy = draft.dm_policy || "open";
  const hasCredentials =
    draft.app_id.trim().length > 0 && draft.app_secret.trim().length > 0;
  const allowListLabel = dmPolicy === "allowlist" ? "白名单用户" : "已允许用户";
  const showAllowListEditor =
    dmPolicy === "pairing" || dmPolicy === "allowlist";
  const statusTone = hasCredentials
    ? draft.enabled
      ? "success"
      : "neutral"
    : "warning";
  const statusText = hasCredentials
    ? draft.enabled
      ? "已启用"
      : "已配置"
    : "待补充凭证";
  const policyOptions = [
    {
      value: "pairing",
      title: "配对模式",
      description: "仅已配对会话可用。",
    },
    {
      value: "open",
      title: "开放模式",
      description: "所有人可直接发起对话。",
    },
    {
      value: "allowlist",
      title: "白名单",
      description: "仅名单内用户可发起。",
    },
    {
      value: "disabled",
      title: "关闭私聊",
      description: "保留配置，不接收私聊。",
    },
  ] as const;

  useEffect(() => {
    if (isOpen) {
      setDraft(config);
      setBusy(false);
      setMessage(null);
      setCredentialsOpen(
        !(
          config.app_id?.trim().length > 0 &&
          config.app_secret?.trim().length > 0
        ),
      );
      setAdvancedOpen(false);
    }
  }, [config, isOpen]);

  const handleProbe = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await feishuChannelProbe({
        accountId: draft.default_account?.trim() || undefined,
      });
      setMessage({
        type: result.ok ? "success" : "error",
        text: result.message || (result.ok ? "飞书连接正常" : "飞书连接失败"),
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    if (draft.enabled && (!draft.app_id.trim() || !draft.app_secret.trim())) {
      setMessage({
        type: "error",
        text: "启用飞书前请先填写应用 ID 和应用密钥",
      });
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      const nextAllowFrom =
        draft.dm_policy === "open" && (draft.allow_from ?? []).length === 0
          ? ["*"]
          : (draft.allow_from ?? []);
      await onSave(
        {
          ...draft,
          app_id: draft.app_id.trim(),
          app_secret: draft.app_secret.trim(),
          verification_token: draft.verification_token?.trim() || undefined,
          encrypt_key: draft.encrypt_key?.trim() || undefined,
          connection_mode: connectionMode,
          webhook_host: draft.webhook_host?.trim() || undefined,
          webhook_port:
            typeof draft.webhook_port === "number" && draft.webhook_port > 0
              ? draft.webhook_port
              : undefined,
          webhook_path: draft.webhook_path?.trim() || undefined,
          allow_from: nextAllowFrom,
        },
        "飞书配置已保存",
      );
      onClose();
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="max-w-2xl"
      className="lime-workbench-theme-scope border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text)]"
    >
      <ModalHeader>配置飞书</ModalHeader>
      <ModalBody className="space-y-4">
        <div
          className={cn(
            "rounded-2xl border px-4 py-4",
            hasCredentials
              ? "border-emerald-200 bg-emerald-50"
              : "border-amber-200 bg-amber-50",
          )}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-sm font-medium text-slate-900">
                  配置状态
                </div>
                <StatusBadge tone={statusTone}>{statusText}</StatusBadge>
              </div>
              <p className="text-sm leading-6 text-slate-600">
                {hasCredentials
                  ? "凭证已就绪，可直接检查接入。"
                  : "先填应用 ID 和应用密钥。"}
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={draft.enabled}
              onClick={() =>
                setDraft((current) => ({
                  ...current,
                  enabled: !current.enabled,
                }))
              }
              className={cn(
                "relative mt-1 inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
                draft.enabled ? "bg-emerald-500" : "bg-slate-300",
              )}
            >
              <span
                className={cn(
                  "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                  draft.enabled ? "translate-x-6" : "translate-x-1",
                )}
              />
            </button>
          </div>
        </div>

        <Collapsible open={credentialsOpen} onOpenChange={setCredentialsOpen}>
          <div className="rounded-2xl border border-slate-200 bg-white">
            <div className="space-y-3 px-4 py-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="text-sm font-medium text-slate-900">
                    当前机器人
                  </div>
                  <p className="text-xs text-slate-500">默认只看摘要。</p>
                </div>
                <button
                  type="button"
                  onClick={() => setCredentialsOpen((current) => !current)}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                  {credentialsOpen
                    ? "收起凭证"
                    : hasCredentials
                      ? "编辑凭证"
                      : "填写凭证"}
                  {credentialsOpen ? (
                    <ChevronDown className="h-4 w-4 text-slate-500" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-slate-500" />
                  )}
                </button>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                <div className="grid gap-3 md:grid-cols-3">
                  <div>
                    <div className="text-xs text-slate-500">应用 ID</div>
                    <div className="mt-1 text-sm font-medium text-slate-900">
                      {maskMiddle(draft.app_id, 5)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">应用密钥</div>
                    <div className="mt-1 text-sm font-medium text-slate-900">
                      {maskMiddle(draft.app_secret, 6)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">默认模型</div>
                    <div className="mt-1 text-sm font-medium text-slate-900">
                      {draft.default_model || "跟随全局默认"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <CollapsibleContent className="space-y-4 border-t border-slate-100 px-4 py-4">
              <div className="grid gap-4 md:grid-cols-2">
                <TextField
                  label="应用 ID"
                  value={draft.app_id}
                  onChange={(value) =>
                    setDraft((current) => ({ ...current, app_id: value }))
                  }
                  placeholder="cli_xxxx"
                  mono
                />
                <SecretField
                  label="应用密钥"
                  value={draft.app_secret}
                  onChange={(value) =>
                    setDraft((current) => ({ ...current, app_secret: value }))
                  }
                  placeholder="飞书应用密钥"
                />
              </div>
              <ModelSelect
                value={draft.default_model}
                onChange={(value) =>
                  setDraft((current) => ({ ...current, default_model: value }))
                }
              />
            </CollapsibleContent>
          </div>
        </Collapsible>

        <div className="space-y-3 rounded-2xl border border-slate-200 bg-white px-4 py-4">
          <div className="space-y-1">
            <div className="text-sm font-medium text-slate-900">准入策略</div>
            <p className="text-xs text-slate-500">选择私聊如何进入。</p>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {policyOptions.map((option) => {
              const selected = dmPolicy === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() =>
                    setDraft((current) => ({
                      ...current,
                      dm_policy: option.value,
                      allow_from:
                        option.value === "open"
                          ? current.allow_from
                          : current.allow_from?.length === 1 &&
                              current.allow_from[0] === "*"
                            ? []
                            : current.allow_from,
                    }))
                  }
                  className={cn(
                    "rounded-2xl border px-4 py-4 text-left transition",
                    selected
                      ? "border-emerald-200 bg-[image:var(--lime-home-card-surface-strong)] text-slate-800 shadow-sm shadow-emerald-950/10"
                      : "border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300 hover:bg-white",
                  )}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-medium">{option.title}</div>
                    <span
                      className={cn(
                        "inline-flex h-5 w-5 items-center justify-center rounded-full border text-[11px]",
                        selected
                          ? "border-emerald-200 bg-white/90 text-emerald-700"
                          : "border-slate-300 text-slate-400",
                      )}
                    >
                      {selected ? "✓" : ""}
                    </span>
                  </div>
                  <p
                    className={cn(
                      "mt-2 text-xs leading-5",
                      selected ? "text-slate-600" : "text-slate-500",
                    )}
                  >
                    {option.description}
                  </p>
                </button>
              );
            })}
          </div>
          {showAllowListEditor ? (
            <StringListEditor
              label={allowListLabel}
              values={(draft.allow_from ?? []).filter(
                (item) => item.trim() !== "*",
              )}
              onChange={(values) =>
                setDraft((current) => ({ ...current, allow_from: values }))
              }
              placeholder="输入 open_id / user_id"
            />
          ) : (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-500">
              {dmPolicy === "open"
                ? "开放模式下不需要名单。"
                : "关闭后不再接收飞书私聊。"}
            </div>
          )}
        </div>

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <div className="rounded-2xl border border-slate-200 bg-white">
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex w-full items-center justify-between px-4 py-3 text-left"
              >
                <div>
                  <div className="text-sm font-medium text-slate-900">
                    高级配置
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    回调与兼容参数。
                  </p>
                </div>
                {advancedOpen ? (
                  <ChevronDown className="h-4 w-4 text-slate-500" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-slate-500" />
                )}
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 border-t border-slate-100 px-4 py-4">
              <div className="grid gap-4 md:grid-cols-2">
                <TextField
                  label="验证 Token"
                  value={draft.verification_token || ""}
                  onChange={(value) =>
                    setDraft((current) => ({
                      ...current,
                      verification_token: value || undefined,
                    }))
                  }
                  placeholder="事件订阅验证 Token"
                />
                <SecretField
                  label="加密密钥"
                  value={draft.encrypt_key || ""}
                  onChange={(value) =>
                    setDraft((current) => ({
                      ...current,
                      encrypt_key: value || undefined,
                    }))
                  }
                  placeholder="事件加密密钥"
                />
              </div>
              <div>
                <FieldLabel label="回调方式" />
                <select
                  value={connectionMode}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      connection_mode: event.target.value,
                    }))
                  }
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                >
                  <option value="webhook">Webhook 回调</option>
                  <option value="websocket">WebSocket</option>
                </select>
                {connectionMode === "websocket" ? (
                  <p className="mt-2 text-xs text-amber-600">
                    当前仍建议优先用 Webhook 回调。
                  </p>
                ) : null}
              </div>
              {connectionMode === "webhook" ? (
                <div className="grid gap-4 md:grid-cols-3">
                  <TextField
                    label="回调主机"
                    value={draft.webhook_host || ""}
                    onChange={(value) =>
                      setDraft((current) => ({
                        ...current,
                        webhook_host: value || undefined,
                      }))
                    }
                    placeholder="127.0.0.1"
                  />
                  <TextField
                    label="回调端口"
                    value={draft.webhook_port ? String(draft.webhook_port) : ""}
                    onChange={(value) =>
                      setDraft((current) => ({
                        ...current,
                        webhook_port:
                          Number.isFinite(Number(value)) && Number(value) > 0
                            ? Number(value)
                            : undefined,
                      }))
                    }
                    placeholder="3000"
                    type="number"
                  />
                  <TextField
                    label="回调路径"
                    value={draft.webhook_path || ""}
                    onChange={(value) =>
                      setDraft((current) => ({
                        ...current,
                        webhook_path: value || undefined,
                      }))
                    }
                    placeholder="/feishu/default"
                    mono
                  />
                </div>
              ) : null}
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <FieldLabel label="流式返回" />
                  <select
                    value={streamingMode}
                    onChange={(event) =>
                      setDraft((current) => ({
                        ...current,
                        streaming: event.target.value,
                      }))
                    }
                    className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                  >
                    <option value="partial">增量返回</option>
                    <option value="block">阻塞返回</option>
                    <option value="off">关闭</option>
                  </select>
                </div>
                <div>
                  <FieldLabel label="引用回复" />
                  <select
                    value={replyToMode}
                    onChange={(event) =>
                      setDraft((current) => ({
                        ...current,
                        reply_to_mode: event.target.value,
                      }))
                    }
                    className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                  >
                    <option value="off">关闭</option>
                    <option value="first">首条</option>
                    <option value="all">全部</option>
                  </select>
                </div>
              </div>
            </CollapsibleContent>
          </div>
        </Collapsible>
        <ProbePanel
          description="配置后直接检测，不通再看排障。"
          message={message}
          busy={busy}
          onProbe={handleProbe}
          onOpenAdvanced={onOpenAdvanced}
        />
      </ModalBody>
      <ModalFooter>
        <ActionButton onClick={onClose}>取消</ActionButton>
        <ActionButton
          kind="primary"
          onClick={() => void handleSave()}
          disabled={busy}
        >
          保存
        </ActionButton>
      </ModalFooter>
    </Modal>
  );
}

interface WechatConfigDialogProps {
  isOpen: boolean;
  config: WechatBotConfig;
  onClose: () => void;
  onSave: (config: WechatBotConfig, successText: string) => Promise<void>;
  onOpenAdvanced?: () => void;
}

function WechatConfigDialog({
  isOpen,
  config,
  onClose,
  onSave,
  onOpenAdvanced,
}: WechatConfigDialogProps) {
  const [draft, setDraft] = useState(config);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<FlashMessage>(null);
  const [qrValue, setQrValue] = useState("");
  const [runtimeAccounts, setRuntimeAccounts] = useState<
    WechatConfiguredAccount[]
  >([]);
  const [loginResult, setLoginResult] = useState<WechatLoginWaitResult | null>(
    null,
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [accountsOpen, setAccountsOpen] = useState(false);
  const [accountEditorOpen, setAccountEditorOpen] = useState(false);
  const [activeAccountId, setActiveAccountId] = useState("");
  const scanRequestIdRef = useRef(0);
  const accountIds = useMemo(
    () => Object.keys(draft.accounts ?? {}),
    [draft.accounts],
  );
  const activeAccount = activeAccountId
    ? draft.accounts?.[activeAccountId]
    : undefined;
  const groupPolicy = normalizeWechatGroupPolicy(draft.group_policy);
  const streamingMode = normalizeStreamingMode(draft.streaming);
  const replyToMode = normalizeReplyToMode(draft.reply_to_mode);

  useEffect(() => {
    if (!isOpen) {
      scanRequestIdRef.current += 1;
      return;
    }

    setDraft(config);
    setBusy(false);
    setMessage(null);
    setQrValue("");
    setLoginResult(null);
    setAdvancedOpen(false);
    setAccountsOpen(false);
    setAccountEditorOpen(false);
    setActiveAccountId(
      config.default_account?.trim() ||
        Object.keys(config.accounts ?? {})[0] ||
        "",
    );

    const requestId = scanRequestIdRef.current + 1;
    scanRequestIdRef.current = requestId;

    void wechatChannelListAccounts()
      .then((accounts) => {
        if (scanRequestIdRef.current === requestId) {
          setRuntimeAccounts(accounts);
        }
      })
      .catch(() => {
        if (scanRequestIdRef.current === requestId) {
          setRuntimeAccounts([]);
        }
      });

    return () => {
      scanRequestIdRef.current += 1;
    };
  }, [config, isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const preferredAccountId =
      draft.default_account?.trim() || accountIds[0] || "";
    if (!preferredAccountId) {
      if (activeAccountId) {
        setActiveAccountId("");
      }
      if (accountEditorOpen) {
        setAccountEditorOpen(false);
      }
      return;
    }

    if (!activeAccountId || !accountIds.includes(activeAccountId)) {
      setActiveAccountId(preferredAccountId);
    }
  }, [
    accountEditorOpen,
    accountIds,
    activeAccountId,
    draft.default_account,
    isOpen,
  ]);

  const patchAccount = useCallback(
    (
      accountId: string,
      updater: (
        current: NonNullable<WechatBotConfig["accounts"]>[string],
      ) => NonNullable<WechatBotConfig["accounts"]>[string],
    ) => {
      setDraft((current) => {
        const baseAccounts = current.accounts ?? {};
        const currentAccountDraft = baseAccounts[accountId] ?? {
          enabled: true,
        };
        return {
          ...current,
          accounts: {
            ...baseAccounts,
            [accountId]: updater(currentAccountDraft),
          },
        };
      });
    },
    [],
  );

  const addAccountDraft = () => {
    const baseAccounts = draft.accounts ?? {};
    let index = 1;
    let nextAccountId = "wechat_1";
    while (baseAccounts[nextAccountId]) {
      index += 1;
      nextAccountId = `wechat_${index}`;
    }

    setDraft((current) => ({
      ...current,
      default_account: current.default_account || nextAccountId,
      accounts: {
        ...(current.accounts ?? {}),
        [nextAccountId]: {
          enabled: true,
          name: `微信账号 ${index}`,
        },
      },
    }));
    setAccountsOpen(true);
    setAccountEditorOpen(true);
    setActiveAccountId(nextAccountId);
  };

  const removeAccountDraft = (accountId: string) => {
    const baseAccounts = { ...(draft.accounts ?? {}) };
    delete baseAccounts[accountId];
    const nextAccountIds = Object.keys(baseAccounts);
    const nextDefaultAccount =
      draft.default_account === accountId
        ? nextAccountIds[0] || undefined
        : draft.default_account;

    setDraft((current) => ({
      ...current,
      default_account: nextDefaultAccount,
      accounts: baseAccounts,
    }));
    if (nextAccountIds.length === 0) {
      setAccountEditorOpen(false);
    }
    setActiveAccountId(nextAccountIds[0] || "");
  };

  const handleOpenAccountEditor = (accountId: string) => {
    setActiveAccountId(accountId);
    setAccountEditorOpen(true);
  };

  const handleStartScan = async (baseUrlOverride?: string) => {
    const requestId = scanRequestIdRef.current + 1;
    scanRequestIdRef.current = requestId;
    const baseUrl =
      (baseUrlOverride ?? draft.base_url).trim() || DEFAULT_WECHAT_BASE_URL;

    setBusy(true);
    setMessage(null);
    setQrValue("");
    setLoginResult(null);

    try {
      const result = await wechatChannelLoginStart({
        baseUrl,
        botType: DEFAULT_WECHAT_BOT_TYPE,
      });
      if (scanRequestIdRef.current !== requestId) {
        return;
      }
      setQrValue(result.qrcodeUrl);
      setMessage({
        type: "success",
        text: "二维码已生成，扫码后会自动等待登录结果",
      });

      const waitResult = await wechatChannelLoginWait({
        sessionKey: result.sessionKey,
        baseUrl,
        botType: DEFAULT_WECHAT_BOT_TYPE,
        timeoutMs: 60_000,
      });
      if (scanRequestIdRef.current !== requestId) {
        return;
      }

      setLoginResult(waitResult);
      if (waitResult.connected) {
        const nextAccountId = waitResult.accountId?.trim() || "default";
        setDraft((current) => ({
          ...current,
          enabled: true,
          bot_token: waitResult.botToken?.trim() || current.bot_token,
          base_url:
            waitResult.baseUrl?.trim() ||
            current.base_url ||
            DEFAULT_WECHAT_BASE_URL,
          cdn_base_url: current.cdn_base_url || DEFAULT_WECHAT_CDN_BASE_URL,
          scanner_user_id: waitResult.userId?.trim() || current.scanner_user_id,
          default_account: nextAccountId,
          accounts: {
            ...(current.accounts ?? {}),
            [nextAccountId]: {
              ...(current.accounts?.[nextAccountId] ?? {}),
              enabled: true,
              bot_token:
                waitResult.botToken?.trim() ||
                current.accounts?.[nextAccountId]?.bot_token,
              scanner_user_id:
                waitResult.userId?.trim() ||
                current.accounts?.[nextAccountId]?.scanner_user_id,
              base_url:
                waitResult.baseUrl?.trim() ||
                current.accounts?.[nextAccountId]?.base_url ||
                current.base_url ||
                DEFAULT_WECHAT_BASE_URL,
              cdn_base_url:
                current.accounts?.[nextAccountId]?.cdn_base_url ||
                current.cdn_base_url ||
                DEFAULT_WECHAT_CDN_BASE_URL,
            },
          },
        }));
        setMessage({
          type: "success",
          text: "微信已扫码成功，可以直接保存并启用",
        });
        const accounts = await wechatChannelListAccounts();
        if (scanRequestIdRef.current === requestId) {
          setRuntimeAccounts(accounts);
        }
      } else {
        setMessage({
          type: "error",
          text: waitResult.message || "未检测到扫码成功，请重新生成二维码",
        });
      }
    } catch (error) {
      if (scanRequestIdRef.current !== requestId) {
        return;
      }
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (scanRequestIdRef.current === requestId) {
        setBusy(false);
      }
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void handleStartScan(config.base_url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.base_url, isOpen]);

  const handleProbe = async () => {
    const accountId =
      draft.default_account?.trim() ||
      loginResult?.accountId?.trim() ||
      runtimeAccounts[0]?.accountId;
    if (!accountId) {
      setMessage({ type: "error", text: "请先扫码或选择默认账号" });
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      const result = await wechatChannelProbe({ accountId });
      setMessage({
        type: result.ok ? "success" : "error",
        text: result.message || (result.ok ? "微信连接正常" : "微信连接失败"),
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    const nextAccounts = Object.entries(draft.accounts ?? {}).reduce<
      NonNullable<WechatBotConfig["accounts"]>
    >((result, [accountId, account]) => {
      const normalizedAccountId = accountId.trim();
      if (!normalizedAccountId) {
        return result;
      }

      result[normalizedAccountId] = {
        ...account,
        enabled: account.enabled !== false,
        name: account.name?.trim() || undefined,
        bot_token: account.bot_token?.trim() || undefined,
        scanner_user_id: account.scanner_user_id?.trim() || undefined,
        base_url: account.base_url?.trim() || undefined,
        cdn_base_url: account.cdn_base_url?.trim() || undefined,
      };
      return result;
    }, {});
    const nextConfig: WechatBotConfig = {
      ...draft,
      enabled: draft.enabled || Object.keys(nextAccounts).length > 0,
      bot_token: draft.bot_token.trim(),
      base_url: draft.base_url.trim() || DEFAULT_WECHAT_BASE_URL,
      cdn_base_url: draft.cdn_base_url.trim() || DEFAULT_WECHAT_CDN_BASE_URL,
      scanner_user_id: draft.scanner_user_id?.trim() || undefined,
      default_account: draft.default_account?.trim() || undefined,
      group_allow_from: (draft.group_allow_from ?? [])
        .map((item) => item.trim())
        .filter(Boolean),
      allow_from:
        draft.dm_policy === "open" && (draft.allow_from ?? []).length === 0
          ? ["*"]
          : (draft.allow_from ?? []).map((item) => item.trim()).filter(Boolean),
      accounts: nextAccounts,
    };

    setBusy(true);
    setMessage(null);
    try {
      await onSave(nextConfig, "微信配置已保存");
      const accountId =
        nextConfig.default_account?.trim() ||
        loginResult?.accountId?.trim() ||
        runtimeAccounts[0]?.accountId;
      if (accountId) {
        await gatewayChannelStart({ channel: "wechat", accountId });
      }
      onClose();
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="max-w-4xl"
      className="lime-workbench-theme-scope border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] text-[color:var(--lime-text)]"
    >
      <ModalHeader>微信扫码配置</ModalHeader>
      <ModalBody className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="space-y-3">
            <QrCodePreview value={qrValue} />
            <div className="text-center text-xs leading-5 text-slate-500">
              微信扫码后会自动接入。
            </div>
            <ActionButton
              onClick={() => void handleStartScan()}
              disabled={busy}
            >
              重新生成二维码
            </ActionButton>
          </div>

          <div className="space-y-5">
            <SwitchField
              checked={draft.enabled}
              onChange={(checked) =>
                setDraft((current) => ({ ...current, enabled: checked }))
              }
              title="启用微信"
              description="扫码后自动回填，保存时尝试启动。"
            />
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-slate-900">
                  当前账号
                </span>
                <StatusBadge
                  tone={draft.default_account?.trim() ? "success" : "warning"}
                >
                  {draft.default_account?.trim() || "等待扫码"}
                </StatusBadge>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                扫码后自动绑定默认账号。
              </p>
            </div>
            <ProbePanel
              description="扫码后直接检测，不通再看排障。"
              message={message}
              busy={busy}
              onProbe={handleProbe}
              onOpenAdvanced={onOpenAdvanced}
            />
            <div>
              <FieldLabel label="私聊接入策略" />
              <select
                value={draft.dm_policy || "pairing"}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    dm_policy: event.target.value,
                  }))
                }
                className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
              >
                <option value="pairing">配对模式</option>
                <option value="allowlist">白名单</option>
                <option value="open">开放模式</option>
                <option value="disabled">关闭私聊</option>
              </select>
            </div>
            {runtimeAccounts.length > 0 ? (
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                <div className="text-sm font-medium text-slate-900">
                  已接入账号
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {runtimeAccounts.map((account) => (
                    <button
                      key={account.accountId}
                      type="button"
                      onClick={() =>
                        setDraft((current) => ({
                          ...current,
                          default_account: account.accountId,
                        }))
                      }
                      className={cn(
                        "inline-flex items-center rounded-full bg-white px-3 py-1 text-xs ring-1 transition",
                        draft.default_account === account.accountId
                          ? "text-slate-900 ring-slate-400"
                          : "text-slate-600 ring-slate-200 hover:ring-slate-300",
                      )}
                    >
                      {account.name || account.accountId}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <div className="rounded-2xl border border-slate-200 bg-white">
                <CollapsibleTrigger asChild>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between px-4 py-3 text-left"
                  >
                    <div>
                      <div className="text-sm font-medium text-slate-900">
                        高级参数
                      </div>
                      <p className="mt-1 text-xs text-slate-500">
                        群组、账号与兼容参数。
                      </p>
                    </div>
                    {advancedOpen ? (
                      <ChevronDown className="h-4 w-4 text-slate-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-slate-500" />
                    )}
                  </button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 border-t border-slate-100 px-4 py-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <TextField
                      label="扫码用户 ID"
                      value={draft.scanner_user_id || ""}
                      onChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          scanner_user_id: value || undefined,
                        }))
                      }
                      placeholder="扫码绑定后的 user_id"
                      mono
                    />
                    <TextField
                      label="默认账号 ID"
                      value={draft.default_account || ""}
                      onChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          default_account: value,
                        }))
                      }
                      placeholder="默认账号 ID"
                    />
                    <ModelSelect
                      value={draft.default_model}
                      onChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          default_model: value,
                        }))
                      }
                    />
                    <div>
                      <FieldLabel label="群组策略" />
                      <select
                        value={groupPolicy}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            group_policy: event.target.value,
                          }))
                        }
                        className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                      >
                        <option value="allowlist">白名单</option>
                        <option value="open">开放模式</option>
                        <option value="disabled">已关闭</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <TextField
                      label="服务地址"
                      value={draft.base_url}
                      onChange={(value) =>
                        setDraft((current) => ({ ...current, base_url: value }))
                      }
                      placeholder={DEFAULT_WECHAT_BASE_URL}
                      mono
                    />
                    <TextField
                      label="资源地址"
                      value={draft.cdn_base_url}
                      onChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          cdn_base_url: value,
                        }))
                      }
                      placeholder={DEFAULT_WECHAT_CDN_BASE_URL}
                      mono
                    />
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <FieldLabel label="流式返回" />
                      <select
                        value={streamingMode}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            streaming: event.target.value,
                          }))
                        }
                        className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                      >
                        <option value="off">关闭</option>
                        <option value="partial">增量返回</option>
                        <option value="block">阻塞返回</option>
                      </select>
                    </div>
                    <div>
                      <FieldLabel label="引用回复" />
                      <select
                        value={replyToMode}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            reply_to_mode: event.target.value,
                          }))
                        }
                        className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                      >
                        <option value="off">关闭</option>
                        <option value="first">首条</option>
                        <option value="all">全部</option>
                      </select>
                    </div>
                  </div>
                  <StringListEditor
                    label="已允许用户"
                    values={draft.allow_from ?? []}
                    onChange={(values) =>
                      setDraft((current) => ({
                        ...current,
                        allow_from: values,
                      }))
                    }
                    placeholder="输入用户 ID 或 *"
                  />
                  <StringListEditor
                    label="群组允许用户"
                    values={draft.group_allow_from ?? []}
                    onChange={(values) =>
                      setDraft((current) => ({
                        ...current,
                        group_allow_from: values,
                      }))
                    }
                    placeholder="输入用户 ID 或 *"
                  />
                  <Collapsible
                    open={accountsOpen}
                    onOpenChange={setAccountsOpen}
                  >
                    <div className="rounded-2xl border border-slate-200 bg-slate-50">
                      <div className="flex flex-wrap items-start justify-between gap-3 px-4 py-4">
                        <div className="space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="text-sm font-medium text-slate-900">
                              账号参数
                            </div>
                            <StatusBadge
                              tone={
                                accountIds.length > 0 ? "success" : "neutral"
                              }
                            >
                              {accountIds.length > 0
                                ? `${accountIds.length} 个账号`
                                : "暂无账号"}
                            </StatusBadge>
                            {draft.default_account?.trim() ? (
                              <StatusBadge tone="neutral">
                                {`默认 ${draft.default_account.trim()}`}
                              </StatusBadge>
                            ) : null}
                          </div>
                          <p className="text-xs text-slate-500">
                            扫码账号会自动进来，按需展开编辑。
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <ActionButton onClick={addAccountDraft}>
                            新增账号草稿
                          </ActionButton>
                          <button
                            type="button"
                            onClick={() =>
                              setAccountsOpen((current) => !current)
                            }
                            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                          >
                            {accountsOpen ? "收起账号" : "展开账号"}
                            {accountsOpen ? (
                              <ChevronDown className="h-4 w-4 text-slate-500" />
                            ) : (
                              <ChevronRight className="h-4 w-4 text-slate-500" />
                            )}
                          </button>
                        </div>
                      </div>
                      <CollapsibleContent className="space-y-4 border-t border-slate-100 px-4 py-4">
                        {accountIds.length > 0 ? (
                          <>
                            <div className="grid gap-3">
                              {accountIds.map((accountId) => {
                                const account = draft.accounts?.[accountId];
                                if (!account) {
                                  return null;
                                }

                                const selected = activeAccountId === accountId;
                                return (
                                  <button
                                    key={accountId}
                                    type="button"
                                    onClick={() =>
                                      handleOpenAccountEditor(accountId)
                                    }
                                    data-testid={`wechat-account-summary-${accountId}`}
                                    className={cn(
                                      "rounded-2xl border bg-white px-4 py-3 text-left transition",
                                      selected
                                        ? "border-slate-300 shadow-sm shadow-slate-950/5"
                                        : "border-slate-200 hover:border-slate-300",
                                    )}
                                  >
                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                      <div className="space-y-1">
                                        <div className="text-sm font-medium text-slate-900">
                                          {account.name?.trim() || accountId}
                                        </div>
                                        <div className="text-xs text-slate-500">
                                          {account.name?.trim()
                                            ? accountId
                                            : "未命名账号"}
                                        </div>
                                      </div>
                                      <div className="flex flex-wrap items-center gap-2">
                                        {draft.default_account === accountId ? (
                                          <StatusBadge tone="success">
                                            默认账号
                                          </StatusBadge>
                                        ) : null}
                                        <StatusBadge
                                          tone={
                                            account.enabled !== false
                                              ? "success"
                                              : "warning"
                                          }
                                        >
                                          {account.enabled !== false
                                            ? "已启用"
                                            : "已停用"}
                                        </StatusBadge>
                                      </div>
                                    </div>
                                    <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500">
                                      <span>
                                        模型：
                                        {describeWechatAccountModel(account)}
                                      </span>
                                      <span>
                                        地址：
                                        {describeWechatAccountRoute(
                                          account,
                                          draft.base_url ||
                                            DEFAULT_WECHAT_BASE_URL,
                                        )}
                                      </span>
                                    </div>
                                  </button>
                                );
                              })}
                            </div>
                            {activeAccount ? (
                              <Collapsible
                                open={accountEditorOpen}
                                onOpenChange={setAccountEditorOpen}
                              >
                                <div className="rounded-2xl border border-slate-200 bg-white">
                                  <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4">
                                    <div className="space-y-1">
                                      <div className="flex flex-wrap items-center gap-2">
                                        <div className="text-sm font-medium text-slate-900">
                                          编辑账号详情
                                        </div>
                                        <StatusBadge
                                          tone={
                                            draft.default_account ===
                                            activeAccountId
                                              ? "success"
                                              : "neutral"
                                          }
                                        >
                                          {activeAccount.name?.trim() ||
                                            activeAccountId}
                                        </StatusBadge>
                                      </div>
                                      <p className="text-xs text-slate-500">
                                        按需补模型、地址或扫码用户 ID。
                                      </p>
                                    </div>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setAccountEditorOpen(
                                          (current) => !current,
                                        )
                                      }
                                      className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                                    >
                                      {accountEditorOpen
                                        ? "收起详情"
                                        : "展开详情"}
                                      {accountEditorOpen ? (
                                        <ChevronDown className="h-4 w-4 text-slate-500" />
                                      ) : (
                                        <ChevronRight className="h-4 w-4 text-slate-500" />
                                      )}
                                    </button>
                                  </div>
                                  <CollapsibleContent className="space-y-4 border-t border-slate-100 px-4 py-4">
                                    <div className="flex flex-wrap items-center justify-between gap-3">
                                      <div className="flex flex-wrap items-center gap-3">
                                        <div className="text-sm font-medium text-slate-900">
                                          当前账号
                                        </div>
                                        <StatusBadge
                                          tone={
                                            draft.default_account ===
                                            activeAccountId
                                              ? "success"
                                              : "neutral"
                                          }
                                        >
                                          {draft.default_account ===
                                          activeAccountId
                                            ? "默认账号"
                                            : "账号草稿"}
                                        </StatusBadge>
                                      </div>
                                      <div className="flex flex-wrap gap-2">
                                        <ActionButton
                                          onClick={() =>
                                            setDraft((current) => ({
                                              ...current,
                                              default_account: activeAccountId,
                                            }))
                                          }
                                        >
                                          设为默认
                                        </ActionButton>
                                        <ActionButton
                                          onClick={() =>
                                            removeAccountDraft(activeAccountId)
                                          }
                                        >
                                          删除
                                        </ActionButton>
                                      </div>
                                    </div>
                                    <SwitchField
                                      checked={activeAccount.enabled !== false}
                                      onChange={(checked) =>
                                        patchAccount(
                                          activeAccountId,
                                          (current) => ({
                                            ...current,
                                            enabled: checked,
                                          }),
                                        )
                                      }
                                      title="启用该账号"
                                      description="关闭后保留参数，但不参与运行。"
                                    />
                                    <div className="grid gap-4 md:grid-cols-2">
                                      <TextField
                                        label="账号名称"
                                        value={activeAccount.name || ""}
                                        onChange={(value) =>
                                          patchAccount(
                                            activeAccountId,
                                            (current) => ({
                                              ...current,
                                              name: value || undefined,
                                            }),
                                          )
                                        }
                                        placeholder="主微信 / 运营号"
                                      />
                                      <ModelSelect
                                        value={activeAccount.default_model}
                                        onChange={(value) =>
                                          patchAccount(
                                            activeAccountId,
                                            (current) => ({
                                              ...current,
                                              default_model: value,
                                            }),
                                          )
                                        }
                                      />
                                    </div>
                                    <div className="grid gap-4 md:grid-cols-2">
                                      <TextField
                                        label="账号服务地址"
                                        value={activeAccount.base_url || ""}
                                        onChange={(value) =>
                                          patchAccount(
                                            activeAccountId,
                                            (current) => ({
                                              ...current,
                                              base_url: value || undefined,
                                            }),
                                          )
                                        }
                                        placeholder={DEFAULT_WECHAT_BASE_URL}
                                        mono
                                      />
                                      <TextField
                                        label="账号资源地址"
                                        value={activeAccount.cdn_base_url || ""}
                                        onChange={(value) =>
                                          patchAccount(
                                            activeAccountId,
                                            (current) => ({
                                              ...current,
                                              cdn_base_url: value || undefined,
                                            }),
                                          )
                                        }
                                        placeholder={
                                          DEFAULT_WECHAT_CDN_BASE_URL
                                        }
                                        mono
                                      />
                                    </div>
                                    <SecretField
                                      label="账号机器人 Token"
                                      value={activeAccount.bot_token || ""}
                                      onChange={(value) =>
                                        patchAccount(
                                          activeAccountId,
                                          (current) => ({
                                            ...current,
                                            bot_token: value || undefined,
                                          }),
                                        )
                                      }
                                      placeholder="扫码成功后回填的 token"
                                    />
                                    <TextField
                                      label="账号扫码用户 ID"
                                      value={
                                        activeAccount.scanner_user_id || ""
                                      }
                                      onChange={(value) =>
                                        patchAccount(
                                          activeAccountId,
                                          (current) => ({
                                            ...current,
                                            scanner_user_id: value || undefined,
                                          }),
                                        )
                                      }
                                      placeholder="扫码后的微信 user_id"
                                      mono
                                    />
                                  </CollapsibleContent>
                                </div>
                              </Collapsible>
                            ) : null}
                          </>
                        ) : (
                          <div className="text-xs text-slate-500">
                            还没有账号。可先扫码或手工新增。
                          </div>
                        )}
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                </CollapsibleContent>
              </div>
            </Collapsible>
          </div>
        </div>
      </ModalBody>
      <ModalFooter>
        <ActionButton onClick={onClose}>取消</ActionButton>
        <ActionButton
          kind="primary"
          onClick={() => void handleSave()}
          disabled={busy}
        >
          保存并启用
        </ActionButton>
      </ModalFooter>
    </Modal>
  );
}

export function ImConfigPage() {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<FlashMessage>(null);
  const [activeDialog, setActiveDialog] = useState<ManagedChannel | null>(null);
  const [debugToolsOpen, setDebugToolsOpen] = useState(false);
  const [debugChannel, setDebugChannel] = useState<ManagedChannel>("telegram");
  const advancedToolsRef = useRef<HTMLElement | null>(null);

  const loadConfig = useCallback(async () => {
    setLoading(true);
    try {
      const nextConfig = await getConfig();
      setConfig(nextConfig);
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    if (!message) {
      return;
    }

    const timeoutId = window.setTimeout(() => setMessage(null), 2500);
    return () => window.clearTimeout(timeoutId);
  }, [message]);

  const channels = useMemo(
    () => normalizeChannelsConfig(config?.channels),
    [config?.channels],
  );
  const enabledChannelCount = [
    channels.telegram.enabled,
    channels.feishu.enabled,
    channels.wechat.enabled,
  ].filter(Boolean).length;
  const wechatAccountCount = Object.keys(channels.wechat.accounts ?? {}).length;

  const persistChannel = async <K extends keyof ChannelsConfig>(
    channel: K,
    nextChannelConfig: ChannelsConfig[K],
    successText: string,
  ) => {
    const nextChannels = {
      ...channels,
      [channel]: nextChannelConfig,
    } as ChannelsConfig;
    const nextConfig = buildNextConfig(config, nextChannels);
    await saveConfig(nextConfig);
    setConfig(nextConfig);
    setMessage({ type: "success", text: successText });
  };

  const revealAdvancedTools = (channel: ManagedChannel) => {
    setActiveDialog(null);
    setDebugChannel(channel);
    setDebugToolsOpen(true);
    window.setTimeout(() => {
      const section = advancedToolsRef.current;
      if (section && typeof section.scrollIntoView === "function") {
        section.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 0);
  };

  if (loading) {
    return (
      <div
        data-testid="im-config-page"
        className="lime-workbench-theme-scope mx-auto flex w-full max-w-[1260px] flex-col gap-6 pb-10"
      >
        <div className="h-36 animate-pulse rounded-[30px] border border-slate-200 bg-[image:var(--lime-card-subtle)]" />
        <div className="grid gap-5 lg:grid-cols-3">
          <div className="h-[280px] animate-pulse rounded-[28px] border border-slate-200 bg-white" />
          <div className="h-[280px] animate-pulse rounded-[28px] border border-slate-200 bg-white" />
          <div className="h-[280px] animate-pulse rounded-[28px] border border-slate-200 bg-white" />
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="lime-workbench-theme-scope mx-auto flex w-full max-w-[1260px] flex-col gap-6 pb-10">
        <section className="overflow-hidden rounded-[32px] border border-slate-200 bg-[image:var(--lime-card-subtle)] px-7 py-7 shadow-sm shadow-slate-950/5">
          <div className="max-w-3xl space-y-4">
            <div className="max-w-2xl space-y-3">
              <span className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200">
                <Sparkles className="h-3.5 w-3.5" />
                系统入口
              </span>
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
                    消息渠道
                  </h1>
                  <WorkbenchInfoTip
                    ariaLabel="消息渠道说明"
                    content="把 Telegram、飞书、微信接进 Lime，让消息接收、回复和结果投递继续留在熟悉客户端；接入检查放在各自配置里，日志和运行检查按需展开。"
                    tone="mint"
                  />
                </div>
                <p className="max-w-3xl text-sm leading-7 text-slate-600">
                  把外部消息接进
                  Lime，让消息接收、回复和结果投递继续留在熟悉客户端。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
                3 个常用渠道
              </span>
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
                已启用 {enabledChannelCount} 个
              </span>
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
                微信 {wechatAccountCount} 个账号
              </span>
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
                日志与检查可展开
              </span>
            </div>
          </div>
        </section>

        <SectionMessage message={message} />

        <div className="grid gap-5 lg:grid-cols-3">
          <ChannelCard
            icon={<Send className="h-5 w-5" />}
            title="Telegram"
            subtitle="Token 接入"
            status={channels.telegram.enabled ? "已启用" : "未启用"}
            statusTone={channels.telegram.enabled ? "success" : "neutral"}
            details={[
              {
                label: "机器人 Token",
                value: channels.telegram.bot_token.trim() ? "已填写" : "未填写",
              },
              {
                label: "接入范围",
                value: describeTelegramScope(channels.telegram),
              },
            ]}
            actions={
              <>
                <ActionButton
                  kind="primary"
                  onClick={() => setActiveDialog("telegram")}
                  ariaLabel="打开 Telegram 配置"
                  dataTestId="telegram-config-button"
                >
                  配置
                </ActionButton>
              </>
            }
          />

          <ChannelCard
            icon={<Bot className="h-5 w-5" />}
            title="飞书"
            subtitle="应用凭证与准入"
            status={channels.feishu.enabled ? "已启用" : "未启用"}
            statusTone={channels.feishu.enabled ? "success" : "neutral"}
            details={[
              {
                label: "应用 ID",
                value: maskMiddle(channels.feishu.app_id, 5),
              },
              {
                label: "准入策略",
                value: describeDmPolicy(channels.feishu.dm_policy),
              },
            ]}
            actions={
              <>
                <ActionButton
                  kind="primary"
                  onClick={() => setActiveDialog("feishu")}
                  ariaLabel="打开飞书配置"
                  dataTestId="feishu-config-button"
                >
                  配置
                </ActionButton>
              </>
            }
          />

          <ChannelCard
            icon={<ScanQrCode className="h-5 w-5" />}
            title="微信"
            subtitle="扫码接入与账号"
            status={describeWechatStatus(channels.wechat)}
            statusTone={
              channels.wechat.enabled
                ? Object.keys(channels.wechat.accounts ?? {}).length > 0
                  ? "success"
                  : "warning"
                : "neutral"
            }
            details={[
              {
                label: "默认账号",
                value: channels.wechat.default_account || "未设置",
              },
              {
                label: "私聊策略",
                value: describeDmPolicy(channels.wechat.dm_policy),
              },
            ]}
            actions={
              <>
                <ActionButton
                  kind="primary"
                  onClick={() => setActiveDialog("wechat")}
                  ariaLabel="打开微信扫码配置"
                  dataTestId="wechat-config-button"
                >
                  配置
                </ActionButton>
              </>
            }
          />
        </div>

        <section className="rounded-[24px] border border-dashed border-slate-200 bg-slate-50/70 px-5 py-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-slate-700">
                <span>即将开放</span>
                <WorkbenchInfoTip
                  ariaLabel="即将开放渠道说明"
                  content="这些渠道先保留同级占位，灰态展示，不开放点击。"
                  tone="slate"
                />
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <div
                aria-disabled="true"
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500"
              >
                <MessageSquare className="h-3.5 w-3.5" />
                Discord
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-400">
                  即将开放
                </span>
              </div>
              <div
                aria-disabled="true"
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500"
              >
                <ShieldCheck className="h-3.5 w-3.5" />
                钉钉
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-400">
                  即将开放
                </span>
              </div>
            </div>
          </div>
        </section>

        <Collapsible open={debugToolsOpen} onOpenChange={setDebugToolsOpen}>
          <section
            ref={advancedToolsRef}
            className="rounded-[28px] border border-slate-200 bg-white shadow-sm shadow-slate-950/5"
          >
            <div className="flex items-start gap-3 px-6 py-5">
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-center justify-between text-left"
                >
                  <div className="space-y-1">
                    <div className="text-lg font-semibold tracking-tight text-slate-900">
                      日志与检查
                    </div>
                  </div>
                  {debugToolsOpen ? (
                    <ChevronDown className="h-5 w-5 text-slate-500" />
                  ) : (
                    <ChevronRight className="h-5 w-5 text-slate-500" />
                  )}
                </button>
              </CollapsibleTrigger>
              <WorkbenchInfoTip
                ariaLabel="日志与检查说明"
                content="接入日志和运行检查都在这里，平时不用一直展开。"
                tone="slate"
              />
            </div>
            <CollapsibleContent className="border-t border-slate-100 px-6 py-5">
              <Suspense
                fallback={
                  <div className="rounded-[24px] border border-slate-200 bg-slate-50 px-5 py-6 text-sm text-slate-500">
                    正在加载日志与检查...
                  </div>
                }
              >
                <ChannelsDebugWorkbench
                  key={debugChannel}
                  initialTab={debugChannel}
                  initialSubPage="logs"
                  initialDebugTab={debugChannel}
                  onConfigSaved={() => {
                    void loadConfig();
                  }}
                />
              </Suspense>
            </CollapsibleContent>
          </section>
        </Collapsible>
      </div>

      <TelegramConfigDialog
        isOpen={activeDialog === "telegram"}
        config={channels.telegram}
        onClose={() => setActiveDialog(null)}
        onSave={(nextConfig, successText) =>
          persistChannel("telegram", nextConfig, successText)
        }
        onOpenAdvanced={() => revealAdvancedTools("telegram")}
      />

      <FeishuConfigDialog
        isOpen={activeDialog === "feishu"}
        config={channels.feishu}
        onClose={() => setActiveDialog(null)}
        onSave={(nextConfig, successText) =>
          persistChannel("feishu", nextConfig, successText)
        }
        onOpenAdvanced={() => revealAdvancedTools("feishu")}
      />

      <WechatConfigDialog
        isOpen={activeDialog === "wechat"}
        config={channels.wechat}
        onClose={() => setActiveDialog(null)}
        onSave={(nextConfig, successText) =>
          persistChannel("wechat", nextConfig, successText)
        }
        onOpenAdvanced={() => revealAdvancedTools("wechat")}
      />
    </>
  );
}
