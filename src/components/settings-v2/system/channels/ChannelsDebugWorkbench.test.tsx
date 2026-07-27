import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const {
  mockGetConfig,
  mockSaveConfig,
  mockUseConfiguredProviders,
  mockDiscordChannelProbe,
  mockFeishuChannelProbe,
  mockGatewayChannelStart,
  mockGatewayChannelStatus,
  mockGatewayChannelStop,
  mockGatewayTunnelCreate,
  mockGatewayTunnelDetectCloudflared,
  mockGatewayTunnelInstallCloudflared,
  mockGatewayTunnelProbe,
  mockGatewayTunnelRestart,
  mockGatewayTunnelStart,
  mockGatewayTunnelStatus,
  mockGatewayTunnelStop,
  mockGatewayTunnelSyncWebhookUrl,
  mockTelegramChannelProbe,
  mockWechatChannelListAccounts,
  mockWechatChannelLoginStart,
  mockWechatChannelLoginWait,
  mockWechatChannelProbe,
  mockWechatChannelRemoveAccount,
  mockQrCodeToDataUrl,
} = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
  mockUseConfiguredProviders: vi.fn(),
  mockDiscordChannelProbe: vi.fn(),
  mockFeishuChannelProbe: vi.fn(),
  mockGatewayChannelStart: vi.fn(),
  mockGatewayChannelStatus: vi.fn(),
  mockGatewayChannelStop: vi.fn(),
  mockGatewayTunnelCreate: vi.fn(),
  mockGatewayTunnelDetectCloudflared: vi.fn(),
  mockGatewayTunnelInstallCloudflared: vi.fn(),
  mockGatewayTunnelProbe: vi.fn(),
  mockGatewayTunnelRestart: vi.fn(),
  mockGatewayTunnelStart: vi.fn(),
  mockGatewayTunnelStatus: vi.fn(),
  mockGatewayTunnelStop: vi.fn(),
  mockGatewayTunnelSyncWebhookUrl: vi.fn(),
  mockTelegramChannelProbe: vi.fn(),
  mockWechatChannelListAccounts: vi.fn(),
  mockWechatChannelLoginStart: vi.fn(),
  mockWechatChannelLoginWait: vi.fn(),
  mockWechatChannelProbe: vi.fn(),
  mockWechatChannelRemoveAccount: vi.fn(),
  mockQrCodeToDataUrl: vi.fn(),
}));

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
}));

vi.mock("@/hooks/useConfiguredProviders", () => ({
  useConfiguredProviders: mockUseConfiguredProviders,
  findConfiguredProviderBySelection: (
    providers: Array<{ key: string; providerId?: string }>,
    selection?: string | null,
  ) => {
    const normalizedSelection = (selection || "").trim().toLowerCase();
    const keyMatch =
      providers.find(
        (provider) => provider.key.trim().toLowerCase() === normalizedSelection,
      ) ?? null;
    const providerIdMatch =
      providers.find(
        (provider) =>
          (provider.providerId || "").trim().toLowerCase() ===
          normalizedSelection,
      ) ?? null;

    if (keyMatch && providerIdMatch && keyMatch !== providerIdMatch) {
      if (!keyMatch.providerId && providerIdMatch.providerId) {
        return providerIdMatch;
      }
    }

    return keyMatch ?? providerIdMatch ?? null;
  },
}));

vi.mock("@/lib/api/channelsRuntime", () => ({
  discordChannelProbe: mockDiscordChannelProbe,
  feishuChannelProbe: mockFeishuChannelProbe,
  gatewayChannelStart: mockGatewayChannelStart,
  gatewayChannelStatus: mockGatewayChannelStatus,
  gatewayChannelStop: mockGatewayChannelStop,
  gatewayTunnelCreate: mockGatewayTunnelCreate,
  gatewayTunnelDetectCloudflared: mockGatewayTunnelDetectCloudflared,
  gatewayTunnelInstallCloudflared: mockGatewayTunnelInstallCloudflared,
  gatewayTunnelProbe: mockGatewayTunnelProbe,
  gatewayTunnelRestart: mockGatewayTunnelRestart,
  gatewayTunnelStart: mockGatewayTunnelStart,
  gatewayTunnelStatus: mockGatewayTunnelStatus,
  gatewayTunnelStop: mockGatewayTunnelStop,
  gatewayTunnelSyncWebhookUrl: mockGatewayTunnelSyncWebhookUrl,
  telegramChannelProbe: mockTelegramChannelProbe,
  wechatChannelListAccounts: mockWechatChannelListAccounts,
  wechatChannelLoginStart: mockWechatChannelLoginStart,
  wechatChannelLoginWait: mockWechatChannelLoginWait,
  wechatChannelProbe: mockWechatChannelProbe,
  wechatChannelRemoveAccount: mockWechatChannelRemoveAccount,
}));

vi.mock("./ChannelLogTailPanel", () => ({
  ChannelLogTailPanel: () => <div data-testid="channel-log-tail-panel" />,
}));

vi.mock("qrcode", () => ({
  default: {
    toDataURL: mockQrCodeToDataUrl,
  },
}));

import { ChannelsDebugWorkbench } from "./ChannelsDebugWorkbench";

interface MountedPage {
  container: HTMLDivElement;
  root: Root;
}

const mountedPages: MountedPage[] = [];

function createConfig() {
  return {
    channels: {
      telegram: {
        enabled: true,
        bot_token: "123456:telegram-token",
        allowed_user_ids: ["10001"],
        default_model: "openai/gpt-4.1",
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
        enabled: true,
        app_id: "cli_test_demo",
        app_secret: "secret",
        default_model: "openai/gpt-4.1",
        dm_policy: "pairing",
        allow_from: ["ou_test"],
        group_policy: "allowlist",
        group_allow_from: [],
      },
      wechat: {
        enabled: true,
        bot_token: "",
        base_url: "https://ilinkai.weixin.qq.com",
        cdn_base_url: "https://novac2c.cdn.weixin.qq.com/c2c",
        default_model: "openai/gpt-4.1",
        default_account: "wechat_1",
        accounts: {
          wechat_1: {
            enabled: true,
            name: "微信一号",
            base_url: "https://ilinkai.weixin.qq.com",
          },
        },
        dm_policy: "pairing",
        allow_from: ["wx_user_1"],
        group_policy: "allowlist",
        group_allow_from: [],
      },
    },
    gateway: {
      tunnel: {
        enabled: false,
        provider: "cloudflare",
        mode: "managed",
        local_host: "127.0.0.1",
        local_port: 3000,
        cloudflare: {},
      },
    },
  } as any;
}

function renderPage(
  props: Partial<ComponentProps<typeof ChannelsDebugWorkbench>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ChannelsDebugWorkbench initialSubPage="config" {...props} />);
  });

  mountedPages.push({ container, root });
  return container;
}

async function flushEffects(times = 6) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("ChannelsDebugWorkbench", () => {
  beforeEach(async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    await changeLimeLocale("en-US");

    mockGetConfig.mockResolvedValue(createConfig());
    mockSaveConfig.mockResolvedValue(undefined);
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "openai",
          label: "OpenAI",
          type: "openai",
          registryId: "openai",
          models: ["gpt-4.1"],
        },
      ],
      loading: false,
    });

    mockDiscordChannelProbe.mockResolvedValue({ ok: true });
    mockFeishuChannelProbe.mockResolvedValue({ ok: true });
    mockGatewayChannelStart.mockResolvedValue({ ok: true });
    mockGatewayChannelStatus.mockResolvedValue({ ok: true });
    mockGatewayChannelStop.mockResolvedValue({ ok: true });
    mockGatewayTunnelCreate.mockResolvedValue({ ok: true });
    mockGatewayTunnelDetectCloudflared.mockResolvedValue({ ok: true });
    mockGatewayTunnelInstallCloudflared.mockResolvedValue({ ok: true });
    mockGatewayTunnelProbe.mockResolvedValue({ ok: true });
    mockGatewayTunnelRestart.mockResolvedValue({ ok: true });
    mockGatewayTunnelStart.mockResolvedValue({ ok: true });
    mockGatewayTunnelStatus.mockResolvedValue({ ok: true });
    mockGatewayTunnelStop.mockResolvedValue({ ok: true });
    mockGatewayTunnelSyncWebhookUrl.mockResolvedValue({ ok: true });
    mockTelegramChannelProbe.mockResolvedValue({ ok: true });
    mockWechatChannelListAccounts.mockResolvedValue([]);
    mockWechatChannelLoginStart.mockResolvedValue({
      sessionKey: "wechat-session",
      qrcodeUrl: "wechat-login-qr",
      message: "ok",
    });
    mockWechatChannelLoginWait.mockResolvedValue({ connected: false });
    mockWechatChannelProbe.mockResolvedValue({ ok: true });
    mockWechatChannelRemoveAccount.mockResolvedValue({ ok: true });
    mockQrCodeToDataUrl.mockResolvedValue("data:image/png;base64,qr");
  });

  afterEach(async () => {
    while (mountedPages.length > 0) {
      const mounted = mountedPages.pop();
      if (!mounted) {
        continue;
      }

      act(() => {
        mounted.root.unmount();
      });
      mounted.container.remove();
    }

    vi.clearAllMocks();
    vi.unstubAllGlobals();
    await changeLimeLocale("zh-CN");
  });

  it("旧的概览或配置入口应直接落到日志与检查区，且不再暴露渠道配置表单", async () => {
    renderPage();
    await flushEffects(10);

    const text = document.body.textContent ?? "";
    expect(text).toContain("Logs and Checks");
    expect(text).toContain("Only gateway, logs, and runtime checks stay here");
    expect(text).toContain("Logs");
    expect(text).toContain("Runtime");
    expect(text).not.toContain("settings.channels.workbench");
    expect(text).not.toContain("日志与检查");
    expect(
      document.body.querySelector('[data-testid="channel-log-tail-panel"]'),
    ).not.toBeNull();
    expect(text).not.toContain("配置说明（通用）");
    expect(text).not.toContain("当前渠道：");
    expect(text).not.toContain("启用 Discord Bot");
    expect(text).not.toContain("当前配置摘要");
    expect(text).not.toContain("配置提示");
    expect(text).not.toContain("探测账号");
    expect(text).not.toContain("扫码登录");
    expect(text).not.toContain("Discord Gateway 调试工具");
  });

  it("网关与隧道表单应通过 settings namespace 渲染外层文案", async () => {
    renderPage({ initialSubPage: "gateway" });
    await flushEffects(10);

    const text = document.body.textContent ?? "";
    expect(text).toContain("Gateway and Tunnel");
    expect(text).toContain("Gateway Public Tunnel");
    expect(text).toContain("Enable tunnel");
    expect(
      Array.from(document.body.querySelectorAll("input")).some(
        (input) => input.placeholder === "Use cloudflared from PATH by default",
      ),
    ).toBe(true);
    expect(text).toContain("Sync Feishu Callback URL");
    expect(text).toContain("Latest Result");
  });

  it("Telegram 与 Feishu 运行控制应通过 settings namespace 渲染", async () => {
    renderPage({ initialDebugTab: "feishu" });
    await flushEffects(10);

    const text = document.body.textContent ?? "";
    expect(text).toContain("Runtime");
    expect(text).toContain("Feishu Gateway Runtime Control");
    expect(text).toContain("Use this for status queries, start, stop");
    expect(text).toContain("Account ID");
    expect(text).toContain("Query Status");
    expect(text).toContain("Start");
    expect(text).toContain("Stop");
    expect(text).toContain("Restart");
    expect(text).toContain("Latest Result");
  });

  it("微信运行控制和兼容扫码排障应通过 settings namespace 渲染", async () => {
    renderPage({ initialDebugTab: "wechat" });
    await flushEffects(10);

    const text = document.body.textContent ?? "";
    expect(text).toContain("WeChat Gateway Runtime Control");
    expect(text).toContain("Leave Account ID empty to use the default config");
    expect(text).toContain("List Accounts");
    expect(text).toContain("Compatibility QR Troubleshooting");
    expect(text).toContain("Generate QR Code");
    expect(text).toContain("Configured Accounts");
    expect(text).toContain("Clear local cached data when deleting");
    expect(text).toContain("Latest Result");
  });
});
