import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockGetConfig,
  mockSaveConfig,
  mockUseConfiguredProviders,
  mockTelegramChannelProbe,
  mockFeishuChannelProbe,
  mockWechatChannelListAccounts,
  mockWechatChannelLoginStart,
  mockWechatChannelLoginWait,
  mockWechatChannelProbe,
  mockGatewayChannelStart,
  mockQrCodeToDataUrl,
} = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
  mockUseConfiguredProviders: vi.fn(),
  mockTelegramChannelProbe: vi.fn(),
  mockFeishuChannelProbe: vi.fn(),
  mockWechatChannelListAccounts: vi.fn(),
  mockWechatChannelLoginStart: vi.fn(),
  mockWechatChannelLoginWait: vi.fn(),
  mockWechatChannelProbe: vi.fn(),
  mockGatewayChannelStart: vi.fn(),
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
  telegramChannelProbe: mockTelegramChannelProbe,
  feishuChannelProbe: mockFeishuChannelProbe,
  wechatChannelListAccounts: mockWechatChannelListAccounts,
  wechatChannelLoginStart: mockWechatChannelLoginStart,
  wechatChannelLoginWait: mockWechatChannelLoginWait,
  wechatChannelProbe: mockWechatChannelProbe,
  gatewayChannelStart: mockGatewayChannelStart,
}));

vi.mock("qrcode", () => ({
  default: {
    toDataURL: mockQrCodeToDataUrl,
  },
}));

vi.mock(
  "@/components/settings-v2/system/channels/ChannelsDebugWorkbench",
  () => ({
    ChannelsDebugWorkbench: () => (
      <div data-testid="channels-debug-workbench" />
    ),
  }),
);

import { ImConfigPage } from "./ImConfigPage";

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
  } as any;
}

function renderPage() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ImConfigPage />);
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

function findButtonByTestId(testId: string): HTMLButtonElement {
  const button = document.body.querySelector(`[data-testid="${testId}"]`);
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`未找到按钮: ${testId}`);
  }
  return button;
}

function findButtonByText(text: string): HTMLButtonElement {
  const button = Array.from(document.body.querySelectorAll("button")).find(
    (item) => item.textContent?.includes(text),
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`未找到按钮文案: ${text}`);
  }
  return button;
}

async function clickButton(button: HTMLButtonElement) {
  await act(async () => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();
  });
}

function getBodyText() {
  return document.body.textContent ?? "";
}

async function hoverTip(ariaLabel: string) {
  const trigger = document.body.querySelector(
    `button[aria-label='${ariaLabel}']`,
  );
  expect(trigger).toBeInstanceOf(HTMLButtonElement);

  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    await Promise.resolve();
  });

  return trigger as HTMLButtonElement;
}

async function leaveTip(trigger: HTMLButtonElement | null) {
  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseout", { bubbles: true }));
    await Promise.resolve();
  });
}

describe("ImConfigPage", () => {
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

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
    mockTelegramChannelProbe.mockResolvedValue({
      ok: true,
      message: "ok",
      account_id: "default",
    });
    mockFeishuChannelProbe.mockResolvedValue({
      ok: true,
      message: "ok",
      account_id: "default",
    });
    mockWechatChannelListAccounts.mockResolvedValue([
      {
        accountId: "wechat_1",
        enabled: true,
        name: "微信一号",
        hasToken: true,
      },
    ]);
    mockWechatChannelLoginStart.mockResolvedValue({
      sessionKey: "wechat-session",
      qrcodeUrl: "wechat-login-qr",
      message: "ok",
    });
    mockWechatChannelLoginWait.mockResolvedValue({
      connected: false,
      message: "waiting",
    });
    mockWechatChannelProbe.mockResolvedValue({
      ok: true,
      message: "ok",
      accountId: "wechat_1",
    });
    mockGatewayChannelStart.mockResolvedValue({
      channel: "wechat",
      status: { ok: true },
    });
    mockQrCodeToDataUrl.mockResolvedValue("data:image/png;base64,qr");
  });

  afterEach(() => {
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
  });

  it("应将 Telegram、飞书、微信作为同级主入口展示", async () => {
    renderPage();
    await flushEffects();

    expect(
      document.body.querySelector(".lime-workbench-theme-scope"),
    ).not.toBeNull();

    const text = document.body.textContent ?? "";
    expect(text).toContain("消息渠道");
    expect(text).toContain("系统入口");
    expect(text).toContain("Telegram");
    expect(text).toContain("飞书");
    expect(text).toContain("微信");
    expect(text).toContain("即将开放");
    expect(text).toContain("Discord");
    expect(text).toContain("钉钉");
    expect(text).toContain("日志与检查");
    expect(text).not.toContain("调试配置");
  });

  it("应把首页说明和日志与检查说明收进 tips", async () => {
    renderPage();
    await flushEffects();

    expect(getBodyText()).not.toContain(
      "把 Telegram、飞书、微信接进 Lime，让消息接收、回复和结果投递继续留在熟悉客户端；接入检查放在各自配置里，日志和运行检查按需展开。",
    );
    expect(getBodyText()).not.toContain(
      "接入日志和运行检查都在这里，平时不用一直展开。",
    );

    const introTip = await hoverTip("消息渠道说明");
    expect(getBodyText()).toContain(
      "把 Telegram、飞书、微信接进 Lime，让消息接收、回复和结果投递继续留在熟悉客户端；接入检查放在各自配置里，日志和运行检查按需展开。",
    );
    await leaveTip(introTip);

    const debugTip = await hoverTip("日志与检查说明");
    expect(getBodyText()).toContain(
      "接入日志和运行检查都在这里，平时不用一直展开。",
    );
    await leaveTip(debugTip);
  });

  it("点击 Telegram 配置后应打开弹窗并支持保存", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByTestId("telegram-config-button"));
    await flushEffects();

    expect(
      document.body.querySelector('[role="dialog"].lime-workbench-theme-scope'),
    ).not.toBeNull();
    expect(document.body.textContent).toContain("Telegram 配置");
    expect(document.body.textContent).toContain("机器人 Token");
    expect(document.body.textContent).toContain("接入检查");
    expect(document.body.textContent).toContain("查看日志与检查");

    await clickButton(findButtonByText("保存"));
    await flushEffects();

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(mockSaveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        channels: expect.objectContaining({
          telegram: expect.objectContaining({
            bot_token: "123456:telegram-token",
            enabled: true,
          }),
        }),
      }),
    );
  });

  it("飞书配置弹窗应承接高级接入参数", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByTestId("feishu-config-button"));
    await flushEffects();

    let text = document.body.textContent ?? "";
    expect(text).toContain("配置飞书");
    expect(text).toContain("配置状态");
    expect(text).toContain("当前机器人");
    expect(text).toContain("准入策略");
    expect(text).toContain("已允许用户");

    await clickButton(findButtonByText("高级配置"));
    await flushEffects();

    text = document.body.textContent ?? "";
    expect(text).toContain("回调方式");
    expect(text).toContain("回调路径");
    expect(text).toContain("验证 Token");
  });

  it("飞书名单编辑应并回配置弹窗，不再暴露独立入口", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByTestId("feishu-config-button"));
    await flushEffects();

    const text = document.body.textContent ?? "";
    expect(text).toContain("已允许用户");
    expect(text).toContain("配对模式");
    expect(text).not.toContain("飞书配对管理");
  });

  it("点击微信扫码配置后应生成二维码并开始登录流程", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByTestId("wechat-config-button"));
    await flushEffects(10);
    await clickButton(findButtonByText("高级参数"));
    await flushEffects();

    expect(mockWechatChannelLoginStart).toHaveBeenCalledTimes(1);
    expect(mockWechatChannelLoginWait).toHaveBeenCalledTimes(1);
    expect(mockQrCodeToDataUrl).toHaveBeenCalledWith(
      "wechat-login-qr",
      expect.objectContaining({
        width: 320,
      }),
    );
    expect(
      document.body.querySelector('img[alt="微信扫码二维码"]'),
    ).not.toBeNull();
    expect(document.body.textContent).toContain("群组策略");
    expect(document.body.textContent).toContain("账号参数");
    expect(document.body.textContent).toContain("接入检查");
    expect(document.body.textContent).toContain("展开账号");
    expect(document.body.textContent).not.toContain("账号名称");

    await clickButton(findButtonByText("展开账号"));
    await flushEffects();
    await clickButton(findButtonByText("微信一号"));
    await flushEffects();

    expect(document.body.textContent).toContain("编辑账号详情");
    expect(document.body.textContent).toContain("账号名称");
  });

  it("配置弹窗里的日志与检查入口应展开旧的高级工作台", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByTestId("telegram-config-button"));
    await flushEffects();
    await clickButton(findButtonByText("查看日志与检查"));
    await flushEffects();

    expect(document.body.textContent).toContain("日志与检查");
    expect(
      document.body.querySelector('[data-testid="channels-debug-workbench"]'),
    ).not.toBeNull();
  });

  it("展开高级区后应保留旧的调试工作台", async () => {
    renderPage();
    await flushEffects();

    await clickButton(findButtonByText("日志与检查"));
    await flushEffects();

    expect(
      document.body.querySelector('[data-testid="channels-debug-workbench"]'),
    ).not.toBeNull();
  });
});
