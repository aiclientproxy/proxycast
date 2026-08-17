import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type {
  ScheduledTaskNotificationSubscription,
  ScheduledTaskRunUpdatedNotification,
} from "@/lib/api/scheduledTasks";
import { ScheduledTaskNotificationBridge } from "./ScheduledTaskNotificationBridge";

const {
  showDesktopNotificationMock,
  subscribeScheduledTaskNotificationsMock,
  toastErrorMock,
} = vi.hoisted(() => ({
  showDesktopNotificationMock: vi.fn(),
  subscribeScheduledTaskNotificationsMock: vi.fn(),
  toastErrorMock: vi.fn(),
}));

vi.mock("@/lib/api/appServerBridgeAvailability", () => ({
  isAppServerBridgeAvailable: () => true,
}));

vi.mock("@/lib/api/desktopNotification", () => ({
  showDesktopNotification: showDesktopNotificationMock,
}));

vi.mock("@/lib/api/scheduledTasks", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/scheduledTasks")
  >("@/lib/api/scheduledTasks");
  return {
    ...actual,
    subscribeScheduledTaskNotifications:
      subscribeScheduledTaskNotificationsMock,
  };
});

vi.mock("sonner", () => ({
  toast: { error: toastErrorMock },
}));

let subscription: ScheduledTaskNotificationSubscription | undefined;
let root: Root | undefined;
let container: HTMLDivElement | undefined;

function notification(
  overrides: Partial<ScheduledTaskRunUpdatedNotification> = {},
): ScheduledTaskRunUpdatedNotification {
  return {
    attention: false,
    notificationPolicy: "all_runs",
    runId: "run-1",
    status: "success",
    taskId: "task-1",
    title: "每日项目简报",
    ...overrides,
  };
}

async function renderBridge(): Promise<void> {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(<ScheduledTaskNotificationBridge />);
    await Promise.resolve();
  });
}

async function publish(
  value: ScheduledTaskRunUpdatedNotification,
): Promise<void> {
  await act(async () => {
    subscription?.onRunUpdated?.(value);
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("ScheduledTaskNotificationBridge", () => {
  beforeEach(async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.clearAllMocks();
    subscription = undefined;
    subscribeScheduledTaskNotificationsMock.mockImplementation((value) => {
      subscription = value;
      return vi.fn();
    });
    showDesktopNotificationMock.mockResolvedValue({ status: "sent" });
    await changeLimeLocale("zh-CN");
  });

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    root = undefined;
    container = undefined;
    vi.unstubAllGlobals();
  });

  it("按 all_runs 策略发送本地化系统通知并按 run 去重", async () => {
    await renderBridge();
    const update = notification();
    await publish(update);
    await publish(update);

    expect(showDesktopNotificationMock).toHaveBeenCalledOnce();
    expect(showDesktopNotificationMock).toHaveBeenCalledWith({
      body: "“每日项目简报”本次运行状态：已完成。",
      tag: "scheduled-task:task-1:run-1",
      title: "已安排任务状态更新",
    });
  });

  it("failures 只发送 attention 终态，none 永不发送", async () => {
    await renderBridge();
    await publish(
      notification({ notificationPolicy: "failures", runId: "run-success" }),
    );
    await publish(
      notification({
        attention: true,
        error: "模型服务暂不可用",
        notificationPolicy: "failures",
        runId: "run-error",
        status: "error",
      }),
    );
    await publish(
      notification({ notificationPolicy: "none", runId: "run-none" }),
    );

    expect(showDesktopNotificationMock).toHaveBeenCalledOnce();
    expect(showDesktopNotificationMock).toHaveBeenCalledWith(
      expect.objectContaining({
        body: "“每日项目简报”本次运行状态：失败。模型服务暂不可用",
        title: "已安排任务需要处理",
      }),
    );
  });

  it("Host 不支持系统通知时显示可见错误", async () => {
    showDesktopNotificationMock.mockResolvedValueOnce({
      reason: "electron_notification_unsupported",
      status: "unsupported",
    });
    await renderBridge();
    await publish(notification());

    expect(toastErrorMock).toHaveBeenCalledWith("无法发送系统通知", {
      description:
        "当前系统或桌面环境不支持系统通知。任务状态仍可在已安排任务中查看。",
      duration: 12_000,
    });
  });

  it("Host 调用失败时不伪造成功并展示失败原因", async () => {
    showDesktopNotificationMock.mockRejectedValueOnce(
      new Error("notification blocked"),
    );
    await renderBridge();
    await publish(notification());

    expect(toastErrorMock).toHaveBeenCalledWith("无法发送系统通知", {
      description:
        "系统通知发送失败：notification blocked。任务状态仍可在已安排任务中查看。",
      duration: 12_000,
    });
  });
});
