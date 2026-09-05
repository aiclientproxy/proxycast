import { afterEach, describe, expect, it, vi } from "vitest";
import {
  readFileSystemChangedNotification,
  startFileSystemWatch,
} from "./fileSystemWatch";
import type { AppServerEventBusSubscription } from "./appServerEventBus";

describe("fileSystemWatch gateway", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("只消费匹配 watchId 的 typed fs/changed，并幂等停止", async () => {
    const watch = vi.fn().mockResolvedValue({
      result: { path: "/workspace" },
    });
    const unwatch = vi.fn().mockResolvedValue({ result: {} });
    const client = { watch, unwatch };
    const handler = vi.fn();
    const unsubscribe = vi.fn();
    let subscription: AppServerEventBusSubscription | undefined;
    const subscribeNotifications = vi.fn((value) => {
      subscription = value;
      return unsubscribe;
    });

    const stop = await startFileSystemWatch("/workspace", handler, {
      client,
      subscribeNotifications,
      watchId: "file-manager-test",
    });

    expect(watch).toHaveBeenCalledWith({
      path: "/workspace",
      watchId: "file-manager-test",
    });
    subscription?.onNotifications?.([
      {
        method: "fs/changed",
        params: {
          changedPaths: ["/workspace/README.md"],
          watchId: "other",
        },
      },
      {
        method: "fs/changed",
        params: {
          changedPaths: ["/workspace/src"],
          watchId: "file-manager-test",
        },
      },
      {
        method: "fs/changed",
        params: { changedPaths: [], watchId: "file-manager-test" },
      },
    ]);
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith({
      changedPaths: ["/workspace/src"],
      watchId: "file-manager-test",
    });

    await stop();
    await stop();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
    expect(unwatch).toHaveBeenCalledTimes(1);
    expect(unwatch).toHaveBeenCalledWith({ watchId: "file-manager-test" });
  });

  it("watch 失败时撤销通知订阅，并拒绝相对路径", async () => {
    const unsubscribe = vi.fn();
    const subscribeNotifications = vi.fn(() => unsubscribe);
    const watch = vi.fn().mockRejectedValue(new Error("watch failed"));
    const client = {
      watch,
      unwatch: vi.fn(),
    };

    await expect(
      startFileSystemWatch("/workspace", vi.fn(), {
        client,
        subscribeNotifications,
        watchId: "file-manager-failure",
      }),
    ).rejects.toThrow("watch failed");
    expect(unsubscribe).toHaveBeenCalledTimes(1);
    expect(client.unwatch).not.toHaveBeenCalled();

    await expect(
      startFileSystemWatch("relative", vi.fn(), {
        client,
        subscribeNotifications,
      }),
    ).rejects.toThrow("absolute path");
    expect(subscribeNotifications).toHaveBeenCalledTimes(1);
  });

  it("malformed fs/changed payload fail closed", () => {
    expect(
      readFileSystemChangedNotification({
        method: "fs/changed",
        params: { changedPaths: ["/workspace/a"], watchId: "watch" },
      }),
    ).toBeTruthy();
    expect(
      readFileSystemChangedNotification({
        method: "fs/changed",
        params: { changedPaths: [""], watchId: "watch" },
      }),
    ).toBeUndefined();
  });
});
