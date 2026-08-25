import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import {
  projectStrictReviewStatus,
  useStrictReviewStatus,
} from "./useStrictReviewStatus";

const strictNotification = {
  jsonrpc: "2.0" as const,
  method: "autoApprovalReview/strictReviewRequired",
  params: {
    startedAtMs: 1_783_814_400_100,
    threadId: "thread-1",
    turnId: "turn-1",
  },
};

const completedNotification = {
  jsonrpc: "2.0" as const,
  method: "item/autoApprovalReview/completed",
  params: {
    action: {
      command: "git status --short",
      cwd: "/workspace",
      source: "shell",
      type: "command",
    },
    completedAtMs: 1_783_814_401_100,
    decisionSource: "agent",
    review: { status: "approved" },
    reviewId: "review-1",
    startedAtMs: 1_783_814_400_100,
    targetItemId: "item-1",
    threadId: "thread-1",
    turnId: "turn-1",
  },
};

describe("useStrictReviewStatus", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("projects exact notification and clears on the same Guardian completion", () => {
    const status = projectStrictReviewStatus(
      null,
      strictNotification,
      "thread-1",
    );
    expect(status).toEqual({
      startedAtMs: 1_783_814_400_100,
      threadId: "thread-1",
      turnId: "turn-1",
    });
    expect(
      projectStrictReviewStatus(status, completedNotification, "thread-1"),
    ).toBeNull();
    expect(
      projectStrictReviewStatus(null, strictNotification, "thread-other"),
    ).toBeNull();
  });

  it("subscribes per Thread and resets when the active Thread changes", async () => {
    let subscription: AppServerEventBusSubscription | null = null;
    const unsubscribe = vi.fn();
    const subscribeNotifications = vi.fn(
      (next: AppServerEventBusSubscription) => {
        subscription = next;
        return unsubscribe;
      },
    );
    let current: ReturnType<typeof useStrictReviewStatus> = null;
    const container = document.createElement("div");
    const root = createRoot(container);

    function Harness({ threadId }: { threadId: string }) {
      current = useStrictReviewStatus({ threadId, subscribeNotifications });
      return null;
    }

    try {
      await act(async () => root.render(<Harness threadId="thread-1" />));
      expect(subscribeNotifications).toHaveBeenCalledWith(
        expect.objectContaining({
          getDrainOptions: expect.any(Function),
        }),
      );
      expect(subscription?.getDrainOptions?.()).toEqual({
        includeRecent: true,
      });
      act(() => subscription?.onNotifications?.([strictNotification]));
      expect(current).toMatchObject({ threadId: "thread-1", turnId: "turn-1" });

      await act(async () => root.render(<Harness threadId="thread-2" />));
      expect(current).toBeNull();
      expect(unsubscribe).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => root.unmount());
    }
  });
});
