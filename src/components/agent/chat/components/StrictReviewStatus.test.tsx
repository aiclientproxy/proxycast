import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StrictReviewStatus } from "./StrictReviewStatus";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "en-US", resolvedLanguage: "en-US" },
    t: (key: string, values?: { time?: string }) =>
      values?.time ? `${key}:${values.time}` : key,
  }),
}));

describe("StrictReviewStatus", () => {
  const mounted: Array<{
    container: HTMLDivElement;
    root: ReturnType<typeof createRoot>;
  }> = [];

  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
  });

  afterEach(() => {
    while (mounted.length > 0) {
      const instance = mounted.pop();
      if (instance) act(() => instance.root.unmount());
    }
  });

  it("exposes exact protocol and canonical Thread/Turn identity", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() => {
      root.render(
        <StrictReviewStatus
          status={{
            startedAtMs: 1_783_814_400_100,
            threadId: "thread-1",
            turnId: "turn-1",
          }}
        />,
      );
    });
    mounted.push({ container, root });

    const status = container.querySelector(
      '[data-testid="strict-review-status"]',
    );
    expect(status?.getAttribute("data-protocol-method")).toBe(
      "autoApprovalReview/strictReviewRequired",
    );
    expect(status?.getAttribute("data-thread-id")).toBe("thread-1");
    expect(status?.getAttribute("data-turn-id")).toBe("turn-1");
    expect(container.textContent).toContain("agentChat.strictReview.title");
    expect(container.textContent).toContain("agentChat.strictReview.nextStep");
  });
});
