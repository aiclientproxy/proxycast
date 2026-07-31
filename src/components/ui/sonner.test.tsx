import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "sonner";
import { Toaster } from "./sonner";

interface MountedToaster {
  container: HTMLDivElement;
  root: Root;
}

const mountedToasters: MountedToaster[] = [];

describe("Toaster", () => {
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  });

  afterEach(() => {
    toast.dismiss();
    for (const mounted of mountedToasters.splice(0)) {
      act(() => mounted.root.unmount());
      mounted.container.remove();
    }
    document.body.replaceChildren();
    vi.unstubAllGlobals();
  });

  it("挂载后应把 toast.error 展示为真实 DOM 提示", async () => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn(
        () =>
          ({
            matches: false,
            media: "(prefers-color-scheme: dark)",
            onchange: null,
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(() => false),
          }) as unknown as MediaQueryList,
      ),
    );
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedToasters.push({ container, root });

    act(() => {
      root.render(<Toaster />);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('section[aria-label^="Notifications"]'),
    ).not.toBeNull();

    await act(async () => {
      toast.error("发送失败：当前模型通道暂时不可用");
      await Promise.resolve();
    });

    await vi.waitFor(() => {
      expect(document.body.querySelector("[data-sonner-toast]")).not.toBeNull();
    });

    expect(
      document.body.querySelector("[data-sonner-toast]")?.textContent,
    ).toContain("发送失败：当前模型通道暂时不可用");
  });
});
