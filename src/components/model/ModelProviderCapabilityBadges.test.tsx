import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { ModelProviderCapabilityBadges } from "./ModelProviderCapabilityBadges";

const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  while (mounted.length > 0) {
    const item = mounted.pop();
    if (!item) break;
    act(() => item.root.unmount());
    item.container.remove();
  }
});

describe("ModelProviderCapabilityBadges", () => {
  it("应逐项展示 exact provider capability 快照", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(
        <ModelProviderCapabilityBadges
          capabilities={{
            namespaceTools: true,
            imageGeneration: false,
            webSearch: true,
          }}
        />,
      );
    });
    mounted.push({ root, container });

    expect(container.textContent).toContain("工具命名空间");
    expect(container.textContent).toContain("无图片生成");
    expect(container.textContent).toContain("网页搜索");
  });
});
