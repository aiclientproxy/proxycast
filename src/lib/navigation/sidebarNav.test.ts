import { describe, expect, it } from "vitest";
import {
  FOOTER_SIDEBAR_NAV_ITEMS,
  MAIN_SIDEBAR_NAV_ITEMS,
  resolveEnabledSidebarNavItems,
} from "./sidebarNav";

describe("sidebarNav", () => {
  it("应把主导航与底部系统入口收口为一级列表", () => {
    expect(MAIN_SIDEBAR_NAV_ITEMS.map((item) => item.label)).toEqual([
      "新建任务",
      "专家",
      "Skills",
      "插件",
    ]);

    expect(FOOTER_SIDEBAR_NAV_ITEMS.map((item) => item.label)).toEqual([
      "设置",
      "项目资料",
      "持续流程",
      "消息渠道",
    ]);
    expect(FOOTER_SIDEBAR_NAV_ITEMS.map((item) => item.id)).not.toContain(
      "memory",
    );
  });

  it("恢复导航设置时应过滤固定系统入口和已下线 companion", () => {
    expect(
      resolveEnabledSidebarNavItems([
        "video",
        "image-gen",
        "terminal",
        "tools",
        "home-general",
        "automation",
        "channels",
        "plugins",
        "companion",
      ]),
    ).toEqual([]);
  });

  it("没有显式设置时不应默认恢复任何可选入口", () => {
    expect(resolveEnabledSidebarNavItems()).toEqual([]);
    expect(resolveEnabledSidebarNavItems(["skills", "resources"])).toEqual([]);
  });

  it("旧 schema 中的 companion 不应被当作显式开启", () => {
    expect(resolveEnabledSidebarNavItems(["companion"], 2)).toEqual([]);
  });
});
