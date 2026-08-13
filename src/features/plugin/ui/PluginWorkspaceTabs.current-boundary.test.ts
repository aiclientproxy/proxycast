import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

describe("PluginWorkspaceTabs current boundary", () => {
  it("顶部 Tab 应越过 Electron 拖拽层并声明 no-drag", () => {
    const source = readFileSync(
      join(cwd(), "src/features/plugin/ui/PluginWorkspaceTabs.tsx"),
      "utf8",
    );

    expect(source).toContain("z-[1001]");
    expect(source).toContain("[app-region:no-drag]");
    expect(source).toContain("[-webkit-app-region:no-drag]");
  });
});
