import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const CANVAS_SESSION_OVERVIEW_SOURCES = [
  "src/components/agent/chat/components/CanvasSessionOverviewPanel.tsx",
  "src/components/agent/chat/components/CanvasSessionOverviewPanel.test.tsx",
] as const;

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), "utf8");
}

describe("canvas session overview current owner boundary", () => {
  it("会话概览 DTO 不得回绕 agentRuntime compat 根 barrel", () => {
    for (const relativePath of CANVAS_SESSION_OVERVIEW_SOURCES) {
      expect(readSource(relativePath), relativePath).not.toContain(
        'from "@/lib/api/agentRuntime"',
      );
    }
  });

  it("turn 与 action DTO 必须直接从聊天域 current types 获取", () => {
    expect(
      readSource(
        "src/components/agent/chat/components/CanvasSessionOverviewPanel.tsx",
      ),
    ).toContain('from "../types"');
  });
});
