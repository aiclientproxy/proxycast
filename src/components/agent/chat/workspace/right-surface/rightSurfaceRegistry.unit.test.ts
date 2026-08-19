import { describe, expect, it } from "vitest";
import {
  buildWorkspaceRightSurfaceDefinitions,
  getWorkspaceRightSurfaceSpec,
  WORKSPACE_RIGHT_SURFACE_SPECS,
} from "./rightSurfaceRegistry";
import type { WorkspaceRightSurfaceKind } from "./rightSurfaceTypes";

const EXPECTED_SURFACE_KINDS: WorkspaceRightSurfaceKind[] = [
  "workbench",
  "appSurface",
  "articleWorkspace",
  "expertInfo",
  "browser",
  "files",
  "shell",
  "harness",
  "trace",
];

describe("rightSurfaceRegistry", () => {
  it("应登记所有 Right Surface 骨架入口", () => {
    expect(WORKSPACE_RIGHT_SURFACE_SPECS.map((spec) => spec.kind)).toEqual(
      EXPECTED_SURFACE_KINDS,
    );
    expect(
      WORKSPACE_RIGHT_SURFACE_SPECS.every(
        (spec) => spec.exclusiveGroup === "workspaceRightSurface",
      ),
    ).toBe(true);
    expect(
      WORKSPACE_RIGHT_SURFACE_SPECS.every(
        (spec) => spec.collapseTarget === "topToolbar",
      ),
    ).toBe(true);
  });

  it("应按 kind 读取 surface 元数据", () => {
    expect(getWorkspaceRightSurfaceSpec("expertInfo")).toMatchObject({
      kind: "expertInfo",
      slot: "canvasPanel",
      openSources: ["user", "route", "runtime"],
    });
  });

  it("应只为已注册 renderer 的 surface 生成 definition", () => {
    const definitions = buildWorkspaceRightSurfaceDefinitions({
      expertInfo: () => "expert",
      shell: () => "shell",
    });

    expect(definitions.map((definition) => definition.kind)).toEqual([
      "expertInfo",
      "shell",
    ]);
    expect(definitions[0]?.render({ activeSurface: "expertInfo" })).toBe(
      "expert",
    );
  });
});
