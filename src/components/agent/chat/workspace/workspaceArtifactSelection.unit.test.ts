import { describe, expect, it } from "vitest";
import type { Artifact, ArtifactType } from "@/lib/artifact/types";
import { resolveWorkspaceSelectedArtifactIdCorrection } from "./workspaceArtifactSelection";

function artifact(id: string, type: ArtifactType = "code"): Artifact {
  return {
    id,
    type,
    title: id,
    content: "",
    status: "complete",
    meta: {},
    position: { start: 0, end: 0 },
    createdAt: 1,
    updatedAt: 1,
  };
}

function baseInput(
  overrides: Partial<
    Parameters<typeof resolveWorkspaceSelectedArtifactIdCorrection>[0]
  > = {},
): Parameters<typeof resolveWorkspaceSelectedArtifactIdCorrection>[0] {
  const selectedArtifact = artifact("artifact-1");
  return {
    activeTheme: "general",
    artifacts: [selectedArtifact],
    selectedArtifact,
    selectedArtifactId: selectedArtifact.id,
    defaultSelectedArtifactId: selectedArtifact.id,
    preferGeneralCanvasFilePreview: false,
    ...overrides,
  };
}

describe("resolveWorkspaceSelectedArtifactIdCorrection", () => {
  it("非 general 主题应清空选中的 artifact", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({ activeTheme: "image" }),
      ),
    ).toBeNull();
  });

  it("已清空时不重复更新非 general 主题", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({ activeTheme: "image", selectedArtifactId: null }),
      ),
    ).toBeUndefined();
  });

  it("General 文件预览优先时应清空 artifact 选中态", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({ preferGeneralCanvasFilePreview: true }),
      ),
    ).toBeNull();
  });

  it("没有 artifacts 时应清空 artifact 选中态", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({ artifacts: [], selectedArtifact: null }),
      ),
    ).toBeNull();
  });

  it("没有 selectedArtifact 时应回退到默认 artifact", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({
          selectedArtifact: null,
          selectedArtifactId: "missing",
          defaultSelectedArtifactId: "artifact-1",
        }),
      ),
    ).toBe("artifact-1");
  });

  it("选中 artifact 已不存在时应回退到默认 artifact", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(
        baseInput({
          artifacts: [artifact("artifact-2")],
          selectedArtifact: artifact("artifact-1"),
          selectedArtifactId: "artifact-1",
          defaultSelectedArtifactId: "artifact-2",
        }),
      ),
    ).toBe("artifact-2");
  });

  it("当前选中 artifact 仍存在时不更新", () => {
    expect(
      resolveWorkspaceSelectedArtifactIdCorrection(baseInput()),
    ).toBeUndefined();
  });
});
