/* global process */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import {
  collectWorkspaceArticlePatchRecordsFromArtifactLike,
  hasWorkspaceArticlePatchMetadata,
  isWorkspaceArticlePatchArtifactKind,
  isWorkspaceArticlePatchArtifactPath,
  readWorkspaceArticlePatchRecordFromMetadata,
  readWorkspaceArticleRecordFromMetadata,
} from "./workspaceArticleWorkspaceMetadata";

const CURRENT_PATCH = {
  objects: [{ ref: { kind: "articleDraft", id: "current" } }],
};
const PRODUCTION_SCAN_DIRS = ["src/components/agent/chat"] as const;
function collectProductionSourceFiles(dir: string): string[] {
  const entries = readdirSync(dir);
  const files: string[] = [];
  for (const entry of entries) {
    const fullPath = join(dir, entry);
    const stats = statSync(fullPath);
    if (stats.isDirectory()) {
      files.push(...collectProductionSourceFiles(fullPath));
      continue;
    }
    if (!fullPath.endsWith(".ts") && !fullPath.endsWith(".tsx")) {
      continue;
    }
    if (
      fullPath.endsWith(".test.ts") ||
      fullPath.endsWith(".test.tsx") ||
      fullPath.endsWith(".unit.test.ts") ||
      fullPath.endsWith(".unit.test.tsx")
    ) {
      continue;
    }
    files.push(fullPath);
  }
  return files;
}

describe("workspaceArticleWorkspaceMetadata", () => {
  it("读取 patch metadata 时应优先使用 current workspacePatch", () => {
    expect(
      readWorkspaceArticlePatchRecordFromMetadata({
        workspacePatch: CURRENT_PATCH,
      }),
    ).toBe(CURRENT_PATCH);
  });

  it("不得读取旧 contentFactoryWorkspacePatch 字段", () => {
    expect(
      readWorkspaceArticlePatchRecordFromMetadata({
        contentFactoryWorkspacePatch: CURRENT_PATCH,
      }),
    ).toBeNull();
  });

  it("读取 article workspace metadata 时不混入 patch fallback", () => {
    const articleWorkspace = { objectKind: "articleDraft" };
    expect(
      readWorkspaceArticleRecordFromMetadata({
        article_workspace: articleWorkspace,
        workspacePatch: CURRENT_PATCH,
      }),
    ).toBe(articleWorkspace);
  });

  it("应从 artifact-like 结构集中收集 patch 候选", () => {
    const contentPatch = {
      objects: [{ ref: { kind: "articleDraft", id: "content" } }],
    };
    expect(
      collectWorkspaceArticlePatchRecordsFromArtifactLike({
        meta: { workspace_patch: CURRENT_PATCH },
        artifact: {
          content: JSON.stringify(contentPatch),
        },
      }),
    ).toEqual([CURRENT_PATCH, contentPatch]);
  });

  it("只识别 current workspace patch kind", () => {
    expect(isWorkspaceArticlePatchArtifactKind("workspace_patch")).toBe(true);
    expect(
      isWorkspaceArticlePatchArtifactKind("content_factory.workspace_patch"),
    ).toBe(false);
    expect(isWorkspaceArticlePatchArtifactKind("articleDraft")).toBe(false);
  });

  it("只识别 current workspace patch path", () => {
    expect(
      isWorkspaceArticlePatchArtifactPath(
        ".lime/artifacts/article-workspace/workspace-patch.json",
      ),
    ).toBe(true);
    expect(
      isWorkspaceArticlePatchArtifactPath(
        "/workspace/demo/.lime/artifacts/content-factory/workspace-patch.json",
      ),
    ).toBe(true);
    expect(
      isWorkspaceArticlePatchArtifactPath(
        ".lime/artifacts/content-factory-workspace-patch.json",
      ),
    ).toBe(false);
    expect(
      isWorkspaceArticlePatchArtifactPath(
        "exports/article/workspace-patch.json",
      ),
    ).toBe(false);
  });

  it("生产代码不得读取旧 workspace patch metadata 字段", () => {
    const offenders: string[] = [];

    for (const dir of PRODUCTION_SCAN_DIRS) {
      for (const filePath of collectProductionSourceFiles(
        join(process.cwd(), dir),
      )) {
        const relativePath = relative(process.cwd(), filePath);
        const source = readFileSync(filePath, "utf8");
        const hasLegacyFieldWriteOrRead =
          /\.\s*contentFactoryWorkspacePatch\b/.test(source) ||
          /\bcontentFactoryWorkspacePatch\s*:/.test(source) ||
          /content_factory_workspace_patch/.test(source);
        if (hasLegacyFieldWriteOrRead) {
          offenders.push(relativePath);
        }
      }
    }

    expect(offenders).toEqual([]);
  });

  it("生产代码不得恢复旧 workspace patch raw artifact 命名", () => {
    const offenders: string[] = [];

    for (const dir of PRODUCTION_SCAN_DIRS) {
      for (const filePath of collectProductionSourceFiles(
        join(process.cwd(), dir),
      )) {
        const relativePath = relative(process.cwd(), filePath);
        const source = readFileSync(filePath, "utf8");
        const hasLegacyRawArtifactName =
          /content_factory\.workspace_patch/.test(source) ||
          /content-factory-workspace-patch/.test(source) ||
          /content-factory\/workspace-patch/.test(source);
        if (hasLegacyRawArtifactName) {
          offenders.push(relativePath);
        }
      }
    }

    expect(offenders).toEqual([]);
  });

  it("应通过统一 helper 识别 articleWorkspace / workspacePatch metadata", () => {
    expect(
      hasWorkspaceArticlePatchMetadata({ workspace_patch: CURRENT_PATCH }),
    ).toBe(true);
    expect(hasWorkspaceArticlePatchMetadata({ articleWorkspace: {} })).toBe(
      true,
    );
    expect(hasWorkspaceArticlePatchMetadata({ title: "普通产物" })).toBe(false);
  });
});
