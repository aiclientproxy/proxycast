import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const CURRENT_MEMORY_STORE_METHODS = [
  "memoryStore/list",
  "memoryStore/read",
  "memoryStore/search",
  "memoryStore/addNote",
  "memoryStore/consolidate",
  "memoryStore/review/list",
  "memoryStore/review/resolve",
  "memoryStore/health",
  "memory/reset",
  "memoryStore/index/rebuild",
];

const CURRENT_MEMORY_STORE_METHOD_CONSTANTS = [
  "METHOD_MEMORY_STORE_LIST",
  "METHOD_MEMORY_STORE_READ",
  "METHOD_MEMORY_STORE_SEARCH",
  "METHOD_MEMORY_STORE_ADD_NOTE",
  "METHOD_MEMORY_STORE_CONSOLIDATE",
  "METHOD_MEMORY_STORE_REVIEW_LIST",
  "METHOD_MEMORY_STORE_REVIEW_RESOLVE",
  "METHOD_MEMORY_STORE_HEALTH",
  "METHOD_MEMORY_RESET",
  "METHOD_MEMORY_STORE_INDEX_REBUILD",
];

const CURRENT_MEMORY_STORE_FRONTEND_EXPORTS = [
  "listMemoryStore",
  "readMemoryStore",
  "searchMemoryStore",
  "addMemoryStoreNote",
  "consolidateMemoryStore",
  "listMemoryStoreReviewNotes",
  "resolveMemoryStoreReviewNote",
  "getMemoryStoreHealth",
  "resetMemory",
  "rebuildMemoryStoreIndex",
];

const RETIRED_MEMORY_API_SNIPPETS = [
  "safeInvoke(",
  "unifiedMemory",
  "unified_memory",
  "memoryRuntime",
  "memory_runtime",
  "MemoryPage",
  "components/memory",
  "saveSceneAppExecutionAsInspiration",
  "messageInspirationDraft",
  "sceneAppExecutionInspirationDraft",
  "memoryStore/reset",
  "resetMemoryStore",
  "METHOD_MEMORY_STORE_RESET",
];

const RETIRED_MEMORY_FILES = [
  "src/lib/api/memory.ts",
  "src/lib/api/memory.test.ts",
  "src/lib/api/unifiedMemory.ts",
  "src/lib/api/unifiedMemory.test.ts",
  "src/lib/api/memoryRuntime.ts",
  "src/lib/api/memoryRuntime.test.ts",
  "src/lib/api/memoryRuntimeTypes.ts",
  "src/lib/runtimeMemoryPrefetchHistory.ts",
  "src/lib/runtimeMemoryPrefetchHistory.test.ts",
  "src/components/memory/index.ts",
  "src/components/memory/MemoryPage.tsx",
  "src/components/memory/MemoryPage.test.tsx",
  "src/components/memory/inspirationProjection.ts",
  "src/components/memory/memoryLayerMetrics.ts",
  "src/components/memory/memoryLayerMetrics.test.ts",
  "src/components/memory/MemoryCuratedTaskSuggestionPanel.tsx",
  "src/components/memory/MemoryCuratedTaskSuggestionPanel.test.tsx",
  "src/components/memory/MemoryFeedback.tsx",
  "src/components/memory/FeedbackStats.tsx",
  "src/components/agent/chat/components/AgentThreadMemoryPrefetchPreview.tsx",
  "src/components/agent/chat/components/AgentThreadMemoryPrefetchPreview.test.tsx",
  "src/components/agent/chat/components/AgentThreadMemoryPrefetchBaselineCard.tsx",
  "src/components/agent/chat/utils/messageInspirationDraft.ts",
  "src/components/agent/chat/utils/messageInspirationDraft.test.ts",
  "src/components/agent/chat/utils/saveSceneAppExecutionAsInspiration.ts",
  "src/components/agent/chat/utils/saveSceneAppExecutionAsInspiration.test.ts",
  "src/components/agent/chat/utils/sceneAppExecutionInspirationDraft.ts",
  "src/components/agent/chat/utils/sceneAppExecutionInspirationDraft.test.ts",
  "src/lib/api/memoryFeedback.ts",
  "src/lib/api/memoryFeedback.test.ts",
  "lime-rs/crates/app-server/src/local_data_source/unified_memory.rs",
  "lime-rs/crates/app-server/src/processor/unified.rs",
  "lime-rs/crates/memory/Cargo.toml",
  "lime-rs/crates/memory/migrations/001_unified_memory.sql",
  "lime-rs/crates/memory/migrations/003_feedback.sql",
  "lime-rs/crates/memory/src/extractor.rs",
  "lime-rs/crates/memory/src/feedback.rs",
  "lime-rs/crates/memory/src/gatekeeper.rs",
  "lime-rs/crates/memory/src/lib.rs",
  "lime-rs/crates/memory/src/migration.rs",
  "lime-rs/crates/memory/src/migrations/mod.rs",
  "lime-rs/crates/memory/src/migrations/v1_unified_memory.rs",
  "lime-rs/crates/memory/src/migrations/v1_unified_memory.sql",
  "lime-rs/crates/memory/src/models/mod.rs",
  "lime-rs/crates/memory/src/models/unified.rs",
  "lime-rs/crates/memory/src/search.rs",
];

const RETIRED_MEMORY_DIRECTORIES = [
  "src/components/memory",
  "lime-rs/crates/memory",
];

const I18N_AGENT_RESOURCE_FILES = [
  "src/i18n/resources/en-US/agent.json",
  "src/i18n/resources/zh-CN/agent.json",
  "src/i18n/resources/zh-TW/agent.json",
  "src/i18n/resources/ja-JP/agent.json",
  "src/i18n/resources/ko-KR/agent.json",
];

function readRepoFile(path: string): string {
  return readFileSync(resolve(cwd(), path), "utf8");
}

function repoPathExists(path: string): boolean {
  return existsSync(resolve(cwd(), path));
}

function expectSnippetsAbsent(source: string, snippets: string[]): void {
  for (const snippet of snippets) {
    expect(source).not.toContain(snippet);
  }
}

describe("memoryStore current App Server boundary", () => {
  it("前端 memoryStore API 固定走 App Server current method", () => {
    const source = readRepoFile("src/lib/api/memoryStore.ts");

    expect(source).toContain("new AppServerClient()");
    for (const methodConstant of CURRENT_MEMORY_STORE_METHOD_CONSTANTS) {
      expect(source).toContain(methodConstant);
    }
    for (const exportName of CURRENT_MEMORY_STORE_FRONTEND_EXPORTS) {
      expect(source).toContain(`export async function ${exportName}`);
    }
    expectSnippetsAbsent(source, RETIRED_MEMORY_API_SNIPPETS);
  });

  it("App Server protocol / client 只暴露 memoryStore current 方法", () => {
    const sources = [
      readRepoFile(
        "packages/app-server-client/src/generated/protocol-types.ts",
      ),
      readRepoFile("packages/app-server-client/src/index.ts"),
      readRepoFile(
        "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
      ),
      readRepoFile("lime-rs/crates/app-server/src/processor/memory_store.rs"),
    ].join("\n");

    for (const method of CURRENT_MEMORY_STORE_METHODS) {
      expect(sources).toContain(`"${method}"`);
    }
    for (const methodConstant of CURRENT_MEMORY_STORE_METHOD_CONSTANTS) {
      expect(sources).toContain(methodConstant);
    }
    expectSnippetsAbsent(sources, [
      "unifiedMemory/",
      "memoryRuntime/",
      "unified_memory_",
      "memory_runtime_",
      "memoryStore/reset",
      "METHOD_MEMORY_STORE_RESET",
      "MemoryStoreReset",
    ]);
  });

  it("旧记忆与旧灵感库文件保持 deleted / forbidden-to-restore", () => {
    const restoredFiles = RETIRED_MEMORY_FILES.filter(repoPathExists);

    expect(restoredFiles).toEqual([]);
  });

  it("旧记忆与旧灵感库目录保持 deleted / forbidden-to-restore", () => {
    const restoredDirectories =
      RETIRED_MEMORY_DIRECTORIES.filter(repoPathExists);

    expect(restoredDirectories).toEqual([]);
  });

  it("旧 MemoryPage 记忆库 i18n 资源不得回流", () => {
    const restoredResourceFiles = I18N_AGENT_RESOURCE_FILES.filter((file) =>
      readRepoFile(file).includes('"memoryLibrary.'),
    );

    expect(restoredResourceFiles).toEqual([]);
  });

  it("旧 MemoryPage 不得恢复到页面路由或侧边栏入口", () => {
    const sources = [
      readRepoFile("src/types/page.ts"),
      readRepoFile("src/components/AppPageContent.tsx"),
      readRepoFile("src/lib/navigation/sidebarNav.ts"),
    ].join("\n");

    expect(sources).not.toContain('"memory"');
    expect(sources).not.toContain("MemoryPage");
    expect(sources).not.toContain("components/memory");
  });

  it("治理目录册封住旧 MemoryPage、旧 unifiedMemory 和旧 memoryRuntime surface", () => {
    const source = readRepoFile("src/lib/governance/legacySurfaceCatalog.json");

    expect(source).toContain('"memory-page-inspiration-library-surface"');
    expect(source).toContain('"memory-runtime-retired-gateway-surface"');
    expect(source).toContain('"memory-app-server-unified-retired-surface"');
    expect(source).toContain('"memory-feedback-frontend-surface"');
    expect(source).toContain('"memory-crate-retired-sqlite-surface"');
    expect(source).toContain('"classification": "dead"');
    for (const retiredFile of RETIRED_MEMORY_FILES) {
      expect(source).toContain(`"${retiredFile}"`);
    }
  });

  it("开发守卫和续测指南不得把旧记忆网关或旧灵感库当 current 入口", () => {
    const eslintSource = readRepoFile("eslint.config.js");
    const playwrightGuide = readRepoFile(
      "internal/aiprompts/playwright-e2e.md",
    );
    const memoryGuide = readRepoFile("internal/aiprompts/memory-compaction.md");

    expect(eslintSource).not.toContain(
      "通过 `src/lib/api/memory.ts` 暴露的网关函数调用",
    );
    expect(eslintSource).not.toContain(
      "通过 `src/lib/api/memoryRuntime.ts` 暴露的网关函数调用",
    );
    expect(eslintSource).not.toContain(
      "通过 `src/lib/api/memoryFeedback.ts` 暴露的网关函数调用",
    );
    expect(eslintSource).not.toContain(
      "通过 `src/lib/api/contextMemory.ts` 暴露的网关函数调用",
    );
    expect(eslintSource).not.toContain('"src/lib/api/memory.ts"');
    expect(eslintSource).not.toContain('"src/lib/api/memoryRuntime.ts"');
    expect(eslintSource).not.toContain('"src/lib/api/memoryFeedback.ts"');
    expect(eslintSource).not.toContain('"src/lib/api/contextMemory.ts"');

    expect(playwrightGuide).toContain("### 记忆设置验证");
    expect(playwrightGuide).toContain("设置");
    expect(playwrightGuide).toContain("日常记忆");
    expect(playwrightGuide).not.toContain("### 记忆工作台验证");
    expect(playwrightGuide).not.toContain("从左侧导航进入 `灵感库`");
    expect(playwrightGuide).not.toContain("打开 `长期记忆`");

    expect(memoryGuide).toContain("src/lib/api/memoryConfigTypes.ts");
    expect(memoryGuide).not.toContain("src/lib/api/memoryRuntimeTypes.ts");
  });
});
