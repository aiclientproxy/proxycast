import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const { mockGetDefaultProject } = vi.hoisted(() => ({
  mockGetDefaultProject: vi.fn(),
}));
const {
  mockAddMemoryStoreNote,
  mockConsolidateMemoryStore,
  mockGetMemoryStoreHealth,
  mockListMemoryStore,
  mockListMemoryStoreReviewNotes,
  mockReadMemoryStore,
  mockRebuildMemoryStoreIndex,
  mockResolveMemoryStoreReviewNote,
  mockResetMemory,
} = vi.hoisted(() => ({
  mockAddMemoryStoreNote: vi.fn(),
  mockConsolidateMemoryStore: vi.fn(),
  mockGetMemoryStoreHealth: vi.fn(),
  mockListMemoryStore: vi.fn(),
  mockListMemoryStoreReviewNotes: vi.fn(),
  mockReadMemoryStore: vi.fn(),
  mockRebuildMemoryStoreIndex: vi.fn(),
  mockResolveMemoryStoreReviewNote: vi.fn(),
  mockResetMemory: vi.fn(),
}));

vi.mock("@/lib/api/project", () => ({
  getDefaultProject: mockGetDefaultProject,
}));

vi.mock("@/lib/api/memoryStore", () => ({
  addMemoryStoreNote: mockAddMemoryStoreNote,
  consolidateMemoryStore: mockConsolidateMemoryStore,
  getMemoryStoreHealth: mockGetMemoryStoreHealth,
  listMemoryStore: mockListMemoryStore,
  listMemoryStoreReviewNotes: mockListMemoryStoreReviewNotes,
  readMemoryStore: mockReadMemoryStore,
  rebuildMemoryStoreIndex: mockRebuildMemoryStoreIndex,
  resolveMemoryStoreReviewNote: mockResolveMemoryStoreReviewNote,
  resetMemory: mockResetMemory,
}));

import { MemoryStoreStatusPanel } from "./MemoryStoreStatusPanel";

interface Mounted {
  container: HTMLDivElement;
  root: Root;
}

const mounted: Mounted[] = [];

function renderPanel(): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <MemoryStoreStatusPanel
        vectorSearchEnabled={false}
        memoryStatusDescriptionKey="settings.memory.embedding.status.fullTextOnly"
        showMessage={vi.fn()}
      />,
    );
  });
  mounted.push({ container, root });
  return container;
}

async function flushEffects(times = 3) {
  for (let index = 0; index < times; index += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find(
    (element) => element.textContent?.includes(text),
  );
  if (!button) {
    throw new Error(`未找到按钮: ${text}`);
  }
  return button as HTMLButtonElement;
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  await changeLimeLocale("en-US");

  mockGetMemoryStoreHealth.mockResolvedValue({
    rootScope: "global",
    rootPath: "/data/memories",
    initialized: true,
    fileCount: 2,
    totalBytes: 1536,
    summaryExists: true,
    summaryBytes: 512,
    memoryExists: true,
    memoryBytes: 1024,
    notesCount: 1,
  });
  mockGetDefaultProject.mockResolvedValue({
    id: "default",
    name: "Default workspace",
    workspaceType: "general",
    rootPath: "/repo/default",
    isDefault: true,
    createdAt: 1,
    updatedAt: 1,
    isFavorite: false,
    isArchived: false,
    tags: [],
  });
  mockListMemoryStoreReviewNotes.mockResolvedValue({
    rootScope: "global",
    rootPath: "/data/memories",
    notes: [],
    truncated: false,
    nextCursor: null,
  });
  mockListMemoryStore.mockResolvedValue({
    rootScope: "workspace",
    path: "rollout_summaries",
    entries: [],
    truncated: false,
    nextCursor: null,
  });
  mockReadMemoryStore.mockResolvedValue({
    path: "rollout_summaries/20260619T010203Z-handoff.md",
    startLineNumber: 1,
    content: "",
    truncated: false,
    citation: {
      path: "rollout_summaries/20260619T010203Z-handoff.md",
      startLineNumber: 1,
      endLineNumber: 1,
    },
  });
  mockRebuildMemoryStoreIndex.mockResolvedValue({
    rootScope: "global",
    rootPath: "/data/memories",
    manifestPath: "index/manifest.json",
    schemaVersion: "memory-index-manifest/v1",
    sourceFileCount: 2,
    sourceTotalBytes: 1536,
    sourceChecksum: "feedface",
    indexedAt: "2026-06-19T10:00:00Z",
    rebuilt: true,
  });
  mockConsolidateMemoryStore.mockResolvedValue({
    rootScope: "workspace",
    rootPath: "/repo/default/.lime/memories",
    processedNotes: 1,
    skippedNotes: 0,
    archivedNotes: 1,
    memoryPath: "MEMORY.md",
    summaryPath: "memory_summary.md",
    warnings: [],
    updated: true,
  });
});

afterEach(async () => {
  while (mounted.length > 0) {
    const target = mounted.pop();
    if (!target) {
      break;
    }
    act(() => {
      target.root.unmount();
    });
    target.container.remove();
  }
  vi.restoreAllMocks();
  await changeLimeLocale("zh-CN");
});

describe("MemoryStoreStatusPanel rollout candidates", () => {
  it("应从日常记忆面板重建派生索引并刷新状态", async () => {
    const container = renderPanel();
    await flushEffects();

    await act(async () => {
      findButton(container, "Rebuild index").click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(mockRebuildMemoryStoreIndex).toHaveBeenCalledWith({
      scope: "global",
    });
    expect(mockGetMemoryStoreHealth).toHaveBeenCalledTimes(2);
  });

  it("应从日常记忆面板整理修正笔记并刷新状态", async () => {
    const container = renderPanel();
    await flushEffects();

    await act(async () => {
      findButton(container, "Consolidate notes").click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(mockConsolidateMemoryStore).toHaveBeenCalledWith({
      scope: "global",
    });
    expect(mockGetMemoryStoreHealth).toHaveBeenCalledTimes(2);
    expect(mockListMemoryStoreReviewNotes).toHaveBeenCalledTimes(2);
    expect(mockListMemoryStore).toHaveBeenCalledTimes(2);
  });

  it("应展示默认工作区运行摘要候选并可显式整理", async () => {
    mockListMemoryStore.mockResolvedValue({
      rootScope: "workspace",
      path: "rollout_summaries",
      entries: [
        {
          path: "rollout_summaries/processed/old.md",
          entryType: "file",
          size: 64,
          modifiedAt: 1,
        },
        {
          path: "rollout_summaries/20260619T010203Z-handoff.md",
          entryType: "file",
          size: 256,
          modifiedAt: 3,
        },
      ],
      truncated: false,
      nextCursor: null,
    });
    mockReadMemoryStore.mockResolvedValue({
      path: "rollout_summaries/20260619T010203Z-handoff.md",
      startLineNumber: 1,
      content: [
        "# Handoff summary",
        "",
        "## Export Evidence",
        "- exportKind: `handoff_bundle`",
        "- exportRoot: `.lime/harness/sessions/session-1`",
        "- exportedAt: `2026-06-19T10:00:00Z`",
        "",
        "## Referenced Artifacts",
        "- Handoff Draft `.app-server/artifacts/handoff.md` (markdown)",
        "",
        "## Candidate Metadata",
        "- source: `agentSession/handoffBundle/export`",
        "- status: `candidate`",
      ].join("\n"),
      truncated: false,
      citation: {
        path: "rollout_summaries/20260619T010203Z-handoff.md",
        startLineNumber: 1,
        endLineNumber: 13,
      },
    });

    const container = renderPanel();
    await flushEffects();

    const bodyText = document.body.textContent ?? "";
    expect(bodyText).toContain("Run summary candidates");
    expect(bodyText).toContain("Default workspace");
    expect(bodyText).toContain("Handoff summary");
    expect(bodyText).toContain("agentSession/handoffBundle/export");
    expect(bodyText).toContain(".lime/harness/sessions/session-1");
    expect(bodyText).toContain("Handoff Draft");
    expect(bodyText).not.toContain("processed/old.md");
    expect(mockListMemoryStore).toHaveBeenCalledWith({
      scope: "workspace",
      workspaceRoot: "/repo/default",
      path: "rollout_summaries",
      maxResults: 20,
    });
    expect(mockReadMemoryStore).toHaveBeenCalledWith({
      scope: "workspace",
      workspaceRoot: "/repo/default",
      path: "rollout_summaries/20260619T010203Z-handoff.md",
      maxLines: 120,
      maxTokens: 2048,
    });

    await act(async () => {
      findButton(container, "Consolidate candidates").click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(mockConsolidateMemoryStore).toHaveBeenCalledWith({
      scope: "workspace",
      workspaceRoot: "/repo/default",
    });
  });

  it("运行摘要候选加载失败时不阻断主面板", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockListMemoryStore.mockRejectedValueOnce(new Error("store unavailable"));

    renderPanel();
    await flushEffects();

    const bodyText = document.body.textContent ?? "";
    expect(bodyText).toContain("Everyday memory status");
    expect(bodyText).toContain("Run summary candidates are unavailable.");
    expect(bodyText).toContain("No memory notes need review.");
  });
});
