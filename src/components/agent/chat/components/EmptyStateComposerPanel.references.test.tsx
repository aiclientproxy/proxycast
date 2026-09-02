import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  createGithubSearchServiceSkill,
  createSkillSelection,
  renderPanel,
} from "./EmptyStateComposerPanel.testFixtures";

function openKnowledgePanel(container: HTMLDivElement) {
  const plusTrigger = container.querySelector(
    '[data-testid="inputbar-plus-trigger"]',
  ) as HTMLButtonElement | null;
  expect(plusTrigger).toBeTruthy();

  act(() => {
    plusTrigger?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  const knowledgeTrigger = document.body.querySelector(
    '[data-testid="inputbar-plus-knowledge"]',
  ) as HTMLButtonElement | null;
  expect(knowledgeTrigger).toBeTruthy();

  act(() => {
    knowledgeTrigger?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  const knowledgePanel = document.body.querySelector(
    '[data-testid="inputbar-plus-panel-knowledge"]',
  ) as HTMLElement | null;
  expect(knowledgePanel).toBeTruthy();
  if (!knowledgePanel) {
    throw new Error("Expected knowledge panel to be open");
  }
  return knowledgePanel;
}

describe("EmptyStateComposerPanel", () => {
  it("首页空态输入区应从加号菜单打开项目资料中枢", () => {
    const container = renderPanel({
      knowledgePackSelection: {
        enabled: false,
        packName: "team-notes",
        workingDir: "workspace-root",
        label: "团队资料",
        status: "ready",
      },
      knowledgePackOptions: [
        {
          packName: "team-notes",
          label: "团队资料",
          status: "ready",
          defaultForWorkspace: true,
        },
      ],
      onToggleKnowledgePack: vi.fn(),
      onSelectKnowledgePack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
    });

    const knowledgePanel = openKnowledgePanel(container);

    expect(
      knowledgePanel.querySelector('[data-testid="inputbar-knowledge-hub"]'),
    ).toBeTruthy();
    expect(
      knowledgePanel.querySelector(
        '[data-testid="inputbar-knowledge-hub-dismiss"]',
      ),
    ).toBeNull();
    expect(knowledgePanel.textContent).toContain("选择项目资料");
  });

  it("@资料兼容触发时不应渲染普通命令标签", () => {
    const container = renderPanel({
      activeCapability: {
        kind: "builtin_command",
        command: {
          key: "knowledge_pack",
          label: "资料",
          mentionLabel: "资料",
          commandPrefix: "@资料",
          description: "打开项目资料。",
          aliases: [],
        },
      },
      knowledgePackSelection: {
        enabled: false,
        packName: "team-notes",
        workingDir: "workspace-root",
        label: "团队资料",
        status: "ready",
      },
      onToggleKnowledgePack: vi.fn(),
      onStartKnowledgeOrganize: vi.fn(),
      onManageKnowledgePacks: vi.fn(),
    });

    expect(
      container.querySelector('[data-testid="inputbar-builtin-command-badge"]'),
    ).toBeNull();
    expect(
      openKnowledgePanel(container).querySelector(
        '[data-testid="inputbar-knowledge-hub"]',
      ),
    ).toBeTruthy();
  });

  it("首页空态输入区应展示本地路径 chip 并支持移除", () => {
    const onRemovePathReference = vi.fn();
    const container = renderPanel({
      pathReferences: [
        {
          id: "dir:/Users/lime/Downloads",
          path: "/Users/lime/Downloads",
          name: "Downloads",
          isDir: true,
          size: null,
          mimeType: null,
          source: "file_manager",
        },
      ],
      onRemovePathReference,
    });

    expect(
      container.querySelector('[data-testid="inputbar-path-reference-chip"]')
        ?.textContent,
    ).toContain("Downloads");
    expect(container.textContent).not.toContain("/Users/lime/Downloads");

    const removeButton = container.querySelector(
      'button[aria-label="移除路径 Downloads"]',
    ) as HTMLButtonElement | null;

    act(() => {
      removeButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onRemovePathReference).toHaveBeenCalledWith(
      "dir:/Users/lime/Downloads",
    );
  });

  it("首页空态输入区的文本文件 chip 应支持设为项目资料", () => {
    const onImportPathReferenceAsKnowledge = vi.fn();
    const reference = {
      id: "file:/Users/lime/brief.md",
      path: "/Users/lime/brief.md",
      name: "brief.md",
      isDir: false,
      size: 128,
      mimeType: "text/markdown",
      source: "file_manager" as const,
    };
    const container = renderPanel({
      pathReferences: [reference],
      onImportPathReferenceAsKnowledge,
    });

    expect(container.textContent).toContain("brief.md");
    expect(container.textContent).toContain("本地文件");
    expect(container.textContent).not.toContain("/Users/lime/brief.md");

    const importButton = container.querySelector(
      'button[aria-label="设为项目资料 brief.md"]',
    ) as HTMLButtonElement | null;
    expect(importButton).toBeTruthy();

    act(() => {
      importButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onImportPathReferenceAsKnowledge).toHaveBeenCalledWith(reference);
  });

  it("存在当前带入的灵感时，应在输入区顶部展示被带入的参考对象", () => {
    const container = renderPanel({
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入灵感",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "保留轻盈但专业的表达。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    expect(container.textContent).toContain("参考");
    expect(container.textContent).toContain("品牌风格样本");
  });

  it("首页输入区应直接聚焦输入主路径，不再展示额外起始卡片", () => {
    const container = renderPanel();
    const textarea = container.querySelector("textarea");

    expect(
      container.querySelector('[data-testid="empty-state-kickoff-guide"]'),
    ).toBeNull();
    expect(textarea).toBeTruthy();
  });

  it("带入项目参考时，应继续在输入区顶部展示参考对象", () => {
    const container = renderPanel({
      projectId: "project-1",
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入灵感",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "保留轻盈但专业的表达。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    expect(
      container.querySelector('[data-testid="empty-state-kickoff-guide"]'),
    ).toBeNull();
    expect(container.textContent).toContain("参考");
    expect(container.textContent).toContain("品牌风格样本");
  });

  it("首页输入区已激活复盘模板时，应把 结果基线引用继续透传给 badge", () => {
    const container = renderPanel({
      activeCapability: {
        kind: "curated_task",
        task: {
          id: "account-project-review",
          title: "复盘这个账号/项目",
          summary: "围绕已有结果判断当前该怎么推进。",
          outputHint: "判断摘要 + 下一步建议",
          resultDestination: "判断摘要会先回到当前内容。",
          categoryLabel: "判断与优化",
          prompt: "请帮我判断这个账号或项目当前该怎么推进",
          requiredInputs: ["账号或项目目标", "已有结果或数据"],
          requiredInputFields: [],
          optionalReferences: ["最近内容链接"],
          outputContract: ["判断摘要", "下一轮动作建议"],
          followUpActions: ["继续做趋势摘要"],
          badge: "结果模板",
          actionLabel: "进入生成",
          statusLabel: "可直接开始",
          statusTone: "emerald",
          recentUsedAt: null,
          isRecent: false,
        },
        referenceEntries: [
          {
            id: "memory:content-pack:run:1",
            sourceKind: "memory",
            title: "AI 内容周报",
            summary: "当前已有一轮项目结果，可直接作为复盘基线。",
            category: "experience",
            categoryLabel: "成果",
            tags: ["复盘"],
          },
        ],
      },
    });

    const badge = container.querySelector(
      '[data-testid="empty-state-curated-task-badge"]',
    ) as HTMLDivElement | null;

    expect(badge?.dataset.referenceCount).toBe("1");
    expect(badge?.dataset.firstSourceKind).toBe("memory");
  });

  it("通用对话且存在站点型 service skill 时不应再展示首页专属提示按钮", () => {
    const container = renderPanel({
      input: "",
      isGeneralTheme: true,
      skillSelection: createSkillSelection({
        serviceSkills: [createGithubSearchServiceSkill()],
      }),
    });

    const hint = container.querySelector(
      '[data-testid="empty-state-site-skill-natural-hint"]',
    );

    expect(hint).toBeFalsy();
  });
});
