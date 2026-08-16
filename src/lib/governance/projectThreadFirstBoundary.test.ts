import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { basename, join, relative } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = cwd();

const PROJECT_THREAD_SCHEMA_NAME_PATTERN =
  /^(?:AgentSession|MemoryStore|SessionFile|Workspace|ConversationImportThread|ImportedThread)/u;

const FORBIDDEN_AGENT_FIRST_ID_FIELD_PATTERN =
  /(^|[^A-Za-z0-9_])(agent_id|expert_id|agentId|expertId)(?![A-Za-z0-9_])/gu;

const PROJECT_THREAD_IDENTITY_BOUNDARY_FILES = [
  "lime-rs/crates/core/src/database/schema.rs",
  "lime-rs/crates/core/src/database/agent_session_repository.rs",
  "lime-rs/crates/thread-store/src/session_record.rs",
  "lime-rs/crates/thread-store/src/session_repository.rs",
  "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
  "lime-rs/crates/app-server/src/runtime/session_list_scope.rs",
  "lime-rs/crates/app-server/src/runtime/memory.rs",
  "lime-rs/crates/app-server/src/runtime/memory_prompt.rs",
  "lime-rs/crates/app-server/src/memory_store.rs",
  "lime-rs/crates/app-server/src/processor/agent_session.rs",
  "lime-rs/crates/app-server/src/processor/memory_store.rs",
  "lime-rs/crates/app-server/src/local_data_source/impls/sessions.rs",
  "lime-rs/crates/app-server/src/local_data_source/impls/memory.rs",
  "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
  "lime-rs/crates/app-server-protocol/src/protocol/v0/model.rs",
  "packages/app-server-client/src/protocol.ts",
  "packages/app-server-client/src/index.ts",
  "packages/app-server-client/src/generated/protocol-types.ts",
  "src/lib/api/agentRuntime/sessionClient.ts",
  "src/lib/api/agentRuntime/appServerSessionClient.ts",
  "src/lib/api/agentRuntime/threadClient.ts",
  "src/lib/api/memoryStore.ts",
  "src/lib/api/memoryConfigTypes.ts",
];

const EXPERT_THREAD_FIRST_SURFACE_FILES = [
  "src/types/page.ts",
  "src/components/experts/ExpertPlazaPage.tsx",
  "src/components/agent/chat/AgentChatWorkspace.tsx",
  "src/components/agent/chat/experts/ExpertInfoPanel.tsx",
  "src/components/agent/chat/workspace/workspaceExpertMetadata.ts",
  "src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.ts",
  "src/features/experts/expertAgentInstances.ts",
  "src/i18n/resources/zh-CN/agentExperts.json",
  "src/i18n/resources/zh-TW/agentExperts.json",
  "src/i18n/resources/en-US/agentExperts.json",
  "src/i18n/resources/ja-JP/agentExperts.json",
  "src/i18n/resources/ko-KR/agentExperts.json",
];

const FORBIDDEN_EXPERT_STABLE_SESSION_PATTERNS = [
  "resume_or_create",
  "latestSessionId",
  "expert-new-thread",
  "agentExperts.actions.continue",
  "agentExperts.actions.newThread",
] as const;

const SKILL_THREAD_FIRST_SURFACE_FILES = [
  "src/components/skills/useSkillsWorkspaceProject.ts",
  "src/components/skills/SkillsWorkspacePage.tsx",
  "src/components/skills/workspaceSkillRuntimeLaunch.ts",
  "src/components/agent/chat/workspace/useInitialPendingServiceSkillLaunchRuntime.ts",
  "src/lib/api/skill-execution.ts",
];

const FORBIDDEN_SKILL_FIRST_RUNTIME_PATTERNS = [
  "getOrCreateDefaultProject",
  "executeSkill",
  "execute_skill",
  "skillSessionId",
  "skill_session_id",
] as const;

const PLUGIN_CURRENT_SURFACE_FILES = [
  "src/lib/api/pluginCatalog.ts",
  "src/components/agent/chat/workspace/useWorkspacePluginCatalogSuggestions.ts",
  "src/components/agent/chat/workspace/useWorkspaceSendActions.ts",
  "src/components/agent/chat/workspace/useWorkspaceRightSurfacePendingRuntime.ts",
  "src/components/agent/chat/workspace/WorkspacePluginSurface.tsx",
];

const FORBIDDEN_RENDERER_PLUGIN_RUNTIME_PATTERNS = [
  "@/features/plugin",
  "@/lib/api/plugins",
  "plugin_activation",
  "plugin_activation_intent",
  "runtime_authorization",
] as const;

const AUTOMATION_THREAD_FIRST_SURFACE_FILES = [
  "lime-rs/crates/app-server/src/automation_execution.rs",
  "lime-rs/crates/app-server/src/local_data_source/automation.rs",
  "src/lib/api/automation.ts",
] as const;

const FORBIDDEN_AUTOMATION_FIRST_RUNTIME_PATTERNS = [
  "automation-session-",
  "automation-thread-",
] as const;

const SUBAGENT_THREAD_FIRST_SURFACE_FILES = [
  "lime-rs/crates/app-server/src/runtime/evidence_provider.rs",
  "lime-rs/crates/app-server/src/runtime/tests/evidence_exports/team_facts.rs",
  "src/lib/api/agentRuntime/appServerCanonicalItemReader.ts",
  "src/lib/api/agentRuntime/threadClient.ts",
  "src/components/agent/chat/AgentChatWorkspace.tsx",
  "src/components/agent/chat/workspace/useWorkspaceSubagentNavigationRuntime.ts",
  "src/components/agent/chat/projection/threadItemProjection.ts",
  "packages/agent-runtime-projection/src/threadItems.ts",
  "src/components/agent/chat/components/AgentThreadTimelineItemRenderers.tsx",
  "src/components/agent/chat/projection/teamControlProjection.ts",
] as const;

const FORBIDDEN_SUBAGENT_FIRST_HISTORY_PATTERNS = [
  "subagentHistory",
  "subagent_history",
  "childSubagentHistory",
  "subagentSessionHistory",
] as const;

function repoPath(path: string): string {
  return join(REPO_ROOT, path);
}

function repoRelative(path: string): string {
  return relative(REPO_ROOT, path).replace(/\\/g, "/");
}

function collectFiles(
  root: string,
  predicate: (file: string) => boolean,
): string[] {
  const absoluteRoot = repoPath(root);
  const files: string[] = [];
  for (const entry of readdirSync(absoluteRoot)) {
    const absolutePath = join(absoluteRoot, entry);
    const stats = statSync(absolutePath);
    if (stats.isDirectory()) {
      files.push(...collectFiles(repoRelative(absolutePath), predicate));
      continue;
    }
    if (predicate(absolutePath)) {
      files.push(absolutePath);
    }
  }
  return files;
}

function projectThreadSchemaFiles(): string[] {
  return collectFiles(
    "lime-rs/crates/app-server-protocol/schema/json/v0",
    (file) =>
      file.endsWith(".json") &&
      PROJECT_THREAD_SCHEMA_NAME_PATTERN.test(basename(file)),
  );
}

function forbiddenAgentFirstFieldHits(file: string): string[] {
  const source = readFileSync(file, "utf8");
  return source
    .split(/\r?\n/u)
    .flatMap((line, index) =>
      Array.from(line.matchAll(FORBIDDEN_AGENT_FIRST_ID_FIELD_PATTERN)).map(
        (match) => `${repoRelative(file)}:${index + 1}: ${match[2]}`,
      ),
    );
}

function readJsonFile(path: string): unknown {
  return JSON.parse(readFileSync(repoPath(path), "utf8")) as unknown;
}

describe("Project / Thread-first boundary", () => {
  it("session / memory current schema 不得新增 Agent-first 主索引字段", () => {
    const checkedFiles = [
      ...projectThreadSchemaFiles(),
      ...PROJECT_THREAD_IDENTITY_BOUNDARY_FILES.map(repoPath),
    ].filter((file) => existsSync(file));

    expect(
      checkedFiles.length,
      "Project / Thread-first 守卫必须覆盖 App Server session、memory、workspace schema 与 current API 边界",
    ).toBeGreaterThan(20);

    const hits = checkedFiles.flatMap(forbiddenAgentFirstFieldHits);

    expect(
      hits,
      "expert / agent 只能作为 businessObjectRef.metadata、runtime metadata、thread item 或 evidence metadata；不得成为 session / memory schema 的一等主索引",
    ).toEqual([]);
  });

  it("AgentSessionStartParams 只通过 businessObjectRef.metadata 承载能力来源", () => {
    const schema = readJsonFile(
      "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionStartParams.json",
    ) as {
      $defs?: {
        BusinessObjectRef?: {
          properties?: Record<string, unknown>;
          required?: string[];
        };
      };
      properties?: Record<string, unknown>;
      required?: string[];
    };
    const sessionProperties = Object.keys(schema.properties || {});
    const businessObjectProperties =
      schema.$defs?.BusinessObjectRef?.properties || {};

    expect(sessionProperties).toEqual(
      expect.arrayContaining([
        "appId",
        "workspaceId",
        "sessionId",
        "threadId",
        "businessObjectRef",
      ]),
    );
    expect(sessionProperties).not.toEqual(
      expect.arrayContaining(["agent_id", "expert_id", "agentId", "expertId"]),
    );
    expect(Object.keys(businessObjectProperties)).toEqual(
      expect.arrayContaining(["id", "kind", "metadata", "title", "uri"]),
    );
    expect(businessObjectProperties.metadata).toBe(true);
  });

  it("MemoryStoreScope 只能按 global / workspace 组织长期记忆", () => {
    const schema = readJsonFile(
      "lime-rs/crates/app-server-protocol/schema/json/v0/MemoryStoreScope.json",
    ) as { enum?: string[] };

    expect(schema.enum).toEqual(["global", "workspace"]);
  });

  it("专家入口不得恢复稳定专家会话或继续对话分叉", () => {
    const hits = EXPERT_THREAD_FIRST_SURFACE_FILES.flatMap((file) => {
      const source = readFileSync(repoPath(file), "utf8");
      return FORBIDDEN_EXPERT_STABLE_SESSION_PATTERNS.flatMap((pattern) =>
        source.includes(pattern) ? [`${file}: ${pattern}`] : [],
      );
    });

    expect(
      hits,
      "专家只能作为当前 Project / Thread 的 profile metadata；不得恢复专家 latestSessionId、resume_or_create 或继续对话 UI 分叉",
    ).toEqual([]);
  });

  it("Skills 入口不得恢复默认项目孤岛或独立 Skill 执行流", () => {
    const hits = SKILL_THREAD_FIRST_SURFACE_FILES.flatMap((file) => {
      const source = readFileSync(repoPath(file), "utf8");
      return FORBIDDEN_SKILL_FIRST_RUNTIME_PATTERNS.flatMap((pattern) =>
        source.includes(pattern) ? [`${file}: ${pattern}`] : [],
      );
    });

    expect(
      hits,
      "Skills 只能作为当前 Project / Thread 的 tool/context/workflow 注入；不得自动创建默认项目、恢复 Skill 私有 session 或暴露独立 executeSkill 运行入口",
    ).toEqual([]);
  });

  it("Agent Plugin current 入口不得恢复 Renderer 私有 runtime", () => {
    const hits = PLUGIN_CURRENT_SURFACE_FILES.flatMap((file) => {
      const source = readFileSync(repoPath(file), "utf8");
      return FORBIDDEN_RENDERER_PLUGIN_RUNTIME_PATTERNS.flatMap((pattern) =>
        source.includes(pattern) ? [`${file}: ${pattern}`] : [],
      );
    });

    expect(
      hits,
      "Agent Plugin 只允许使用 typed catalog、App Server runtime 与 Thread/pending projection；不得恢复 Renderer installed-state、activation metadata 或 manifest authorization",
    ).toEqual([]);
  });

  it("Automation job 不得把 job id 拼成私有 session / thread fallback", () => {
    const hits = AUTOMATION_THREAD_FIRST_SURFACE_FILES.flatMap((file) => {
      const source = readFileSync(repoPath(file), "utf8");
      return FORBIDDEN_AUTOMATION_FIRST_RUNTIME_PATTERNS.flatMap((pattern) =>
        source.includes(pattern) ? [`${file}: ${pattern}`] : [],
      );
    });

    expect(
      hits,
      "Automation / Workflow 可以是项目级配置，但运行产物必须显式绑定 Project / Thread lineage；不得用 job id 自动拼出能力私有 session 或 thread",
    ).toEqual([]);
  });

  it("Automation agent_turn API 类型必须要求显式 session / thread lineage", () => {
    const source = readFileSync(repoPath("src/lib/api/automation.ts"), "utf8");

    expect(source).toContain("session_id: string;");
    expect(source).toContain("thread_id: string;");
    expect(source).not.toContain("session_id?:");
    expect(source).not.toContain("thread_id?:");
  });

  it("Thread 内 service skill automation 创建链必须写入当前 session / thread lineage", () => {
    const workspaceSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
      ),
      "utf8",
    );
    const sceneSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
      ),
      "utf8",
    );
    const actionSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/useWorkspaceServiceSkillEntryActions.ts",
      ),
      "utf8",
    );
    const viewModelSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/workspaceServiceSkillEntryActionsViewModel.ts",
      ),
      "utf8",
    );

    expect(workspaceSource).toContain(
      "threadId: threadRead?.thread_id ?? sessionId",
    );
    expect(sceneSource).toContain("threadLineage={");
    expect(actionSource).toContain("ensureSessionForThreadLineage");
    expect(actionSource).toContain("normalizeAutomationThreadLineage");
    expect(viewModelSource).toContain(
      "threadLineage: ServiceSkillAutomationThreadLineage",
    );
    expect(viewModelSource).toContain(
      "session_id: pendingAutomation.threadLineage.sessionId",
    );
    expect(viewModelSource).toContain(
      "thread_id: pendingAutomation.threadLineage.threadId",
    );
  });

  it("子代理和 Team facts 必须挂在 parent thread，不得恢复独立子代理历史入口", () => {
    const hits = SUBAGENT_THREAD_FIRST_SURFACE_FILES.flatMap((file) => {
      const source = readFileSync(repoPath(file), "utf8");
      return FORBIDDEN_SUBAGENT_FIRST_HISTORY_PATTERNS.flatMap((pattern) =>
        source.includes(pattern) ? [`${file}: ${pattern}`] : [],
      );
    });

    expect(
      hits,
      "子代理 / Team 只能作为 parent Thread 的执行层事实；不得恢复独立子代理历史列表或子代理会话一级入口",
    ).toEqual([]);

    const workspaceSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
      ),
      "utf8",
    );
    const canonicalReaderSource = readFileSync(
      repoPath("src/lib/api/agentRuntime/appServerCanonicalItemReader.ts"),
      "utf8",
    );
    const threadClientSource = readFileSync(
      repoPath("src/lib/api/agentRuntime/threadClient.ts"),
      "utf8",
    );
    const navigationSource = readFileSync(
      repoPath(
        "src/components/agent/chat/workspace/useWorkspaceSubagentNavigationRuntime.ts",
      ),
      "utf8",
    );
    const threadItemProjectionSource = readFileSync(
      repoPath("src/components/agent/chat/projection/threadItemProjection.ts"),
      "utf8",
    );
    const sharedProjectionSource = readFileSync(
      repoPath("packages/agent-runtime-projection/src/threadItems.ts"),
      "utf8",
    );
    const timelineRendererSource = readFileSync(
      repoPath(
        "src/components/agent/chat/components/AgentThreadTimelineItemRenderers.tsx",
      ),
      "utf8",
    );
    const teamControlSource = readFileSync(
      repoPath("src/components/agent/chat/projection/teamControlProjection.ts"),
      "utf8",
    );
    expect(workspaceSource).toContain(
      "useWorkspaceSubagentNavigationRuntime({",
    );
    expect(canonicalReaderSource).toContain('case "subAgentActivity"');
    expect(canonicalReaderSource).toContain(
      'readString(item, "agentThreadId")',
    );
    expect(canonicalReaderSource).toContain('type: "subagent_activity"');
    expect(threadClientSource).toContain(
      "async function readThreadSessionId(threadId: string)",
    );
    expect(threadClientSource).toContain("includeTurns: false");
    expect(threadClientSource).toContain("mismatched threadId");
    expect(navigationSource).toContain("const canonicalSessionId");
    expect(navigationSource).toContain(
      ".find((child) => child.threadId.trim() === normalizedTargetId)",
    );
    expect(navigationSource).toContain(
      "await readSessionId(normalizedTargetId)",
    );
    expect(navigationSource).toContain("await switchTopic(sessionId");
    expect(navigationSource).toContain("allowDetachedSession: true");
    expect(navigationSource).not.toContain("isKnownSession");
    expect(threadItemProjectionSource).toContain(
      "buildAgentUiThreadItemEvent(sourceType, item, context)",
    );
    expect(sharedProjectionSource).toContain(
      "buildAgentUiThreadItemSubagentActivityEvent",
    );
    expect(timelineRendererSource).toContain(
      'item.type === "subagent_activity"',
    );
    expect(timelineRendererSource).toContain("subagentThreadId");
    expect(teamControlSource).toContain(
      "threadId: definedString(context.threadId)",
    );
    expect(teamControlSource).toContain(
      "parentThreadId: definedString(context.threadId)",
    );
  });
});
