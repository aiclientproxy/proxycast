/**
 * @file Skill 执行 API 模块
 * @description 封装 Skill 执行相关的 Desktop Host / App Server 命令调用
 *
 * 提供以下功能：
 * - listExecutableSkills: 列出所有可执行的 Skills
 * - getSkillDetail: 获取 Skill 详情
 *
 * @module lib/api/skill-execution
 * @requirements 3.1, 4.1, 5.1
 */

import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_SKILLS_LIST,
  METHOD_SKILL_READ,
  type SkillsListResponse as AppServerSkillsListResponse,
  type SkillReadResponse as AppServerSkillReadResponse,
} from "../../../packages/app-server-client/src/protocol";

type SkillExecutionAppServerClient = Pick<AppServerClient, "request">;

async function requestSkillExecutionAppServer<T>(
  method: string,
  params: unknown,
  appServerClient: SkillExecutionAppServerClient = new AppServerClient(),
): Promise<T> {
  const response = await appServerClient.request<T>(method, params);
  return response.result;
}

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 可执行 Skill 信息
 *
 * 用于 listExecutableSkills 返回的 Skill 列表项
 */
export interface ExecutableSkillInfo {
  /** 跨 workspace/session 稳定的 Skill identity */
  skill_id: string;
  /** Skill 名称（唯一标识） */
  name: string;
  /** 显示名称 */
  display_name: string;
  /** Skill 描述 */
  description: string;
  /** 执行模式：prompt, workflow, agent */
  execution_mode: "prompt" | "workflow" | "agent";
  /** 是否有 workflow 定义 */
  has_workflow: boolean;
  /** 指定的 Provider（可选） */
  provider?: string;
  /** 指定的 Model（可选） */
  model?: string;
  /** 参数提示（可选） */
  argument_hint?: string;
  /** Skill 来源 */
  source: "project" | "user" | "app" | "other";
  /** Skill 权限事实源 */
  authority: "workspace" | "user" | "application" | "external";
  /** Skill 作用域 */
  scope: "project" | "user" | "app" | "other";
  /** 当前是否允许模型选择 */
  enabled: boolean;
  /** 所需 runtime capabilities */
  capabilities: string[];
  /** typed tool dependencies */
  dependencies: SkillDependencyInfo[];
  /** 本地读取 locator，不参与 identity */
  locator: SkillLocatorInfo;
  /** 是否允许隐式选择 */
  allow_implicit_invocation: boolean;
  /** 使用场景说明（可选） */
  when_to_use?: string;
}

export interface SkillDependencyInfo {
  type: string;
  value: string;
  required: boolean;
}

export interface SkillLocatorInfo {
  directory: string;
  skill_file_path: string;
}

/**
 * Workflow 步骤信息
 *
 * 描述 Workflow 中的单个步骤
 */
export interface WorkflowStepInfo {
  /** 步骤 ID */
  id: string;
  /** 步骤名称 */
  name: string;
  /** 依赖的步骤 ID 列表 */
  dependencies: string[];
}

/**
 * Skill 详情信息
 *
 * 包含 Skill 的完整信息，用于 getSkillDetail 返回
 */
export interface SkillDetailInfo extends ExecutableSkillInfo {
  /** Markdown 内容（System Prompt） */
  markdown_content: string;
  /** Workflow 步骤（如果有） */
  workflow_steps?: WorkflowStepInfo[];
  /** 允许的工具列表（可选） */
  allowed_tools?: string[];
  /** 使用场景说明（可选） */
  when_to_use?: string;
}

function normalizeSkillListResponse(
  response: AppServerSkillsListResponse | null | undefined,
): ExecutableSkillInfo[] {
  if (!response || typeof response !== "object") {
    throw new Error("App Server skills/list did not return data");
  }

  if (!Array.isArray(response.data)) {
    throw new Error("App Server skills/list did not return data");
  }

  const seen = new Set<string>();
  const skills: ExecutableSkillInfo[] = [];
  response.data.forEach((entry, entryIndex) => {
    const item = requireRecord(entry, `skills/list data[${entryIndex}]`);
    requireString(item.cwd, `skills/list data[${entryIndex}].cwd`);
    requireArray(item.errors, `skills/list data[${entryIndex}].errors`).forEach(
      (error, errorIndex) => {
        const errorInfo = requireRecord(
          error,
          `skills/list data[${entryIndex}].errors[${errorIndex}]`,
        );
        requireString(
          errorInfo.path,
          `skills/list data[${entryIndex}].errors[${errorIndex}].path`,
        );
        requireString(
          errorInfo.message,
          `skills/list data[${entryIndex}].errors[${errorIndex}].message`,
        );
      },
    );
    requireArray(item.skills, `skills/list data[${entryIndex}].skills`).forEach(
      (skill, skillIndex) => {
        const normalized = normalizeListedSkill(
          skill,
          `skills/list data[${entryIndex}].skills[${skillIndex}]`,
        );
        if (normalized.enabled && !seen.has(normalized.skill_id)) {
          seen.add(normalized.skill_id);
          skills.push(normalized);
        }
      },
    );
  });
  return skills;
}

function normalizeListedSkill(
  value: unknown,
  label: string,
): ExecutableSkillInfo {
  const skill = requireRecord(value, label);
  const name = requireString(skill.name, `${label}.name`);
  const path = requireString(skill.path, `${label}.path`);
  const scope = requireOneOf(
    skill.scope,
    ["repo", "user", "system", "admin"] as const,
    `${label}.scope`,
  );
  const projection = listedScopeProjection(scope);
  const skillInterface =
    skill.interface === undefined
      ? {}
      : requireRecord(skill.interface, `${label}.interface`);
  const dependencies =
    skill.dependencies === undefined
      ? []
      : requireArray(
          requireRecord(skill.dependencies, `${label}.dependencies`).tools,
          `${label}.dependencies.tools`,
        ).map((dependency, index) => {
          const item = requireRecord(
            dependency,
            `${label}.dependencies.tools[${index}]`,
          );
          return {
            type: requireString(
              item.type,
              `${label}.dependencies.tools[${index}].type`,
            ),
            value: requireString(
              item.value,
              `${label}.dependencies.tools[${index}].value`,
            ),
            required: true,
          };
        });

  return {
    skill_id: `${projection.scope}:${name.trim().toLowerCase()}`,
    name,
    display_name:
      optionalString(
        skillInterface.displayName,
        `${label}.interface.displayName`,
      ) ?? name,
    description: requireString(skill.description, `${label}.description`),
    execution_mode: "prompt",
    has_workflow: false,
    source: projection.source,
    authority: projection.authority,
    scope: projection.scope,
    enabled: requireBoolean(skill.enabled, `${label}.enabled`),
    capabilities: dependencies.map((dependency) => dependency.value),
    dependencies,
    locator: {
      directory: path.replace(/[\\/]SKILL\.md$/i, ""),
      skill_file_path: path,
    },
    allow_implicit_invocation: true,
  };
}

function listedScopeProjection(
  scope: "repo" | "user" | "system" | "admin",
): Pick<ExecutableSkillInfo, "scope" | "source" | "authority"> {
  switch (scope) {
    case "repo":
      return { scope: "project", source: "project", authority: "workspace" };
    case "user":
      return { scope: "user", source: "user", authority: "user" };
    case "system":
      return { scope: "app", source: "app", authority: "application" };
    case "admin":
      return { scope: "other", source: "other", authority: "external" };
  }
}

export function resolveExecutableSkillId(
  skills: ReadonlyArray<Pick<ExecutableSkillInfo, "skill_id" | "name">>,
  reference: string,
): string | null {
  const normalizedReference = reference.trim();
  if (!normalizedReference) {
    return null;
  }

  const exactIdMatches = skills.filter(
    (skill) => skill.skill_id === normalizedReference,
  );
  if (exactIdMatches.length === 1) {
    return exactIdMatches[0].skill_id;
  }
  if (exactIdMatches.length > 1) {
    return null;
  }

  const nameMatches = skills.filter(
    (skill) => skill.name === normalizedReference,
  );
  return nameMatches.length === 1 ? nameMatches[0].skill_id : null;
}

function normalizeSkillReadResponse(
  response: AppServerSkillReadResponse | null | undefined,
  expectedSkillId: string,
): SkillDetailInfo {
  if (!response || typeof response !== "object" || !response.skill) {
    throw new Error("App Server skill/read did not return skill");
  }

  const detail = normalizeSkillDetail(response.skill);
  if (detail.skill_id !== expectedSkillId) {
    throw new Error(
      `App Server skill/read returned unexpected skillId: ${detail.skill_id}`,
    );
  }
  return detail;
}

function normalizeSkillSummary(
  skill: unknown,
  label: string,
): ExecutableSkillInfo {
  if (!isRecord(skill)) {
    throw new Error(`${label} is not an object`);
  }
  const skillInterface = requireRecord(skill.interface, `${label}.interface`);
  const dependencies = requireRecord(
    skill.dependencies,
    `${label}.dependencies`,
  );
  const policy = requireRecord(skill.policy, `${label}.policy`);
  const locator = requireRecord(skill.locator, `${label}.locator`);
  const executionMode = requireExecutionMode(
    skillInterface.executionMode,
    `${label}.interface.executionMode`,
  );

  return {
    skill_id: requireString(skill.skillId, `${label}.skillId`),
    name: requireString(skill.name, `${label}.name`),
    display_name: requireString(
      skillInterface.displayName,
      `${label}.interface.displayName`,
    ),
    description: requireString(skill.description, `${label}.description`),
    execution_mode: executionMode,
    has_workflow: executionMode === "workflow",
    provider: optionalString(
      skillInterface.provider,
      `${label}.interface.provider`,
    ),
    model: optionalString(skillInterface.model, `${label}.interface.model`),
    argument_hint: optionalString(
      skillInterface.argumentHint,
      `${label}.interface.argumentHint`,
    ),
    source: requireOneOf(
      skill.source,
      ["project", "user", "app", "other"] as const,
      `${label}.source`,
    ),
    authority: requireOneOf(
      skill.authority,
      ["workspace", "user", "application", "external"] as const,
      `${label}.authority`,
    ),
    scope: requireOneOf(
      skill.scope,
      ["project", "user", "app", "other"] as const,
      `${label}.scope`,
    ),
    enabled: requireBoolean(skill.enabled, `${label}.enabled`),
    capabilities: requireStringArray(
      skill.capabilities,
      `${label}.capabilities`,
    ),
    dependencies: requireArray(
      dependencies.tools,
      `${label}.dependencies.tools`,
    ).map((dependency, index) => {
      const item = requireRecord(
        dependency,
        `${label}.dependencies.tools[${index}]`,
      );
      return {
        type: requireString(
          item.type,
          `${label}.dependencies.tools[${index}].type`,
        ),
        value: requireString(
          item.value,
          `${label}.dependencies.tools[${index}].value`,
        ),
        required: requireBoolean(
          item.required,
          `${label}.dependencies.tools[${index}].required`,
        ),
      };
    }),
    locator: {
      directory: requireString(locator.directory, `${label}.locator.directory`),
      skill_file_path: requireString(
        locator.skillFilePath,
        `${label}.locator.skillFilePath`,
      ),
    },
    allow_implicit_invocation: requireBoolean(
      policy.allowImplicitInvocation,
      `${label}.policy.allowImplicitInvocation`,
    ),
    when_to_use: optionalString(policy.whenToUse, `${label}.policy.whenToUse`),
  };
}

function normalizeSkillDetail(skill: unknown): SkillDetailInfo {
  if (!isRecord(skill)) {
    throw new Error("App Server skill/read skill is not an object");
  }
  const metadata = normalizeSkillSummary(
    skill.metadata,
    "skill/read skill.metadata",
  );
  const workflowSteps = requireArray(
    skill.workflowSteps,
    "skill/read skill.workflowSteps",
  ).map((step, index) => {
    const item = requireRecord(
      step,
      `skill/read skill.workflowSteps[${index}]`,
    );
    return {
      id: requireString(item.id, `skill/read skill.workflowSteps[${index}].id`),
      name: requireString(
        item.name,
        `skill/read skill.workflowSteps[${index}].name`,
      ),
      dependencies: requireStringArray(
        item.dependencies,
        `skill/read skill.workflowSteps[${index}].dependencies`,
      ),
    };
  });

  return {
    ...metadata,
    markdown_content: requireString(
      skill.markdownContent,
      "skill/read skill.markdownContent",
    ),
    workflow_steps: workflowSteps.length > 0 ? workflowSteps : undefined,
    allowed_tools: metadata.capabilities,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${label} is not an object`);
  }
  return value;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} is not an array`);
  }
  return value;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} is not a non-empty string`);
  }
  return value;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return requireString(value, label);
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} is not a boolean`);
  }
  return value;
}

function requireStringArray(value: unknown, label: string): string[] {
  return requireArray(value, label).map((item, index) =>
    requireString(item, `${label}[${index}]`),
  );
}

function requireExecutionMode(
  value: unknown,
  label: string,
): ExecutableSkillInfo["execution_mode"] {
  return requireOneOf(value, ["prompt", "workflow", "agent"] as const, label);
}

function requireOneOf<const T extends readonly string[]>(
  value: unknown,
  allowed: T,
  label: string,
): T[number] {
  if (
    typeof value !== "string" ||
    !(allowed as readonly string[]).includes(value)
  ) {
    throw new Error(`${label} is not a supported value`);
  }
  return value as T[number];
}

/**
 * 步骤执行结果
 *
 * 描述单个步骤的执行结果
 */
export interface StepResult {
  /** 步骤 ID */
  step_id: string;
  /** 步骤名称 */
  step_name: string;
  /** 是否成功 */
  success: boolean;
  /** 输出内容 */
  output?: string;
  /** 错误信息 */
  error?: string;
}

// ============================================================================
// API 函数
// ============================================================================

/**
 * Skill 执行 API
 *
 * 封装 Skill 执行相关的 Desktop Host / App Server 命令调用
 */
export const skillExecutionApi = {
  /**
   * 列出所有可执行的 Skills
   *
   * 返回所有可以执行的 Skills 列表，已过滤掉 disable_model_invocation=true 的 Skills
   *
   * @returns 可执行的 Skills 列表
   *
   * @requirements 4.1, 4.2, 4.3, 4.4
   */
  async listExecutableSkills(): Promise<ExecutableSkillInfo[]> {
    const response =
      await requestSkillExecutionAppServer<AppServerSkillsListResponse>(
        METHOD_SKILLS_LIST,
        {},
      );
    return normalizeSkillListResponse(response);
  },

  /**
   * 获取 Skill 详情
   *
   * @param skillId - typed catalog 暴露的稳定 Skill identity
   * @returns Skill 详情信息
   * @throws 如果 Skill 不存在则抛出错误
   *
   * @requirements 5.1, 5.2, 5.3, 5.4
   */
  async getSkillDetail(skillId: string): Promise<SkillDetailInfo> {
    const response =
      await requestSkillExecutionAppServer<AppServerSkillReadResponse>(
        METHOD_SKILL_READ,
        { skillId },
      );
    return normalizeSkillReadResponse(response, skillId);
  },
};
