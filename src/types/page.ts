/**
 * 页面类型定义
 *
 * 支持静态页面
 * - 静态页面: 预定义的页面标识符
 *
 * @module types/page
 */

import type { SettingsTabs } from "./settings";
import type { SkillScaffoldTarget } from "@/lib/api/skills";
import type { SceneAppExecutionSummaryViewModel } from "@/lib/agent/legacySceneAppExecutionSummary";
import type { InputCapabilitySendRoute } from "@/components/agent/chat/skill-selection/inputCapabilitySelection";

export type WorkspaceTheme = "general";

export type Page =
  | "agent"
  | "experts"
  | "skills"
  | "plugins"
  | "knowledge"
  | "scheduled-tasks"
  | "channels"
  | "resources"
  | "browser-runtime"
  | "settings";

export interface AgentPendingServiceSkillLaunchParams {
  skillId: string;
  skillKey?: string;
  requestKey?: number;
  initialSlotValues?: Record<string, string>;
  prefillHint?: string;
  launchUserInput?: string;
}

export interface AgentInitialInputCapabilityParams {
  capabilityRoute: InputCapabilitySendRoute;
  requestKey?: number;
}

export interface AgentInitialKnowledgePackSelectionParams {
  enabled: boolean;
  packName: string;
  workingDir: string;
  label?: string;
  status?: string;
  companionPacks?: Array<{
    name: string;
    activation?: "explicit" | "implicit" | "resolver-driven";
  }>;
}

/**
 * Agent 页面参数
 * 用于从项目入口跳转到创作界面时传递项目上下文
 */
export interface AgentPageParams {
  projectId?: string;
  contentId?: string;
  /** 进入 Agent 后优先恢复到指定会话 */
  initialSessionId?: string;
  /** 从创作场景进入生成时透传的执行摘要 */
  initialSceneAppExecutionSummary?: SceneAppExecutionSummaryViewModel;
  initialRequestMetadata?: Record<string, unknown>;
  initialAutoSendRequestMetadata?: Record<string, unknown>;
  autoRunInitialPromptOnMount?: boolean;
  /** Agent 入口模式：新建任务或生成 */
  agentEntry?: "new-task" | "claw";
  /** 首页沉浸模式提交后透传的首条图片 */
  initialUserImages?: Array<{
    data: string;
    mediaType: string;
  }>;
  /** 进入 Agent 时自动发送的首条用户消息 */
  initialUserPrompt?: string;
  /** 进入 Agent 时透传的初始创作模式 */
  initialCreationMode?: "guided" | "fast" | "hybrid" | "framework";
  /** 进入 Agent 时优先创建的话题名称 */
  initialSessionName?: string;
  /** 一次性入口提示文案 */
  entryBannerMessage?: string;
  /** 首屏工作区主题（用于直达指定工作区入口） */
  theme?: string;
  /** 是否锁定主题（锁定后不在首屏显示主题切换） */
  lockTheme?: boolean;
  /** 从资源管理页进入（用于沉浸式展示） */
  fromResources?: boolean;
  /** 首页沉浸模式：隐藏左侧应用导航与话题列表，仅保留主工作区 */
  immersiveHome?: boolean;
  /** 进入 Agent 后立即打开浏览器协助 */
  openBrowserAssistOnMount?: boolean;
  /** 进入 Agent 后执行一次站点技能启动 */
  initialSiteSkillLaunch?: AgentSiteSkillLaunchParams;
  /** 进入 Agent 后在当前对话挂起或继续一次服务技能启动 */
  initialPendingServiceSkillLaunch?: AgentPendingServiceSkillLaunchParams;
  /** 进入 Agent 后优先恢复一条输入能力路由 */
  initialInputCapability?: AgentInitialInputCapabilityParams;
  /** 仅把输入能力放进首页输入框，不直接切进会话工作区 */
  preferHomeForInitialInputCapability?: boolean;
  /** 从资料管理进入 Agent 时预启用的项目资料 */
  initialKnowledgePackSelection?: AgentInitialKnowledgePackSelectionParams;
  /** 进入 Agent 后自动打开项目内某个文件 */
  initialProjectFileOpenTarget?: AgentProjectFileOpenTarget;
  /** 首页点击触发的新会话标记（时间戳） */
  newChatAt?: number;
  /** 专家入口身份，只作为当前 Thread 的 profile / metadata 来源 */
  expertAgentLaunch?: ExpertAgentLaunchParams;
}

export interface ExpertsPageParams {
  /** 专家广场启动专家时优先使用的当前项目作用域 */
  projectId?: string;
  currentProjectId?: string;
}

export interface ExpertAgentLaunchParams {
  tenantId: string;
  projectId?: string;
  expertId: string;
  releaseId: string;
  agentInstanceKey: string;
  launchMode: "new_thread";
  catalogVersion?: string;
  title?: string;
  skillRefsOverride?: string[];
}

/**
 * 设置页面参数
 */
export type SettingsProviderView = "settings" | "cloud";

export interface ProviderSettingsFocusContext {
  providerId?: string;
  modelId?: string;
  reasonCode?: string;
  recoveryAction?: string;
  requestKey?: number;
}

export type ExecutionPolicyFocusSection = "workspace" | "shell" | "network";
export type ExecutionPolicyFocusTarget = "command" | "host" | "url";

export interface ExecutionPolicyFocusContext {
  section: ExecutionPolicyFocusSection;
  ruleId?: string;
  target?: ExecutionPolicyFocusTarget;
  value?: string;
  reasonCode?: string;
  requestKey?: number;
}

export interface SettingsPageParams {
  tab?: SettingsTabs;
  providerView?: SettingsProviderView;
  providerFocus?: ProviderSettingsFocusContext;
  executionPolicyFocus?: ExecutionPolicyFocusContext;
}

export interface SkillScaffoldDraft extends Record<string, unknown> {
  target?: SkillScaffoldTarget;
  directory?: string;
  name?: string;
  description?: string;
  whenToUse?: string[];
  inputs?: string[];
  outputs?: string[];
  steps?: string[];
  fallbackStrategy?: string[];
  sourceMessageId?: string;
  sourceExcerpt?: string;
}

export interface SkillsPageParams {
  initialScaffoldDraft?: SkillScaffoldDraft;
  initialScaffoldRequestKey?: number;
  initialSkillPackagePath?: string;
  initialSkillPackageName?: string;
  initialSkillPackageRequestKey?: number;
  /** 进入技能工作台后预填搜索词，用于从专家技能恢复动作定位目标。 */
  initialSearchQuery?: string;
  initialSearchRequestKey?: number;
  creationProjectId?: string;
  highlightCapabilityDraftId?: string;
  initialView?: "store" | "builtin" | "installed" | "discover" | "manage";
}

export interface PluginsPageParams {
  query?: string;
  selectedPluginId?: string;
  statusFilter?: "all" | "installed";
  /** 从项目内进入插件工作台时保留的项目作用域。 */
  currentProjectId?: string;
}

export interface KnowledgePageParams {
  workingDir?: string;
  selectedPackName?: string;
  initialView?: "overview" | "import" | "detail" | "save" | "states";
  saveDraft?: {
    sourceText: string;
    sourceName?: string;
    description?: string | null;
    packType?: string | null;
    requestKey?: number;
  };
}

export interface ScheduledTasksPageParams {
  selectedTaskId?: string;
  projectId?: string;
  sessionId?: string;
  threadId?: string;
}

export interface BrowserRuntimePageParams {
  projectId?: string;
  contentId?: string;
  initialProfileKey?: string;
  initialSessionId?: string;
  initialTargetId?: string;
  initialAdapterName?: string;
  initialArgs?: Record<string, unknown>;
  initialAutoRun?: boolean;
  initialRequireAttachedSession?: boolean;
  initialSaveTitle?: string;
}

export interface ResourcesPageParams {
  projectId?: string;
  contentId?: string;
  focusIntentId?: string;
  focusResourceTitle?: string;
  resourceFolderId?: string;
  resourceCategory?: "all" | "document" | "image" | "audio" | "video";
}

export interface AgentSiteSkillLaunchParams {
  adapterName: string;
  args?: Record<string, unknown>;
  autoRun?: boolean;
  profileKey?: string;
  targetId?: string;
  requireAttachedSession?: boolean;
  preferredBackend?: "lime_extension_bridge" | "cdp_direct";
  autoLaunch?: boolean;
  saveTitle?: string;
  skillTitle?: string;
}

export interface AgentProjectFileOpenTarget {
  relativePath: string;
  requestKey?: number;
}

/**
 * 页面参数联合类型
 */
export type PageParams =
  | AgentPageParams
  | ScheduledTasksPageParams
  | BrowserRuntimePageParams
  | ExpertsPageParams
  | ResourcesPageParams
  | SettingsPageParams
  | SkillsPageParams
  | PluginsPageParams
  | KnowledgePageParams
  | Record<string, unknown>;
