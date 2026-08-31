import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import {
  buildExpertSkillsRuntimeCatalog,
  buildExpertSkillsRuntimeMetadata,
  createExplicitSkillsRuntimeFixtureScenario,
  createExpertSkillsRuntimeFixtureScenario,
  createExpertPanelSkillsRuntimeFixtureScenario,
  createManualEnableSkillsRuntimeFixtureScenario,
  createSkillsRuntimeFixtureScenario,
  EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
  EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_ID,
  EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_TITLE,
  renderSkillsRuntimeBackendEvents,
  SKILLS_RUNTIME_ASSERTION_KEYS,
  SKILLS_RUNTIME_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_QUERY,
  SKILLS_RUNTIME_SKILL_NAME,
  summarizeSkillsRuntimeStatusEvents,
  summarizeSkillsRuntimeThreadRead,
} from "./skills-runtime-fixture-scenario.mjs";
export {
  buildExpertSkillsRuntimeCatalog,
  buildExpertSkillsRuntimeMetadata,
  createExplicitSkillsRuntimeFixtureScenario,
  createExpertSkillsRuntimeFixtureScenario,
  createExpertPanelSkillsRuntimeFixtureScenario,
  createManualEnableSkillsRuntimeFixtureScenario,
  createSkillsRuntimeFixtureScenario,
  EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
  EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_ID,
  EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_TITLE,
  renderSkillsRuntimeBackendEvents,
  SKILLS_RUNTIME_ASSERTION_KEYS,
  SKILLS_RUNTIME_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_QUERY,
  SKILLS_RUNTIME_SKILL_NAME,
  summarizeSkillsRuntimeStatusEvents,
  summarizeSkillsRuntimeThreadRead,
};

export const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "claw-chat-current-fixture",
  ),
  prefix: "claw-chat-current-fixture",
  timeoutMs: 180_000,
  intervalMs: 500,
  keepTemp: false,
  scenario: "complete",
};

export const LOG_PREFIX = "[smoke:claw-chat-current-fixture]";
export const APP_SERVER_HANDLE_JSON_LINES_COMMAND =
  "app_server_handle_json_lines";
export const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
export const APP_SERVER_METHOD_INITIALIZE = "initialize";
export const APP_SERVER_METHOD_INITIALIZED = "initialized";
export const APP_SERVER_METHOD_SESSION_START = "thread/start";
export const APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE =
  "thread/settings/update";
export const APP_SERVER_METHOD_SESSION_TURN_START = "turn/start";
export const APP_SERVER_METHOD_SESSION_TURN_CANCEL = "turn/interrupt";
export const APP_SERVER_METHOD_THREAD_SHELL_COMMAND = "thread/shellCommand";
export const USER_SHELL_INPUT = "!printf ready";
export const USER_SHELL_COMMAND = "printf ready";
export const USER_SHELL_OUTPUT = "ready";
export const APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND =
  "agentSession/action/respond";
export const APP_SERVER_METHOD_SESSION_READ = "thread/read";
export const APP_SERVER_METHOD_SESSION_THREAD_RESUME = "thread/resume";
export const APP_SERVER_METHOD_SESSION_LIST = "thread/list";
export const APP_SERVER_METHOD_WORKFLOW_READ = "workflow/read";
export const APP_SERVER_METHOD_WORKFLOW_RESPOND = "workflow/respond";
export const APP_SERVER_METHOD_WORKFLOW_CANCEL = "workflow/cancel";
export const APP_SERVER_METHOD_WORKFLOW_RETRY = "workflow/retry";
export const APP_SERVER_METHOD_ARTIFACT_READ = "artifact/read";
export const APP_SERVER_METHOD_ARTIFACT_WRITE = "artifact/write";
export const APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE =
  "mediaTaskArtifact/image/create";
export const APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE =
  "mediaTaskArtifact/image/complete";
export const APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET =
  "mediaTaskArtifact/get";
export const APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST =
  "mediaTaskArtifact/list";
export const APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST =
  "diagnostics/trace/list";
export const APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ =
  "diagnostics/trace/read";
export const APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT =
  "diagnostics/trace/export";
export const APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT =
  "diagnostics/supportBundle/export";
export const APP_SERVER_METHOD_WORKSPACE_DEFAULT_ENSURE =
  "workspace/default/ensure";
export const APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST =
  "workspaceRightSurface/request";
export const APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST =
  "workspaceRightSurface/pending/list";
export const RIGHT_SURFACE_VISUAL_MATRIX_SCENARIO =
  "right-surface-visual-matrix";
export const THREAD_ACTIVITY_PANEL_SCENARIO = "thread-activity-panel";
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO =
  "content-factory-article-workspace";
export const CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO =
  "content-factory-inline-image-article-workspace";
export const SOUL_STYLE_SCENARIO = "soul-style";
export const INPUTBAR_RICH_RESTORE_SCENARIO = "inputbar-rich-restore";
export const HOME_HOTPATH_SCENARIO = "home-hotpath";
export const HOME_HOTPATH_GREETING_SCENARIO = "home-hotpath-greeting";
export const NEWS_PROMPT = "整理今天的国际新闻";
export const GREETING_PROMPT = "你好";
export const CONTINUE_PROMPT = "继续输出";
export const PLAN_PROMPT = "先给我一个修复计划，不要直接改代码";
export const GOAL_PROMPT = "本周完成 Goal E2E 修复";
export const INPUTBAR_RICH_RESTORE_PROMPT =
  "请结合这个截图、文件和 Capability Report 技能，先不要输出正文。";
export const INPUTBAR_RICH_RESTORE_PATH_NAME =
  "clawstream-rich-restore-fixture.md";
export const INPUTBAR_RICH_RESTORE_PATH = path.join(
  process.cwd(),
  "internal",
  "roadmap",
  "test",
  "clawstream",
  INPUTBAR_RICH_RESTORE_PATH_NAME,
);
export const INPUTBAR_RICH_RESTORE_SKILL_NAME = "Capability Report";
export const WEB_TOOLS_RENDERING_PROMPT = "验证网页搜索渲染";
export const MCP_STRUCTURED_CONTENT_PROMPT = "验证 MCP structuredContent 展示";
export const ASSISTANT_DONE_TEXT = "CLAW_NEWS_FIXTURE_DONE";
export const GREETING_DONE_TEXT = "CLAW_GREETING_FIXTURE_DONE";
export const GREETING_SUMMARY_TEXT = "你好，我在，可以继续说需求";
export const INPUTBAR_RICH_RESTORE_FORBIDDEN_ASSISTANT_TEXT =
  "CLAW_INPUTBAR_RICH_RESTORE_DONE";
export const CONTINUE_DONE_TEXT = "CLAW_CONTINUE_FIXTURE_DONE";
export const PLAN_DONE_TEXT = "CLAW_PLAN_FIXTURE_DONE";
export const TURN_PLAN_UPDATE_SCENARIO = "turn-plan-update";
export const TURN_PLAN_UPDATE_PROMPT = "请先更新执行清单，再给出本轮处理结果";
export const TURN_PLAN_UPDATE_DONE_TEXT = "CLAW_TURN_PLAN_UPDATE_DONE";
export const TURN_PLAN_UPDATE_EXPLANATION = "继续执行";
export const TURN_PLAN_UPDATE_TOOL_CALL_ID = "call_turn_plan_update";
export const GOAL_DONE_TEXT = "CLAW_GOAL_FIXTURE_DONE";
export const WEB_TOOLS_RENDERING_DONE_TEXT = "CLAW_WEB_TOOLS_RENDERING_DONE";
export const MCP_STRUCTURED_CONTENT_DONE_TEXT =
  "CLAW_MCP_STRUCTURED_CONTENT_DONE";
export const IMAGE_COMMAND_SCENARIO = "image-command";
export const PLAIN_IMAGE_INTENT_SCENARIO = "plain-image-intent";
export const IMAGE_COMMAND_PROMPT =
  "@配图 E2E 图片命令路由测试，请生成一张青柠插画";
export const IMAGE_COMMAND_IMAGE_PROMPT =
  "E2E 图片命令路由测试，请生成一张青柠插画";
export const PLAIN_IMAGE_INTENT_PROMPT = "画一张广州夏天的图";
export const PLAIN_IMAGE_INTENT_ROUTED_PROMPT = `@配图 ${PLAIN_IMAGE_INTENT_PROMPT}`;
export const PLAIN_IMAGE_INTENT_IMAGE_PROMPT = PLAIN_IMAGE_INTENT_PROMPT;
export const IMAGE_COMMAND_DONE_TEXT = "CLAW_IMAGE_COMMAND_FIXTURE_DONE";
export const IMAGE_COMMAND_PRESENTATION_INTRO =
  "好啊，这张图我来处理，先把画面氛围定准。";
export const IMAGE_COMMAND_PRESENTATION_CAPTION =
  "完成了，画面已经生成。想更清爽、更写实或换构图，都可以继续调。";
export const WEB_TOOLS_SEARCH_TITLE = "Lime WebSearch Rendering Source";
export const WEB_TOOLS_SEARCH_URL =
  "https://example.com/lime-websearch-rendering";
export const WEB_TOOLS_SEARCH_SOURCE_LABEL =
  "example.com/lime-websearch-rendering";
export const WEB_TOOLS_SEARCH_SNIPPET =
  "Search source used to verify inline rendering";
export const WEB_TOOLS_MID_THINKING_TEXT =
  "搜索结果还需要继续筛掉广告软文，我先读取有效来源。";
export const REASONING_FIRST_VISIBLE_SCENARIO = "reasoning-first-visible";
export const REASONING_FIRST_VISIBLE_PROMPT = "验证 reasoning 先于最终回答可见";
export const REASONING_FIRST_VISIBLE_TEXT =
  "先确认用户要验证的是展示时序，再给出最终回答。";
export const REASONING_FIRST_VISIBLE_CONTENT_TEXT =
  "完整思考：先确认当前回合要验证 reasoning 展示时序，再检查完成态折叠区展开后仍能读取 canonical content。";
export const REASONING_FIRST_VISIBLE_FINAL_TEXT =
  "最终回答：reasoning 已经先于正文出现在当前回合。";
export const REASONING_FIRST_VISIBLE_DONE_TEXT = "REASONING_FIRST_VISIBLE_DONE";
export const LIVE_TAIL_COMMIT_SCENARIO = "live-tail-commit";
export const ELECTRON_RESIZE_REFLOW_SCENARIO = "electron-resize-reflow";
export const LIVE_TAIL_COMMIT_PROMPT = "验证 live tail 长输出和表格滚动锚点";
export const LIVE_TAIL_COMMIT_FIRST_TEXT =
  "LIVE_TAIL_FIRST_VISIBLE_TOKEN: 第一段长输出已经在完成前可见。";
export const LIVE_TAIL_COMMIT_OVERFLOW_MARKER =
  "LIVE_TAIL_OVERFLOW_COMMIT_MARKER";
export const LIVE_TAIL_COMMIT_TABLE_HEADER = "| 序号 | 校验点 | 状态 |";
export const LIVE_TAIL_COMMIT_TABLE_TAIL =
  "| 24 | table tail reflow | stable |";
export const LIVE_TAIL_COMMIT_DONE_TEXT = "LIVE_TAIL_COMMIT_DONE";
export const APPROVAL_REQUEST_RESUME_SCENARIO = "approval-request-resume";
export const APPROVAL_REQUEST_DECLINE_SCENARIO = "approval-request-decline";
export const APPROVAL_REQUEST_CANCEL_SCENARIO = "approval-request-cancel";
export const APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO =
  "approval-request-host-interrupt";
export const APPROVAL_REQUEST_FULL_ACCESS_SCENARIO =
  "approval-request-full-access";
export const APPROVAL_REQUEST_RESUME_PROMPT = "验证审批请求 hydrate 后允许继续";
export const APPROVAL_REQUEST_FULL_ACCESS_PROMPT =
  "验证完全授权不会显示审批记录";
export const APPROVAL_REQUEST_RESUME_TOOL_NAME = "exec_command";
export const APPROVAL_REQUEST_RESUME_COMMAND =
  "printf 'approval resume fixture\\n'";
export const APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT =
  "允许执行 approval resume fixture？";
export const APPROVAL_REQUEST_RESUME_RESULT_TEXT =
  "approval resume fixture 审批通过后已经继续执行。";
export const APPROVAL_REQUEST_RESUME_DONE_TEXT = "APPROVAL_REQUEST_RESUME_DONE";
export const APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT =
  "full-access fixture 没有生成任何审批记录。";
export const APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT =
  "APPROVAL_REQUEST_FULL_ACCESS_DONE";
export const APPROVAL_REQUEST_DECLINE_RESULT_TEXT =
  "approval decline fixture 已拒绝当前浏览器动作，并改用无浏览器路径继续。";
export const APPROVAL_REQUEST_DECLINE_DONE_TEXT =
  "APPROVAL_REQUEST_DECLINE_DONE";
export const APPROVAL_REQUEST_CANCEL_DONE_TEXT = "APPROVAL_REQUEST_CANCEL_DONE";
export const TERMINAL_STALE_GUARD_SCENARIO = "terminal-stale-guard";
export const TERMINAL_STALE_GUARD_FIRST_PROMPT =
  "验证旧 terminal 不影响下一轮：第一轮";
export const TERMINAL_STALE_GUARD_SECOND_PROMPT =
  "验证旧 terminal 不影响下一轮：第二轮";
export const TERMINAL_STALE_GUARD_FIRST_TEXT =
  "第一轮已经完成，用作后续旧 terminal owner。";
export const TERMINAL_STALE_GUARD_SECOND_TEXT =
  "第二轮在旧 terminal 干扰后继续完成。";
export const TERMINAL_STALE_GUARD_FIRST_DONE_TEXT =
  "TERMINAL_STALE_GUARD_FIRST_DONE";
export const TERMINAL_STALE_GUARD_DONE_TEXT = "TERMINAL_STALE_GUARD_DONE";
export const TERMINAL_STALE_GUARD_STALE_DONE_TEXT =
  "TERMINAL_STALE_GUARD_STALE_DONE";
export const TERMINAL_FAILED_AFTER_ANSWER_SCENARIO =
  "terminal-failed-after-answer";
export const TERMINAL_FAILED_AFTER_ANSWER_PROMPT =
  "验证已输出正文后 turn.failed 不吞正文";
export const TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT =
  "TERMINAL_FAILED_AFTER_ANSWER_PARTIAL: 正文已经先显示，失败终态不能覆盖或重复。";
export const TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT =
  "TERMINAL_FAILED_AFTER_ANSWER_FAILURE: provider stream closed after partial answer";
export const TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO =
  "terminal-canceled-after-answer";
export const TERMINAL_CANCELED_AFTER_ANSWER_PROMPT =
  "验证已输出正文后 turn.canceled 不吞正文";
export const TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT =
  "TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL: 正文已经先显示，取消终态不能覆盖或重复。";
export const TERMINAL_CANCELED_AFTER_ANSWER_CANCELED_TEXT =
  "TERMINAL_CANCELED_AFTER_ANSWER_CANCELED: user canceled after partial answer";
export const TYPED_ERROR_RETRY_SUCCESS_SCENARIO = "typed-error-retry-success";
export const TYPED_ERROR_RETRY_FAILURE_SCENARIO = "typed-error-retry-failure";
export const TYPED_ERROR_RETRY_SUCCESS_PROMPT = "验证插件重试后继续完成";
export const TYPED_ERROR_RETRY_FAILURE_PROMPT = "验证重试耗尽后进入失败终态";
export const TYPED_ERROR_RETRY_MESSAGE = "插件暂时不可用，正在重试当前回合。";
export const TYPED_ERROR_RETRY_SECOND_MESSAGE =
  "第二次插件重试仍在进行，当前回合保持打开。";
export const TYPED_ERROR_RETRY_SUCCESS_TEXT =
  "重试完成后，当前回合继续输出并正常收口。";
export const TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT =
  "TYPED_ERROR_RETRY_SUCCESS_DONE";
export const TYPED_ERROR_RETRY_FAILURE_TEXT =
  "重试次数已耗尽，当前回合保留失败记录。";
export const TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT =
  "失败终态前已保留的部分输出。";
export const TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT =
  "TYPED_ERROR_RETRY_FAILURE_ERROR: plugin worker retry budget exhausted";
export const TYPED_ERROR_RETRY_ASSERTION_KEYS = [
  "typedErrorRetryPromptReachedBackend",
  "typedErrorRetryBackendEventOrder",
  "typedErrorRetryGuiRetryingStatusVisible",
  "typedErrorRetryNoPrematureTerminal",
  "typedErrorRetrySuccessGuiCompleted",
  "typedErrorRetrySuccessReadModelCompleted",
  "typedErrorRetryFailureGuiAwaitedTerminal",
  "typedErrorRetryFailureGuiCompleted",
  "typedErrorRetryFailureReadModelFailed",
  "typedErrorRetryFailureNoPrematureTerminal",
  "typedErrorRetryIdentityConsistent",
];
export const WEB_TOOLS_REASONING_ITEM_SIGNATURE =
  "web-tools-reasoning-item-signature";
export const WEB_TOOLS_REASONING_NATIVE_ITEM_ID = "rs_web_tools_fixture";
export const WEB_TOOLS_REASONING_PROVIDER_BACKEND = "reasoning_fixture";
export const WEB_TOOLS_FETCH_MARKDOWN =
  "WebFetch 正文摘要：页面确认搜索来源可以展开，同时最终正文继续输出。";
export const WEB_TOOLS_BROKEN_MARKDOWN_TEXT = [
  "五年级选购指南###",
  "####如果孩子基础一般，优先看护眼、内容和家长管理。",
  "**推荐 型号 **：Lime 学习机 S30",
  "**理由 **：系统清晰，适合五年级基础巩固。",
  "对比表：",
  "| 品牌 | 型号 | 场景 |",
  "| --- | --- | --- |",
  "| Lime | S30 | 五年级巩固 |",
].join("\n");
export const PLAN_STEPS = [
  { step: "确认计划模式请求进入 App Server", status: "completed" },
  { step: "输出 proposed_plan", status: "in_progress" },
  { step: "验证右侧计划轨显示", status: "pending" },
];
export const TURN_PLAN_UPDATE_STEPS = [
  { step: "读现状", status: "completed" },
  { step: "补主链", status: "in_progress" },
];
export const PROPOSED_PLAN_BLOCK = `<proposed_plan>
${PLAN_STEPS.map((step) => `- ${step.step}`).join("\n")}
</proposed_plan>`;
export const FIXTURE_PROVIDER = "fixture-provider";
export const FIXTURE_MODEL = "fixture-model";
export const TEXT_FIXTURE_PROVIDER_NAME = "Fixture Text Provider";
export const TEXT_PROVIDER_FIXTURE_API_KEY = "sk-claw-text-fixture";
export const IMAGE_FIXTURE_PROVIDER_NAME = "Fixture Image Provider";
export const IMAGE_FIXTURE_MODEL = "fal-ai/nano-banana-pro";
export const SESSION_ID = `claw-chat-current-${Date.now()}-${process.pid}`;
export const THREAD_ID = `${SESSION_ID}-thread`;
export const APPROVAL_REQUEST_RESUME_REQUEST_ID = `${SESSION_ID}:approval:resume`;
export const APPROVAL_REQUEST_RESUME_TOOL_CALL_ID = `${SESSION_ID}:tool:approval-resume`;
export const SESSION_TITLE = "Claw 新闻输入 Electron fixture";
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_ID = `${SESSION_ID}-content-factory-article-workspace`;
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_THREAD_ID = `${CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_ID}-thread`;
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_SESSION_TITLE =
  "内容工厂 Article Editor Fixture";
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_ARTICLE_ARTIFACT_ID =
  "artifact-article-1";
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_IMAGE_ARTIFACT_ID =
  "artifact-image-1";
export const CONTENT_FACTORY_INLINE_IMAGE_SLOT_ID =
  "article-inline-image-slot-e2e";
export const CONTENT_FACTORY_INLINE_IMAGE_TASK_PROMPT =
  "广州夏天午后街景原位配图";
export const CONTENT_FACTORY_INLINE_IMAGE_FILE_PATH = path.join(
  process.cwd(),
  "public",
  "fixtures",
  "article-inline-image-e2e.svg",
);
export const CONTENT_FACTORY_INLINE_IMAGE_URL = pathToFileURL(
  CONTENT_FACTORY_INLINE_IMAGE_FILE_PATH,
).href;
export const WEB_TOOLS_SEARCH_TOOL_CALL_ID = `${SESSION_ID}:tool:websearch-rendering`;
export const WEB_TOOLS_REASONING_ITEM_ID = `${SESSION_ID}:reasoning:web-tools-rendering`;
export const WEB_TOOLS_FETCH_TOOL_CALL_ID = `${SESSION_ID}:tool:webfetch-rendering`;
export const MCP_STRUCTURED_CONTENT_TOOL_CALL_ID = `${SESSION_ID}:tool:mcp-structured-content`;
export const MCP_STRUCTURED_CONTENT_TOOL_NAME = "mcp__docs__diagnostic_probe";
export const MCP_STRUCTURED_CONTENT_TOOL_DISPLAY_LABEL =
  "docs / diagnostic probe";
export const MCP_STRUCTURED_CONTENT_ANSWER =
  "MCP 结构化答案已进入 Agent Chat GUI";
export const MCP_STRUCTURED_CONTENT_REFERENCE_ID = "doc-structured-1";
export const MCP_STRUCTURED_CONTENT_PROTOCOL_OUTPUT = JSON.stringify({
  request_metadata: {
    projection: "mcp_tool_result_projection",
    trace_id: "mcp-structured-content-fixture-trace",
  },
  diagnostics: {
    elapsed_ms: 12,
    raw_transport_payload: "doc-hidden-envelope",
  },
  content: [
    {
      type: "text",
      text: "control-plane envelope only; user answer is stored in structuredContent",
    },
  ],
});
export const MCP_STRUCTURED_CONTENT_RESULT = {
  answer: MCP_STRUCTURED_CONTENT_ANSWER,
  ids: [MCP_STRUCTURED_CONTENT_REFERENCE_ID],
};
export const IMAGE_COMMAND_CREATE_TASK_TOOL_CALL_ID = `${SESSION_ID}:tool:image-create-task`;
export const IMAGE_COMMAND_CREATE_TASK_TOOL_NAME =
  "lime_create_image_generation_task";
export const SKILLS_RUNTIME_SCENARIO =
  createSkillsRuntimeFixtureScenario(SESSION_ID);
export const SKILLS_RUNTIME_EXPLICIT_SCENARIO =
  createExplicitSkillsRuntimeFixtureScenario(SESSION_ID);
export const SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO =
  createManualEnableSkillsRuntimeFixtureScenario(SESSION_ID);
export const EXPERT_SKILLS_RUNTIME_SCENARIO =
  createExpertSkillsRuntimeFixtureScenario(SESSION_ID);
export const EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO =
  createExpertPanelSkillsRuntimeFixtureScenario(SESSION_ID);
export const EXPERT_SKILLS_RUNTIME_SESSION_ID = `${SESSION_ID}-expert-skills`;
export const EXPERT_SKILLS_RUNTIME_THREAD_ID = `${EXPERT_SKILLS_RUNTIME_SESSION_ID}-thread`;
export const EXPERT_SKILLS_RUNTIME_SESSION_TITLE =
  "专家 Skills Runtime Fixture";
export const EVENT_READ_PROBE_PROMPT =
  "验证 direct v2 notification 与 read model 同 turn 对齐。";
export const EVENT_READ_PROBE_TURN_ID = `${SESSION_ID}-event-read-probe`;
export const EVENT_READ_PROBE_READ_TEXT = "事件流 probe 已进入 RuntimeCore";
export const EVENT_READ_PROBE_DONE_TEXT = "EVENT_READ_PROBE_DONE";
export const EVENT_READ_PROBE_TOOL_CALL_ID = `${EVENT_READ_PROBE_TURN_ID}:tool:webfetch`;
export const EVENT_READ_PROBE_TOOL_NAME = "WebFetch";
export const EVENT_READ_PROBE_TOOL_OUTPUT =
  "fixture fetched https://example.com/claw-event-read";
export const WEB_TOOLS_RENDERING_ASSERTION_KEYS = [
  "webToolsRenderingPromptReachedBackend",
  "guiWebToolsRenderingInputSubmitted",
  "guiWebToolsLiveRunningStateCaptured",
  "guiWebToolsLiveNoLegacyTextAfterProcess",
  "guiWebToolsLiveSourcesVisible",
  "guiWebToolsLiveReadPagesVisible",
  "guiWebToolsLiveTimelineOrderPreserved",
  "guiWebToolsCompletedProcessCompacted",
  "guiWebSearchNoiseHidden",
  "guiMarkdownRendered",
  "guiWebToolsFinalTextVisibleAfterCompletion",
  "guiWebFetchTransportEnvelopeHidden",
  "readModelWebToolsRenderingCompleted",
  "readModelWebToolsReasoningProviderMetadataPreserved",
  "guiWebToolsReasoningDidNotOpenPlanRail",
];
export const REASONING_FIRST_VISIBLE_ASSERTION_KEYS = [
  "reasoningFirstVisiblePromptReachedBackend",
  "guiReasoningFirstVisibleInputSubmitted",
  "guiReasoningFirstVisibleBeforeAnswer",
  "guiReasoningFirstVisibleCompleted",
  "readModelReasoningFirstVisibleCompleted",
  "readModelReasoningFirstVisibleItemObserved",
];
export const LIVE_TAIL_COMMIT_ASSERTION_KEYS = [
  "liveTailCommitPromptReachedBackend",
  "guiLiveTailCommitInputSubmitted",
  "guiLiveTailFirstVisibleBeforeCommit",
  "guiLiveTailRunningStatusPreserved",
  "guiLiveTailNoStartupNote",
  "guiLiveTailOverflowCommitted",
  "guiLiveTailTableTailVisible",
  "guiLiveTailCanonicalRepairExactOnce",
  "guiLiveTailScrollAnchorStable",
  "guiLiveTailCompleted",
  "readModelLiveTailCommitCompleted",
  "backendLiveTailCommitRecorded",
];
export const ELECTRON_RESIZE_REFLOW_ASSERTION_KEYS = [
  "electronResizeReflowPromptReachedBackend",
  "guiElectronResizeReflowInputSubmitted",
  "guiElectronResizeReflowCompleted",
  "guiElectronResizeReflowFilesSurfaceOpened",
  "guiElectronResizeReflowViewportSnapshotsCaptured",
  "guiElectronResizeReflowMessageAnchorStable",
  "guiElectronResizeReflowInputbarAnchored",
  "guiElectronResizeReflowRightSurfaceStable",
  "guiElectronResizeReflowNoOverlap",
  "readModelElectronResizeReflowCompleted",
  "backendElectronResizeReflowRecorded",
];
export const APPROVAL_REQUEST_RESUME_ASSERTION_KEYS = [
  "approvalRequestResumePromptReachedBackend",
  "guiApprovalRequestResumeInputSubmitted",
  "guiApprovalRequestResumePendingVisible",
  "readModelApprovalRequestResumePending",
  "approvalRequestResumeUsedCurrentServerRequestResponse",
  "approvalRequestResumeBackendResponseScoped",
  "approvalRequestResumeServerRequestResolved",
  "approvalRequestResumeBackendActionRespondObserved",
  "approvalRequestResumePendingCleared",
  "guiApprovalRequestResumeCompleted",
  "readModelApprovalRequestResumeCompleted",
  "approvalRequestResumeNoLegacyRuntimeRespond",
];
export const APPROVAL_REQUEST_DECISION_ASSERTION_KEYS = [
  "approvalRequestDecisionPromptReachedBackend",
  "guiApprovalRequestDecisionInputSubmitted",
  "guiApprovalRequestDecisionPendingVisible",
  "readModelApprovalRequestDecisionPending",
  "approvalRequestDecisionUsedCurrentServerRequestResponse",
  "approvalRequestDecisionServerRequestResponseScoped",
  "approvalRequestDecisionServerRequestResolved",
  "approvalRequestDecisionBackendActionRespondObserved",
  "approvalRequestDecisionPendingCleared",
  "approvalRequestDeclineNoToolExecuted",
  "guiApprovalRequestDeclineCompleted",
  "readModelApprovalRequestDeclineCompleted",
  "approvalRequestCancelNoToolExecuted",
  "guiApprovalRequestCancelCompleted",
  "readModelApprovalRequestCancelCanceled",
  "approvalRequestDecisionNoRendererActionRespond",
];
export const APPROVAL_REQUEST_HOST_INTERRUPT_ASSERTION_KEYS = [
  "approvalRequestHostInterruptPromptReachedBackend",
  "guiApprovalRequestHostInterruptInputSubmitted",
  "guiApprovalRequestHostInterruptPendingVisible",
  "readModelApprovalRequestHostInterruptPending",
  "approvalRequestHostInterruptUsedCurrentMethod",
  "approvalRequestHostInterruptPayloadScoped",
  "approvalRequestHostInterruptServerRequestResolved",
  "approvalRequestHostInterruptNoRendererResponse",
  "approvalRequestHostInterruptNoBackendActionRespond",
  "approvalRequestHostInterruptPendingCleared",
  "guiApprovalRequestHostInterruptCanceled",
  "readModelApprovalRequestHostInterruptCanceled",
  "approvalRequestHostInterruptCanonicalEventOrder",
];
export const APPROVAL_REQUEST_FULL_ACCESS_ASSERTION_KEYS = [
  "approvalRequestFullAccessPromptReachedBackend",
  "approvalRequestFullAccessUsesFullAccessPolicy",
  "guiApprovalRequestFullAccessInputSubmitted",
  "guiApprovalRequestFullAccessCompleted",
  "guiApprovalRequestFullAccessNoApprovalPrompt",
  "guiApprovalRequestFullAccessNoApprovalRecord",
  "readModelApprovalRequestFullAccessCompleted",
  "readModelApprovalRequestFullAccessNoApprovalRequest",
  "approvalRequestFullAccessNoActionRespond",
  "approvalRequestFullAccessNoLegacyRuntimeRespond",
];
export const TERMINAL_STALE_GUARD_ASSERTION_KEYS = [
  "terminalStaleGuardFirstPromptReachedBackend",
  "terminalStaleGuardSecondPromptReachedBackend",
  "terminalStaleGuardFirstCompleted",
  "terminalStaleGuardSecondInputSubmitted",
  "terminalStaleGuardSecondCompleted",
  "terminalStaleGuardReadModelCompleted",
  "terminalStaleGuardStaleTerminalIgnored",
];
export const TERMINAL_FAILED_AFTER_ANSWER_ASSERTION_KEYS = [
  "terminalFailedAfterAnswerPromptReachedBackend",
  "guiTerminalFailedAfterAnswerInputSubmitted",
  "guiTerminalFailedAfterAnswerPartialRetained",
  "guiTerminalFailedAfterAnswerFailureHiddenFromBody",
  "guiTerminalFailedAfterAnswerNoDuplicates",
  "guiTerminalFailedAfterAnswerInputReady",
  "readModelTerminalFailedAfterAnswerFailed",
  "backendTerminalFailedAfterAnswerRecorded",
];
export const TERMINAL_CANCELED_AFTER_ANSWER_ASSERTION_KEYS = [
  "terminalCanceledAfterAnswerPromptReachedBackend",
  "guiTerminalCanceledAfterAnswerInputSubmitted",
  "guiTerminalCanceledAfterAnswerPartialVisibleBeforeStop",
  "guiTerminalCanceledAfterAnswerStopClicked",
  "guiTerminalCanceledAfterAnswerPartialRetained",
  "guiTerminalCanceledAfterAnswerNoDuplicates",
  "guiTerminalCanceledAfterAnswerInputReady",
  "readModelTerminalCanceledAfterAnswerCanceled",
  "backendTerminalCanceledAfterAnswerRecorded",
];
export const MCP_STRUCTURED_CONTENT_ASSERTION_KEYS = [
  "mcpStructuredContentPromptReachedBackend",
  "guiMcpStructuredContentInputSubmitted",
  "guiMcpStructuredContentVisible",
  "guiMcpStructuredContentEnvelopeHidden",
  "readModelMcpStructuredContentCompleted",
  "readModelMcpStructuredContentObserved",
];
export const IMAGE_COMMAND_ASSERTION_KEYS = [
  "imageCommandPromptReachedBackend",
  "imageCommandMetadataReachedBackend",
  "imageCommandLegacySkillLaunchNotSubmitted",
  "imageCommandUsedCurrentMediaTaskArtifactMethods",
  "imageCommandTaskArtifactWritten",
  "imageCommandTaskArtifactReadable",
  "imageCommandTaskArtifactTerminal",
  "imageCommandTaskArtifactSameTaskUpdated",
  "imageCommandTaskAuditLogWritten",
  "imageCommandTaskAuditLogEventSequence",
  "imageCommandTaskAuditLogNoSensitiveTokens",
  "imageCommandWorkflowAuditReadModelProjected",
  "imageCommandWorkflowAuditStepsProjected",
  "imageCommandWorkflowAuditSummaryRedacted",
  "imageCommandWorkerUsedFixtureProviderAndModel",
  "imageCommandWorkflowToolObserved",
  "imageCommandCreateTaskToolObserved",
  "guiImageCommandInputSubmitted",
  "guiImageCommandToolProcessVisible",
  "guiImageCommandTaskCardVisible",
  "guiImageCommandTaskCardTerminal",
  "guiImageCommandSingleTaskCard",
  "guiImageCommandRestoredAfterReload",
  "guiImageCommandNoDraftCard",
  "guiImageCommandNoTemplateTaskId",
  "readModelImageCommandCompleted",
  "readModelImageCommandTaskPreviewObserved",
];
export const INPUTBAR_RICH_RESTORE_ASSERTION_KEYS = [
  "inputbarRichRestorePromptReachedBackend",
  "inputbarRichRestoreDraftPrepared",
  "inputbarRichRestoreInputSubmitted",
  "inputbarRichRestoreBackendInputSummaryReached",
  "inputbarRichRestoreUsedCurrentTurnCancel",
  "inputbarRichRestoreBackendCanceled",
  "inputbarRichRestoreGuiCanceled",
  "inputbarRichRestoreTextRestored",
  "inputbarRichRestoreImageRestored",
  "inputbarRichRestorePathRestored",
  "inputbarRichRestoreSkillRestored",
  "inputbarRichRestoreNoVisibleAssistantOutput",
  "inputbarRichRestoreReadModelCanceled",
];
export const RIGHT_SURFACE_VISUAL_MATRIX_ASSERTION_KEYS = [
  "rightSurfaceVisualMatrixRequestedThroughAppServer",
  "rightSurfaceVisualMatrixFilesSurfaceVisible",
  "rightSurfaceVisualMatrixObjectCanvasSurfaceVisible",
  "rightSurfaceVisualMatrixExpertSurfaceVisible",
  "rightSurfaceVisualMatrixBrowserSurfaceVisible",
  "rightSurfaceVisualMatrixSurfacesMutuallyExclusive",
  "rightSurfaceVisualMatrixHostsFillRightSide",
  "rightSurfaceVisualMatrixPendingConsumeKeepsSurfaceOpen",
  "rightSurfaceVisualMatrixDoesNotUseModelTurn",
];
export const CONTENT_FACTORY_ARTICLE_WORKSPACE_ASSERTION_KEYS = [
  "contentFactoryArticleWorkspaceArtifactWritePersisted",
  "contentFactoryArticleWorkspaceRightSurfaceRequested",
  "contentFactoryArticleWorkspaceSessionOpenedFromSidebar",
  "contentFactoryArticleWorkspaceRightSurfaceVisible",
  "contentFactoryArticleWorkspaceFinalArticleFrameVisible",
  "contentFactoryArticleWorkspacePageShowsObjects",
  "contentFactoryArticleWorkspaceReadModelProjected",
  "contentFactoryArticleWorkspaceArtifactsProjected",
  "contentFactoryArticleWorkspaceRendererArtifactsProjected",
  "contentFactoryArticleWorkspaceArtifactReadContent",
  "contentFactoryArticleWorkspaceArticleCanvasSurfaceVisible",
  "contentFactoryArticleWorkspaceEditedDraftRestored",
  "contentFactoryArticleWorkspaceWorkspacePatchEvidenceProjected",
  "contentFactoryArticleWorkspaceWorkspacePatchEvidenceScoped",
  "contentFactoryArticleWorkspaceActionResultPatchProjected",
  "contentFactoryArticleWorkspaceStoryboardRendererContractPreserved",
  "contentFactoryArticleWorkspaceDoesNotUseModelTurn",
];
