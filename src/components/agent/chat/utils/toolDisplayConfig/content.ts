import { FilePlus, FileText, Globe, Settings, Wrench } from "lucide-react";
import type { ToolDisplayConfig } from "../toolDisplayTypes";

const CONTENT_TASK_GROUP_TITLE = "内容任务";
const LIME_CREATE_TASK_ACTIONS = {
  failed: "发起失败",
  completed: "已发起",
  running: "发起中",
} as const;
const LIME_CREATE_TASK_ACTION_KEYS = {
  failed: "action.createTask.failed",
  completed: "action.createTask.completed",
  running: "action.createTask.running",
} as const;
const DIRECT_CONTENT_GENERATION_ACTION_KEYS = {
  failed: "action.generate.failed",
  completed: "action.generate.completed",
  running: "action.generate.running",
} as const;
const SERVICE_SKILL_RUN_ACTION_KEYS = {
  failed: "toolCall.action.serviceSkillRun.failed",
  completed: "toolCall.action.serviceSkillRun.completed",
  running: "toolCall.action.serviceSkillRun.running",
} as const;

export const CONTENT_EXACT_TOOL_CONFIGS = [
  [
    "generateimage",
    {
      family: "task",
      label: "图片生成",
      labelKey: "label.imageGeneration",
      verb: "生成",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: {
        failed: "生成失败",
        completed: "已生成",
        running: "生成中",
      },
      actionKeys: DIRECT_CONTENT_GENERATION_ACTION_KEYS,
    },
  ],
  [
    "requestuserinput",
    {
      family: "generic",
      label: "用户输入",
      verb: "收集",
      icon: Wrench,
      groupTitle: "交互",
      actionKey: "generic",
      actions: {
        failed: "收集失败",
        completed: "已收集",
        running: "等待输入",
      },
    },
  ],
  [
    "structuredoutput",
    {
      family: "generic",
      label: "最终答复",
      verb: "整理",
      icon: FileText,
      groupTitle: "回复",
      actionKey: "generic",
      actions: {
        failed: "整理失败",
        completed: "已整理最终答复",
        running: "整理最终答复中",
      },
    },
  ],
  [
    "agent",
    {
      family: "subagent",
      label: "创建子任务",
      verb: "创建",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "subagent",
    },
  ],
  [
    "sendmessage",
    {
      family: "subagent",
      label: "补充说明",
      verb: "发送",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "subagent",
    },
  ],
  [
    "teamcreate",
    {
      family: "subagent",
      label: "创建子代理组",
      verb: "创建",
      icon: Globe,
      groupTitle: "创建子代理组",
      actionKey: "subagent",
      actions: {
        failed: "创建失败",
        completed: "已创建",
        running: "创建中",
      },
    },
  ],
  [
    "teamdelete",
    {
      family: "subagent",
      label: "删除子代理组",
      verb: "删除",
      icon: Globe,
      groupTitle: "删除子代理组",
      actionKey: "subagent",
      actions: {
        failed: "删除失败",
        completed: "已删除",
        running: "删除中",
      },
    },
  ],
  [
    "listpeers",
    {
      family: "list",
      label: "子任务",
      verb: "查看",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "list",
    },
  ],
  [
    "waitagent",
    {
      family: "subagent",
      label: "查看任务进展",
      verb: "查看",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "subagent",
    },
  ],
  [
    "resumeagent",
    {
      family: "subagent",
      label: "继续处理",
      verb: "继续",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "subagent",
    },
  ],
  [
    "closeagent",
    {
      family: "subagent",
      label: "暂停处理",
      verb: "暂停",
      icon: Globe,
      groupTitle: "子任务",
      actionKey: "subagent",
    },
  ],
  [
    "socialgeneratecoverimage",
    {
      family: "task",
      label: "封面图生成",
      labelKey: "label.coverImageGeneration",
      verb: "生成",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: {
        failed: "生成失败",
        completed: "已生成",
        running: "生成中",
      },
      actionKeys: DIRECT_CONTENT_GENERATION_ACTION_KEYS,
    },
  ],
  [
    "videogenerate",
    {
      family: "task",
      label: "视频生成",
      labelKey: "label.videoGeneration",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreateaudiogenerationtask",
    {
      family: "task",
      label: "配音生成",
      labelKey: "label.audioGeneration",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreatetranscriptiontask",
    {
      family: "task",
      label: "转写",
      labelKey: "label.transcription",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreatebroadcastgenerationtask",
    {
      family: "task",
      label: "口播生成",
      labelKey: "label.broadcastGeneration",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreatecovergenerationtask",
    {
      family: "task",
      label: "封面生成",
      labelKey: "label.coverGeneration",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreateresourcesearchtask",
    {
      family: "task",
      label: "素材检索",
      labelKey: "label.resourceSearch",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreatemodalresourcesearchtask",
    {
      family: "task",
      label: "素材检索",
      labelKey: "label.resourceSearch",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreateimagegenerationtask",
    {
      family: "task",
      label: "图片生成",
      labelKey: "label.imageGeneration",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreateurlparsetask",
    {
      family: "task",
      label: "链接解析",
      labelKey: "label.urlParse",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limecreatetypesettingtask",
    {
      family: "task",
      label: "排版",
      labelKey: "label.typesetting",
      verb: "发起",
      icon: FilePlus,
      groupTitle: CONTENT_TASK_GROUP_TITLE,
      groupTitleKey: "groupTitle",
      actionKey: "task",
      actions: LIME_CREATE_TASK_ACTIONS,
      actionKeys: LIME_CREATE_TASK_ACTION_KEYS,
    },
  ],
  [
    "limerunserviceskill",
    {
      family: "skill",
      label: "服务技能执行",
      verb: "执行",
      icon: Settings,
      groupTitle: "服务技能",
      actionKey: "skill",
      actions: {
        failed: "服务技能执行失败",
        completed: "已执行服务技能",
        running: "执行服务技能中",
      },
      actionKeys: SERVICE_SKILL_RUN_ACTION_KEYS,
    },
  ],
] as const satisfies ReadonlyArray<readonly [string, ToolDisplayConfig]>;

export const CONTENT_CREATE_TASK_TOOL_KEYS = new Set([
  "videogenerate",
  "limecreateaudiogenerationtask",
  "limecreatetranscriptiontask",
  "limecreatebroadcastgenerationtask",
  "limecreatecovergenerationtask",
  "limecreateresourcesearchtask",
  "limecreatemodalresourcesearchtask",
  "limecreateimagegenerationtask",
  "limecreateurlparsetask",
  "limecreatetypesettingtask",
]);

export const DIRECT_CONTENT_GENERATION_TOOL_KEYS = new Set([
  "socialgeneratecoverimage",
  "generateimage",
]);

export const CONTENT_TOOL_USER_FACING_COPY: Partial<
  Record<string, { key: string; defaultValue: string }>
> = {
  generateimage: {
    key: "userFacing.imageGeneration",
    defaultValue: "Generate image",
  },
  socialgeneratecoverimage: {
    key: "userFacing.coverImageGeneration",
    defaultValue: "Generate cover image",
  },
  videogenerate: {
    key: "userFacing.videoGeneration",
    defaultValue: "Generate video",
  },
  limecreateaudiogenerationtask: {
    key: "userFacing.audioGeneration",
    defaultValue: "Generate voice",
  },
  limecreatetranscriptiontask: {
    key: "userFacing.transcription",
    defaultValue: "Transcribe audio",
  },
  limecreatebroadcastgenerationtask: {
    key: "userFacing.broadcastGeneration",
    defaultValue: "Generate broadcast",
  },
  limecreatecovergenerationtask: {
    key: "userFacing.coverGeneration",
    defaultValue: "Generate cover",
  },
  limecreateresourcesearchtask: {
    key: "userFacing.resourceSearch",
    defaultValue: "Search assets",
  },
  limecreatemodalresourcesearchtask: {
    key: "userFacing.resourceSearch",
    defaultValue: "Search assets",
  },
  limecreateimagegenerationtask: {
    key: "userFacing.imageGeneration",
    defaultValue: "Generate image",
  },
  limecreateurlparsetask: {
    key: "userFacing.urlParse",
    defaultValue: "Parse URL",
  },
  limecreatetypesettingtask: {
    key: "userFacing.typesetting",
    defaultValue: "Typeset content",
  },
};

export const DIRECT_CONTENT_GROUP_LABEL_COPY: Partial<
  Record<string, { key: string; defaultValue: string }>
> = {
  generateimage: {
    key: "label.image",
    defaultValue: "image",
  },
  socialgeneratecoverimage: {
    key: "label.coverImage",
    defaultValue: "cover image",
  },
};
