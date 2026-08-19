import type {
  BaseSetupCatalogProjection,
  BaseSetupCommandBinding,
  BaseSetupPackage,
  BaseSetupRenderContract,
} from "./types";
import { resolveImageCapabilityModels } from "@/lib/imageGen/catalog";
import { SEEDED_SERVICE_SKILL_CATALOG_VERSION } from "./seededServiceSkillPackage";

const SEEDED_COMMAND_PACKAGE_ID = "lime-seeded-command-catalog";
const SEEDED_COMMAND_BUNDLE_ID = "seeded-command-catalog-bundle";
const SEEDED_COMMAND_SLOT_PROFILE_ID = "seeded-command-input";
const SEEDED_COMMAND_SCORECARD_PROFILE_ID = "seeded-command-scorecard";
const SEEDED_COMMAND_POLICY_PROFILE_ID = "seeded-command-policy";
const SEEDED_COMMAND_TIMELINE_ARTIFACT_PROFILE_ID =
  "seeded-command-timeline-artifact";
const SEEDED_COMMAND_MEDIA_ARTIFACT_PROFILE_ID =
  "seeded-command-media-artifact";
const SEEDED_COMMAND_ARTIFACT_PROFILE_ID = "seeded-command-artifact-bundle";
const SEEDED_COMMAND_FORM_ARTIFACT_PROFILE_ID = "seeded-command-form-artifact";

const COMMAND_TIMELINE_JSON_CONTRACT: BaseSetupRenderContract = {
  resultKind: "tool_timeline",
  detailKind: "json",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_TIMELINE_TASK_CONTRACT: BaseSetupRenderContract = {
  resultKind: "tool_timeline",
  detailKind: "task_detail",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_TIMELINE_SCENE_CONTRACT: BaseSetupRenderContract = {
  resultKind: "tool_timeline",
  detailKind: "scene_detail",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_MEDIA_DETAIL_CONTRACT: BaseSetupRenderContract = {
  resultKind: "tool_timeline",
  detailKind: "media_detail",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_IMAGE_GALLERY_CONTRACT: BaseSetupRenderContract = {
  resultKind: "image_gallery",
  detailKind: "media_detail",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_ARTIFACT_CONTRACT: BaseSetupRenderContract = {
  resultKind: "artifact",
  detailKind: "artifact_detail",
  supportsStreaming: true,
  supportsTimeline: true,
};

const COMMAND_FORM_CONTRACT: BaseSetupRenderContract = {
  resultKind: "form",
  detailKind: "json",
  supportsStreaming: true,
  supportsTimeline: true,
};

interface SeededCommandProjectionSpec {
  commandKey: string;
  title: string;
  summary: string;
  aliases: string[];
  trigger: string;
  triggerHints?: string[];
  category: string;
  outputHint: string;
  commandBinding?: BaseSetupCommandBinding;
  commandRenderContract: BaseSetupRenderContract;
}

function normalizeCatalogLabel(value: string): string {
  return value.trim().toLowerCase();
}

function requireSeededImageCommandModelId(params: {
  providerId: string;
  providerType: string;
  modelName: string;
}): string {
  const models = resolveImageCapabilityModels({
    id: params.providerId,
    type: params.providerType,
  });
  const model = models.find(
    (candidate) =>
      normalizeCatalogLabel(candidate.name) ===
      normalizeCatalogLabel(params.modelName),
  );
  if (!model) {
    throw new Error(
      `missing seeded image command model in image capability catalog: ${params.providerId}/${params.modelName}`,
    );
  }

  return model.id;
}

const SEEDED_NANOBANANA_PRO_MODEL_ID = requireSeededImageCommandModelId({
  providerId: "fal",
  providerType: "fal",
  modelName: "Nano Banana Pro",
});

const SEEDED_GPT_IMAGES_2_MODEL_ID = requireSeededImageCommandModelId({
  providerId: "new-api",
  providerType: "openai",
  modelName: "GPT Images 2",
});

const SEEDED_COMMAND_PROJECTION_SPECS: SeededCommandProjectionSpec[] = [
  {
    commandKey: "image_generate",
    title: "配图",
    summary: "根据文字描述生成新的图片结果。",
    aliases: ["image", "img", "vision 1", "nanobanana", "图片", "生图"],
    trigger: "@配图",
    triggerHints: ["@Vision 1", "@image", "/image"],
    category: "图像生成",
    outputHint: "图片结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
      },
      intentConfirmation: {
        id: "plain_image_generation",
        ruleKey: "agentChat.inputIntent.imageGeneration.rules",
        confirmationKey: "agentChat.inputIntent.imageGeneration.confirm",
        systemPromptKey: "agentChat.inputIntent.imageGeneration.systemPrompt",
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "image_generate_nanobanana_pro",
    title: "Nanobanana Pro",
    summary: "使用 Nanobanana Pro 生成新的图片结果。",
    aliases: [
      "nanobanana",
      "nano banana",
      "nano-banana-pro",
      "banana pro",
      "图片",
      "生图",
    ],
    trigger: "@Nanobanana Pro",
    category: "图像生成",
    outputHint: "Nanobanana Pro 图片结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
        providerId: "fal",
        model: SEEDED_NANOBANANA_PRO_MODEL_ID,
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "image_generate_gpt_images_2",
    title: "GPT Images 2",
    summary: "使用 GPT Images 2 生成新的图片结果。",
    aliases: [
      "gpt images",
      "gpt-images-2",
      "gpt image",
      "openai image",
      "图片",
      "生图",
    ],
    trigger: "@GPT Images 2",
    category: "图像生成",
    outputHint: "GPT Images 2 图片结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
        providerId: "new-api",
        model: SEEDED_GPT_IMAGES_2_MODEL_ID,
        executorMode: "responses_image_generation",
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "image_storyboard",
    title: "分镜",
    summary: "根据主题一次生成多张分镜画面，适合九宫格与镜头草图场景。",
    aliases: ["storyboard", "fenjing", "分镜", "九宫格", "分镜图", "多图配图"],
    trigger: "@分镜",
    category: "图像生成",
    outputHint: "分镜结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
        layoutHint: "storyboard_3x3",
        count: "9",
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "cover_generate",
    title: "封面",
    summary: "根据主题生成平台封面图任务。",
    aliases: ["cover", "fengmian", "封面", "封面图", "头图"],
    trigger: "@封面",
    category: "图像生成",
    outputHint: "封面图结果集",
    commandBinding: {
      skillId: "cover_generate",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "poster_generate",
    title: "海报",
    summary: "围绕活动、产品或主题生成可直接使用的海报视觉。",
    aliases: ["poster", "haibao", "flyer 3", "海报", "活动海报", "宣传海报"],
    trigger: "@海报",
    triggerHints: ["@Flyer 3"],
    category: "图像生成",
    outputHint: "海报图结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "image_edit",
    title: "修图",
    summary: "编辑已有图片并生成新的结果图。",
    aliases: ["edit", "xiutu", "修图", "改图", "图片编辑"],
    trigger: "@修图",
    category: "图像生成",
    outputHint: "修图结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
        mode: "edit",
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "image_variation",
    title: "重绘",
    summary: "基于已有图片或参考图继续重绘新的结果图。",
    aliases: ["variation", "variant", "zhonghui", "重绘", "图片重绘", "变体"],
    trigger: "@重绘",
    category: "图像生成",
    outputHint: "重绘结果集",
    commandBinding: {
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
        mode: "variation",
      },
    },
    commandRenderContract: COMMAND_IMAGE_GALLERY_CONTRACT,
  },
  {
    commandKey: "video_generate",
    title: "视频",
    summary: "根据文字描述发起视频生成。",
    aliases: ["video", "shipin", "视频", "短视频", "生成视频"],
    trigger: "@视频",
    category: "视频生成",
    outputHint: "视频任务与媒体结果",
    commandBinding: {
      skillId: "video_generate",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_MEDIA_DETAIL_CONTRACT,
  },
  {
    commandKey: "voice_runtime",
    title: "配音",
    summary: "把视频或旁白需求切到本地配音技能主链，优先整理首版配音稿。",
    aliases: [
      "voice",
      "dubbing",
      "dub",
      "website voiceover",
      "peiyin",
      "配音",
      "旁白",
      "视频配音",
      "语音配音",
    ],
    trigger: "@配音",
    triggerHints: ["@Website Voiceover"],
    category: "视频创作",
    outputHint: "配音稿与工作区执行结果",
    commandBinding: {
      skillId: "cloud-video-dubbing",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_SCENE_CONTRACT,
  },
  {
    commandKey: "growth_runtime",
    title: "增长跟踪",
    summary:
      "围绕目标账号先产出首版增长策略，再把后续节奏、指标和告警条件挂回持续跟踪主链。",
    aliases: [
      "growth",
      "growth expert",
      "zhangzhang",
      "增长",
      "增长跟踪",
      "账号增长",
      "涨粉",
      "账号表现",
    ],
    trigger: "@增长",
    triggerHints: ["@Growth Expert"],
    category: "内容运营",
    outputHint: "增长策略 + 跟踪指标",
    commandBinding: {
      skillId: "account-performance-tracking",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_SCENE_CONTRACT,
  },
  {
    commandKey: "browser_runtime",
    title: "浏览器",
    summary:
      "把本次输入显式切到真实浏览器执行主链，而不是退回 WebSearch 或普通聊天。",
    aliases: [
      "browser",
      "browse",
      "browser agent",
      "mini tester",
      "web scheduler",
      "web manage",
      "liulanqi",
      "浏览器",
      "网页操作",
      "打开网页",
      "网页任务",
    ],
    trigger: "@浏览器",
    triggerHints: [
      "@Browser Agent",
      "@Mini Tester",
      "@Web Scheduler",
      "@Web Manage",
    ],
    category: "浏览器执行",
    outputHint: "浏览器操作 timeline",
    commandBinding: {
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "presentation_generate",
    title: "PPT",
    summary: "根据目标说明生成一份可直接讲述和继续导出的演示稿草稿。",
    aliases: [
      "ppt",
      "presentation",
      "slides",
      "deck",
      "sales 1",
      "yanjiang",
      "演示",
      "演示稿",
      "路演",
    ],
    trigger: "@PPT",
    triggerHints: ["@Sales 1"],
    category: "创作输出",
    outputHint: "演示稿 artifact",
    commandBinding: {
      skillId: "presentation_generate",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "form_generate",
    title: "表单",
    summary: "根据目标说明生成可直接在聊天区渲染的 A2UI 表单。",
    aliases: ["form", "survey", "biaodan", "wenjuan", "表单", "问卷", "报名表"],
    trigger: "@表单",
    category: "创作输出",
    outputHint: "表单 JSON",
    commandBinding: {
      skillId: "form_generate",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_FORM_CONTRACT,
  },
  {
    commandKey: "webpage_generate",
    title: "网页",
    summary: "根据目标说明生成可直接预览的单文件网页。",
    aliases: [
      "webpage",
      "web",
      "wangye",
      "网页",
      "落地页",
      "landing",
      "web composer",
      "web style",
      "html preview",
      "官网",
      "活动页",
    ],
    trigger: "@网页",
    triggerHints: ["@Web Composer", "@HTML Preview", "@Web Style"],
    category: "创作输出",
    outputHint: "网页 artifact",
    commandBinding: {
      skillId: "webpage_generate",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "writing_runtime",
    title: "写作",
    summary: "把当前输入整理成可继续修改的写作主稿，统一复用现有内容成稿主链。",
    aliases: [
      "writing",
      "write",
      "writing partner",
      "writers 1",
      "blog 1",
      "newsletters pro",
      "web copy",
      "wenan",
      "文案",
      "写稿",
      "起草",
      "blog",
      "newsletter",
    ],
    trigger: "@文案",
    triggerHints: [
      "@Writing Partner",
      "@Writers 1",
      "@Blog 1",
      "@Newsletters Pro",
      "@Web Copy",
    ],
    category: "创作输出",
    outputHint: "写作 artifact",
    commandBinding: {
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "code_runtime",
    title: "代码",
    summary: "把本次输入切到代码编排主链，优先调度工具、子代理与代码团队协作。",
    aliases: ["code", "coding", "code agent", "kaifa", "daima", "代码", "开发"],
    trigger: "@代码",
    triggerHints: ["@code", "@coding", "@开发", "@Code Agent"],
    category: "代码执行",
    outputHint: "代码执行 timeline",
    commandBinding: {
      executionKind: "agent_turn",
      requestDefaults: {
        executionStrategy: "react",
      },
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "channel_preview_runtime",
    title: "渠道预览",
    summary: "把当前内容整理成适合指定平台预览的预览稿与封面建议。",
    aliases: [
      "preview",
      "qudaoyulan",
      "预览",
      "渠道预览",
      "平台预览",
      "首屏预览",
    ],
    trigger: "@渠道预览",
    triggerHints: [
      "@Instagram Preview",
      "@TikTok Preview",
      "@Twitter Preview",
      "@YouTube Preview",
    ],
    category: "内容发布",
    outputHint: "渠道预览 artifact",
    commandBinding: {
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "upload_runtime",
    title: "上传",
    summary: "把当前内容整理成适合目标平台直接上传的上传稿与素材清单。",
    aliases: ["upload", "shangchuan", "上传", "上架", "内容上传", "平台上传"],
    trigger: "@上传",
    category: "内容发布",
    outputHint: "上传稿 artifact",
    commandBinding: {
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "publish_runtime",
    title: "发布",
    summary: "把当前内容导入发布工作流，继续整理发布稿、发布包与平台检查。",
    aliases: [
      "publish",
      "fabu",
      "fawen",
      "投稿",
      "发文",
      "发布",
      "发布稿",
      "发布前检查",
    ],
    trigger: "@发布",
    triggerHints: ["@TikTok Publish", "@Twitter Publish", "@YouTube Publish"],
    category: "内容发布",
    outputHint: "发布包 artifact",
    commandBinding: {
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "broadcast_generate",
    title: "播报",
    summary: "把现有文稿整理成适合口播或播客转换的文本任务。",
    aliases: [
      "broadcast",
      "bobao",
      "speaker 1",
      "播报",
      "播客",
      "口播",
      "podcast",
    ],
    trigger: "@播报",
    triggerHints: ["@Speaker 1"],
    category: "音频创作",
    outputHint: "播报任务 timeline",
    commandBinding: {
      skillId: "broadcast_generate",
      executionKind: "cli",
    },
    commandRenderContract: COMMAND_TIMELINE_TASK_CONTRACT,
  },
  {
    commandKey: "modal_resource_search",
    title: "素材",
    summary: "为当前内容提交图片、BGM、音效等资源检索任务。",
    aliases: [
      "resource",
      "sucai",
      "image search",
      "video search",
      "pinterest image search",
      "fetch image",
      "素材",
      "资源",
      "素材检索",
      "资源检索",
    ],
    trigger: "@素材",
    triggerHints: [
      "@Image Search",
      "@Fetch Image",
      "@Pinterest Image Search",
      "@Video Search",
    ],
    category: "素材检索",
    outputHint: "素材检索任务 timeline",
    commandBinding: {
      skillId: "modal_resource_search",
      executionKind: "cli",
    },
    commandRenderContract: COMMAND_TIMELINE_TASK_CONTRACT,
  },
  {
    commandKey: "research",
    title: "搜索",
    summary: "针对当前主题执行联网检索与轻量调研，返回可引用结论与来源。",
    aliases: [
      "search",
      "research",
      "search agent",
      "instagram research",
      "google search",
      "daily search",
      "sousuo",
      "搜索",
      "检索",
      "调研",
      "联网搜索",
      "最新信息",
    ],
    trigger: "@搜索",
    triggerHints: [
      "@Search",
      "@Google Search",
      "@Daily Search",
      "@Search Agent",
      "@Instagram Research",
    ],
    category: "研究分析",
    outputHint: "搜索 timeline",
    commandBinding: {
      skillId: "research",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "deep_search",
    title: "深搜",
    summary: "针对当前主题执行多轮扩搜与深度调研，输出事实、推断与待确认项。",
    aliases: [
      "deep",
      "deepsearch",
      "researchers pro",
      "shensou",
      "深搜",
      "深度搜索",
      "深入调研",
      "多轮搜索",
    ],
    trigger: "@深搜",
    triggerHints: ["@Researchers Pro"],
    category: "研究分析",
    outputHint: "深搜 timeline",
    commandBinding: {
      skillId: "research",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "research_report",
    title: "研报",
    summary: "围绕主题执行多轮调研并输出结构化研究报告。",
    aliases: [
      "report",
      "report search",
      "research_report",
      "yanbao",
      "研报",
      "研究报告",
      "行业报告",
      "竞品报告",
    ],
    trigger: "@研报",
    triggerHints: ["@Report Search"],
    category: "研究分析",
    outputHint: "研报 artifact",
    commandBinding: {
      skillId: "report_generate",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "competitor_research",
    title: "竞品",
    summary: "围绕竞品对象执行多轮调研，并输出结构化竞品分析与对比结论。",
    aliases: [
      "competitor",
      "competitive",
      "product search",
      "jingpin",
      "竞品",
      "竞品分析",
      "竞品研究",
      "产品对比",
      "竞对",
    ],
    trigger: "@竞品",
    triggerHints: ["@Product Search"],
    category: "研究分析",
    outputHint: "竞品分析 artifact",
    commandBinding: {
      skillId: "report_generate",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "read_pdf",
    title: "读PDF",
    summary:
      "读取本地或工作区 PDF，并通过真实文件读取 timeline 输出结构化解读。",
    aliases: [
      "pdf",
      "read_pdf",
      "dupdf",
      "读PDF",
      "PDF 解读",
      "PDF 阅读",
      "PDF 总结",
    ],
    trigger: "@读PDF",
    category: "研究分析",
    outputHint: "PDF 解读 timeline",
    commandBinding: {
      skillId: "pdf_read",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "file_read_runtime",
    title: "读文件",
    summary:
      "读取本地或工作区文件，并继续挂回 summary 当前主链输出结构化解读。",
    aliases: [
      "read_file",
      "duwenjian",
      "读文件",
      "文件读取",
      "read file content",
    ],
    trigger: "@读文件",
    triggerHints: ["@Read File Content"],
    category: "研究分析",
    outputHint: "文件解读 timeline",
    commandBinding: {
      skillId: "summary",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "knowledge_pack",
    title: "资料",
    summary: "兼容打开输入框里的项目资料面板。",
    aliases: [
      "knowledge",
      "knowledge pack",
      "project knowledge",
      "xiangmuziliao",
      "ziliao",
      "知识库",
      "知识包",
      "项目资料",
      "使用资料",
    ],
    trigger: "@资料",
    category: "项目资料",
    outputHint: "资料引用",
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "knowledge_settle",
    title: "沉淀资料",
    summary: "把当前输入或对话结果整理成可复用的项目资料。",
    aliases: [
      "settle knowledge",
      "save knowledge",
      "knowledge import",
      "chendianziliao",
      "沉淀",
      "沉淀资料",
      "生成资料",
      "添加项目资料",
    ],
    trigger: "@沉淀资料",
    category: "项目资料",
    outputHint: "项目资料草稿",
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "summary",
    title: "总结",
    summary: "提炼当前文本、对话或上下文中的关键结论与重点。",
    aliases: ["summary", "summarize", "zongjie", "总结", "摘要", "提炼重点"],
    trigger: "@总结",
    category: "研究分析",
    outputHint: "总结 timeline",
    commandBinding: {
      skillId: "summary",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "translation",
    title: "翻译",
    summary: "翻译当前文本、对话或文件内容，保留真实 skill / tool timeline。",
    aliases: [
      "translate",
      "translation",
      "write translate",
      "fanyi",
      "翻译",
      "中译英",
      "英译中",
      "多语言翻译",
    ],
    trigger: "@翻译",
    triggerHints: ["@Write Translate"],
    category: "研究分析",
    outputHint: "翻译 timeline",
    commandBinding: {
      skillId: "translation",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "analysis",
    title: "分析",
    summary: "分析当前文本、对话或文件内容，输出判断、依据与待确认项。",
    aliases: ["analysis", "analyze", "fenxi", "分析", "拆解", "研判"],
    trigger: "@分析",
    category: "研究分析",
    outputHint: "分析 timeline",
    commandBinding: {
      skillId: "analysis",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "logo_decomposition",
    title: "Logo拆解",
    summary: "拆解图片或 Logo 的构图、元素、配色与可复用视觉结构。",
    aliases: [
      "logo decomposition",
      "image logo decomposition",
      "logo analyze",
      "logo拆解",
      "图形拆解",
      "logo分析",
    ],
    trigger: "@Logo拆解",
    triggerHints: ["@Image Logo Decomposition"],
    category: "研究分析",
    outputHint: "视觉拆解 timeline",
    commandBinding: {
      skillId: "analysis",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "publish_compliance",
    title: "发布合规",
    summary: "围绕广告法、版权与平台发布风险检查当前内容是否适合发布。",
    aliases: [
      "compliance",
      "hegui",
      "发布合规",
      "内容合规",
      "广告法",
      "版权风险",
    ],
    trigger: "@发布合规",
    category: "研究分析",
    outputHint: "合规检查 timeline",
    commandBinding: {
      skillId: "analysis",
      executionKind: "agent_turn",
    },
    commandRenderContract: COMMAND_TIMELINE_JSON_CONTRACT,
  },
  {
    commandKey: "transcription_generate",
    title: "转写",
    summary: "把音频或视频来源提交为转写任务。",
    aliases: [
      "transcribe",
      "audio extractor",
      "zhuanxie",
      "转写",
      "逐字稿",
      "字幕",
      "语音转文字",
    ],
    trigger: "@转写",
    triggerHints: ["@Audio Extractor"],
    category: "音频创作",
    outputHint: "转写 artifact",
    commandBinding: {
      skillId: "transcription_generate",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "web_scrape",
    title: "抓取",
    summary: "抓取网页正文并提交为可追踪的网页内容任务。",
    aliases: [
      "scrape",
      "web_scrape",
      "fetch",
      "zhuaqu",
      "抓取",
      "网页抓取",
      "网页正文",
      "Web Scrape",
    ],
    trigger: "@抓取",
    triggerHints: ["@Fetch"],
    category: "网页处理",
    outputHint: "网页抓取 artifact",
    commandBinding: {
      skillId: "url_parse",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "webpage_read",
    title: "网页读取",
    summary: "读取页面或链接内容，并提交为可追踪的网页阅读任务。",
    aliases: [
      "web_read",
      "page_read",
      "url summarize",
      "read webpage",
      "get homepage",
      "wangyeduqu",
      "网页读取",
      "读取网页",
      "页面读取",
      "网页总结",
    ],
    trigger: "@网页读取",
    triggerHints: ["@Read Webpage", "@Get Homepage", "@URL Summarize"],
    category: "网页处理",
    outputHint: "网页读取 artifact",
    commandBinding: {
      skillId: "url_parse",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "url_parse",
    title: "链接解析",
    summary: "解析网页链接并提交为可追踪的文本任务。",
    aliases: ["url", "url_parse", "链接", "链接解析", "网页解析"],
    trigger: "@链接解析",
    category: "网页处理",
    outputHint: "链接解析 artifact",
    commandBinding: {
      skillId: "url_parse",
      executionKind: "task_queue",
    },
    commandRenderContract: COMMAND_ARTIFACT_CONTRACT,
  },
  {
    commandKey: "typesetting",
    title: "排版",
    summary: "把现有文稿整理成更适合发布与阅读的排版任务。",
    aliases: ["typesetting", "paiban", "排版", "排版优化", "整理排版"],
    trigger: "@排版",
    category: "内容整理",
    outputHint: "排版任务 timeline",
    commandBinding: {
      skillId: "typesetting",
      executionKind: "cli",
    },
    commandRenderContract: COMMAND_TIMELINE_TASK_CONTRACT,
  },
];

function cloneRenderContract(
  contract: BaseSetupRenderContract,
): BaseSetupRenderContract {
  return { ...contract };
}

function resolveBindingProfileRef(
  binding?: BaseSetupCommandBinding,
): BaseSetupCatalogProjection["bindingProfileRef"] {
  switch (binding?.executionKind) {
    case "native_skill":
      return "native-skill-instant";
    default:
      return "agent-turn-instant";
  }
}

function resolveArtifactProfileRef(
  contract: BaseSetupRenderContract,
): BaseSetupCatalogProjection["artifactProfileRef"] {
  if (
    contract.resultKind === "image_gallery" ||
    contract.detailKind === "media_detail"
  ) {
    return SEEDED_COMMAND_MEDIA_ARTIFACT_PROFILE_ID;
  }
  if (
    contract.resultKind === "artifact" ||
    contract.resultKind === "table_report"
  ) {
    return SEEDED_COMMAND_ARTIFACT_PROFILE_ID;
  }
  if (contract.resultKind === "form") {
    return SEEDED_COMMAND_FORM_ARTIFACT_PROFILE_ID;
  }
  return SEEDED_COMMAND_TIMELINE_ARTIFACT_PROFILE_ID;
}

function buildSeededCommandProjection(
  spec: SeededCommandProjectionSpec,
): BaseSetupCatalogProjection {
  return {
    id: `seeded-command-${spec.commandKey}`,
    targetCatalog: "command_catalog",
    entryKey: spec.commandKey,
    skillKey: spec.commandKey,
    title: spec.title,
    summary: spec.summary,
    category: spec.category,
    outputHint: spec.outputHint,
    bundleRefId: SEEDED_COMMAND_BUNDLE_ID,
    slotProfileRef: SEEDED_COMMAND_SLOT_PROFILE_ID,
    bindingProfileRef: resolveBindingProfileRef(spec.commandBinding),
    artifactProfileRef: resolveArtifactProfileRef(spec.commandRenderContract),
    scorecardProfileRef: SEEDED_COMMAND_SCORECARD_PROFILE_ID,
    policyProfileRef: SEEDED_COMMAND_POLICY_PROFILE_ID,
    aliases: [...spec.aliases],
    triggerHints: [spec.trigger, ...(spec.triggerHints ?? [])],
    commandBinding: spec.commandBinding
      ? { ...spec.commandBinding }
      : undefined,
    commandRenderContract: cloneRenderContract(spec.commandRenderContract),
    version: SEEDED_SERVICE_SKILL_CATALOG_VERSION,
  };
}

const SEEDED_COMMAND_PACKAGE: BaseSetupPackage = {
  id: SEEDED_COMMAND_PACKAGE_ID,
  version: SEEDED_SERVICE_SKILL_CATALOG_VERSION,
  title: "Lime Seeded Command Catalog",
  summary: "Lime 内置命令目录的基础设置包事实源。",
  bundleRefs: [
    {
      id: SEEDED_COMMAND_BUNDLE_ID,
      source: "builtin",
      pathOrUri: "seeded://command-catalog/current",
      kind: "skill_bundle",
    },
  ],
  catalogProjections: SEEDED_COMMAND_PROJECTION_SPECS.map((spec) =>
    buildSeededCommandProjection(spec),
  ),
  slotProfiles: [
    {
      id: SEEDED_COMMAND_SLOT_PROFILE_ID,
      slots: [
        {
          key: "user_input",
          label: "用户输入",
          type: "textarea",
          required: false,
          placeholder:
            "命令命中后继续保留原始输入，由当前输入主链解析具体参数。",
          helpText:
            "当前字段只用于保持 command_catalog projection 的统一装配结构。",
        },
      ],
    },
  ],
  bindingProfiles: [
    {
      id: "agent-turn-instant",
      bindingFamily: "agent_turn",
      runnerType: "instant",
      executionLocation: "client_default",
    },
    {
      id: "native-skill-instant",
      bindingFamily: "native_skill",
      runnerType: "instant",
      executionLocation: "client_default",
    },
  ],
  artifactProfiles: [
    {
      id: SEEDED_COMMAND_TIMELINE_ARTIFACT_PROFILE_ID,
      deliveryContract: "artifact_bundle",
      requiredParts: ["timeline.json"],
      viewerKind: "document",
      outputDestination: "用于承接 timeline / json 类命令结果的兼容装配合同。",
    },
    {
      id: SEEDED_COMMAND_MEDIA_ARTIFACT_PROFILE_ID,
      deliveryContract: "artifact_bundle",
      requiredParts: ["media.json"],
      viewerKind: "artifact_bundle",
      outputDestination: "用于承接图片或媒体详情类命令结果的兼容装配合同。",
    },
    {
      id: SEEDED_COMMAND_ARTIFACT_PROFILE_ID,
      deliveryContract: "artifact_bundle",
      requiredParts: ["index.md"],
      viewerKind: "artifact_bundle",
      outputDestination: "用于承接 artifact 类命令结果的兼容装配合同。",
    },
    {
      id: SEEDED_COMMAND_FORM_ARTIFACT_PROFILE_ID,
      deliveryContract: "artifact_bundle",
      requiredParts: ["form.json"],
      viewerKind: "document",
      outputDestination: "用于承接表单类命令结果的兼容装配合同。",
    },
  ],
  scorecardProfiles: [
    {
      id: SEEDED_COMMAND_SCORECARD_PROFILE_ID,
      metrics: ["activation_rate", "completion_rate"],
    },
  ],
  policyProfiles: [
    {
      id: SEEDED_COMMAND_POLICY_PROFILE_ID,
      enabled: true,
      surfaceScopes: ["mention", "workspace"],
      rolloutStage: "seeded",
    },
  ],
  compatibility: {
    minAppVersion: "1.11.0",
    requiredKernelCapabilities: [
      "agent_turn",
      "native_skill",
      "artifact_viewer",
      "timeline",
    ],
    seededFallback: true,
    compatCatalogProjection: true,
  },
};

export function createSeededCommandCatalogBaseSetupPackage(): BaseSetupPackage {
  return JSON.parse(JSON.stringify(SEEDED_COMMAND_PACKAGE)) as BaseSetupPackage;
}
