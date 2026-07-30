import i18n from "i18next";
import { describe, expect, it } from "vitest";
import commonResources from "../resources/zh-CN/common.json";

type CommonResourceKey = keyof typeof commonResources;

function assertI18nKeyTypes() {
  i18n.t("common.save", { ns: "common" });
  i18n.t("common.cancel", { ns: "common" });
  i18n.t("common.confirm", { ns: "common" });
  i18n.t("common.app.loadingPage", { ns: "common" });
  i18n.t("common.splashScreen.slogan", { ns: "common" });
  i18n.t("common.splashScreen.subtitle", { ns: "common" });
  i18n.t("common.startupLoading.title", { ns: "common", brand: "Lime" });
  i18n.t("common.startupLoading.description", { ns: "common" });
  i18n.t("common.confirmDialog.title", { ns: "common" });
  i18n.t("common.projectSelector.placeholder.project", { ns: "common" });
  i18n.t("common.projectSelector.management.title.workspace", {
    ns: "common",
  });
  i18n.t("common.projectSelector.toast.renameFailed", {
    ns: "common",
    message: "name exists",
  });
  i18n.t("common.projectSelector.remove.dangerDescription", { ns: "common" });
  i18n.t("common.createProjectDialog.title", { ns: "common" });
  i18n.t("common.createProjectDialog.path.select", { ns: "common" });
  i18n.t("common.createProjectDialog.path.conflict", {
    ns: "common",
    name: "Project Alpha",
  });
  i18n.t("common.createProjectDialog.projectType.general", { ns: "common" });
  i18n.t("common.projects.rename.nameRequired", { ns: "common" });
  i18n.t("common.createProjectDialog.error.unknown", { ns: "common" });
  i18n.t("common.deepLink.referral.saved.title", { ns: "common" });
  i18n.t("common.deepLink.referral.claimed.title", { ns: "common" });
  i18n.t("common.deepLink.referral.tenantMismatch.description", {
    ns: "common",
  });
  i18n.t("common.deepLink.referral.claimFailed.fallback", { ns: "common" });
  i18n.t("common.deepLink.oauth.success.description", {
    ns: "common",
    providerName: "Lime Hub",
  });
  i18n.t("common.deepLink.payment.returned.title", { ns: "common" });
  i18n.t("common.app.startup.windows.blockingTitle", { ns: "common" });
  i18n.t("common.app.startup.windows.warningTitle", { ns: "common" });
  i18n.t("common.updateNotification.version.new", { ns: "common" });
  i18n.t("common.execution.latestRunStatus.status.success", { ns: "common" });
  i18n.t("common.oemCloudAccess.auth.googleSynced", { ns: "common" });
  i18n.t("common.oemCloudAccess.auth.browserPreopenTitle", { ns: "common" });
  i18n.t("common.oemCloudAccess.auth.systemBrowserOpenFailedWithMessage", {
    ns: "common",
  });
  i18n.t("common.oemCloudAccess.payment.returnSyncing", { ns: "common" });
  i18n.t("common.oemCloudAccess.session.refreshSuccess", { ns: "common" });
  i18n.t("common.oemCloudAccess.emailCode.sent", { ns: "common" });
  i18n.t("common.oemCloudAccess.apiKey.createSuccess", { ns: "common" });
  i18n.t("common.oemCloudAccess.label.accessMode.session", {
    ns: "common",
  });
  i18n.t("common.oemLimeHubProviderSync.managedKeyAlias", { ns: "common" });
  i18n.t("common.oemLimeHubProviderSync.cloudTokenName", { ns: "common" });
  i18n.t("common.modelSelector.header.title", { ns: "common" });
  i18n.t("common.modelSelector.header.themeFilter", {
    ns: "common",
    theme: "General chat",
  });
  i18n.t("common.modelSelector.provider.section.cloud", { ns: "common" });
  i18n.t("common.modelSelector.model.incompatibleHidden", {
    ns: "common",
    count: 2,
  });
  i18n.t("common.modelSelector.noProvider.dismissAria", { ns: "common" });
  i18n.t("common.modelSelector.state.loginRequired", { ns: "common" });
  i18n.t("common.modelCapabilities.reasoning.active", { ns: "common" });
  i18n.t("common.modelCapabilities.vision.inactive", { ns: "common" });
  i18n.t("skills.contentDialog.loading", { ns: "agent" });
  i18n.t("skills.contentDialog.section.metadata", { ns: "agent" });
  i18n.t("skills.contentDialog.emptyInspection", { ns: "agent" });
  i18n.t("skills.repoManager.title", { ns: "agent" });
  i18n.t("skills.repoManager.action.add", { ns: "agent" });
  i18n.t("skills.repoManager.message.addFailed", {
    ns: "agent",
    message: "network down",
  });
  i18n.t("agentChat.messageList.historicalAssistantPreview.longNotice", {
    ns: "agent",
    countLabel: "12,000",
  });
  i18n.t("agentChat.messageList.historicalTimeline.meta", {
    ns: "agent",
    stepCountLabel: "3",
    meta: "2 tool steps",
  });
  i18n.t("agentChat.messageList.actions.saveToKnowledge", { ns: "agent" });
  i18n.t("agentChat.toolCall.skillContent.action.view", { ns: "agent" });
  i18n.t("agentChat.toolCall.skillContent.action.expandBody", {
    ns: "agent",
  });
  i18n.t("agentChat.toolCall.skillContent.action.collapseBody", {
    ns: "agent",
  });
  i18n.t("agentChat.toolCall.skillContent.meta.bytes", {
    ns: "agent",
    count: 1024,
  });
  i18n.t("agentChat.toolCall.commandSummary.title", { ns: "agent" });
  i18n.t("agentChat.toolCall.commandSummary.command", { ns: "agent" });
  i18n.t("agentChat.toolCall.commandSummary.cwd", { ns: "agent" });
  i18n.t("agentChat.toolCall.commandSummary.exitCode", {
    ns: "agent",
    value: 0,
  });
  i18n.t("agentChat.toolCall.commandSummary.stdout", {
    ns: "agent",
    value: 128,
  });
  i18n.t("agentChat.toolCall.commandSummary.stderr", {
    ns: "agent",
    value: 0,
  });
  i18n.t("agentChat.toolCall.commandSummary.sandboxEnabled", { ns: "agent" });
  i18n.t("agentChat.toolCall.commandSummary.sandboxEnabledWithType", {
    ns: "agent",
    type: "workspace-write",
  });
  i18n.t("agentChat.toolCall.commandSummary.sandboxDisabled", { ns: "agent" });
  i18n.t("agentChat.toolCall.commandSummary.truncated", { ns: "agent" });
  i18n.t("agentChat.messageList.artifact.saveDocument", { ns: "agent" });
  i18n.t("agentChat.messageList.artifact.checkpointBadge", { ns: "agent" });
  i18n.t("agentChat.messageList.artifact.diffBadge", {
    ns: "agent",
    count: 1,
  });
  i18n.t("agentChat.messageList.artifact.validationBadge", {
    ns: "agent",
    count: 2,
  });
  i18n.t("agentChat.messageList.artifact.agentUiProjectionTitle", {
    ns: "agent",
  });
  i18n.t("agentChat.messageList.artifact.agentUiBadgePrefix", {
    ns: "agent",
  });
  i18n.t("agentChat.messageList.artifact.blockCount", {
    ns: "agent",
    count: 2,
  });
  i18n.t("agentChat.messageList.artifact.sourceCount", {
    ns: "agent",
    count: 1,
  });
  i18n.t("agentChat.messageList.artifact.locateBlock", {
    ns: "agent",
    label: "摘要",
  });
  i18n.t("agentChat.messageList.artifact.documentSyncedPreview", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.queuedTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.permissionHandledTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.submittedTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.submittedAnswerLabel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.queuedDescription", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.submittedDescription", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.infoRequiredTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.defaultInfoPrompt", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.answerLabel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.answerPlaceholder", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.supplementLabel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.supplementPlaceholder", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.assistantQuestionTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.otherLabel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.otherPlaceholder", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.multiSelectHint", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.fallbackQueuedDescription", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.permissionRequestTitle", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.assistantWantsUse", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.unknownTool", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.permission.inputbarOnlyHint", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.action.submitting", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.submit", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.cancelling", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.cancel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.recording", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.recordAnswer", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.action.submitAnswer", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.runtimePermission.title", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.runtimePermission.description", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.runtimePermission.submittedTitle", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.runtimePermission.submittedDescription", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.runtimePermission.deniedDescription", {
    ns: "agent",
  });
  i18n.t("agentChat.decisionPanel.permission.resultLabel", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.permission.result.allowed", { ns: "agent" });
  i18n.t("agentChat.decisionPanel.permission.result.denied", { ns: "agent" });
  i18n.t("agentChat.messageList.history.windowEntrySummaryRestored", {
    ns: "agent",
    loaded: "40",
    hidden: "30",
  });
  i18n.t("agentChat.messageList.history.persistedEntrySummary", {
    ns: "agent",
    loaded: "40",
  });
  i18n.t("agentChat.messageList.taskCenterEmpty.title", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.badge.recent", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.badge.browserAssist", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.badge.readyMade", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.badge.installed", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.badge.customScene", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.catalogSource.seeded", { ns: "agent" });
  i18n.t("agentChat.serviceSkills.catalogSource.synced", { ns: "agent" });
  i18n.t("agentExperts.title", { ns: "agent" });
  i18n.t("agentExperts.info.sections.workflow", { ns: "agent" });
  i18n.t("agentExperts.info.boundary.soul", { ns: "agent" });
  i18n.t("agentExperts.chat.runtimePrompt", {
    ns: "agent",
    title: "合同审阅",
    summary: "审查合同风险",
    personaRef: "expert-persona:contract-reviewer@1.0.0",
    memoryTemplateRef: "memory-template:contract-reviewer@1.0.0",
    skillRefs: "skill:document-review",
    workflowRefs: "workflow:contract-risk-review",
    starter: "帮我审这份协议",
  });
  i18n.t("agentChat.home.composer.placeholder", { ns: "agent" });
  i18n.t("agentChat.home.composer.autoLaunchExample", {
    ns: "agent",
    example: "open example.com",
  });
  i18n.t("agentChat.home.composer.pathReferenceFallbackPrompt", {
    ns: "agent",
  });
  i18n.t("agentChat.home.composer.guideHelpDefaultLabel", { ns: "agent" });
  i18n.t("agentChat.home.composer.guideHelpClose", { ns: "agent" });
  i18n.t("agentChat.home.composer.guideHelpCloseWithLabel", {
    ns: "agent",
    label: "Lime guide help",
  });
  i18n.t("agentChat.home.composer.advancedSettings.label", { ns: "agent" });
  i18n.t("agentChat.home.composer.advancedSettings.expand", { ns: "agent" });
  i18n.t("agentChat.home.composer.advancedSettings.collapse", {
    ns: "agent",
  });
  i18n.t("agentChat.home.composer.currentModel.label", { ns: "agent" });
  i18n.t("agentChat.home.composer.currentModel.title", {
    ns: "agent",
    model: "OpenAI / gpt-4.1",
  });
  i18n.t("agentChat.home.composer.fileManager.open", { ns: "agent" });
  i18n.t("agentChat.home.composer.fileManager.close", { ns: "agent" });
  i18n.t("agentChat.home.composer.creationMode.label", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.dictation.recording", {
    ns: "agent",
    duration: "0:12",
  });
  i18n.t("agentChat.inputbar.core.dictation.recordingLabel", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.dictation.transcribing", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.dictation.polishing", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.dictation.liveTranscript", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.core.dictation.polishingTitle", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.core.dictation.transcribingTitle", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.core.dictation.stopRecording", {
    ns: "agent",
    label: "Recording 0:12",
  });
  i18n.t("agentChat.inputbar.core.dictation.start", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.dictation.disabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.placeholder.fullscreen", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.placeholder.default", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.image.previewAlt", {
    ns: "agent",
    index: 1,
  });
  i18n.t("agentChat.inputbar.core.image.remove", {
    ns: "agent",
    index: 1,
  });
  i18n.t("agentChat.inputbar.core.image.add", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.path.containerLabel", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.path.localFolder", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.path.localFile", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.path.importAsKnowledge", {
    ns: "agent",
    name: "brief.txt",
  });
  i18n.t("agentChat.inputbar.core.path.importAction", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.path.remove", {
    ns: "agent",
    name: "brief.txt",
  });
  i18n.t("agentChat.inputbar.core.suggestion.acceptTitle", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.textarea.expand", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.textarea.collapse", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.action.defer", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.action.stop", { ns: "agent" });
  i18n.t("agentChat.inputbar.core.action.send", { ns: "agent" });
  i18n.t("agentChat.inputbar.dictation.feedback.unavailable", { ns: "agent" });
  i18n.t("agentChat.inputbar.dictation.feedback.enableInSettings", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.dictation.feedback.downloadVoiceModel", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.dictation.feedback.startFailed", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.dictation.feedback.tooShort", { ns: "agent" });
  i18n.t("agentChat.inputbar.dictation.feedback.emptyTranscript", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.dictation.feedback.recognitionFailed", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.controller.toast.systemPathDropUnsupported", {
    ns: "agent",
  });
  i18n.t(
    "agentChat.inputbar.controller.curatedTask.reviewSuggestionPrefillHint",
    { ns: "agent" },
  );
  i18n.t("agentChat.inputbar.composer.advancedSettings.label", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.composer.advancedSettings.expand", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.composer.advancedSettings.collapse", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.composer.currentModel.label", { ns: "agent" });
  i18n.t("agentChat.inputbar.composer.currentModel.title", {
    ns: "agent",
    model: "OpenAI / gpt-4.1",
  });
  i18n.t("agentChat.inputbar.composer.fileManager.open", { ns: "agent" });
  i18n.t("agentChat.inputbar.composer.fileManager.close", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.open", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.addFiles", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.attachKnowledge", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.planMode", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.objective", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.plugins", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.skills", { ns: "agent" });
  i18n.t("agentChat.inputbar.plusMenu.unavailable", { ns: "agent" });
  i18n.t("agentChat.inputbar.pluginChip.empty", { ns: "agent" });
  i18n.t("agentChat.inputbar.pluginChip.remove", {
    ns: "agent",
    name: "内容工厂",
  });
  i18n.t("agentChat.inputbar.pluginChip.selectorTitle", { ns: "agent" });
  i18n.t("agentChat.inputbar.pluginChip.unavailable", { ns: "agent" });
  i18n.t("agentChat.inputbar.composer.workspacePlaceholder.waiting", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.composer.workspacePlaceholder.taskCenter", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.composer.workspacePlaceholder.default", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.tools.task.label", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.task.title.enabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.task.title.disabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.task.toast.enabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.task.toast.disabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.subagent.label", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.subagent.title.enabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.subagent.title.disabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.subagent.toast.enabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.subagent.toast.disabled", { ns: "agent" });
  i18n.t("agentChat.inputbar.tools.fullscreen.toast.entered", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.tools.fullscreen.toast.exited", { ns: "agent" });
  i18n.t("agentChat.inputbar.imageAttachments.unnamedImage", { ns: "agent" });
  i18n.t("agentChat.inputbar.imageAttachments.imageReadFailed", {
    ns: "agent",
    fileName: "cover.png",
  });
  i18n.t("agentChat.inputbar.imageAttachments.imageAdded", {
    ns: "agent",
    fileName: "cover.png",
  });
  i18n.t("agentChat.inputbar.imageAttachments.imagePasted", { ns: "agent" });
  i18n.t("agentChat.inputbar.knowledge.fallbackPackLabel", { ns: "agent" });
  i18n.t("agentChat.inputbar.knowledge.toggle.label.enabledWithCompanions", {
    ns: "agent",
    count: 1,
    label: "Brand pack",
  });
  i18n.t("agentChat.inputbar.knowledge.state.using.title", {
    ns: "agent",
    label: "Brand pack",
  });
  i18n.t("agentChat.inputbar.knowledge.pendingNotice", {
    ns: "agent",
    count: 1,
  });
  i18n.t("agentChat.inputbar.knowledge.action.check", { ns: "agent" });
  i18n.t("agentChat.inputbar.knowledge.meta.companion", { ns: "agent" });
  i18n.t("agentChat.inputbar.workflow.status.active", { ns: "agent" });
  i18n.t("agentChat.inputbar.workflow.status.waitingDecision", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.workflow.summary.currentProgress", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.workflow.summary.activeWithTrailing", {
    ns: "agent",
    count: 2,
  });
  i18n.t("agentChat.inputbar.workflow.progress.completed", {
    ns: "agent",
    completed: 1,
    total: 4,
  });
  i18n.t("agentChat.inputbar.workflow.queue.title", { ns: "agent" });
  i18n.t("agentChat.inputbar.workflow.queue.hiddenSuffix", {
    ns: "agent",
    count: 2,
    progressLabel: "Completed 1/4",
  });
  i18n.t("agentChat.inputbar.workflow.queue.itemMetaWithProgress", {
    ns: "agent",
    status: "In progress",
    completed: 1,
    total: 4,
  });
  i18n.t("agentChat.inputbar.workflow.quickAction.topicOptions.label", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.workflow.quickAction.topicOptions.prompt", {
    ns: "agent",
  });
  i18n.t("agentChat.inputbar.workflow.quickAction.nextStep.label", {
    ns: "agent",
  });
  i18n.t("agentChat.home.hero.slogan", { ns: "agent" });
  i18n.t("agentChat.home.toast.imageReadFailed", {
    ns: "agent",
    fileName: "cover.png",
  });
  i18n.t("agentChat.home.curatedTask.reviewSuggestionPrefillHint", {
    ns: "agent",
  });
  i18n.t("agentChat.home.quickActions.title", { ns: "agent" });
  i18n.t("agentChat.home.quickActions.badge", {
    ns: "agent",
    icon: "✨",
  });
  i18n.t("agentChat.home.quickActions.preset.generateImage.prompt", {
    ns: "agent",
  });
  i18n.t("agentChat.home.quickActions.preset.researchMode.label", {
    ns: "agent",
  });
  i18n.t("agentChat.home.guideHelp.contextLabelWithStarter", {
    ns: "agent",
    label: "Guide help",
  });
  i18n.t("agentChat.home.starter.rowLabel", { ns: "agent" });
  i18n.t("agentChat.home.starter.guideHelp.label", { ns: "agent" });
  i18n.t("agentChat.home.starter.ppt.prompt", { ns: "agent" });
  i18n.t("agentChat.home.guide.longTermPlan.title", { ns: "agent" });
  i18n.t("agentChat.home.guideCards.label", { ns: "agent" });
  i18n.t("agentChat.home.gallery.title", { ns: "agent" });
  i18n.t("agentChat.home.category.recent", { ns: "agent" });
  i18n.t("agentChat.home.sceneManager.title.list", { ns: "agent" });
  i18n.t("agentChat.home.sceneManager.action.addScene", { ns: "agent" });
  i18n.t("agentChat.home.sceneManager.error.syncSignInRequired", {
    ns: "agent",
  });
  i18n.t("agentChat.home.sceneManager.row.moveUp", {
    ns: "agent",
    title: "趋势简报",
  });
  i18n.t("agentChat.home.sceneManager.field.sceneName.placeholder", {
    ns: "agent",
  });
  i18n.t("agentChat.home.sceneManager.action.done", { ns: "agent" });
  i18n.t("skills.skillCard.source.builtin", { ns: "agent" });
  i18n.t("skills.skillCard.action.install", { ns: "agent" });
  i18n.t("skills.skillCard.compliance.standard", { ns: "agent" });
  i18n.t("skills.workflowProgress.empty", { ns: "agent" });
  i18n.t("skills.workflowProgress.status.retrying", { ns: "agent" });
  i18n.t("skills.workflowProgress.summary.running", {
    ns: "agent",
    current: 1,
    total: 3,
  });
  i18n.t("skills.page.title", { ns: "agent" });
  i18n.t("skills.page.action.refresh", { ns: "agent" });
  i18n.t("skills.page.sections.remote.empty", { ns: "agent" });
  i18n.t("skills.workspace.header.title", { ns: "agent" });
  i18n.t("skills.workspace.serviceSkill.runner.instant.label", {
    ns: "agent",
  });
  i18n.t("skills.workspace.serviceSkill.factItems.separator", { ns: "agent" });
  i18n.t("skills.workspace.serviceSkill.prefill.recentServiceHint", {
    ns: "agent",
    title: "Deep Research",
  });
  i18n.t("skills.workspace.serviceSkill.prefill.recentSceneHint", {
    ns: "agent",
    title: "/x export",
  });
  i18n.t("skills.workspace.serviceSkill.prefill.filledWithMore", {
    ns: "agent",
    items: "Topic",
    remaining: "1",
    total: "2",
  });
  i18n.t("skills.workspace.serviceSkill.prefill.creationReplay.hint", {
    ns: "agent",
    fields: "Topic",
    source: "the current idea",
  });
  i18n.t("sceneGate.title", {
    ns: "agent",
    title: "X Article Export",
  });
  i18n.t("sceneGate.project.label", { ns: "agent" });
  i18n.t("sceneGate.validation.missingSlotsAndProject", {
    ns: "agent",
    fields: "Article URL",
    project: "Project workspace",
  });
  i18n.t("sceneGate.toast.launchFailed", {
    ns: "agent",
    message: "network down",
  });
  i18n.t("curatedTask.sceneAppReference.followUpBanner", {
    ns: "agent",
    title: "Break Down a Viral Piece",
  });
  i18n.t("curatedTask.sceneAppReference.highlight.status", {
    ns: "agent",
    value: "Ready to review",
  });
  i18n.t("curatedTask.sceneAppReference.resultSummaryFallback", {
    ns: "agent",
  });
  i18n.t("sceneAppExecutionSummary.title", { ns: "agent" });
  i18n.t("sceneAppExecutionSummary.overview.referenceCount", {
    ns: "agent",
    count: 2,
  });
  i18n.t("sceneAppExecutionSummary.followup.action.saveAsSkill", {
    ns: "agent",
  });
  i18n.t("generalWorkbench.workflow.current.title", { ns: "agent" });
  i18n.t("generalWorkbench.workflow.queue.hiddenCount", {
    ns: "agent",
    visible: 2,
    hidden: 1,
  });
  i18n.t("generalWorkbench.workflow.outputs.copyPath", { ns: "agent" });
  i18n.t("generalWorkbench.workflow.runDetail.artifactCount", {
    ns: "agent",
    count: 3,
  });
  i18n.t("generalWorkbench.workflow.baseline.badge", { ns: "agent" });
  i18n.t("generalWorkbench.workflow.outputs.summary.latestTitle", {
    ns: "agent",
    label: "Drafts",
  });
  i18n.t("generalWorkbench.workflow.activity.summary.latestTitle", {
    ns: "agent",
    name: "Tool run",
  });
  i18n.t("generalWorkbench.workflow.activity.viewRun", {
    ns: "agent",
    run: "abc123",
  });
  i18n.t("generalWorkbench.workflow.runDetail.summary.curatedTask", {
    ns: "agent",
    title: "Review",
  });
  i18n.t("generalWorkbench.workflow.followUp.applyAria", {
    ns: "agent",
    action: "Expand",
  });
  i18n.t("generalWorkbench.workflow.branch.status.inProgress", {
    ns: "agent",
  });
  i18n.t("generalWorkbench.workflow.branch.summary.multiple.draft", {
    ns: "agent",
    title: "Draft A",
    count: 2,
  });
  i18n.t("generalWorkbench.workflow.handoff.defaultContinuing", {
    ns: "agent",
    branchTitle: "Continuable drafts",
  });
  i18n.t("generalWorkbench.workflow.followUp.prompt.withTask", {
    ns: "agent",
    title: "Review",
    action: "Expand",
  });
  i18n.t("generalWorkbench.workflow.reviewFeedback.followUpBanner", {
    ns: "agent",
    title: "Next task",
  });
  i18n.t("generalWorkbench.context.search.title", { ns: "agent" });
  i18n.t("generalWorkbench.context.search.placeholder", { ns: "agent" });
  i18n.t("generalWorkbench.context.detail.query", {
    ns: "agent",
    query: "brand 2026",
  });
  i18n.t("generalWorkbench.context.detail.notFound", { ns: "agent" });
  i18n.t("generalWorkbench.context.detail.sourceTokens", {
    ns: "agent",
    source: "Web search",
    tokens: 1200,
  });
  i18n.t("generalWorkbench.context.list.summary", {
    ns: "agent",
    active: 1,
    limit: 12,
    searchCount: 2,
    estimatedTokens: 600,
    tokenLimit: 32000,
  });
  i18n.t("generalWorkbench.context.modal.add.uploadAria", { ns: "agent" });
  i18n.t("generalWorkbench.context.modal.text.placeholder", { ns: "agent" });
  i18n.t("skills.workspace.managedJob.created", {
    ns: "agent",
    name: "Daily Review",
  });
  i18n.t("curatedTask.templates.recommendation.recentReviewReasonLabel", {
    ns: "agent",
  });
  i18n.t("curatedTask.badge.review.suggestedTask", {
    ns: "agent",
    title: "Review",
  });
  i18n.t("curatedTask.badge.sceneApp.failureSignal", {
    ns: "agent",
    value: "Needs review",
  });
  i18n.t("skills.scaffoldDialog.title", { ns: "agent" });
  i18n.t("skills.scaffoldDialog.action.create", { ns: "agent" });
  i18n.t("skills.scaffoldDialog.sourceHint", {
    ns: "agent",
    hint: "research result",
  });
  i18n.t("skills.scaffoldDialog.validation.directory", { ns: "agent" });
  i18n.t("agentChat.sessionOverview.panel.title", { ns: "agent" });
  i18n.t("agentChat.sessionOverview.status.turn.running", { ns: "agent" });
  i18n.t("agentChat.sessionOverview.timeline.empty", { ns: "agent" });
  i18n.t("agentChat.sessionOverview.metrics.inProgress", {
    ns: "agent",
    countLabel: "2",
  });
  i18n.t("agentChat.imageWorkbenchPreview.media.alt", { ns: "agent" });
  i18n.t("agentChat.imageWorkbenchPreview.media.open", { ns: "agent" });
  i18n.t("agentChat.imageWorkbenchPreview.action.retry", { ns: "agent" });
  i18n.t("agentChat.imageWorkbenchPreview.placeholder.generating", {
    ns: "agent",
  });
  i18n.t("agentChat.imageWorkbenchPreview.placeholder.failed", {
    ns: "agent",
  });
  i18n.t("agentChat.contentWorkbenchTools.groupTitle", { ns: "agent" });
  i18n.t("agentChat.contentWorkbenchTools.label.audioGeneration", {
    ns: "agent",
  });
  i18n.t("agentChat.contentWorkbenchTools.failure.videoGeneration", {
    ns: "agent",
  });
  i18n.t("agentChat.contentWorkbenchTools.group.multiple.completed", {
    ns: "agent",
    count: 2,
  });
  i18n.t("agentChat.searchResultPreview.expandMore", {
    ns: "agent",
    countLabel: "2",
  });
  i18n.t("agentChat.searchResultPreview.openAria", {
    ns: "agent",
    title: "Example",
  });
  i18n.t("agentChat.harness.codeWorkbench.title", { ns: "agent" });
  i18n.t("agentChat.harness.codeWorkbench.stage.review.description", {
    ns: "agent",
    pending: 2,
    total: 3,
  });
  i18n.t("agentChat.harness.codeWorkbench.metric.fileChanges", {
    ns: "agent",
    pending: 2,
    total: 3,
  });
  i18n.t("agentChat.harness.codeWorkbench.metric.checkpoints", {
    ns: "agent",
  });
  i18n.t("agentChat.harness.codeReview.description", {
    ns: "agent",
    files: 1,
    outputs: 2,
    checkpoints: 3,
  });
  i18n.t("agentChat.harness.codeReview.metric.outputsPassing", {
    ns: "agent",
    count: 1,
  });
  i18n.t("agentChat.harness.codeReview.metric.checkpointsEmpty", {
    ns: "agent",
  });
  i18n.t("agentChat.harnessVerification.section.title", { ns: "agent" });
  i18n.t("agentChat.harnessVerification.artifact.description", {
    ns: "agent",
    fallbackCount: "0",
    issueCount: "2",
    recordCount: "1",
    repairedCount: "1",
  });
  i18n.t("agentChat.agentUiProjection.eventType.artifact.preview.ready", {
    ns: "agent",
  });
  i18n.t("agentChat.agentUiProjection.sourceType.artifact_snapshot", {
    ns: "agent",
  });
  i18n.t("agentChat.agentUiProjection.phase.completed", { ns: "agent" });
  i18n.t("agentChat.agentUiProjection.control.unknown", { ns: "agent" });
  i18n.t("agentChat.agentUiProjection.lane.team-topology.label", {
    ns: "agent",
  });
  i18n.t("agentChat.agentUiProjection.surface.work_board.description", {
    ns: "agent",
  });
  i18n.t("agentChat.incidentPanel.empty", { ns: "agent" });
  i18n.t("agentChat.incidentPanel.priorityBadge", {
    ns: "agent",
    severity: "High",
  });
  i18n.t("navigation.sidebar.items.homeGeneral", { ns: "navigation" });
  i18n.t(
    "workspace.browserExistingSession.presentation.status.attached.label",
    {
      ns: "workspace",
    },
  );
  i18n.t("workspace.video.promptInput.title", { ns: "workspace" });
  i18n.t("workspace.video.canvas.sidebar.collapse", { ns: "workspace" });
  i18n.t("workspace.video.workspace.summary.currentModel.label", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.recentTasks.title", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.taskStatus.success", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.taskSync.saved.label", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.session.title", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.session.preview.title", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.focusedTask.title", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.focusedTask.source.label", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.generate.submitted", {
    ns: "workspace",
  });
  i18n.t("workspace.video.workspace.reference.unsupportedFormat", {
    ns: "workspace",
  });
  i18n.t("workspace.video.sidebar.intro.title", { ns: "workspace" });
  i18n.t("workspace.video.sidebar.reference.start.title", { ns: "workspace" });
  i18n.t("workspace.video.sidebar.reference.empty.action", {
    ns: "workspace",
  });
  i18n.t("workspace.video.sidebar.controls.seed.placeholder", {
    ns: "workspace",
  });
  i18n.t("workspace.video.sidebar.controls.cameraFixed.tipContent", {
    ns: "workspace",
  });
  i18n.t("workspace.video.sidebar.model.panel.title", {
    ns: "workspace",
  });
  i18n.t("workspace.video.sidebar.model.meta.sora2Pro.description", {
    ns: "workspace",
  });
  i18n.t("workspace.document.editor.placeholder", { ns: "workspace" });
  i18n.t("workspace.document.editor.slashCommand.items.image.title", {
    ns: "workspace",
  });
  i18n.t("workspace.document.editor.slashCommand.prompt.imageUrl", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactToolbar.type.code", { ns: "workspace" });
  i18n.t("workspace.artifactToolbar.writePhase.streaming", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactToolbar.action.copyContent", { ns: "workspace" });
  i18n.t("workspace.artifactRenderer.empty.failed.title", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.fallback.unknownType", {
    ns: "workspace",
  });
  i18n.t("workspace.canvasAdapter.action.openFullEditor", {
    ns: "workspace",
  });
  i18n.t("workspace.canvasBreadcrumb.backHome", { ns: "workspace" });
  i18n.t("workspace.canvasAdapter.designPreview.visibleLayers", {
    ns: "workspace",
  });
  i18n.t("workspace.browserAssistRenderer.launching.title", {
    ns: "workspace",
  });
  i18n.t("workspace.browserAssistRenderer.launching.detail", {
    ns: "workspace",
  });
  i18n.t("workspace.browserAssistRenderer.failed.attachedOnlyWithDetail", {
    ns: "workspace",
    target: "浏览器协助",
    detail: "Chrome Bridge 未连接",
  });
  i18n.t("workspace.browserAssistRenderer.replay.title", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactDocument.kind.report", { ns: "workspace" });
  i18n.t("workspace.artifactDocument.stats.blocks.label", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactDocumentBlock.citation.titleFallback", {
    ns: "workspace",
  });
  i18n.t("workspace.a2uiSubmissionNotice.action.expand", {
    ns: "workspace",
  });
  i18n.t("workspace.pendingA2UI.status.progressStep", {
    ns: "workspace",
  });
  i18n.t("workspace.resourceManager.pdf.missingTitle", { ns: "workspace" });
  i18n.t("workspace.resourceManager.pdf.frameTitle", { ns: "workspace" });
  i18n.t("workspace.resourceManager.media.titleFallback", {
    ns: "workspace",
  });
  i18n.t("workspace.resourceManager.media.error.description", {
    ns: "workspace",
  });
  i18n.t("workspace.resourceManager.unsupported.title", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewFallback.systemOpen.title", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewFallback.unsupported.detail", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewSource.source.url", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewSource.source.databaseRecord", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewSource.source.app", {
    ns: "workspace",
  });
  i18n.t("workspace.artifactRenderer.previewSource.empty", {
    ns: "workspace",
  });
  i18n.t("workspace.resourceManager.unsupported.mimeType", {
    ns: "workspace",
    mimeType: "application/octet-stream",
  });
  i18n.t("settings.layout.sidebar.experimentalBadge", { ns: "settings" });
  i18n.t("settings.layout.action.backHome", { ns: "settings" });
  i18n.t("settings.layout.loading.profile", { ns: "settings" });
  i18n.t("settings.profile.field.nickname.label", { ns: "settings" });
  i18n.t("settings.profile.field.tipAria", {
    ns: "settings",
    label: "Nickname",
  });
  i18n.t("settings.profile.field.tipsHint", { ns: "settings" });
  i18n.t("settings.profile.action.save", { ns: "settings" });
  i18n.t("settings.profile.message.saved", { ns: "settings" });
  i18n.t("settings.profile.message.saveFailed", {
    ns: "settings",
    error: "network",
  });
  i18n.t("settings.profile.message.fileTooLarge", {
    ns: "settings",
    size: "6",
    max: "5",
  });
  i18n.t("settings.profile.message.avatarWip", { ns: "settings" });
  i18n.t("settings.profile.message.uploadFailed", {
    ns: "settings",
    error: "network",
  });
  i18n.t("settings.profile.status.complete", { ns: "settings" });
  i18n.t("settings.profile.suggestedTag.programming", { ns: "settings" });
  i18n.t("settings.profile.hero.status", {
    ns: "settings",
    status: "Complete profile",
  });
  i18n.t("settings.profile.avatar.extraTags", {
    ns: "settings",
    countLabel: "2",
  });
  i18n.t("settings.profile.basic.itemCount", {
    ns: "settings",
    countLabel: "3",
  });
  i18n.t("settings.profile.tags.selectedCount", {
    ns: "settings",
    countLabel: "2",
  });
  i18n.t("settings.profile.tags.removeAria", {
    ns: "settings",
    tag: "Programming",
  });
  i18n.t("settings.profile.usage.tipTags", { ns: "settings" });
  i18n.t("settings.layout.placeholder.notFound", { ns: "settings" });
  i18n.t("settings.layout.sidebar.floatingNav.openAria", { ns: "settings" });
  i18n.t("settings.about.version.label", {
    ns: "settings",
    version: "1.10.0",
    build: "1.10.0",
  });
  i18n.t("settings.about.update.errorDownload", { ns: "settings" });
  i18n.t("settings.browserConnection.title", { ns: "settings" });
  i18n.t("settings.browserConnection.action.check", { ns: "settings" });
  i18n.t("settings.browserConnection.error.unavailable", { ns: "settings" });
  i18n.t("settings.group.system", { ns: "settings" });
  i18n.t("settings.tab.developerLab", { ns: "settings" });
  i18n.t("settings.developerLab.title", { ns: "settings" });
  i18n.t("settings.developerLab.tabs.experimental", { ns: "settings" });
  i18n.t("settings.developer.title", { ns: "settings" });
  i18n.t("settings.developer.serviceSkill.title", { ns: "settings" });
  i18n.t("settings.developer.siteAdapterCatalog.message.injected", {
    ns: "settings",
    count: 2,
  });
  i18n.t("settings.appearance.hero.title", { ns: "settings" });
  i18n.t("settings.appearance.colorScheme.options.lime-classic.label", {
    ns: "settings",
  });
  i18n.t("settings.appearance.language.title", { ns: "settings" });
  i18n.t("settings.experimental.updateCheck.title", { ns: "settings" });
  i18n.t("settings.experimental.updateCheck.autoCheck.title", {
    ns: "settings",
  });
  i18n.t("settings.experimental.updateCheck.metrics.updateNow", {
    ns: "settings",
    shown: "20",
    updateNow: "3",
    rate: "15",
  });
  i18n.t("settings.experimental.updateCheck.remindLaterUntil", {
    ns: "settings",
    time: "Oct 9, 2025, 4:53 PM",
  });
  i18n.t("settings.experimental.title", { ns: "settings" });
  i18n.t("settings.experimental.webMcp.title", { ns: "settings" });
  i18n.t("settings.experimental.message.webMcpEnabled", { ns: "settings" });
  i18n.t("settings.experimental.crashReporting.environment.label", {
    ns: "settings",
  });
  i18n.t("settings.environment.hero.title", { ns: "settings" });
  i18n.t("settings.environment.summary.shellImport", {
    ns: "settings",
    status: "Imported",
  });
  i18n.t("settings.environment.preview.overriddenSources", {
    ns: "settings",
    sources: "Shell environment import",
  });
  i18n.t("settings.home.group.tipAria", { ns: "settings" });
  i18n.t("settings.home.title", { ns: "settings" });
  i18n.t("settings.home.summary.groups", { ns: "settings" });
  i18n.t("settings.home.quickAccess.cardTipAria", { ns: "settings" });
  i18n.t("settings.home.current.title", { ns: "settings" });
  i18n.t("settings.memory.title", { ns: "settings" });
  i18n.t("settings.memory.soul.title", { ns: "settings" });
  i18n.t("settings.memory.soul.styleProfile.title", { ns: "settings" });
  i18n.t("settings.memory.soul.import.warning.projectRules", {
    ns: "settings",
  });
  i18n.t("settings.memory.embedding.provider.ollama.label", {
    ns: "settings",
  });
  i18n.t("settings.memory.embedding.status.configured", { ns: "settings" });
  i18n.t("settings.memory.embedding.providerSelect.description", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.providers.googleEngine.label", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.title", { ns: "settings" });
  i18n.t("settings.webSearch.description", { ns: "settings" });
  i18n.t("settings.webSearch.hero.tipAria", { ns: "settings" });
  i18n.t("settings.webSearch.summary.status", {
    ns: "settings",
    status: "saved",
  });
  i18n.t("settings.webSearch.tabs.searchChain", { ns: "settings" });
  i18n.t("settings.webSearch.action.save", { ns: "settings" });
  i18n.t("settings.webSearch.message.saved", { ns: "settings" });
  i18n.t("settings.webSearch.message.loadFailed", {
    ns: "settings",
    message: "network",
  });
  i18n.t("settings.webSearch.secret.hide", { ns: "settings" });
  i18n.t("settings.webSearch.status.credential", {
    ns: "settings",
    name: "Tavily",
    status: "filled",
  });
  i18n.t("settings.webSearch.providerChain.auto", { ns: "settings" });
  i18n.t("settings.webSearch.searchChain.title", { ns: "settings" });
  i18n.t("settings.webSearch.providers.title", { ns: "settings" });
  i18n.t("settings.webSearch.providers.tavily.apply", { ns: "settings" });
  i18n.t("settings.webSearch.providers.bing.placeholder", { ns: "settings" });
  i18n.t("settings.webSearch.providers.googleApi.tipAria", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.providers.googleEngine.create", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.action.viewDocs", { ns: "settings" });
  i18n.t("settings.webSearch.mse.customTemplate.placeholder", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.mse.customTemplateStatus", {
    ns: "settings",
    status: "ready",
  });
  i18n.t("settings.webSearch.mse.suggestion.title", { ns: "settings" });
  i18n.t("settings.webSearch.mse.templateStatus.ready", {
    ns: "settings",
    name: "hn",
  });
  i18n.t("settings.webSearch.mse.action.viewDesign", { ns: "settings" });
  i18n.t("settings.webSearch.images.title", { ns: "settings" });
  i18n.t("settings.webSearch.images.pexels.apply", { ns: "settings" });
  i18n.t("settings.webSearch.images.pexels.noteTip", {
    ns: "settings",
    applyUrl: "https://www.pexels.com/api/new/",
  });
  i18n.t("settings.webSearch.images.pixabay.apply", { ns: "settings" });
  i18n.t("settings.webSearch.images.pixabay.noteTip", {
    ns: "settings",
    applyUrl: "https://pixabay.com/accounts/register/",
  });
  i18n.t("settings.webSearch.observability.title", { ns: "settings" });
  i18n.t("settings.webSearch.observability.mseCustomTemplateStatus", {
    ns: "settings",
    status: "ready",
  });
  i18n.t("settings.webSearch.observability.providerChain.title", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.observability.imageKeys.ready", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.searchChain.engine.label", { ns: "settings" });
  i18n.t("settings.webSearch.searchChain.provider.label", { ns: "settings" });
  i18n.t("settings.webSearch.searchChain.providerPriority.placeholder", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.searchChain.fallbackPreview.tip", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.searchChain.suggestion.title", {
    ns: "settings",
  });
  i18n.t("settings.webSearch.engine.xiaohongshu", { ns: "settings" });
  i18n.t("settings.automation.main.load.timeout", {
    ns: "settings",
    label: "Scheduler config",
    timeoutMs: 8000,
  });
  i18n.t("settings.automation.main.load.label.schedulerConfig", {
    ns: "settings",
  });
  i18n.t("settings.automation.main.load.error.schedulerConfig", {
    ns: "settings",
  });
  i18n.t("settings.automation.main.toast.partialLoad", {
    ns: "settings",
    details: "health",
  });
  i18n.t("settings.automation.main.toast.loadFailed", {
    ns: "settings",
    message: "network",
  });
  i18n.t("settings.automation.main.hero.title.settings", {
    ns: "settings",
  });
  i18n.t("settings.automation.main.hero.description.workspace", {
    ns: "settings",
  });
  i18n.t("settings.automation.main.summary.scheduler", {
    ns: "settings",
    status: "Enabled",
  });
  i18n.t("settings.automation.main.tab.overview", { ns: "settings" });
  i18n.t("settings.automation.overview.title", { ns: "settings" });
  i18n.t("settings.automation.overview.tipAria", { ns: "settings" });
  i18n.t("settings.automation.overview.tipContent", { ns: "settings" });
  i18n.t("settings.automation.overview.latestRunStatusLabel", {
    ns: "settings",
  });
  i18n.t("settings.automation.overview.metric.scheduler", { ns: "settings" });
  i18n.t("settings.automation.overview.status.running", { ns: "settings" });
  i18n.t("settings.automation.overview.status.stopped", { ns: "settings" });
  i18n.t("settings.automation.overview.metric.lastPoll", { ns: "settings" });
  i18n.t("settings.automation.overview.metric.nextPoll", { ns: "settings" });
  i18n.t("settings.automation.overview.metric.active", { ns: "settings" });
  i18n.t("settings.automation.overview.active.empty", { ns: "settings" });
  i18n.t("settings.automation.scheduler.title", { ns: "settings" });
  i18n.t("settings.automation.scheduler.action.save", { ns: "settings" });
  i18n.t("settings.automation.scheduler.pollWindow", {
    ns: "settings",
    last: "10:00",
    next: "10:05",
  });
  i18n.t("settings.automation.scheduler.toast.saveFailed", {
    ns: "settings",
    message: "permission denied",
  });
  i18n.t("settings.automation.tasks.kicker", { ns: "settings" });
  i18n.t("settings.automation.tasks.title", { ns: "settings" });
  i18n.t("settings.automation.tasks.description", { ns: "settings" });
  i18n.t("settings.automation.tasks.action.blankStart", { ns: "settings" });
  i18n.t("settings.automation.tasks.legacyBrowserNotice", {
    ns: "settings",
    count: 2,
  });
  i18n.t("settings.automation.tasks.template.tipAria", {
    ns: "settings",
    name: "Daily Summary",
  });
  i18n.t("settings.automation.tasks.template.dailyBrief.name", {
    ns: "settings",
  });
  i18n.t(
    "settings.automation.tasks.template.structuredDelivery.initial.prompt",
    { ns: "settings" },
  );
  i18n.t("settings.automation.tasks.template.blank.action", {
    ns: "settings",
  });
  i18n.t("settings.automation.tasks.list.title", { ns: "settings" });
  i18n.t("settings.automation.tasks.list.count", {
    ns: "settings",
    count: 3,
  });
  i18n.t("settings.automation.tasks.list.column.actions", { ns: "settings" });
  i18n.t("settings.automation.tasks.list.descriptionFallback", {
    ns: "settings",
  });
  i18n.t("settings.automation.tasks.list.serviceSkillSlotPreview", {
    ns: "settings",
    summary: "Platform: X",
  });
  i18n.t("settings.automation.tasks.list.nextRun", {
    ns: "settings",
    time: "10:00",
  });
  i18n.t("settings.automation.tasks.list.action.run", { ns: "settings" });
  i18n.t("settings.automation.tasks.empty.title", { ns: "settings" });
  i18n.t("settings.automation.tasks.toast.runCompleted", {
    ns: "settings",
    success: 1,
    failed: 0,
    timeout: 0,
  });
  i18n.t("settings.automation.tasks.confirm.delete", {
    ns: "settings",
    name: "Daily Summary",
  });
  i18n.t("settings.automation.tasks.list.badge.serviceSkillLegacyCompat", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.serviceSkill.executionCompatNote", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.title", { ns: "settings" });
  i18n.t("settings.automation.details.badge.workspace", {
    ns: "settings",
    workspace: "Default Workspace",
  });
  i18n.t("settings.automation.details.legacy.message", { ns: "settings" });
  i18n.t("settings.automation.details.sceneApp.retired.title", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.sceneApp.retired.description", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.schedule.hours", {
    ns: "settings",
    count: 2,
  });
  i18n.t("settings.automation.details.legacy.payload.targetId", {
    ns: "settings",
    targetId: "page-1",
  });
  i18n.t("settings.automation.details.delivery.schema.links", {
    ns: "settings",
  });
  i18n.t("settings.automation.history.action.refresh", { ns: "settings" });
  i18n.t("settings.automation.history.toast.loadFailed", {
    ns: "settings",
    message: "network",
  });
  i18n.t("settings.automation.serviceSkill.runner.instant", {
    ns: "settings",
  });
  i18n.t("settings.automation.serviceSkill.unknown", { ns: "settings" });
  i18n.t("settings.automation.serviceSkill.slotFallback", {
    ns: "settings",
    index: 1,
  });
  i18n.t("settings.automation.serviceSkill.source.cloudCatalog", {
    ns: "settings",
  });
  i18n.t("settings.automation.jobDialog.title.create", { ns: "settings" });
  i18n.t("settings.automation.jobDialog.badge.startMethod", {
    ns: "settings",
    method: "Agent Conversation",
  });
  i18n.t("settings.automation.jobDialog.legacy.title", { ns: "settings" });
  i18n.t("settings.automation.jobDialog.footer.create", { ns: "settings" });
  i18n.t("settings.automation.jobDialog.accessMode.policy.current", {
    ns: "settings",
  });
  i18n.t("settings.automation.jobDialog.executionMode.intelligent", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.statusDetail.blocking", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.status.queued", { ns: "settings" });
  i18n.t("settings.automation.details.statusDetail.running", {
    ns: "settings",
  });
  i18n.t("settings.automation.details.serviceSkill.taskLine", {
    ns: "settings",
    title: "Daily Summary",
  });
  i18n.t("settings.automation.details.serviceSkill.moreItems", {
    ns: "settings",
    count: 2,
  });
  i18n.t("settings.automation.focus.label", { ns: "settings" });
  i18n.t("settings.automation.focus.title", { ns: "settings" });
  i18n.t("settings.automation.health.title", { ns: "settings" });
  i18n.t("settings.automation.health.status.waitingForHuman", {
    ns: "settings",
  });
  i18n.t("settings.stats.heatmap.rangeValue", { ns: "settings" });
  i18n.t("settings.stats.range.week.label", { ns: "settings" });
  i18n.t("settings.stats.segment.messages", { ns: "settings" });
  i18n.t("settings.channels.logTail.title", { ns: "settings" });
  i18n.t("settings.channels.logTail.error.invalidRegex", { ns: "settings" });
  i18n.t("settings.channels.workbench.title", { ns: "settings" });
  i18n.t("settings.channels.workbench.currentScope", {
    ns: "settings",
    scope: "Logs",
  });
  i18n.t("settings.channels.gatewayTunnel.confirm.installCloudflared", {
    ns: "settings",
  });
  i18n.t("settings.channels.gatewayTunnel.state.running", {
    ns: "settings",
    action: "Start",
  });
  i18n.t("settings.channels.gatewayRuntime.message.actionFailed", {
    ns: "settings",
    error: "network down",
  });
  i18n.t("settings.channels.wechatRuntime.confirm.removeAccount", {
    ns: "settings",
    accountId: "wx-main",
    purgeAction: "",
    purgeScope: "",
    purgeRisk: "",
  });
  i18n.t("settings.channels.wechatRuntime.accounts.runningCount", {
    ns: "settings",
    count: 1,
  });
  i18n.t("settings.system.clipboardPermission.macos.title", {
    ns: "settings",
  });
  i18n.t(
    "settings.system.clipboardPermission.message.openSettingsFailedWithMessage",
    {
      ns: "settings",
    },
  );
  i18n.t("settings.system.workspaceRepair.title", { ns: "settings" });
  i18n.t("settings.system.workspaceRepair.message.copiedAll", {
    ns: "settings",
  });
  i18n.t("settings.mediaServices.title", { ns: "settings" });
  i18n.t("settings.mediaServices.sections.responsiveChat.title", {
    ns: "settings",
  });
  i18n.t("settings.mediaServices.card.tipAria", {
    ns: "settings",
    title: "Fast Response Chat Assistant",
  });
  i18n.t("settings.providers.importExport.title", { ns: "settings" });
  i18n.t("settings.providers.importExport.export.action.generate", {
    ns: "settings",
  });
  i18n.t("settings.providers.importExport.import.result.importedProviders", {
    ns: "settings",
    count: 2,
  });
  i18n.t("settings.providers.importExport.import.result.errors", {
    ns: "settings",
    errors: "OpenAI exists",
  });
  i18n.t("settings.mediaGeneration.image.title", { ns: "settings" });
  i18n.t("settings.mediaGeneration.video.hint.ready", { ns: "settings" });
  i18n.t("settings.userCenterSession.value.expiresAt.unknown", {
    ns: "settings",
  });
  i18n.t("settings.userCenterSession.title", { ns: "settings" });
  i18n.t("settings.userCenterSession.login.google.title", { ns: "settings" });
  i18n.t("settings.agent.skills.advancedEntry.title", { ns: "settings" });
  i18n.t("settings.agent.skills.advancedEntry.tipAria", { ns: "settings" });
  i18n.t("settings.voice.model.action.download", { ns: "settings" });
  i18n.t("settings.voice.processing.title", { ns: "settings" });
  i18n.t("settings.voice.media.title", { ns: "settings" });
  i18n.t("settings.voice.model.message.downloaded", {
    ns: "settings",
    model: "SenseVoice Small",
  });
  i18n.t("errors.artifactFallback.title.renderFailed", { ns: "errors" });
  i18n.t("errors.artifactFallback.action.copyErrorReport", { ns: "errors" });
  i18n.t("errors.svgRenderer.error.renderFailed", { ns: "errors" });
  i18n.t("errors.codeRenderer.action.copyCode", { ns: "errors" });
  i18n.t("errors.codeRenderer.error.emptyContent", { ns: "errors" });
  i18n.t("errors.htmlRenderer.error.renderFailed", { ns: "errors" });
  i18n.t("errors.htmlRenderer.action.refreshPreview", { ns: "errors" });
  i18n.t("errors.mermaidRenderer.error.syntax", { ns: "errors" });
  i18n.t("errors.mermaidRenderer.action.exportDiagram", { ns: "errors" });
  i18n.t("errors.crashRecovery.title", { ns: "errors" });
  i18n.t("errors.crashRecovery.action.forceReload", { ns: "errors" });
  i18n.t("errors.crashRecovery.action.retryOnly", { ns: "errors" });
  i18n.t("errors.crashRecovery.message.diagnosticExportedShort", {
    ns: "errors",
  });
  i18n.t("errors.crashRecovery.moduleImportFailure.prefix", { ns: "errors" });

  // @ts-expect-error common namespace key 必须来自已迁移的 zh-CN source resource。
  const missingCommonKey: CommonResourceKey = "common.__missing__";
  void missingCommonKey;
}

describe("i18n type binding", () => {
  it("应把类型断言保留在 tsc 覆盖范围内", () => {
    expect(typeof assertI18nKeyTypes).toBe("function");
  });
});
