import { useMemo } from "react";
import type { ProjectMemory } from "@/lib/api/projectMemory";
import {
  generateGeneralWorkbenchPrompt,
  generateProjectMemoryPrompt,
} from "@/lib/workspace/workbenchPrompt";
import type { ThemeType } from "@/lib/workspace/workbenchContract";
import type { CreationMode } from "../components/types";
import {
  shouldUseCompactGeneralPromptForPreferences,
  type ChatToolPreferences,
} from "../utils/chatToolPreferences";
import {
  buildGeneralAgentSystemPrompt,
  resolveAgentChatMode,
} from "../utils/generalAgentPrompt";

interface UseWorkspaceSystemPromptRuntimeParams {
  chatToolPreferences: ChatToolPreferences;
  contentId?: string | null;
  creationMode: CreationMode;
  isSpecializedThemeMode: boolean;
  mappedTheme: ThemeType;
  projectMemory?: ProjectMemory | null;
}

/** 对话模式与 system prompt 由同一 projection owner 派生。 */
export function useWorkspaceSystemPromptRuntime({
  chatToolPreferences,
  contentId,
  creationMode,
  isSpecializedThemeMode,
  mappedTheme,
  projectMemory,
}: UseWorkspaceSystemPromptRuntimeParams) {
  const chatMode = useMemo(
    () => resolveAgentChatMode(mappedTheme, isSpecializedThemeMode),
    [isSpecializedThemeMode, mappedTheme],
  );
  const shouldUseCompactGeneralSystemPrompt =
    shouldUseCompactGeneralPromptForPreferences({
      chatMode,
      contentId,
      preferences: chatToolPreferences,
    });
  const systemPrompt = useMemo(() => {
    let prompt = "";

    if (chatMode === "general") {
      prompt = buildGeneralAgentSystemPrompt(mappedTheme, {
        compact: shouldUseCompactGeneralSystemPrompt,
        toolPreferences: chatToolPreferences,
        harness: {
          contentId: contentId || null,
        },
      });
    } else if (isSpecializedThemeMode) {
      prompt = generateGeneralWorkbenchPrompt(mappedTheme, creationMode);
    }

    if (projectMemory) {
      const memoryPrompt = generateProjectMemoryPrompt(projectMemory);
      if (memoryPrompt) {
        prompt = prompt ? `${prompt}\n\n${memoryPrompt}` : memoryPrompt;
      }
    }

    return prompt || undefined;
  }, [
    chatMode,
    chatToolPreferences,
    contentId,
    creationMode,
    isSpecializedThemeMode,
    mappedTheme,
    projectMemory,
    shouldUseCompactGeneralSystemPrompt,
  ]);

  return {
    chatMode,
    generalHarnessEntryEnabled: chatMode === "general",
    systemPrompt,
  };
}
