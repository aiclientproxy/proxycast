import React, { memo, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import styled from "styled-components";
import { toast } from "sonner";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { VideoCanvasProps } from "./types";
import { VideoSidebar, type VideoProviderOption } from "./VideoSidebar";
import { VideoWorkspace } from "./VideoWorkspace";
import { useGlobalMediaGenerationDefaults } from "@/hooks/useGlobalMediaGenerationDefaults";
import {
  apiKeyProviderApi,
  providerModelIds,
} from "@/lib/api/apiKeyProvider";
import {
  ackCanvasImageInsertRequest,
  emitCanvasImageInsertAck,
  getPendingCanvasImageInsertRequests,
  matchesCanvasImageInsertTarget,
  onCanvasImageInsertRequest,
  type CanvasImageInsertRequest,
} from "@/lib/canvasImageInsertBus";
import { normalizeMediaGenerationPreference } from "@/lib/mediaGeneration";

const VIDEO_MODEL_PRESETS: Record<string, string[]> = {
  doubao: ["seedance-1-5-pro-251215", "seedance-1-5-lite-250428"],
  volcengine: ["seedance-1-5-pro-251215", "seedance-1-5-lite-250428"],
  dashscope: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  alibaba: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  qwen: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  sora: ["sora-2", "sora-2-pro"],
  openai: ["sora-2", "sora-2-pro"],
  veo: ["veo-3.1"],
  google: ["veo-3.1"],
  vertex: ["veo-3.1"],
  kling: ["kling-2.6"],
  minimax: ["minimax-hailuo-2.3", "minimax-hailuo-02"],
  hailuo: ["minimax-hailuo-2.3", "minimax-hailuo-02"],
  runway: ["runway-gen-4-turbo"],
};

function isVideoProvider(providerId: string): boolean {
  const normalized = providerId.toLowerCase();
  const isAudioOnlyOpenai =
    normalized.includes("openai") &&
    (normalized.includes("tts") ||
      normalized.includes("voice") ||
      normalized.includes("audio"));
  return (
    !isAudioOnlyOpenai &&
    (normalized.includes("doubao") ||
      normalized.includes("volc") ||
      normalized.includes("dashscope") ||
      normalized.includes("alibaba") ||
      normalized.includes("qwen") ||
      normalized.includes("openai") ||
      normalized.includes("video") ||
      normalized.includes("runway") ||
      normalized.includes("minimax") ||
      normalized.includes("kling") ||
      normalized.includes("sora") ||
      normalized.includes("veo"))
  );
}

function resolveProviderModels(provider: VideoProviderOption): string[] {
  if (provider.models.length > 0) {
    return provider.models;
  }

  const normalizedId = provider.id.toLowerCase();
  for (const [key, models] of Object.entries(VIDEO_MODEL_PRESETS)) {
    if (normalizedId.includes(key)) {
      return models;
    }
  }

  return [];
}

const Root = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  width: 100%;
  padding: 12px;
  gap: 12px;
  overflow: hidden;
  background: linear-gradient(
    180deg,
    hsl(206 55% 97%) 0%,
    hsl(0 0% 100%) 42%,
    hsl(200 43% 96%) 100%
  );

  &::before,
  &::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    pointer-events: none;
    filter: blur(72px);
    opacity: 0.85;
  }

  &::before {
    width: 340px;
    height: 340px;
    top: -110px;
    left: 6%;
    background: hsl(154 62% 84% / 0.4);
  }

  &::after {
    width: 320px;
    height: 320px;
    right: -80px;
    bottom: -140px;
    background: hsl(203 88% 84% / 0.32);
  }

  > * {
    position: relative;
    z-index: 1;
  }

  @media (max-width: 1100px) {
    padding: 10px;
    gap: 10px;
  }
`;

const Body = styled.div`
  display: flex;
  flex: 1;
  min-height: 0;
  width: 100%;

  @media (max-width: 1100px) {
    flex-direction: column;
  }
`;

const SidebarContainer = styled.div<{ $collapsed: boolean }>`
  width: ${({ $collapsed }) => ($collapsed ? "0px" : "332px")};
  flex-shrink: 0;
  height: 100%;
  min-height: 0;
  background: transparent;
  border-radius: ${({ $collapsed }) => ($collapsed ? "0" : "28px 0 0 28px")};
  border: 1px solid
    ${({ $collapsed }) =>
      $collapsed ? "transparent" : "hsl(var(--border) / 0.7)"};
  border-right: none;
  box-shadow:
    0 16px 40px hsl(215 40% 10% / 0.06),
    inset 0 1px 0 hsl(0 0% 100% / 0.72);
  overflow-y: auto;
  overflow-x: hidden;
  opacity: ${({ $collapsed }) => ($collapsed ? 0 : 1)};
  pointer-events: ${({ $collapsed }) => ($collapsed ? "none" : "auto")};
  transition:
    width 0.24s ease,
    opacity 0.24s ease,
    max-height 0.24s ease;

  @media (max-width: 1100px) {
    width: 100%;
    height: auto;
    max-height: ${({ $collapsed }) => ($collapsed ? "0px" : "780px")};
    border-radius: ${({ $collapsed }) => ($collapsed ? "0" : "28px")};
    border-right: 1px solid
      ${({ $collapsed }) =>
        $collapsed ? "transparent" : "hsl(var(--border) / 0.7)"};
  }
`;

const Splitter = styled.div`
  position: relative;
  width: 0;
  flex-shrink: 0;
  z-index: 2;

  @media (max-width: 1100px) {
    width: 100%;
    height: 0;
  }
`;

const SplitterButton = styled.button`
  position: absolute;
  top: 14px;
  left: 0;
  transform: translateX(-50%);
  width: 32px;
  height: 48px;
  border-radius: 999px;
  border: 1px solid hsl(var(--border) / 0.85);
  background: hsl(var(--background) / 0.88);
  color: hsl(var(--muted-foreground));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 24px hsl(215 32% 12% / 0.08);
  transition:
    color 0.2s ease,
    border-color 0.2s ease,
    transform 0.2s ease,
    box-shadow 0.2s ease;

  &:hover {
    color: hsl(var(--foreground));
    border-color: hsl(214 68% 38% / 0.35);
    transform: translateY(-1px);
    box-shadow: 0 14px 28px hsl(215 32% 12% / 0.12);
  }

  @media (max-width: 1100px) {
    top: 0;
    left: 18px;
    transform: translateY(-50%);
    width: 48px;
    height: 32px;
  }
`;

const MainContainer = styled.div`
  flex: 1;
  height: 100%;
  min-height: 0;
  background: transparent;
  overflow: hidden;
  position: relative;
`;

const WorkspaceFrame = styled.div`
  flex: 1;
  min-height: 0;
  display: flex;
  border-radius: 0 32px 32px 0;
  border: 1px solid hsl(var(--border) / 0.75);
  background: transparent;
  box-shadow:
    0 20px 48px hsl(215 32% 12% / 0.07),
    inset 0 1px 0 hsl(0 0% 100% / 0.7);
  overflow: hidden;

  @media (max-width: 1100px) {
    border-radius: 32px;
  }
`;

export const VideoCanvas: React.FC<VideoCanvasProps> = memo(
  ({
    state,
    onStateChange,
    projectRootPath,
    projectId,
    contentId,
    onClose: _onClose,
  }) => {
    const { t } = useTranslation("workspace");
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [providers, setProviders] = useState<VideoProviderOption[]>([]);
    const { mediaDefaults } = useGlobalMediaGenerationDefaults();

    useEffect(() => {
      let active = true;
      const loadProviders = async () => {
        try {
          const allProviders = await apiKeyProviderApi.getProviders();
          if (!active) {
            return;
          }

          const availableProviders = allProviders
            .filter(
              (provider) =>
                provider.enabled &&
                provider.api_key_count > 0 &&
                isVideoProvider(provider.id),
            )
            .map((provider) => ({
              id: provider.id,
              name: provider.name,
              models: providerModelIds(provider.models),
            }));

          setProviders(availableProviders);
        } catch (error) {
          console.error("[VideoCanvas] 加载视频 Provider 失败:", error);
          if (active) {
            setProviders([]);
          }
        }
      };

      void loadProviders();
      return () => {
        active = false;
      };
    }, []);

    const selectedProvider = useMemo(() => {
      return (
        providers.find((provider) => provider.id === state.providerId) ?? null
      );
    }, [providers, state.providerId]);

    const availableModels = useMemo(() => {
      if (!selectedProvider) {
        return [];
      }
      return resolveProviderModels(selectedProvider);
    }, [selectedProvider]);
    const preferredVideoSelection = useMemo(
      () => normalizeMediaGenerationPreference(mediaDefaults.video),
      [mediaDefaults.video],
    );

    useEffect(() => {
      if (providers.length === 0) {
        return;
      }

      const currentProvider =
        providers.find((provider) => provider.id === state.providerId) ?? null;
      const preferredProvider = preferredVideoSelection.preferredProviderId
        ? (providers.find(
            (provider) =>
              provider.id === preferredVideoSelection.preferredProviderId,
          ) ?? null)
        : null;
      const nextProvider = currentProvider ?? preferredProvider ?? providers[0];
      if (!nextProvider) {
        return;
      }

      const nextModels = resolveProviderModels(nextProvider);
      const currentModelValid =
        currentProvider?.id === nextProvider.id &&
        nextModels.includes(state.model);
      const preferredModelValid =
        preferredProvider?.id === nextProvider.id &&
        Boolean(preferredVideoSelection.preferredModelId) &&
        nextModels.includes(preferredVideoSelection.preferredModelId || "");
      const nextModel = currentModelValid
        ? state.model
        : preferredModelValid
          ? (preferredVideoSelection.preferredModelId ?? "")
          : (nextModels[0] ?? "");

      if (state.providerId === nextProvider.id && state.model === nextModel) {
        return;
      }

      onStateChange({
        ...state,
        providerId: nextProvider.id,
        model: nextModel,
      });
    }, [
      onStateChange,
      preferredVideoSelection.preferredModelId,
      preferredVideoSelection.preferredProviderId,
      providers,
      state,
    ]);

    const matchesRequestTarget = useMemo(
      () => (request: CanvasImageInsertRequest) =>
        matchesCanvasImageInsertTarget(request, {
          projectId: projectId || null,
          contentId: contentId || null,
          canvasType: "video",
        }),
      [contentId, projectId],
    );

    useEffect(() => {
      const processInsertRequest = (request: CanvasImageInsertRequest) => {
        if (!matchesRequestTarget(request)) {
          return;
        }

        const imageUrl = request.image.contentUrl?.trim();
        if (!imageUrl) {
          emitCanvasImageInsertAck({
            requestId: request.requestId,
            success: false,
            canvasType: "video",
            reason: "invalid_image_url",
          });
          ackCanvasImageInsertRequest(request.requestId);
          return;
        }

        onStateChange({
          ...state,
          startImage: imageUrl,
        });
        toast.success(
          t("workspace.video.canvas.imageInsert.success", {
            defaultValue: "已设置为视频起始参考图",
          }),
        );

        emitCanvasImageInsertAck({
          requestId: request.requestId,
          success: true,
          canvasType: "video",
          locationLabel: t("workspace.video.canvas.imageInsert.locationLabel", {
            defaultValue: "起始画面参考图",
          }),
        });
        ackCanvasImageInsertRequest(request.requestId);
      };

      const unsubscribe = onCanvasImageInsertRequest((request) => {
        processInsertRequest(request);
      });
      getPendingCanvasImageInsertRequests().forEach((request) => {
        processInsertRequest(request);
      });
      return unsubscribe;
    }, [matchesRequestTarget, onStateChange, state, t]);

    return (
      <Root>
        <Body>
          <SidebarContainer $collapsed={sidebarCollapsed}>
            <VideoSidebar
              state={state}
              providers={providers}
              availableModels={availableModels}
              onStateChange={onStateChange}
            />
          </SidebarContainer>

          <Splitter>
            <SplitterButton
              onClick={() => setSidebarCollapsed((previous) => !previous)}
              title={
                sidebarCollapsed
                  ? t("workspace.video.canvas.sidebar.expand", {
                      defaultValue: "展开侧栏",
                    })
                  : t("workspace.video.canvas.sidebar.collapse", {
                      defaultValue: "收起侧栏",
                    })
              }
            >
              {sidebarCollapsed ? (
                <PanelLeftOpen size={12} />
              ) : (
                <PanelLeftClose size={12} />
              )}
            </SplitterButton>
          </Splitter>

          <WorkspaceFrame>
            <MainContainer>
              <VideoWorkspace
                state={state}
                projectRootPath={projectRootPath}
                projectId={projectId}
                onStateChange={onStateChange}
              />
            </MainContainer>
          </WorkspaceFrame>
        </Body>
      </Root>
    );
  },
);

VideoCanvas.displayName = "VideoCanvas";
