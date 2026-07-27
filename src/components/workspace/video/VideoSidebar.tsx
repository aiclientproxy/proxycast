import React, { memo, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import styled from "styled-components";
import { Check, ChevronDown, Dices, ImagePlus, X } from "lucide-react";
import { VideoAspectRatio, VideoCanvasState, VideoResolution } from "./types";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import { formatNumber } from "@/i18n/format";

export interface VideoProviderOption {
  id: string;
  name: string;
  models: string[];
}

interface VideoSidebarProps {
  state: VideoCanvasState;
  providers: VideoProviderOption[];
  availableModels: string[];
  onStateChange: (state: VideoCanvasState) => void;
}

const SidebarWrapper = styled.div`
  height: 100%;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  background: transparent;

  @media (max-width: 1100px) {
    padding: 16px;
  }
`;

const PanelIntro = styled.div`
  border-radius: 24px;
  border: 1px solid hsl(152 30% 86%);
  background: linear-gradient(
    135deg,
    hsl(154 48% 96%) 0%,
    hsl(0 0% 100%) 48%,
    hsl(201 62% 97%) 100%
  );
  padding: 18px;
  box-shadow:
    0 14px 32px hsl(200 38% 16% / 0.06),
    inset 0 1px 0 hsl(0 0% 100% / 0.78);
`;

const PanelEyebrow = styled.span`
  display: inline-flex;
  align-items: center;
  width: fit-content;
  border-radius: 999px;
  border: 1px solid hsl(154 36% 82%);
  background: hsl(0 0% 100% / 0.8);
  padding: 5px 10px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: hsl(154 50% 28%);
`;

const PanelTitle = styled.h2`
  margin: 0;
  font-size: 22px;
  line-height: 1.2;
  font-weight: 700;
  color: hsl(var(--foreground));
`;

const PanelTitleRow = styled.div`
  margin: 10px 0 6px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const PanelMetaGrid = styled.div`
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
`;

const PanelMetaCard = styled.div`
  border-radius: 18px;
  border: 1px solid hsl(var(--border) / 0.8);
  background: hsl(var(--background) / 0.86);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 5px;
`;

const PanelMetaLabel = styled.span`
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: hsl(var(--muted-foreground));
`;

const PanelMetaValue = styled.span`
  font-size: 14px;
  line-height: 1.45;
  font-weight: 600;
  color: hsl(var(--foreground));
  word-break: break-word;
`;

const Section = styled.section`
  display: flex;
  flex-direction: column;
  gap: 12px;
  border-radius: 22px;
  border: 1px solid hsl(var(--border) / 0.78);
  background: hsl(var(--background) / 0.86);
  padding: 14px;
  box-shadow:
    0 10px 28px hsl(215 30% 14% / 0.04),
    inset 0 1px 0 hsl(0 0% 100% / 0.75);
`;

const SectionTitle = styled.div`
  font-size: 14px;
  line-height: 1.3;
  font-weight: 700;
  color: hsl(var(--foreground));
`;

const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const ModelTrigger = styled.button`
  width: 100%;
  min-height: 62px;
  border-radius: 18px;
  border: 1px solid hsl(var(--border));
  background: linear-gradient(
    180deg,
    hsl(var(--background)),
    hsl(var(--muted) / 0.12)
  );
  color: hsl(var(--foreground));
  padding: 12px 14px;
  outline: none;
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease,
    transform 0.2s ease;

  &:hover {
    border-color: hsl(214 68% 38% / 0.35);
    transform: translateY(-1px);
    box-shadow: 0 12px 28px hsl(215 32% 12% / 0.08);
  }

  &:focus-visible {
    border-color: hsl(214 68% 38% / 0.4);
    box-shadow: 0 0 0 4px hsl(211 100% 96%);
  }
`;

const ModelTriggerBody = styled.div`
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
`;

const ModelTriggerLabel = styled.span`
  font-size: 14px;
  font-weight: 700;
  color: hsl(var(--foreground));
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
`;

const ModelTriggerMeta = styled.span`
  font-size: 12px;
  color: hsl(var(--muted-foreground));
`;

const ModelPanelMask = styled.div`
  position: fixed;
  inset: 0;
  background: hsl(220 36% 8% / 0.24);
  backdrop-filter: blur(12px);
  z-index: 2100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
`;

const ModelPanel = styled.div`
  width: min(880px, calc(100vw - 36px));
  max-height: min(820px, calc(100vh - 36px));
  border-radius: 30px;
  border: 1px solid hsl(var(--border) / 0.85);
  background: linear-gradient(
    180deg,
    hsl(var(--background) / 0.98),
    hsl(204 38% 98% / 0.98)
  );
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-shadow: 0 28px 80px hsl(215 32% 12% / 0.18);
`;

const ModelPanelHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const ModelPanelEyebrow = styled.span`
  display: inline-flex;
  align-items: center;
  width: fit-content;
  border-radius: 999px;
  border: 1px solid hsl(203 82% 88%);
  background: hsl(200 100% 97%);
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: hsl(211 58% 38%);
`;

const ModelPanelTitle = styled.h3`
  margin: 0;
  font-size: 28px;
  line-height: 1.15;
  font-weight: 700;
  color: hsl(var(--foreground));
`;

const ModelPanelDescription = styled.p`
  margin: 0;
  font-size: 14px;
  line-height: 1.65;
  color: hsl(var(--muted-foreground));
  max-width: 720px;
`;

const ModelPanelDivider = styled.div`
  height: 1px;
  background: hsl(var(--border) / 0.9);
`;

const ModelPanelList = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  overflow: auto;
  padding-right: 2px;

  @media (max-width: 860px) {
    grid-template-columns: 1fr;
  }
`;

const ModelPanelItem = styled.button<{ $active: boolean }>`
  width: 100%;
  border: 1px solid
    ${(props) =>
      props.$active ? "hsl(214 68% 38% / 0.35)" : "hsl(var(--border) / 0.88)"};
  border-radius: 24px;
  background: ${(props) =>
    props.$active
      ? "linear-gradient(180deg, hsl(211 100% 98%), hsl(203 100% 97%))"
      : "hsl(var(--background) / 0.92)"};
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 16px;
  gap: 14px;
  color: hsl(var(--foreground));
  text-align: left;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    transform 0.2s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;
  box-shadow: ${(props) =>
    props.$active ? "0 16px 36px hsl(204 68% 46% / 0.12)" : "none"};

  &:hover {
    border-color: hsl(214 68% 38% / 0.3);
    transform: translateY(-1px);
  }
`;

const ModelPanelBody = styled.div`
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const ModelPanelName = styled.div`
  font-size: 18px;
  line-height: 1.25;
  font-weight: 700;
  color: hsl(var(--foreground));
`;

const ModelPanelMetaRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const ModelPanelCost = styled.span`
  font-size: 12px;
  font-weight: 700;
  color: hsl(211 58% 38%);
`;

const ModelProviderTag = styled.span`
  display: inline-flex;
  align-items: center;
  height: 24px;
  border-radius: 999px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--muted) / 0.18);
  padding: 0 10px;
  font-size: 11px;
  font-weight: 600;
  color: hsl(var(--muted-foreground));
`;

const ModelPanelDesc = styled.div`
  font-size: 13px;
  line-height: 1.6;
  color: hsl(var(--muted-foreground));
`;

const ModelPanelSelected = styled.div<{ $active: boolean }>`
  width: 32px;
  height: 32px;
  border-radius: 999px;
  border: 1px solid
    ${(props) =>
      props.$active ? "hsl(214 68% 38% / 0.4)" : "hsl(var(--border) / 0.9)"};
  background: ${(props) =>
    props.$active ? "hsl(221 39% 16%)" : "hsl(var(--background))"};
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: ${(props) =>
    props.$active ? "hsl(var(--background))" : "hsl(var(--muted-foreground))"};
  flex-shrink: 0;
`;

const ImageUploadArea = styled.div<{ $dragging?: boolean }>`
  min-height: 134px;
  border-radius: 18px;
  background: ${(props) =>
    props.$dragging
      ? "hsl(211 100% 96%)"
      : "linear-gradient(180deg, hsl(var(--muted) / 0.18), hsl(var(--background)))"};
  border: 1px dashed
    ${(props) =>
      props.$dragging
        ? "hsl(214 68% 38% / 0.46)"
        : "hsl(var(--border) / 0.95)"};
  display: flex;
  align-items: center;
  justify-content: center;
  color: hsl(var(--muted-foreground));
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    transform 0.2s ease;
  padding: 12px;
  cursor: pointer;

  &:hover {
    background: hsl(var(--muted) / 0.24);
    border-color: hsl(214 68% 38% / 0.35);
  }
`;

const UploadPrompt = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
  font-size: 12px;
  color: hsl(var(--muted-foreground));
  line-height: 1.55;

  svg {
    margin-bottom: 6px;
    color: hsl(211 58% 38%);
  }
`;

const UploadActionButton = styled.button`
  margin-top: 8px;
  height: 32px;
  border-radius: 999px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  padding: 0 12px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    transform 0.2s ease;

  &:hover {
    border-color: hsl(214 68% 38% / 0.35);
    transform: translateY(-1px);
  }
`;

const PreviewBox = styled.div`
  width: 100%;
  min-height: 134px;
  border-radius: 18px;
  overflow: hidden;
  position: relative;
  background: hsl(var(--muted) / 0.18);
  border: 1px solid hsl(var(--border));

  img {
    display: block;
    width: 100%;
    max-height: 160px;
    object-fit: cover;
  }
`;

const ReplaceHint = styled.div`
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  font-size: 11px;
  color: hsl(var(--foreground));
  background: linear-gradient(transparent, hsl(var(--background) / 0.92));
  padding: 22px 10px 10px;
  text-align: center;
`;

const RemovePreviewButton = styled.button`
  position: absolute;
  top: 10px;
  right: 10px;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 999px;
  background: hsl(var(--background) / 0.92);
  color: hsl(var(--foreground));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 8px 16px hsl(215 30% 14% / 0.12);
`;

const ReplacePreviewButton = styled.button`
  position: absolute;
  top: 10px;
  left: 10px;
  height: 28px;
  border: none;
  border-radius: 999px;
  background: hsl(var(--background) / 0.94);
  color: hsl(var(--foreground));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0 12px;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 8px 16px hsl(215 30% 14% / 0.12);
`;

const RatioGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
`;

const RatioItem = styled.button<{ $active?: boolean }>`
  min-height: 64px;
  border-radius: 16px;
  background: ${(props) =>
    props.$active ? "hsl(211 100% 96%)" : "hsl(var(--muted) / 0.12)"};
  border: 1px solid
    ${(props) =>
      props.$active ? "hsl(214 68% 38% / 0.3)" : "hsl(var(--border) / 0.75)"};
  color: ${(props) =>
    props.$active ? "hsl(211 58% 38%)" : "hsl(var(--muted-foreground))"};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 11px;
  font-weight: 700;
  transition:
    background 0.2s ease,
    border-color 0.2s ease,
    transform 0.2s ease;

  &:hover {
    border-color: hsl(214 68% 38% / 0.25);
    transform: translateY(-1px);
  }
`;

const RatioShape = styled.div<{ $active?: boolean }>`
  width: 16px;
  height: 16px;
  border: 1px solid
    ${(props) =>
      props.$active ? "hsl(211 58% 38%)" : "hsl(var(--muted-foreground))"};
  border-radius: 4px;
  margin-bottom: 6px;
`;

const ResolutionGroup = styled.div`
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
`;

const ResolutionButton = styled.button<{ $active?: boolean }>`
  height: 38px;
  border-radius: 14px;
  border: 1px solid
    ${(props) =>
      props.$active ? "hsl(221 39% 16%)" : "hsl(var(--border) / 0.88)"};
  background: ${(props) =>
    props.$active
      ? "linear-gradient(180deg, hsl(221 39% 16%), hsl(216 34% 12%))"
      : "hsl(var(--muted) / 0.12)"};
  color: ${(props) =>
    props.$active ? "hsl(var(--background))" : "hsl(var(--muted-foreground))"};
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-1px);
    border-color: hsl(214 68% 38% / 0.3);
  }
`;

const DurationRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const DurationSlider = styled.input`
  flex: 1;
  accent-color: hsl(221 39% 16%);
  cursor: pointer;
`;

const DurationValue = styled.input`
  width: 64px;
  height: 42px;
  border-radius: 14px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  text-align: center;
  font-size: 14px;
  font-weight: 700;
  outline: none;
`;

const SeedRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const SeedInput = styled.input`
  flex: 1;
  height: 42px;
  border-radius: 14px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  padding: 0 12px;
  font-size: 13px;
  outline: none;
`;

const SeedRandomButton = styled.button`
  width: 42px;
  height: 42px;
  border-radius: 14px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--background));
  color: hsl(var(--muted-foreground));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    color 0.2s ease;

  &:hover {
    color: hsl(var(--foreground));
    border-color: hsl(214 68% 38% / 0.3);
    transform: translateY(-1px);
  }
`;

const ToggleRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const ToggleCopy = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const ToggleTitleRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const ToggleTitle = styled.span`
  font-size: 14px;
  line-height: 1.2;
  font-weight: 700;
  color: hsl(var(--foreground));
`;

const ToggleSwitch = styled.button<{ $checked: boolean }>`
  width: 46px;
  height: 26px;
  border: none;
  border-radius: 999px;
  padding: 3px;
  cursor: pointer;
  background: ${(props) =>
    props.$checked ? "hsl(221 39% 16%)" : "hsl(var(--border))"};
  display: flex;
  align-items: center;
  justify-content: ${(props) => (props.$checked ? "flex-end" : "flex-start")};
  transition: all 0.2s ease;
`;

const ToggleDot = styled.span`
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: hsl(var(--background));
`;

const RATIOS: { label: string; value: VideoAspectRatio }[] = [
  { label: "adaptive", value: "adaptive" },
  { label: "16:9", value: "16:9" },
  { label: "9:16", value: "9:16" },
  { label: "1:1", value: "1:1" },
  { label: "4:3", value: "4:3" },
  { label: "3:4", value: "3:4" },
  { label: "21:9", value: "21:9" },
];

type FrameImageField = "startImage" | "endImage";
type FrameDropArea = "start" | "end";

interface VideoModelOption {
  key: string;
  providerId: string;
  providerName: string;
  model: string;
  label: string;
  cost: string;
  description: string;
}

type VideoModelMetaId =
  | "veo31"
  | "sora2Pro"
  | "sora2"
  | "seedance15Pro"
  | "kling26"
  | "minimaxHailuo23"
  | "minimaxHailuo02"
  | "runwayGen4Turbo"
  | "seedance15Lite"
  | "wanx21T2vTurbo"
  | "wanx21Kf2vPlus"
  | "fallback";

type VideoModelMetaCopy = Record<
  VideoModelMetaId,
  { cost: string; description: string }
>;

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error("文件读取失败"));
    };
    reader.onerror = () => reject(new Error("文件读取失败"));
    reader.readAsDataURL(file);
  });
}

function getModelLabel(model: string): string {
  const normalized = model.toLowerCase();
  if (
    normalized === "sora-2-pro" ||
    normalized.includes("sora-2-pro") ||
    normalized.includes("sora2-pro")
  ) {
    return "Sora-2-Pro";
  }
  if (
    normalized === "veo-3.1" ||
    normalized === "veo 3.1" ||
    normalized.includes("veo-3.1")
  ) {
    return "Veo 3.1";
  }
  if (normalized === "sora-2" || normalized.includes("sora-2")) {
    return "Sora-2";
  }
  if (normalized.includes("seedance-1-5-pro")) {
    return "Seedance 1.5 Pro";
  }
  if (normalized.includes("seedance-1-5-lite")) {
    return "Seedance 1.5 Lite";
  }
  if (normalized === "kling-2.6" || normalized.includes("kling-2.6")) {
    return "Kling 2.6";
  }
  if (
    normalized === "minimax-hailuo-2.3" ||
    normalized.includes("hailuo-2.3")
  ) {
    return "Minimax Hailuo 2.3";
  }
  if (normalized === "minimax-hailuo-02" || normalized.includes("hailuo-02")) {
    return "Minimax Hailuo-02";
  }
  if (
    normalized === "runway-gen-4-turbo" ||
    normalized.includes("runway-gen-4-turbo")
  ) {
    return "Runway Gen-4 Turbo";
  }
  if (normalized.includes("wanx2.1-t2v-turbo")) {
    return "Wanx 2.1 T2V Turbo";
  }
  if (normalized.includes("wanx2.1-kf2v-plus")) {
    return "Wanx 2.1 KF2V Plus";
  }
  return model;
}

function normalizeModelKey(model: string): string {
  return model.toLowerCase().replace(/\s+/g, "");
}

function getModelMetaId(model: string): VideoModelMetaId {
  const normalized = normalizeModelKey(model);
  if (normalized.includes("veo-3.1")) {
    return "veo31";
  }
  if (normalized.includes("sora-2-pro") || normalized.includes("sora2-pro")) {
    return "sora2Pro";
  }
  if (normalized.includes("sora-2")) {
    return "sora2";
  }
  if (normalized.includes("seedance-1-5-pro")) {
    return "seedance15Pro";
  }
  if (normalized.includes("kling-2.6")) {
    return "kling26";
  }
  if (normalized.includes("minimax-hailuo-2.3")) {
    return "minimaxHailuo23";
  }
  if (normalized.includes("minimax-hailuo-02")) {
    return "minimaxHailuo02";
  }
  if (normalized.includes("runway-gen-4-turbo")) {
    return "runwayGen4Turbo";
  }
  if (normalized.includes("seedance-1-5-lite")) {
    return "seedance15Lite";
  }
  if (normalized.includes("wanx2.1-t2v-turbo")) {
    return "wanx21T2vTurbo";
  }
  if (normalized.includes("wanx2.1-kf2v-plus")) {
    return "wanx21Kf2vPlus";
  }
  return "fallback";
}

function nextRandomSeed(): number {
  return Math.floor(Math.random() * 1_000_000_000);
}

export const VideoSidebar: React.FC<VideoSidebarProps> = memo(
  ({ state, providers, availableModels, onStateChange }) => {
    const { t, i18n } = useTranslation("workspace");
    const startFileInputRef = useRef<HTMLInputElement>(null);
    const endFileInputRef = useRef<HTMLInputElement>(null);
    const modelPanelRef = useRef<HTMLDivElement>(null);
    const [modelPanelOpen, setModelPanelOpen] = useState(false);
    const [draggingArea, setDraggingArea] = useState<FrameDropArea | null>(
      null,
    );

    useEffect(() => {
      if (!modelPanelOpen) {
        return;
      }
      const handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          setModelPanelOpen(false);
        }
      };
      window.addEventListener("keydown", handleKeyDown);
      return () => {
        window.removeEventListener("keydown", handleKeyDown);
      };
    }, [modelPanelOpen]);

    const modelMetaCopy = useMemo<VideoModelMetaCopy>(
      () => ({
        veo31: {
          cost: t("workspace.video.sidebar.model.meta.veo31.cost", {
            defaultValue: "30 credits/秒 · 8 秒约 240",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.veo31.description",
            {
              defaultValue: "Google Veo 3.1 支持 1080p/4K、多图参考与首尾帧。",
            },
          ),
        },
        sora2Pro: {
          cost: t("workspace.video.sidebar.model.meta.sora2Pro.cost", {
            defaultValue: "20 credits/秒 · 4 秒约 80",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.sora2Pro.description",
            {
              defaultValue: "Sora-2 Pro 生成时间约 2 分钟，稳定性高。",
            },
          ),
        },
        sora2: {
          cost: t("workspace.video.sidebar.model.meta.sora2.cost", {
            defaultValue: "2.7 credits/秒 · 15 秒约 40.5",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.sora2.description",
            {
              defaultValue: "Sora 2 最长 15 秒，不支持上传人物图。",
            },
          ),
        },
        seedance15Pro: {
          cost: t("workspace.video.sidebar.model.meta.seedance15Pro.cost", {
            defaultValue: "20 credits/秒 · 5 秒约 100",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.seedance15Pro.description",
            {
              defaultValue: "支持文生视频与首帧 / 首尾帧图生视频。",
            },
          ),
        },
        kling26: {
          cost: t("workspace.video.sidebar.model.meta.kling26.cost", {
            defaultValue: "27 credits/秒 · 5 秒约 135",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.kling26.description",
            {
              defaultValue: "支持 1080p 文生视频和图生视频。",
            },
          ),
        },
        minimaxHailuo23: {
          cost: t("workspace.video.sidebar.model.meta.minimaxHailuo23.cost", {
            defaultValue: "25 credits/秒 · 6 秒约 150",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.minimaxHailuo23.description",
            {
              defaultValue: "支持文生视频和图生视频，适合泛场景生成。",
            },
          ),
        },
        minimaxHailuo02: {
          cost: t("workspace.video.sidebar.model.meta.minimaxHailuo02.cost", {
            defaultValue: "25 credits/秒 · 6 秒约 150",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.minimaxHailuo02.description",
            {
              defaultValue: "支持首尾帧与 1080p 输出。",
            },
          ),
        },
        runwayGen4Turbo: {
          cost: t("workspace.video.sidebar.model.meta.runwayGen4Turbo.cost", {
            defaultValue: "30 credits/秒 · 5 秒约 150",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.runwayGen4Turbo.description",
            {
              defaultValue: "仅支持图生视频，适合已有视觉锚点的场景。",
            },
          ),
        },
        seedance15Lite: {
          cost: t("workspace.video.sidebar.model.meta.seedance15Lite.cost", {
            defaultValue: "8 credits/秒 · 5 秒约 40",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.seedance15Lite.description",
            {
              defaultValue: "轻量版 Seedance，速度更快、成本更低。",
            },
          ),
        },
        wanx21T2vTurbo: {
          cost: t("workspace.video.sidebar.model.meta.wanx21T2vTurbo.cost", {
            defaultValue: "18 credits/秒 · 5 秒约 90",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.wanx21T2vTurbo.description",
            {
              defaultValue: "阿里万相文生视频 Turbo 模型。",
            },
          ),
        },
        wanx21Kf2vPlus: {
          cost: t("workspace.video.sidebar.model.meta.wanx21Kf2vPlus.cost", {
            defaultValue: "22 credits/秒 · 5 秒约 110",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.wanx21Kf2vPlus.description",
            {
              defaultValue: "阿里万相关键帧图生视频 Plus 模型。",
            },
          ),
        },
        fallback: {
          cost: t("workspace.video.sidebar.model.meta.fallback.cost", {
            defaultValue: "按服务商计费",
          }),
          description: t(
            "workspace.video.sidebar.model.meta.fallback.description",
            {
              defaultValue: "具体能力与计费以服务商后台为准。",
            },
          ),
        },
      }),
      [t],
    );

    const modelOptions = useMemo(() => {
      const options: VideoModelOption[] = [];
      const seenKeys = new Set<string>();
      for (const provider of providers) {
        const providerModels =
          provider.models.length > 0
            ? provider.models
            : provider.id === state.providerId
              ? availableModels
              : [];
        for (const model of providerModels) {
          const key = `${provider.id}::${model}`;
          if (seenKeys.has(key)) {
            continue;
          }
          seenKeys.add(key);
          const meta = modelMetaCopy[getModelMetaId(model)];
          options.push({
            key,
            providerId: provider.id,
            providerName: provider.name,
            model,
            label: getModelLabel(model),
            cost: meta.cost,
            description: meta.description,
          });
        }
      }
      if (options.length === 0 && state.providerId && state.model) {
        const meta = modelMetaCopy[getModelMetaId(state.model)];
        const fallbackProviderName =
          providers.find((provider) => provider.id === state.providerId)
            ?.name ?? state.providerId;
        options.push({
          key: `${state.providerId}::${state.model}`,
          providerId: state.providerId,
          providerName: fallbackProviderName,
          model: state.model,
          label: getModelLabel(state.model),
          cost: meta.cost,
          description: meta.description,
        });
      }
      return options;
    }, [
      availableModels,
      modelMetaCopy,
      providers,
      state.model,
      state.providerId,
    ]);

    const selectedModelKey = useMemo(() => {
      const currentKey = `${state.providerId}::${state.model}`;
      if (modelOptions.some((item) => item.key === currentKey)) {
        return currentKey;
      }
      return modelOptions[0]?.key ?? "";
    }, [modelOptions, state.model, state.providerId]);

    const selectedModelOption = useMemo(
      () => modelOptions.find((item) => item.key === selectedModelKey) ?? null,
      [modelOptions, selectedModelKey],
    );

    const referenceCount = useMemo(
      () => [state.startImage, state.endImage].filter(Boolean).length,
      [state.endImage, state.startImage],
    );
    const locale = i18n.language;
    const formattedDuration = formatNumber(state.duration, { locale });
    const formattedReferenceCount = formatNumber(referenceCount, { locale });
    const referenceSummary =
      referenceCount > 0
        ? t("workspace.video.sidebar.summary.referenceImages.ready", {
            count: referenceCount,
            defaultValue: "{{value}} 张已就绪",
            value: formattedReferenceCount,
          })
        : t("workspace.video.sidebar.summary.referenceImages.empty", {
            defaultValue: "暂未上传",
          });

    const frameConfigs = useMemo<
      {
        title: string;
        tipAria: string;
        tipContent: string;
        previewAlt: string;
        field: FrameImageField;
        area: FrameDropArea;
      }[]
    >(
      () => [
        {
          title: t("workspace.video.sidebar.reference.start.title", {
            defaultValue: "起始画面",
          }),
          tipAria: t("workspace.video.sidebar.reference.start.tipAria", {
            defaultValue: "起始画面说明",
          }),
          tipContent: t("workspace.video.sidebar.reference.start.tipContent", {
            defaultValue: "用于锁定开场构图、人物与场景氛围。",
          }),
          previewAlt: t("workspace.video.sidebar.reference.start.previewAlt", {
            defaultValue: "起始画面预览",
          }),
          field: "startImage",
          area: "start",
        },
        {
          title: t("workspace.video.sidebar.reference.end.title", {
            defaultValue: "结束画面",
          }),
          tipAria: t("workspace.video.sidebar.reference.end.tipAria", {
            defaultValue: "结束画面说明",
          }),
          tipContent: t("workspace.video.sidebar.reference.end.tipContent", {
            defaultValue: "用于约束收尾镜头，让前后画面更连贯。",
          }),
          previewAlt: t("workspace.video.sidebar.reference.end.previewAlt", {
            defaultValue: "结束画面预览",
          }),
          field: "endImage",
          area: "end",
        },
      ],
      [t],
    );

    const setFrameImage = (field: FrameImageField, value?: string) => {
      if (field === "startImage") {
        onStateChange({ ...state, startImage: value });
        return;
      }
      onStateChange({ ...state, endImage: value });
    };

    const openFramePicker = (field: FrameImageField) => {
      const inputRef =
        field === "startImage" ? startFileInputRef : endFileInputRef;
      inputRef.current?.click();
    };

    const handleUploadFiles = async (
      field: FrameImageField,
      files: FileList | null,
    ) => {
      const imageFile = Array.from(files ?? []).find((file) =>
        file.type.startsWith("image/"),
      );
      if (!imageFile) {
        return;
      }
      try {
        const dataUrl = await fileToDataUrl(imageFile);
        setFrameImage(field, dataUrl);
      } catch (_error) {
        return;
      }
    };

    return (
      <SidebarWrapper>
        <PanelIntro>
          <PanelEyebrow>
            {t("workspace.video.sidebar.intro.eyebrow", {
              defaultValue: "VIDEO CONTROL",
            })}
          </PanelEyebrow>
          <PanelTitleRow>
            <PanelTitle>
              {t("workspace.video.sidebar.intro.title", {
                defaultValue: "生成参数",
              })}
            </PanelTitle>
            <WorkbenchInfoTip
              ariaLabel={t("workspace.video.sidebar.intro.tipAria", {
                defaultValue: "生成参数说明",
              })}
              content={t("workspace.video.sidebar.intro.tipContent", {
                defaultValue:
                  "先确定模型，再补参考图和输出规格。这里保持轻量控制，主创作仍留在右侧画布。",
              })}
              tone="mint"
            />
          </PanelTitleRow>
          <PanelMetaGrid>
            <PanelMetaCard>
              <PanelMetaLabel>
                {t("workspace.video.sidebar.summary.currentModel.label", {
                  defaultValue: "当前模型",
                })}
              </PanelMetaLabel>
              <PanelMetaValue>
                {selectedModelOption?.label ??
                  t("workspace.video.sidebar.summary.currentModel.pending", {
                    defaultValue: "待配置视频模型",
                  })}
              </PanelMetaValue>
            </PanelMetaCard>
            <PanelMetaCard>
              <PanelMetaLabel>
                {t("workspace.video.sidebar.summary.outputSpec.label", {
                  defaultValue: "输出规格",
                })}
              </PanelMetaLabel>
              <PanelMetaValue>
                {state.aspectRatio} · {state.resolution}
              </PanelMetaValue>
            </PanelMetaCard>
            <PanelMetaCard>
              <PanelMetaLabel>
                {t("workspace.video.sidebar.summary.duration.label", {
                  defaultValue: "时长",
                })}
              </PanelMetaLabel>
              <PanelMetaValue>
                {t("workspace.video.sidebar.summary.duration.value", {
                  defaultValue: "{{value}} 秒",
                  value: formattedDuration,
                })}
              </PanelMetaValue>
            </PanelMetaCard>
            <PanelMetaCard>
              <PanelMetaLabel>
                {t("workspace.video.sidebar.summary.referenceImages.label", {
                  defaultValue: "参考图",
                })}
              </PanelMetaLabel>
              <PanelMetaValue>{referenceSummary}</PanelMetaValue>
            </PanelMetaCard>
          </PanelMetaGrid>
        </PanelIntro>

        <Section>
          <SectionHeader>
            <SectionTitle>
              {t("workspace.video.sidebar.model.sectionTitle", {
                defaultValue: "模型",
              })}
            </SectionTitle>
            <WorkbenchInfoTip
              ariaLabel={t("workspace.video.sidebar.model.tipAria", {
                defaultValue: "模型说明",
              })}
              content={t("workspace.video.sidebar.model.tipContent", {
                defaultValue:
                  "模型能力决定可选分辨率、时长和图生视频支持范围。",
              })}
              tone="mint"
            />
          </SectionHeader>
          <ModelTrigger
            type="button"
            onClick={() => setModelPanelOpen(true)}
            title={t("workspace.video.sidebar.model.triggerTitle", {
              defaultValue: "选择视频模型",
            })}
          >
            <ModelTriggerBody>
              <ModelTriggerLabel>
                {selectedModelOption?.label ??
                  t("workspace.video.sidebar.model.emptyLabel", {
                    defaultValue: "暂无可用视频模型",
                  })}
              </ModelTriggerLabel>
              <ModelTriggerMeta>
                {selectedModelOption?.providerName ??
                  t("workspace.video.sidebar.model.providerMissing", {
                    defaultValue: "请先配置支持视频的 Provider",
                  })}
              </ModelTriggerMeta>
            </ModelTriggerBody>
            <ChevronDown size={16} />
          </ModelTrigger>
        </Section>

        {modelPanelOpen ? (
          <ModelPanelMask
            onClick={(event) => {
              if (event.target === event.currentTarget) {
                setModelPanelOpen(false);
              }
            }}
          >
            <ModelPanel ref={modelPanelRef}>
              <ModelPanelHeader>
                <ModelPanelEyebrow>
                  {t("workspace.video.sidebar.model.panel.eyebrow", {
                    defaultValue: "MODEL LIBRARY",
                  })}
                </ModelPanelEyebrow>
                <ModelPanelTitle>
                  {t("workspace.video.sidebar.model.panel.title", {
                    defaultValue: "选择视频模型",
                  })}
                </ModelPanelTitle>
                <ModelPanelDescription>
                  {t("workspace.video.sidebar.model.panel.description", {
                    defaultValue:
                      "统一在一个面板里查看模型能力、成本与 Provider 来源，避免在侧栏里堆叠过多信息。",
                  })}
                </ModelPanelDescription>
              </ModelPanelHeader>
              <ModelPanelDivider />
              <ModelPanelList>
                {modelOptions.length === 0 ? (
                  <ModelPanelItem
                    type="button"
                    $active={false}
                    onClick={() => setModelPanelOpen(false)}
                  >
                    <ModelPanelBody>
                      <ModelPanelName>
                        {t("workspace.video.sidebar.model.empty.title", {
                          defaultValue: "暂无可用视频模型",
                        })}
                      </ModelPanelName>
                      <ModelPanelDesc>
                        {t("workspace.video.sidebar.model.empty.description", {
                          defaultValue: "请先配置支持视频生成的 Provider。",
                        })}
                      </ModelPanelDesc>
                    </ModelPanelBody>
                    <ModelPanelSelected $active={false}>
                      <X size={14} />
                    </ModelPanelSelected>
                  </ModelPanelItem>
                ) : (
                  modelOptions.map((option) => (
                    <ModelPanelItem
                      key={option.key}
                      type="button"
                      $active={option.key === selectedModelKey}
                      onClick={() => {
                        onStateChange({
                          ...state,
                          providerId: option.providerId,
                          model: option.model,
                        });
                        setModelPanelOpen(false);
                      }}
                    >
                      <ModelPanelBody>
                        <ModelPanelName>{option.label}</ModelPanelName>
                        <ModelPanelMetaRow>
                          <ModelPanelCost>{option.cost}</ModelPanelCost>
                          <ModelProviderTag>
                            {option.providerName}
                          </ModelProviderTag>
                        </ModelPanelMetaRow>
                        <ModelPanelDesc>{option.description}</ModelPanelDesc>
                      </ModelPanelBody>
                      <ModelPanelSelected
                        $active={option.key === selectedModelKey}
                      >
                        {option.key === selectedModelKey ? (
                          <Check size={15} />
                        ) : null}
                      </ModelPanelSelected>
                    </ModelPanelItem>
                  ))
                )}
              </ModelPanelList>
            </ModelPanel>
          </ModelPanelMask>
        ) : null}

        {frameConfigs.map((frame) => {
          const previewImage =
            frame.field === "startImage" ? state.startImage : state.endImage;
          const inputRef =
            frame.field === "startImage" ? startFileInputRef : endFileInputRef;

          return (
            <Section key={frame.field}>
              <SectionHeader>
                <SectionTitle>{frame.title}</SectionTitle>
                <WorkbenchInfoTip
                  ariaLabel={frame.tipAria}
                  content={frame.tipContent}
                  tone="mint"
                />
              </SectionHeader>
              <ImageUploadArea
                $dragging={draggingArea === frame.area}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDraggingArea(frame.area);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  const relatedTarget = event.relatedTarget as Node | null;
                  if (
                    relatedTarget &&
                    event.currentTarget.contains(relatedTarget)
                  ) {
                    return;
                  }
                  setDraggingArea((current) =>
                    current === frame.area ? null : current,
                  );
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setDraggingArea(null);
                  void handleUploadFiles(frame.field, event.dataTransfer.files);
                }}
              >
                {previewImage ? (
                  <PreviewBox>
                    <img src={previewImage} alt={frame.previewAlt} />
                    <ReplacePreviewButton
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        openFramePicker(frame.field);
                      }}
                    >
                      {t("workspace.video.sidebar.reference.replaceAction", {
                        defaultValue: "更换",
                      })}
                    </ReplacePreviewButton>
                    <RemovePreviewButton
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        setFrameImage(frame.field, undefined);
                      }}
                    >
                      <X size={14} />
                    </RemovePreviewButton>
                    <ReplaceHint>
                      {t("workspace.video.sidebar.reference.replaceHint", {
                        defaultValue: "拖拽上传即可替换当前图片",
                      })}
                    </ReplaceHint>
                  </PreviewBox>
                ) : (
                  <UploadPrompt>
                    <ImagePlus size={18} />
                    <div>
                      {t("workspace.video.sidebar.reference.empty.title", {
                        defaultValue: "添加图片",
                      })}
                    </div>
                    <div>
                      {t(
                        "workspace.video.sidebar.reference.empty.description",
                        {
                          defaultValue: "拖拽或点击上传参考图",
                        },
                      )}
                    </div>
                    <UploadActionButton
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        openFramePicker(frame.field);
                      }}
                    >
                      {t("workspace.video.sidebar.reference.empty.action", {
                        defaultValue: "选择图片",
                      })}
                    </UploadActionButton>
                  </UploadPrompt>
                )}
              </ImageUploadArea>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(event) => {
                  void handleUploadFiles(frame.field, event.target.files);
                  event.target.value = "";
                }}
              />
            </Section>
          );
        })}

        <Section>
          <SectionTitle>
            {t("workspace.video.sidebar.controls.aspectRatio.title", {
              defaultValue: "宽高比",
            })}
          </SectionTitle>
          <RatioGrid>
            {RATIOS.map((ratio) => {
              const ratioLabel =
                ratio.value === "adaptive"
                  ? t("workspace.video.sidebar.controls.aspectRatio.adaptive", {
                      defaultValue: "自适应",
                    })
                  : ratio.label;
              return (
                <RatioItem
                  key={ratio.value}
                  type="button"
                  $active={state.aspectRatio === ratio.value}
                  onClick={() =>
                    onStateChange({ ...state, aspectRatio: ratio.value })
                  }
                >
                  <RatioShape
                    $active={state.aspectRatio === ratio.value}
                    style={{
                      aspectRatio:
                        ratio.value === "adaptive"
                          ? "1 / 1"
                          : ratio.value.replace(":", "/"),
                      borderStyle:
                        ratio.value === "adaptive" ? "dashed" : "solid",
                    }}
                  />
                  {ratioLabel}
                </RatioItem>
              );
            })}
          </RatioGrid>
        </Section>

        <Section>
          <SectionTitle>
            {t("workspace.video.sidebar.controls.resolution.title", {
              defaultValue: "分辨率",
            })}
          </SectionTitle>
          <ResolutionGroup>
            {(["480p", "720p", "1080p"] as VideoResolution[]).map(
              (resolution) => (
                <ResolutionButton
                  key={resolution}
                  type="button"
                  $active={state.resolution === resolution}
                  onClick={() => onStateChange({ ...state, resolution })}
                >
                  {resolution}
                </ResolutionButton>
              ),
            )}
          </ResolutionGroup>
        </Section>

        <Section>
          <SectionHeader>
            <SectionTitle>
              {t("workspace.video.sidebar.controls.duration.title", {
                defaultValue: "时长",
              })}
            </SectionTitle>
            <WorkbenchInfoTip
              ariaLabel={t(
                "workspace.video.sidebar.controls.duration.tipAria",
                {
                  defaultValue: "时长说明",
                },
              )}
              content={t(
                "workspace.video.sidebar.controls.duration.tipContent",
                {
                  defaultValue:
                    "建议先用 4 到 8 秒验证镜头是否成立，再逐步拉长。",
                },
              )}
              tone="mint"
            />
          </SectionHeader>
          <DurationRow>
            <DurationSlider
              type="range"
              min={1}
              max={20}
              step={1}
              value={state.duration}
              onChange={(event) =>
                onStateChange({
                  ...state,
                  duration: Number.parseInt(event.target.value, 10),
                })
              }
            />
            <DurationValue
              type="number"
              min={1}
              max={20}
              value={state.duration}
              onChange={(event) => {
                const value = Number.parseInt(event.target.value, 10);
                if (!Number.isFinite(value)) {
                  return;
                }
                onStateChange({
                  ...state,
                  duration: Math.min(20, Math.max(1, value)),
                });
              }}
            />
          </DurationRow>
        </Section>

        <Section>
          <SectionHeader>
            <SectionTitle>
              {t("workspace.video.sidebar.controls.seed.title", {
                defaultValue: "种子",
              })}
            </SectionTitle>
            <WorkbenchInfoTip
              ariaLabel={t("workspace.video.sidebar.controls.seed.tipAria", {
                defaultValue: "种子说明",
              })}
              content={t("workspace.video.sidebar.controls.seed.tipContent", {
                defaultValue:
                  "需要复现某次结果时再固定种子；探索阶段保持随机即可。",
              })}
              tone="mint"
            />
          </SectionHeader>
          <SeedRow>
            <SeedInput
              type="number"
              placeholder={t(
                "workspace.video.sidebar.controls.seed.placeholder",
                {
                  defaultValue: "随机",
                },
              )}
              value={state.seed ?? ""}
              onChange={(event) => {
                const raw = event.target.value.trim();
                if (!raw) {
                  onStateChange({ ...state, seed: undefined });
                  return;
                }
                const value = Number.parseInt(raw, 10);
                if (!Number.isFinite(value)) {
                  return;
                }
                onStateChange({
                  ...state,
                  seed: Math.max(0, value),
                });
              }}
            />
            <SeedRandomButton
              type="button"
              title={t("workspace.video.sidebar.controls.seed.randomTitle", {
                defaultValue: "随机种子",
              })}
              onClick={() =>
                onStateChange({
                  ...state,
                  seed: nextRandomSeed(),
                })
              }
            >
              <Dices size={16} />
            </SeedRandomButton>
          </SeedRow>
        </Section>

        <Section>
          <ToggleRow>
            <ToggleCopy>
              <ToggleTitleRow>
                <ToggleTitle>
                  {t("workspace.video.sidebar.controls.generateAudio.title", {
                    defaultValue: "生成音频",
                  })}
                </ToggleTitle>
                <WorkbenchInfoTip
                  ariaLabel={t(
                    "workspace.video.sidebar.controls.generateAudio.tipAria",
                    {
                      defaultValue: "生成音频说明",
                    },
                  )}
                  content={t(
                    "workspace.video.sidebar.controls.generateAudio.tipContent",
                    {
                      defaultValue: "需要环境声或基础配乐时再开启。",
                    },
                  )}
                  tone="mint"
                />
              </ToggleTitleRow>
            </ToggleCopy>
            <ToggleSwitch
              type="button"
              $checked={state.generateAudio}
              onClick={() =>
                onStateChange({ ...state, generateAudio: !state.generateAudio })
              }
            >
              <ToggleDot />
            </ToggleSwitch>
          </ToggleRow>
          <ToggleRow>
            <ToggleCopy>
              <ToggleTitleRow>
                <ToggleTitle>
                  {t("workspace.video.sidebar.controls.cameraFixed.title", {
                    defaultValue: "固定镜头",
                  })}
                </ToggleTitle>
                <WorkbenchInfoTip
                  ariaLabel={t(
                    "workspace.video.sidebar.controls.cameraFixed.tipAria",
                    {
                      defaultValue: "固定镜头说明",
                    },
                  )}
                  content={t(
                    "workspace.video.sidebar.controls.cameraFixed.tipContent",
                    {
                      defaultValue: "减少镜头摇移，适合产品或静态场景。",
                    },
                  )}
                  tone="mint"
                />
              </ToggleTitleRow>
            </ToggleCopy>
            <ToggleSwitch
              type="button"
              $checked={state.cameraFixed}
              onClick={() =>
                onStateChange({ ...state, cameraFixed: !state.cameraFixed })
              }
            >
              <ToggleDot />
            </ToggleSwitch>
          </ToggleRow>
        </Section>
      </SidebarWrapper>
    );
  },
);

VideoSidebar.displayName = "VideoSidebar";
