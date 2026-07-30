import { FileAudio, FileVideo, Image as ImageIcon } from "lucide-react";
import { memo, useState } from "react";
import { resolveLocalFilePreviewUrl } from "@/lib/api/fileSystem";
import { cn } from "@/lib/utils";
import type { Artifact } from "@/lib/artifact/types";
import { PreviewArtifactFallbackSurface } from "./PreviewArtifactFallbackSurface";

export type PreviewMediaContentKind = "image" | "audio" | "video";

interface PreviewMediaRendererProps {
  artifact: Artifact;
  contentKind: PreviewMediaContentKind;
  tone?: "dark" | "light";
}

function readStringMeta(meta: Artifact["meta"], key: string): string | null {
  const value = meta[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function resolvePreviewArtifactFileUrl(artifact: Artifact): string | null {
  const content = artifact.content.trim();
  const previewUrl = readStringMeta(artifact.meta, "previewUrl");
  const sourcePath =
    readStringMeta(artifact.meta, "filePath") ||
    readStringMeta(artifact.meta, "sourcePath");

  if (previewUrl) {
    return previewUrl;
  }
  if (
    content.startsWith("asset://") ||
    content.startsWith("file://") ||
    content.startsWith("http://") ||
    content.startsWith("https://")
  ) {
    return content;
  }
  if (sourcePath) {
    return resolveLocalFilePreviewUrl(sourcePath) || sourcePath;
  }
  return null;
}

export const PreviewMediaRenderer = memo(function PreviewMediaRenderer({
  artifact,
  contentKind,
  tone = "dark",
}: PreviewMediaRendererProps) {
  const mediaUrl = resolvePreviewArtifactFileUrl(artifact);
  const loadIdentity = `${contentKind}:${mediaUrl ?? ""}`;
  const [failedLoadIdentity, setFailedLoadIdentity] = useState<string | null>(
    null,
  );
  const hasLoadError =
    Boolean(mediaUrl) && failedLoadIdentity === loadIdentity;
  const filename =
    readStringMeta(artifact.meta, "filename") || artifact.title || "preview";
  const isLight = tone === "light";
  const shellClassName = cn(
    "flex h-full min-h-[320px] flex-col",
    isLight ? "bg-background" : "bg-[#1e2227]",
  );
  const frameClassName = cn(
    "m-6 flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-2xl border",
    isLight ? "border-border bg-muted/30" : "border-white/10 bg-black/20",
  );

  if (hasLoadError) {
    return (
      <PreviewArtifactFallbackSurface
        artifact={artifact}
        mode="unsupported"
        tone={tone}
      />
    );
  }

  if (!mediaUrl) {
    const Icon =
      contentKind === "audio"
        ? FileAudio
        : contentKind === "video"
          ? FileVideo
          : ImageIcon;
    return (
      <div className={shellClassName}>
        <div className={frameClassName}>
          <div
            className={cn(
              "flex flex-col items-center gap-3 text-sm",
              isLight ? "text-muted-foreground" : "text-gray-400",
            )}
          >
            <Icon className="h-8 w-8" />
            <span>{filename}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={shellClassName}>
      <div className={frameClassName}>
        {contentKind === "image" ? (
          <img
            src={mediaUrl}
            alt={filename}
            className="max-h-full max-w-full object-contain"
            data-testid="preview-artifact-image"
            onError={() => setFailedLoadIdentity(loadIdentity)}
          />
        ) : contentKind === "audio" ? (
          <audio
            src={mediaUrl}
            controls
            className="w-full max-w-xl"
            data-testid="preview-artifact-audio"
            onError={() => setFailedLoadIdentity(loadIdentity)}
          />
        ) : (
          <video
            src={mediaUrl}
            controls
            className="max-h-full max-w-full"
            data-testid="preview-artifact-video"
            onError={() => setFailedLoadIdentity(loadIdentity)}
          />
        )}
      </div>
    </div>
  );
});

PreviewMediaRenderer.displayName = "PreviewMediaRenderer";
