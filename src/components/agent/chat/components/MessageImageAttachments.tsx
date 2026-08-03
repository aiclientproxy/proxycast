import React from "react";
import { useTranslation } from "react-i18next";
import type { Message } from "../types";
import { createAppServerClient } from "@/lib/api/appServer";
import { resolveLocalFilePreviewUrl } from "@/lib/api/fileSystem";
import { buildMessageImageDataUrl } from "../utils/imageAttachments";
import { ImageUnavailablePlaceholder } from "./ImageUnavailablePlaceholder";

const MESSAGE_IMAGE_PREVIEW_MAX_BYTES = 32 * 1024 * 1024;

interface MessageImageAttachmentsProps {
  images: Message["images"];
  threadId?: string | null;
  onOpenImage?: (
    image: NonNullable<Message["images"]>[number],
    index: number,
  ) => void;
}

function isSidecarUri(uri?: string | null): boolean {
  return Boolean(uri?.trim().toLowerCase().startsWith("sidecar://"));
}

function isDirectPreviewUri(uri: string): boolean {
  return (
    /^(?:https?|file|asset|blob|data):/iu.test(uri) || uri.startsWith("//")
  );
}

function resolveMessageImageSrc(image: NonNullable<Message["images"]>[number]) {
  const data = image.data.trim();
  if (data) {
    return data.toLowerCase().startsWith("data:")
      ? data
      : buildMessageImageDataUrl(image);
  }

  const previewUrl = image.previewUrl?.trim();
  if (previewUrl && isDirectPreviewUri(previewUrl)) {
    return previewUrl;
  }
  if (image.sourcePath?.trim()) {
    return resolveLocalFilePreviewUrl(image.sourcePath) || image.sourcePath;
  }
  const sourceUri = image.sourceUri?.trim();
  if (sourceUri && isDirectPreviewUri(sourceUri)) {
    return sourceUri;
  }
  return "";
}

function resolveMessageImageSidecarUri(
  image: NonNullable<Message["images"]>[number],
): string {
  return (
    [image.previewUrl, image.sourceUri]
      .map((value) => value?.trim() || "")
      .find(isSidecarUri) || ""
  );
}

function MessageImageAttachment({
  image,
  index,
  onOpenImage,
  threadId,
}: {
  image: NonNullable<Message["images"]>[number];
  index: number;
  threadId?: string | null;
  onOpenImage?: MessageImageAttachmentsProps["onOpenImage"];
}) {
  const { t } = useTranslation("agent");
  const directSrc = resolveMessageImageSrc(image);
  const sidecarUri = resolveMessageImageSidecarUri(image);
  const normalizedThreadId = threadId?.trim() || "";
  const [sidecarSrc, setSidecarSrc] = React.useState("");
  const [loadFailed, setLoadFailed] = React.useState(false);

  React.useEffect(() => {
    setSidecarSrc("");
    setLoadFailed(false);
    if (directSrc || !sidecarUri || !normalizedThreadId) {
      return;
    }

    const abortController = new AbortController();
    void createAppServerClient()
      .readMedia(
        {
          threadId: normalizedThreadId,
          uri: sidecarUri,
          maxBytes: MESSAGE_IMAGE_PREVIEW_MAX_BYTES,
          length: MESSAGE_IMAGE_PREVIEW_MAX_BYTES,
        },
        { signal: abortController.signal },
      )
      .then(({ result }) => {
        const mimeType = result.mimeType?.trim() || image.mediaType.trim();
        if (
          abortController.signal.aborted ||
          !mimeType.toLowerCase().startsWith("image/") ||
          !result.contentBase64 ||
          result.offset !== 0 ||
          result.hasMore ||
          result.bytes !== result.totalBytes
        ) {
          if (!abortController.signal.aborted) {
            setLoadFailed(true);
          }
          return;
        }
        setSidecarSrc(`data:${mimeType};base64,${result.contentBase64}`);
      })
      .catch(() => {
        if (!abortController.signal.aborted) {
          setLoadFailed(true);
        }
      });

    return () => abortController.abort();
  }, [directSrc, image.mediaType, normalizedThreadId, sidecarUri]);

  const src = directSrc || sidecarSrc;
  const isUnavailable =
    loadFailed || (!src && (!sidecarUri || !normalizedThreadId));
  const imageNode =
    isUnavailable || !src ? (
      <ImageUnavailablePlaceholder
        label={t("agentChat.messageImageAttachments.unavailable")}
        testId={`message-image-attachment-unavailable-${index}`}
        className="h-28 w-40 max-w-xs"
      />
    ) : (
      <img
        src={src}
        className="max-h-64 max-w-xs rounded-lg border border-border object-contain"
        alt={t("agentChat.messageImageAttachments.alt")}
        data-testid={`message-image-attachment-${index}`}
        onError={() => setLoadFailed(true)}
      />
    );

  if (!onOpenImage) {
    return imageNode;
  }

  return (
    <button
      type="button"
      className="rounded-lg text-left transition focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
      data-testid={`message-image-attachment-open-${index}`}
      onClick={() => onOpenImage(image, index)}
    >
      {imageNode}
    </button>
  );
}

export function MessageImageAttachments({
  images,
  onOpenImage,
  threadId,
}: MessageImageAttachmentsProps) {
  if (!images?.length) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {images.map((img, index) => {
        return (
          <MessageImageAttachment
            key={index}
            image={img}
            index={index}
            onOpenImage={onOpenImage}
            threadId={threadId}
          />
        );
      })}
    </div>
  );
}
