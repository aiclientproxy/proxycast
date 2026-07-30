import {
  subscribeAppServerNotifications,
  type AppServerMediaReadParams,
  type AppServerEventBusSubscription,
} from "@/lib/api/appServer";
import {
  emitStreamingMediaReadProgress,
  type MediaReferencePreviewProgress,
} from "./mediaReferencePreviewStreamingProgress";

const MEDIA_REFERENCE_PREVIEW_DRAIN_LIMIT = 20;
const MEDIA_REFERENCE_PREVIEW_DRAIN_INTERVAL_MS = 48;
const MEDIA_REFERENCE_PREVIEW_DRAIN_ACTIVE_INTERVAL_MS = 24;

type SubscribeNotifications = (
  subscription: AppServerEventBusSubscription,
) => () => void;

export function subscribeMediaReferencePreviewReadProgress(params: {
  onProgress?: (progress: MediaReferencePreviewProgress) => void;
  readRequest: AppServerMediaReadParams;
  shouldContinue?: () => boolean;
  subscribeNotifications?: SubscribeNotifications;
}): () => void {
  const threadId = params.readRequest.threadId.trim();
  if (!threadId || !params.onProgress) {
    return () => undefined;
  }

  let disposed = false;
  let streamId: string | undefined;
  const seenEventIds = new Set<string>();
  const subscribeNotifications =
    params.subscribeNotifications ?? subscribeAppServerNotifications;
  const unsubscribe = subscribeNotifications({
    getDrainOptions: () => ({
      activeIntervalMs: MEDIA_REFERENCE_PREVIEW_DRAIN_ACTIVE_INTERVAL_MS,
      includeRecent: true,
      intervalMs: MEDIA_REFERENCE_PREVIEW_DRAIN_INTERVAL_MS,
      limit: MEDIA_REFERENCE_PREVIEW_DRAIN_LIMIT,
    }),
    onNotifications: (notifications) => {
      if (disposed || params.shouldContinue?.() === false) {
        return;
      }
      const result = emitStreamingMediaReadProgress({
        expectedOffset: params.readRequest.offset ?? 0,
        expectedStreamId: streamId,
        expectedUri: params.readRequest.uri,
        notifications,
        onProgress: params.onProgress,
        onStreamId: (nextStreamId) => {
          streamId = nextStreamId;
        },
        seenEventIds,
        threadId,
      });
      streamId = result.streamId ?? streamId;
    },
    shouldDrain: () => !disposed && params.shouldContinue?.() !== false,
  });

  return () => {
    disposed = true;
    unsubscribe();
  };
}
