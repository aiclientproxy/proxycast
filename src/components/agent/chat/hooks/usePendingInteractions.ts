import { useCallback, useEffect, useSyncExternalStore } from "react";
import {
  getDefaultPendingInteractionController,
  type PendingInteractionController,
  type PendingInteractionResponse,
} from "@/lib/api/agentRuntime/pendingInteractionController";

/** 绑定唯一 controller，并向 Composer 暴露当前 typed pending projection。 */
export function usePendingInteractions(
  controller: PendingInteractionController = getDefaultPendingInteractionController(),
) {
  useEffect(() => controller.attach(), [controller]);
  const interactions = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );
  const respond = useCallback(
    (response: PendingInteractionResponse) => controller.respond(response),
    [controller],
  );

  return { interactions, respond };
}
