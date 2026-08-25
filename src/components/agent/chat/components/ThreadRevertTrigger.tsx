import { Undo2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { ThreadRevertRequest } from "@/lib/api/threadRevert";

export interface ThreadRevertTarget extends ThreadRevertRequest {
  messageId: string;
}

interface ThreadRevertTriggerProps {
  target: ThreadRevertTarget;
  onRequest: (target: ThreadRevertTarget) => void;
}

export function ThreadRevertTrigger({
  target,
  onRequest,
}: ThreadRevertTriggerProps) {
  const { t } = useTranslation("agent");
  const label = t("agentChat.messageList.threadRevert.trigger");

  return (
    <button
      type="button"
      className="inline-flex h-6 w-6 items-center justify-center rounded-md text-slate-400 transition hover:bg-rose-50 hover:text-rose-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-200"
      data-testid="thread-revert-trigger"
      data-message-id={target.messageId}
      data-thread-id={target.threadId}
      data-before-turn-id={target.beforeTurnId}
      onClick={() => onRequest(target)}
      aria-label={label}
      title={label}
    >
      <Undo2 size={13} aria-hidden="true" />
    </button>
  );
}
