import { ListOrdered, Loader2, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { QueuedSubmission, UserInput } from "@limecloud/app-server-client";
import { useAgentSessionThreadQueue } from "../hooks/useAgentSessionThreadQueue";

const VISIBLE_SUBMISSION_LIMIT = 3;

function submissionPreview(
  submission: QueuedSubmission,
  fallback: string,
): string {
  const text = submission.input
    .filter(
      (input): input is Extract<UserInput, { type: "text" }> =>
        input.type === "text",
    )
    .map((input) => input.text.trim())
    .filter(Boolean)
    .join(" ");
  return text || fallback;
}

export function ThreadQueueStatus({ threadId }: { threadId?: string | null }) {
  const { t } = useTranslation("agent");
  const queue = useAgentSessionThreadQueue({ threadId });
  const text = (key: string, options?: Record<string, unknown>) =>
    String(t(key as never, options as never));

  if (
    !threadId?.trim() ||
    (!queue.loading && !queue.error && queue.submissions.length === 0)
  ) {
    return null;
  }

  const visible = queue.submissions.slice(0, VISIBLE_SUBMISSION_LIMIT);
  const hiddenCount = Math.max(0, queue.submissions.length - visible.length);

  return (
    <section
      aria-label={text("agentChat.threadQueue.title")}
      className="mx-4 mb-2 border-t border-slate-200/80 bg-slate-50 px-3 py-2 text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
      data-testid="thread-queue-status"
    >
      <div className="flex min-h-7 items-center gap-2">
        <ListOrdered className="h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" />
        <span className="text-xs font-semibold">
          {text("agentChat.threadQueue.title")}
        </span>
        {queue.loading ? (
          <Loader2
            aria-label={text("agentChat.threadQueue.loading")}
            className="h-3.5 w-3.5 animate-spin text-slate-400"
          />
        ) : (
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {text("agentChat.threadQueue.pendingCount", {
              count: queue.submissions.length,
            })}
          </span>
        )}
        {queue.error ? (
          <span className="ml-auto inline-flex items-center gap-1 text-xs text-rose-600 dark:text-rose-400">
            <RefreshCw className="h-3.5 w-3.5" />
            {text("agentChat.threadQueue.refreshFailed")}
          </span>
        ) : null}
      </div>
      {visible.length > 0 ? (
        <ol className="mt-1 grid gap-1" data-testid="thread-queue-items">
          {visible.map((submission, index) => (
            <li
              className="flex min-w-0 items-center gap-2 text-xs"
              key={submission.id}
            >
              <span className="w-4 shrink-0 text-right tabular-nums text-slate-400">
                {index + 1}
              </span>
              <span className="min-w-0 flex-1 truncate">
                {submissionPreview(
                  submission,
                  text("agentChat.threadQueue.itemFallback"),
                )}
              </span>
            </li>
          ))}
          {hiddenCount > 0 ? (
            <li className="pl-6 text-xs text-slate-500 dark:text-slate-400">
              {text("agentChat.threadQueue.more", { count: hiddenCount })}
            </li>
          ) : null}
        </ol>
      ) : null}
    </section>
  );
}
