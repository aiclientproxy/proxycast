import { useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronUp,
  ListOrdered,
  Loader2,
  Pencil,
  Play,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAgentSessionThreadQueue } from "../hooks/useAgentSessionThreadQueue";
import {
  deleteThreadQueue,
  reorderThreadQueue,
  startThreadQueue,
  updateThreadQueue,
} from "@/lib/api/agentRuntime/threadQueueActions";
import type { UserInput } from "@limecloud/app-server-client";

function submissionText(input: readonly UserInput[]): string {
  return input
    .filter(
      (item): item is Extract<UserInput, { type: "text" }> =>
        item.type === "text",
    )
    .map((item) => item.text)
    .join("\n");
}

function replaceSubmissionText(
  input: readonly UserInput[],
  text: string,
): UserInput[] {
  const next = [...input];
  const textIndex = next.findIndex((item) => item.type === "text");
  if (textIndex >= 0) {
    next[textIndex] = { type: "text", text };
  } else {
    next.unshift({ type: "text", text });
  }
  return next;
}

export function ThreadQueueStatus({ threadId }: { threadId?: string | null }) {
  const { t } = useTranslation("agent");
  const queue = useAgentSessionThreadQueue({ threadId });
  const [expanded, setExpanded] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [actionId, setActionId] = useState<string | null>(null);
  const [actionError, setActionError] = useState(false);
  const text = (key: string, options?: Record<string, unknown>) =>
    String(t(key as never, options as never));

  if (
    !threadId?.trim() ||
    (!queue.loading &&
      !queue.error &&
      !actionError &&
      queue.submissions.length === 0)
  ) {
    return null;
  }

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
        {queue.error || actionError ? (
          <span className="ml-auto inline-flex items-center gap-1 text-xs text-rose-600 dark:text-rose-400">
            <RefreshCw className="h-3.5 w-3.5" />
            {text(
              queue.error
                ? "agentChat.threadQueue.refreshFailed"
                : "agentChat.threadQueue.actionFailed",
            )}
          </span>
        ) : null}
        <button
          type="button"
          className="ml-auto inline-flex h-7 w-7 items-center justify-center rounded-md text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-800"
          aria-label={text(
            expanded
              ? "agentChat.threadQueue.collapse"
              : "agentChat.threadQueue.expand",
          )}
          title={text(
            expanded
              ? "agentChat.threadQueue.collapse"
              : "agentChat.threadQueue.expand",
          )}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
      </div>
      {expanded ? (
        <div className="mt-2 space-y-1" data-testid="thread-queue-action-list">
          {queue.submissions.map((submission, index) => {
            const isEditing = editingId === submission.id;
            const busy = actionId === submission.id;
            const textValue = submissionText(submission.input);
            return (
              <div
                key={submission.id}
                className="flex items-start gap-1 rounded-md border border-slate-200/80 bg-white/70 p-1.5 dark:border-slate-700 dark:bg-slate-950/40"
                data-testid="thread-queue-action-item"
              >
                <div className="min-w-0 flex-1">
                  {isEditing ? (
                    <textarea
                      value={editingText}
                      onChange={(event) => setEditingText(event.target.value)}
                      rows={2}
                      className="w-full resize-y rounded border border-slate-300 bg-transparent px-2 py-1 text-xs outline-none focus:border-sky-500 dark:border-slate-600"
                      aria-label={text("agentChat.threadQueue.editInput")}
                    />
                  ) : (
                    <p className="whitespace-pre-wrap break-words px-1 text-xs">
                      {textValue || text("agentChat.threadQueue.nonTextInput")}
                    </p>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-0.5">
                  {isEditing ? (
                    <>
                      <button
                        type="button"
                        className="inline-flex h-6 w-6 items-center justify-center rounded text-emerald-600 hover:bg-emerald-50 dark:hover:bg-emerald-950/30"
                        aria-label={text("agentChat.threadQueue.save")}
                        title={text("agentChat.threadQueue.save")}
                        disabled={busy || !editingText.trim()}
                        onClick={() => {
                          setActionId(submission.id);
                          void updateThreadQueue({
                            threadId: threadId!,
                            queuedSubmissionId: submission.id,
                            input: replaceSubmissionText(
                              submission.input,
                              editingText,
                            ),
                          })
                            .then(() => {
                              setActionError(false);
                              setEditingId(null);
                              queue.refresh?.();
                            })
                            .catch(() => setActionError(true))
                            .finally(() => setActionId(null));
                        }}
                      >
                        <Check size={14} />
                      </button>
                      <button
                        type="button"
                        className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                        aria-label={text("agentChat.threadQueue.cancel")}
                        title={text("agentChat.threadQueue.cancel")}
                        onClick={() => setEditingId(null)}
                      >
                        <X size={14} />
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                      aria-label={text("agentChat.threadQueue.edit")}
                      title={text("agentChat.threadQueue.edit")}
                      onClick={() => {
                        setEditingId(submission.id);
                        setEditingText(textValue);
                      }}
                    >
                      <Pencil size={13} />
                    </button>
                  )}
                  <button
                    type="button"
                    className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
                    aria-label={text("agentChat.threadQueue.moveUp")}
                    title={text("agentChat.threadQueue.moveUp")}
                    disabled={busy || index === 0}
                    onClick={() => {
                      const ids = queue.submissions.map((item) => item.id);
                      [ids[index - 1], ids[index]] = [
                        ids[index],
                        ids[index - 1],
                      ];
                      setActionId(submission.id);
                      void reorderThreadQueue({
                        threadId: threadId!,
                        queuedSubmissionIds: ids,
                      })
                        .then(() => {
                          setActionError(false);
                          queue.refresh?.();
                        })
                        .catch(() => setActionError(true))
                        .finally(() => setActionId(null));
                    }}
                  >
                    <ArrowUp size={13} />
                  </button>
                  <button
                    type="button"
                    className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
                    aria-label={text("agentChat.threadQueue.moveDown")}
                    title={text("agentChat.threadQueue.moveDown")}
                    disabled={busy || index === queue.submissions.length - 1}
                    onClick={() => {
                      const ids = queue.submissions.map((item) => item.id);
                      [ids[index], ids[index + 1]] = [
                        ids[index + 1],
                        ids[index],
                      ];
                      setActionId(submission.id);
                      void reorderThreadQueue({
                        threadId: threadId!,
                        queuedSubmissionIds: ids,
                      })
                        .then(() => {
                          setActionError(false);
                          queue.refresh?.();
                        })
                        .catch(() => setActionError(true))
                        .finally(() => setActionId(null));
                    }}
                  >
                    <ArrowDown size={13} />
                  </button>
                  <button
                    type="button"
                    className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-950/30"
                    aria-label={text("agentChat.threadQueue.delete")}
                    title={text("agentChat.threadQueue.delete")}
                    disabled={busy}
                    onClick={() => {
                      setActionId(submission.id);
                      void deleteThreadQueue({
                        threadId: threadId!,
                        queuedSubmissionId: submission.id,
                      })
                        .then(() => {
                          setActionError(false);
                          queue.refresh?.();
                        })
                        .catch(() => setActionError(true))
                        .finally(() => setActionId(null));
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                  <button
                    type="button"
                    className="inline-flex h-6 w-6 items-center justify-center rounded text-sky-600 hover:bg-sky-50 dark:hover:bg-sky-950/30"
                    aria-label={text("agentChat.threadQueue.sendNow")}
                    title={text("agentChat.threadQueue.sendNow")}
                    disabled={busy}
                    onClick={() => {
                      setActionId(submission.id);
                      void startThreadQueue({
                        threadId: threadId!,
                        queuedSubmissionId: submission.id,
                      })
                        .then(() => {
                          setActionError(false);
                          queue.refresh?.();
                        })
                        .catch(() => setActionError(true))
                        .finally(() => setActionId(null));
                    }}
                  >
                    <Play size={13} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
