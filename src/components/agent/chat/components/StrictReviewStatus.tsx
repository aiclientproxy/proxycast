import { Loader2, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { StrictReviewStatus as StrictReviewStatusValue } from "../hooks/useStrictReviewStatus";

export function StrictReviewStatus({
  status,
}: {
  status: StrictReviewStatusValue;
}) {
  const { i18n, t } = useTranslation("agent");
  const startedAt = new Date(status.startedAtMs).toLocaleTimeString(
    i18n.resolvedLanguage || i18n.language,
    { hour: "2-digit", minute: "2-digit" },
  );

  return (
    <section
      className="flex min-w-0 items-start gap-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-amber-950 shadow-sm shadow-slate-950/5 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100"
      data-testid="strict-review-status"
      data-protocol-method="autoApprovalReview/strictReviewRequired"
      data-thread-id={status.threadId}
      data-turn-id={status.turnId}
      data-started-at-ms={status.startedAtMs}
      aria-live="polite"
    >
      <span className="relative mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-amber-100 text-amber-700 dark:bg-amber-900/60 dark:text-amber-200">
        <ShieldAlert className="h-4 w-4" aria-hidden="true" />
        <Loader2
          className="absolute -bottom-1 -right-1 h-3.5 w-3.5 animate-spin rounded-full bg-amber-50 p-0.5 dark:bg-amber-950"
          aria-hidden="true"
        />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <h2 className="text-sm font-semibold">
            {t("agentChat.strictReview.title")}
          </h2>
          <span className="text-xs text-amber-700 dark:text-amber-300">
            {t("agentChat.strictReview.startedAt", { time: startedAt })}
          </span>
        </div>
        <p className="mt-1 text-xs leading-5 text-amber-800 dark:text-amber-200">
          {t("agentChat.strictReview.description")}
        </p>
        <p className="mt-0.5 text-xs font-medium leading-5 text-amber-900 dark:text-amber-100">
          {t("agentChat.strictReview.nextStep")}
        </p>
      </div>
    </section>
  );
}
