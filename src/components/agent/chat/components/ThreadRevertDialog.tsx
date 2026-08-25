import { FileCheck2, Loader2, MessageSquareText } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ThreadRevertTarget } from "./ThreadRevertTrigger";

export type ThreadRevertStatus = "idle" | "submitting" | "success" | "error";

interface ThreadRevertDialogProps {
  target: ThreadRevertTarget | null;
  status: ThreadRevertStatus;
  onClose: () => void;
  onConfirm: () => void | Promise<void>;
}

export function ThreadRevertDialog({
  target,
  status,
  onClose,
  onConfirm,
}: ThreadRevertDialogProps) {
  const { t } = useTranslation("agent");
  const submitting = status === "submitting";

  return (
    <Dialog
      open={Boolean(target)}
      onOpenChange={(open) => {
        if (!open && !submitting) onClose();
      }}
    >
      <DialogContent
        maxWidth="max-w-[440px]"
        className="border border-slate-200 bg-white p-0"
      >
        <DialogHeader className="border-b border-slate-200 px-6 py-5 pr-12 text-left">
          <DialogTitle className="text-base leading-6 tracking-normal text-slate-950">
            {t("agentChat.messageList.threadRevert.title")}
          </DialogTitle>
          <DialogDescription className="leading-5 text-slate-600">
            {t("agentChat.messageList.threadRevert.description")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 px-6 py-5">
          <p className="text-sm leading-6 text-slate-700">
            {t("agentChat.messageList.threadRevert.historyEffect")}
          </p>
          <ul className="space-y-3 text-sm leading-5 text-slate-600">
            <li className="flex items-start gap-2.5">
              <MessageSquareText
                className="mt-0.5 h-4 w-4 shrink-0 text-slate-500"
                aria-hidden="true"
              />
              <span>{t("agentChat.messageList.threadRevert.keepThread")}</span>
            </li>
            <li className="flex items-start gap-2.5">
              <FileCheck2
                className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600"
                aria-hidden="true"
              />
              <span>{t("agentChat.messageList.threadRevert.keepFiles")}</span>
            </li>
          </ul>
          {status === "error" ? (
            <p
              className="text-sm leading-5 text-rose-700"
              data-testid="thread-revert-error"
              role="alert"
            >
              {t("agentChat.messageList.threadRevert.failed")}
            </p>
          ) : null}
        </div>

        <DialogFooter className="gap-2 border-t border-slate-200 px-6 py-4 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={submitting}
            onClick={onClose}
          >
            {t("agentChat.messageList.threadRevert.cancel")}
          </Button>
          <Button
            type="button"
            variant="destructive"
            size="sm"
            disabled={submitting}
            data-testid="thread-revert-confirm"
            onClick={() => void onConfirm()}
          >
            {submitting ? (
              <Loader2
                className="mr-2 h-4 w-4 animate-spin"
                aria-hidden="true"
              />
            ) : null}
            {t(
              submitting
                ? "agentChat.messageList.threadRevert.submitting"
                : "agentChat.messageList.threadRevert.confirm",
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
