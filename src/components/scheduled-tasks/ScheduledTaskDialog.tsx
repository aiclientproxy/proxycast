import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import {
  scheduledTasksApi,
  type ScheduledTaskSchedule,
} from "@/lib/api/scheduledTasks";
import { ScheduledTaskEditor } from "./ScheduledTaskEditor";
import {
  validateScheduledTaskForm,
  type ScheduledTaskFormErrors,
  type ScheduledTaskFormState,
} from "./scheduledTaskViewModel";

interface ScheduledTaskDialogProps {
  open: boolean;
  initialForm: ScheduledTaskFormState | null;
  saving: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (form: ScheduledTaskFormState) => Promise<void>;
}

export function ScheduledTaskDialog({
  open,
  initialForm,
  saving,
  onOpenChange,
  onSubmit,
}: ScheduledTaskDialogProps) {
  const { i18n, t } = useTranslation("workspace");
  const [form, setForm] = useState<ScheduledTaskFormState | null>(initialForm);
  const [errors, setErrors] = useState<ScheduledTaskFormErrors>({});
  const [preview, setPreview] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);

  useEffect(() => {
    if (!open) {
      return;
    }
    setForm(initialForm);
    setErrors({});
    setPreview([]);
  }, [initialForm, open]);

  const handlePreview = async (schedule: ScheduledTaskSchedule) => {
    setPreviewLoading(true);
    try {
      const response = await scheduledTasksApi.previewSchedule(schedule);
      setPreview(response.nextRunAt);
    } catch (error) {
      toast.error(
        t("scheduledTasks.error.preview", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSave = async () => {
    if (!form) {
      return;
    }
    const nextErrors = validateScheduledTaskForm(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      toast.error(t("scheduledTasks.editor.validation.fix"));
      return;
    }
    await onSubmit(form);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        maxWidth="max-w-[820px]"
        className="lime-workbench-theme-scope h-[min(860px,calc(100vh-32px))] overflow-hidden rounded-lg border border-slate-200 bg-white p-0"
      >
        {form ? (
          <ScheduledTaskEditor
            mode="create"
            form={form}
            errors={errors}
            preview={preview}
            previewLoading={previewLoading}
            saving={saving}
            locale={i18n.language}
            t={t}
            onChange={setForm}
            onPreview={(schedule) => void handlePreview(schedule)}
            onSave={() => void handleSave()}
            onCancel={() => onOpenChange(false)}
          />
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
