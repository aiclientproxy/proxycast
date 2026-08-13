import { ScheduledTasksPage } from "@/components/scheduled-tasks/ScheduledTasksPage";
import type { AutomationPageParams, Page, PageParams } from "@/types/page";

interface AutomationPageProps {
  onNavigate?: (page: Page, params?: PageParams) => void;
  pageParams?: AutomationPageParams;
}

export function AutomationPage({
  onNavigate,
  pageParams,
}: AutomationPageProps) {
  return <ScheduledTasksPage onNavigate={onNavigate} pageParams={pageParams} />;
}
