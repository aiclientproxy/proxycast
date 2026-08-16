import type { ComponentProps } from "react";
import { AgentThreadReliabilityPanel } from "./AgentThreadReliabilityPanel";
import { HarnessActivitySections } from "./HarnessActivitySections";
import { HarnessActiveWritesSection } from "./HarnessActiveWritesSection";
import { ThreadGoalPanel } from "./ThreadGoalPanel";
import { HarnessFileReviewSection } from "./HarnessFileReviewSection";
import { HarnessOutputSignalsSection } from "./HarnessOutputSignalsSection";
import {
  HarnessAgentUiProjectionSection,
  HarnessRuntimeFactsSection,
  HarnessRuntimeTaskSection,
} from "./HarnessRuntimeOverviewSections";
import { HarnessToolInventorySection } from "./HarnessToolInventorySection";
import {
  HarnessStatusSection as Section,
  type HarnessSectionKey,
} from "./HarnessStatusSectionFrame";
import { agentText } from "./harnessPanelText";

interface HarnessObjectiveSectionModel {
  title: string;
  badge: string;
  panelProps: ComponentProps<typeof ThreadGoalPanel>;
}

interface HarnessReliabilitySectionModel {
  shouldRender: boolean;
  statusLabel: string;
  panelProps: ComponentProps<typeof AgentThreadReliabilityPanel>;
}

interface HarnessFileReviewSectionModel {
  title: string;
  badge: string;
  props: ComponentProps<typeof HarnessFileReviewSection>;
}

interface HarnessActiveWritesSectionModel {
  count: number;
  props: ComponentProps<typeof HarnessActiveWritesSection>;
}

interface HarnessOutputSignalsSectionModel {
  totalCount: number;
  filteredCount: number;
  props: ComponentProps<typeof HarnessOutputSignalsSection>;
}

export interface HarnessStatusPanelSectionsProps {
  registerSectionRef: (
    key: HarnessSectionKey,
    node: HTMLElement | null,
  ) => void;
  runtimeTaskSectionProps: ComponentProps<
    typeof HarnessRuntimeTaskSection
  > | null;
  objectiveSection: HarnessObjectiveSectionModel | null;
  reliabilitySection: HarnessReliabilitySectionModel;
  fileReviewSection: HarnessFileReviewSectionModel | null;
  agentUiProjectionSectionProps: ComponentProps<
    typeof HarnessAgentUiProjectionSection
  > | null;
  runtimeFactsSectionProps: ComponentProps<
    typeof HarnessRuntimeFactsSection
  > | null;
  activeWritesSection: HarnessActiveWritesSectionModel | null;
  outputSignalsSection: HarnessOutputSignalsSectionModel | null;
  toolInventorySectionProps: ComponentProps<typeof HarnessToolInventorySection>;
  activitySectionsProps: ComponentProps<typeof HarnessActivitySections>;
}

export function HarnessStatusPanelSections({
  registerSectionRef,
  runtimeTaskSectionProps,
  objectiveSection,
  reliabilitySection,
  fileReviewSection,
  agentUiProjectionSectionProps,
  runtimeFactsSectionProps,
  activeWritesSection,
  outputSignalsSection,
  toolInventorySectionProps,
  activitySectionsProps,
}: HarnessStatusPanelSectionsProps) {
  return (
    <>
      {runtimeTaskSectionProps ? (
        <HarnessRuntimeTaskSection {...runtimeTaskSectionProps} />
      ) : null}

      {objectiveSection ? (
        <Section
          sectionKey="objective"
          title={objectiveSection.title}
          badge={objectiveSection.badge}
          registerRef={registerSectionRef}
        >
          <ThreadGoalPanel {...objectiveSection.panelProps} />
        </Section>
      ) : null}

      {reliabilitySection.shouldRender ? (
        <Section
          sectionKey="reliability"
          title={agentText(
            "agentChat.harness.generated.8f2d0db713",
            "线程可靠性",
          )}
          badge={reliabilitySection.statusLabel}
          registerRef={registerSectionRef}
        >
          <AgentThreadReliabilityPanel
            className="mb-0 border-border bg-background shadow-none"
            {...reliabilitySection.panelProps}
          />
        </Section>
      ) : null}

      {fileReviewSection ? (
        <Section
          sectionKey="file_review"
          title={fileReviewSection.title}
          badge={fileReviewSection.badge}
          registerRef={registerSectionRef}
        >
          <HarnessFileReviewSection {...fileReviewSection.props} />
        </Section>
      ) : null}

      {agentUiProjectionSectionProps ? (
        <HarnessAgentUiProjectionSection {...agentUiProjectionSectionProps} />
      ) : null}

      {runtimeFactsSectionProps ? (
        <HarnessRuntimeFactsSection {...runtimeFactsSectionProps} />
      ) : null}

      {activeWritesSection ? (
        <Section
          sectionKey="writes"
          title={agentText(
            "agentChat.harness.generated.e36ba9e753",
            "当前文件写入",
          )}
          badge={`${activeWritesSection.count} 条`}
          registerRef={registerSectionRef}
        >
          <HarnessActiveWritesSection {...activeWritesSection.props} />
        </Section>
      ) : null}

      {outputSignalsSection ? (
        <Section
          sectionKey="outputs"
          title={agentText(
            "agentChat.harness.generated.fb7edd231f",
            "工具输出",
          )}
          badge={
            outputSignalsSection.filteredCount ===
            outputSignalsSection.totalCount
              ? `${outputSignalsSection.totalCount} 条`
              : `${outputSignalsSection.filteredCount} / ${outputSignalsSection.totalCount} 条`
          }
          registerRef={registerSectionRef}
        >
          <HarnessOutputSignalsSection {...outputSignalsSection.props} />
        </Section>
      ) : null}

      <HarnessToolInventorySection {...toolInventorySectionProps} />
      <HarnessActivitySections {...activitySectionsProps} />
    </>
  );
}
