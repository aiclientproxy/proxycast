import React from "react";
import { Braces, Globe, Image as ImageIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type { ModelProviderCapabilities } from "@/lib/api/modelRegistry";

interface ModelProviderCapabilityBadgesProps {
  capabilities: ModelProviderCapabilities;
  className?: string;
  compact?: boolean;
}

function CapabilityBadge({
  active,
  icon,
  label,
  compact,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  compact: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-medium",
        compact ? "text-[10px] leading-4" : "text-[11px] leading-4",
        active
          ? "border-emerald-200/90 bg-emerald-50/90 text-emerald-700"
          : "border-slate-200/80 bg-slate-100/80 text-slate-500",
      )}
    >
      {icon}
      <span>{label}</span>
    </span>
  );
}

export const ModelProviderCapabilityBadges: React.FC<
  ModelProviderCapabilityBadgesProps
> = ({ capabilities, className, compact = false }) => {
  const { t } = useTranslation("common");
  const iconClassName = compact ? "h-3 w-3" : "h-3.5 w-3.5";

  return (
    <div
      className={cn("flex flex-wrap items-center gap-1.5", className)}
      data-testid="model-provider-capabilities"
    >
      <CapabilityBadge
        active={capabilities.namespaceTools}
        icon={<Braces className={iconClassName} />}
        label={t(
          capabilities.namespaceTools
            ? "common.modelProviderCapabilities.namespaceTools.active"
            : "common.modelProviderCapabilities.namespaceTools.inactive",
        )}
        compact={compact}
      />
      <CapabilityBadge
        active={capabilities.imageGeneration}
        icon={<ImageIcon className={iconClassName} />}
        label={t(
          capabilities.imageGeneration
            ? "common.modelProviderCapabilities.imageGeneration.active"
            : "common.modelProviderCapabilities.imageGeneration.inactive",
        )}
        compact={compact}
      />
      <CapabilityBadge
        active={capabilities.webSearch}
        icon={<Globe className={iconClassName} />}
        label={t(
          capabilities.webSearch
            ? "common.modelProviderCapabilities.webSearch.active"
            : "common.modelProviderCapabilities.webSearch.inactive",
        )}
        compact={compact}
      />
    </div>
  );
};
