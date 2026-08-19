import { useCallback, useEffect, useMemo, useState } from "react";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { useWorkspaceSkillBindingsRuntime } from "./useWorkspaceSkillBindingsRuntime";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readFirstString(
  candidates: Array<Record<string, unknown> | null>,
  keys: readonly string[],
): string | undefined {
  for (const candidate of candidates) {
    for (const key of keys) {
      const value = candidate?.[key];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return undefined;
}

export interface ExpertWorkspaceSkillRuntimeEnableInput {
  workspaceRoot?: string | null;
  bindings: AgentRuntimeWorkspaceSkillBinding[];
}

interface UseExpertWorkspaceSkillRuntimeParams {
  activeTheme: string;
  requestMetadata: unknown;
  workspaceRoot?: string | null;
  deferredDelayMs?: number;
  onOpenSkillsManage?: () => void;
}

export function workspaceSkillDirectoryFromRef(ref: string): string {
  const raw = ref.trim().replace(/^workspace_skill:/i, "");
  return raw.split("@")[0]?.trim() || raw;
}

function normalizeWorkspaceSkillBindingKey(value: string): string {
  return value.trim().toLowerCase();
}

export function findWorkspaceSkillBindingByRef(
  bindings: readonly AgentRuntimeWorkspaceSkillBinding[],
  ref: string,
): AgentRuntimeWorkspaceSkillBinding | null {
  const directory = workspaceSkillDirectoryFromRef(ref);
  if (!directory) {
    return null;
  }

  const normalized = normalizeWorkspaceSkillBindingKey(directory);
  return (
    bindings.find((binding) =>
      [
        binding.key,
        binding.directory,
        binding.registered_skill_directory,
        binding.name,
      ]
        .filter((value): value is string => Boolean(value))
        .some((value) => {
          const normalizedValue = normalizeWorkspaceSkillBindingKey(value);
          return (
            normalizedValue === normalized ||
            normalizedValue === `workspace_skill:${normalized}` ||
            normalizedValue.endsWith(`/${normalized}`)
          );
        }),
    ) ?? null
  );
}

export function resolveExpertWorkspaceSkillRuntimeKey(
  requestMetadata: unknown,
): string {
  const metadata = asRecord(requestMetadata);
  const harness = asRecord(metadata?.harness);
  const expert = asRecord(metadata?.expert);
  const harnessExpert = asRecord(harness?.expert);
  const id =
    readFirstString([expert, harnessExpert], ["id", "expertId", "expert_id"]) ||
    "";
  const release =
    readFirstString(
      [expert, harnessExpert],
      ["releaseId", "release_id", "version"],
    ) || "";

  return expert || harnessExpert ? `${id}:${release}` : "";
}

export function filterExpertWorkspaceSkillRuntimeEnableRefsForSkillRefs(
  currentRefs: readonly string[],
  skillRefs: readonly string[],
): string[] {
  const nextWorkspaceSkillDirectories = new Set(
    skillRefs
      .filter((ref) => /^workspace_skill:/i.test(ref.trim()))
      .map(workspaceSkillDirectoryFromRef),
  );

  return currentRefs.filter((ref) =>
    nextWorkspaceSkillDirectories.has(workspaceSkillDirectoryFromRef(ref)),
  );
}

export function appendExpertWorkspaceSkillRuntimeEnableRef(
  currentRefs: readonly string[],
  ref: string,
): string[] {
  const directory = workspaceSkillDirectoryFromRef(ref);
  if (
    currentRefs.some(
      (item) => workspaceSkillDirectoryFromRef(item) === directory,
    )
  ) {
    return [...currentRefs];
  }

  return [...currentRefs, ref];
}

export function resolveExpertWorkspaceSkillRuntimeEnableBindings(
  bindings: readonly AgentRuntimeWorkspaceSkillBinding[],
  refs: readonly string[],
): AgentRuntimeWorkspaceSkillBinding[] {
  const enabled = new Map<string, AgentRuntimeWorkspaceSkillBinding>();
  for (const ref of refs) {
    const binding = findWorkspaceSkillBindingByRef(bindings, ref);
    if (binding?.binding_status !== "ready_for_manual_enable") {
      continue;
    }
    enabled.set(binding.key || binding.directory, binding);
  }

  return [...enabled.values()];
}

export function useExpertWorkspaceSkillRuntime({
  activeTheme,
  requestMetadata,
  workspaceRoot,
  deferredDelayMs,
  onOpenSkillsManage,
}: UseExpertWorkspaceSkillRuntimeParams) {
  const runtimeKey = useMemo(
    () => resolveExpertWorkspaceSkillRuntimeKey(requestMetadata),
    [requestMetadata],
  );
  const bindingsRuntime = useWorkspaceSkillBindingsRuntime({
    enabled:
      activeTheme === "general" &&
      Boolean(runtimeKey) &&
      Boolean(workspaceRoot?.trim()),
    workspaceRoot,
    deferredDelayMs,
  });
  const [enabledRefs, setEnabledRefs] = useState<string[]>([]);

  useEffect(() => {
    setEnabledRefs((current) => (current.length === 0 ? current : []));
  }, [runtimeKey, workspaceRoot]);

  const pruneEnabledRefsForSkillRefs = useCallback((skillRefs: string[]) => {
    setEnabledRefs((current) =>
      filterExpertWorkspaceSkillRuntimeEnableRefsForSkillRefs(
        current,
        skillRefs,
      ),
    );
  }, []);

  const enabledBindings = useMemo(
    () =>
      resolveExpertWorkspaceSkillRuntimeEnableBindings(
        bindingsRuntime.bindings,
        enabledRefs,
      ),
    [bindingsRuntime.bindings, enabledRefs],
  );
  const enableInput = useMemo<ExpertWorkspaceSkillRuntimeEnableInput | null>(
    () =>
      enabledBindings.length > 0
        ? {
            workspaceRoot,
            bindings: enabledBindings,
          }
        : null,
    [enabledBindings, workspaceRoot],
  );
  const handleEnableWorkspaceSkillRuntime = useCallback(
    (ref: string) => {
      const binding = findWorkspaceSkillBindingByRef(
        bindingsRuntime.bindings,
        ref,
      );
      if (binding?.binding_status !== "ready_for_manual_enable") {
        onOpenSkillsManage?.();
        return;
      }

      setEnabledRefs((current) =>
        appendExpertWorkspaceSkillRuntimeEnableRef(current, ref),
      );
    },
    [bindingsRuntime.bindings, onOpenSkillsManage],
  );

  return {
    runtimeKey,
    bindingsRuntime,
    enabledRefs,
    enabledBindings,
    enableInput,
    handleEnableWorkspaceSkillRuntime,
    pruneEnabledRefsForSkillRefs,
  };
}
