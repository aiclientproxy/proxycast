export const CORE_NAMESPACES = [
  "common",
  "navigation",
  "settings",
  "workspace",
  "agent",
  "errors",
] as const;

export type LimeNamespace = (typeof CORE_NAMESPACES)[number];

export const BUNDLED_NAMESPACE_RESOURCE_PARTS = {
  agent: [
    "agent",
    "agentHome",
    "agentInputbar",
    "agentMessageList",
    "agentRuntime",
    "agentSkills",
    "agentExperts",
    "agentTeamWorkspace",
  ],
  workspace: ["workspace", "scheduledTasks"],
} as const satisfies Partial<Record<LimeNamespace, readonly string[]>>;

function hasResourceParts(
  namespace: LimeNamespace,
): namespace is keyof typeof BUNDLED_NAMESPACE_RESOURCE_PARTS {
  return namespace in BUNDLED_NAMESPACE_RESOURCE_PARTS;
}

export function getBundledNamespaceResourceParts(
  namespace: LimeNamespace,
): readonly string[] {
  return hasResourceParts(namespace)
    ? BUNDLED_NAMESPACE_RESOURCE_PARTS[namespace]
    : [namespace];
}
