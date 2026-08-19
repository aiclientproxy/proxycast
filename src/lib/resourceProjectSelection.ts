const RESOURCE_PROJECT_STORAGE_KEY = "lime-resource-project-id";
const LEGACY_IMAGE_GEN_PROJECT_KEY = "image-gen-target-project-id";
const RESOURCE_PROJECT_CHANGE_EVENT = "lime:resource-project-change";

export type ResourceProjectChangeSource =
  | "resources"
  | "general-chat"
  | "image-gen-save"
  | "unknown";

export interface ResourceProjectChangeDetail {
  projectId: string | null;
  source: ResourceProjectChangeSource;
}

interface SetStoredResourceProjectIdOptions {
  source?: ResourceProjectChangeSource;
  syncLegacy?: boolean;
  emitEvent?: boolean;
}

interface GetStoredResourceProjectIdOptions {
  includeLegacy?: boolean;
}

const normalizeProjectId = (
  value: string | null | undefined,
): string | null => {
  if (!value) return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const hasWindow = (): boolean => {
  return typeof window !== "undefined";
};

export const getStoredResourceProjectId = (
  options?: GetStoredResourceProjectIdOptions,
): string | null => {
  if (!hasWindow()) {
    return null;
  }

  const normalizedPrimary = normalizeProjectId(
    localStorage.getItem(RESOURCE_PROJECT_STORAGE_KEY),
  );
  if (normalizedPrimary) {
    return normalizedPrimary;
  }

  if (!options?.includeLegacy) {
    return null;
  }

  return normalizeProjectId(localStorage.getItem(LEGACY_IMAGE_GEN_PROJECT_KEY));
};

export const setStoredResourceProjectId = (
  projectId: string | null | undefined,
  options?: SetStoredResourceProjectIdOptions,
): void => {
  if (!hasWindow()) {
    return;
  }

  const normalizedProjectId = normalizeProjectId(projectId);

  if (normalizedProjectId) {
    localStorage.setItem(RESOURCE_PROJECT_STORAGE_KEY, normalizedProjectId);
  } else {
    localStorage.removeItem(RESOURCE_PROJECT_STORAGE_KEY);
  }

  if (options?.syncLegacy) {
    if (normalizedProjectId) {
      localStorage.setItem(LEGACY_IMAGE_GEN_PROJECT_KEY, normalizedProjectId);
    } else {
      localStorage.removeItem(LEGACY_IMAGE_GEN_PROJECT_KEY);
    }
  }

  if (options?.emitEvent === false) {
    return;
  }

  window.dispatchEvent(
    new CustomEvent<ResourceProjectChangeDetail>(
      RESOURCE_PROJECT_CHANGE_EVENT,
      {
        detail: {
          projectId: normalizedProjectId,
          source: options?.source ?? "unknown",
        },
      },
    ),
  );
};

export const onResourceProjectChange = (
  listener: (detail: ResourceProjectChangeDetail) => void,
): (() => void) => {
  if (!hasWindow()) {
    return () => undefined;
  }

  const handler = (event: Event) => {
    if (!(event instanceof CustomEvent)) {
      return;
    }

    const detail = (event as CustomEvent<ResourceProjectChangeDetail>).detail;
    listener({
      projectId: normalizeProjectId(detail?.projectId),
      source: detail?.source ?? "unknown",
    });
  };

  const eventHandler: (event: Event) => void = handler;
  window.addEventListener(RESOURCE_PROJECT_CHANGE_EVENT, eventHandler);
  return () => {
    window.removeEventListener(RESOURCE_PROJECT_CHANGE_EVENT, eventHandler);
  };
};
