import { useCallback, useLayoutEffect, useState } from "react";
import {
  buildClawAgentParams,
  buildHomeAgentParams,
} from "@/lib/workspace/navigation";
import type { AgentPageParams, Page, PageParams } from "@/types/page";

const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";

interface UseAppNavigationResult {
  currentPage: Page;
  pageParams: PageParams;
  requestedPage: Page;
  requestedPageParams: PageParams;
  navigationRequestId: number;
  isNavigating: boolean;
  handleNavigate: (page: Page, params?: PageParams) => void;
}

function normalizePageParams(params?: PageParams): PageParams {
  return params ? { ...params } : {};
}

type SerializableNavigationValue =
  | null
  | boolean
  | number
  | string
  | SerializableNavigationValue[]
  | { [key: string]: SerializableNavigationValue };

function normalizeNavigationParamValue(
  value: unknown,
  insideArray = false,
): SerializableNavigationValue | undefined {
  if (
    value === undefined ||
    typeof value === "function" ||
    typeof value === "symbol"
  ) {
    return insideArray ? null : undefined;
  }

  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }

  if (typeof value === "bigint") {
    throw new TypeError("BigInt values are not supported in navigation params");
  }

  if (Array.isArray(value)) {
    return value.map(
      (item) => normalizeNavigationParamValue(item, true) ?? null,
    );
  }

  if (typeof value === "object") {
    const toJson = (value as { toJSON?: unknown }).toJSON;
    if (typeof toJson === "function") {
      return normalizeNavigationParamValue(
        (toJson as () => unknown).call(value),
        insideArray,
      );
    }

    const record = value as Record<string, unknown>;
    const normalized: { [key: string]: SerializableNavigationValue } = {};
    for (const key of Object.keys(record).sort()) {
      const normalizedValue = normalizeNavigationParamValue(record[key]);
      if (normalizedValue !== undefined) {
        normalized[key] = normalizedValue;
      }
    }
    return normalized;
  }

  return insideArray ? null : undefined;
}

function serializePageParams(params: PageParams): string {
  return JSON.stringify(normalizeNavigationParamValue(params) ?? {});
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function pickRestorableAgentParams(params: unknown): AgentPageParams | null {
  if (!isRecord(params)) {
    return null;
  }

  const initialSessionId = readString(params.initialSessionId);
  if (!initialSessionId) {
    return null;
  }

  return buildClawAgentParams({
    initialSessionId,
    projectId: readString(params.projectId),
    contentId: readString(params.contentId),
    theme: readString(params.theme),
    lockTheme:
      typeof params.lockTheme === "boolean" ? params.lockTheme : undefined,
  });
}

function clearRestoredNavigationState(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.sessionStorage.removeItem(NAVIGATION_RESTORE_STORAGE_KEY);
  } catch {
    // sessionStorage 不可用时退化为普通首页导航。
  }
}

function readRestoredNavigationState(): {
  page: Page;
  params: PageParams;
} | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const raw = window.sessionStorage.getItem(NAVIGATION_RESTORE_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) {
      clearRestoredNavigationState();
      return null;
    }

    const page = parsed.page;
    if (page !== "agent") {
      clearRestoredNavigationState();
      return null;
    }

    const params = pickRestorableAgentParams(parsed.params);
    if (!params) {
      clearRestoredNavigationState();
      return null;
    }

    return {
      page,
      params,
    };
  } catch {
    clearRestoredNavigationState();
    return null;
  }
}

function persistRestorableNavigation(page: Page, params: PageParams): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const restorableParams =
      page === "agent" ? pickRestorableAgentParams(params) : null;
    if (!restorableParams) {
      clearRestoredNavigationState();
      return;
    }

    window.sessionStorage.setItem(
      NAVIGATION_RESTORE_STORAGE_KEY,
      JSON.stringify({
        page,
        params: restorableParams,
      }),
    );
  } catch {
    // sessionStorage 只是 reload 恢复优化，失败时不应阻断主导航。
  }
}

export function useAppNavigation(): UseAppNavigationResult {
  const [navigationState, setNavigationState] = useState<{
    currentPage: Page;
    currentPageParams: PageParams;
    currentPageParamsKey: string;
    requestedPage: Page;
    requestedPageParams: PageParams;
    requestedPageParamsKey: string;
    navigationRequestId: number;
  }>(() => {
    const restored = readRestoredNavigationState();
    if (restored) {
      const restoredPageParamsKey = serializePageParams(restored.params);
      return {
        currentPage: restored.page,
        currentPageParams: restored.params,
        currentPageParamsKey: restoredPageParamsKey,
        requestedPage: restored.page,
        requestedPageParams: restored.params,
        requestedPageParamsKey: restoredPageParamsKey,
        navigationRequestId: 0,
      };
    }

    const initialPageParams = buildHomeAgentParams();
    const initialPageParamsKey = serializePageParams(initialPageParams);
    return {
      currentPage: "agent",
      currentPageParams: initialPageParams,
      currentPageParamsKey: initialPageParamsKey,
      requestedPage: "agent",
      requestedPageParams: initialPageParams,
      requestedPageParamsKey: initialPageParamsKey,
      navigationRequestId: 0,
    };
  });

  const handleNavigate = useCallback((page: Page, params?: PageParams) => {
    const nextPageParams = normalizePageParams(params);
    const nextPageParamsKey = serializePageParams(nextPageParams);
    persistRestorableNavigation(page, nextPageParams);

    setNavigationState((current) => {
      if (
        current.requestedPage === page &&
        current.requestedPageParamsKey === nextPageParamsKey
      ) {
        return current;
      }

      return {
        ...current,
        requestedPage: page,
        requestedPageParams: nextPageParams,
        requestedPageParamsKey: nextPageParamsKey,
        navigationRequestId: current.navigationRequestId + 1,
      };
    });
  }, []);

  useLayoutEffect(() => {
    setNavigationState((current) => {
      if (
        current.currentPage === current.requestedPage &&
        current.currentPageParamsKey === current.requestedPageParamsKey
      ) {
        return current;
      }

      return {
        ...current,
        currentPage: current.requestedPage,
        currentPageParams: current.requestedPageParams,
        currentPageParamsKey: current.requestedPageParamsKey,
      };
    });
  }, [navigationState.requestedPage, navigationState.requestedPageParamsKey]);

  const isNavigating =
    navigationState.currentPage !== navigationState.requestedPage ||
    navigationState.currentPageParamsKey !==
      navigationState.requestedPageParamsKey;

  return {
    currentPage: navigationState.currentPage,
    pageParams: navigationState.currentPageParams,
    requestedPage: navigationState.requestedPage,
    requestedPageParams: navigationState.requestedPageParams,
    navigationRequestId: navigationState.navigationRequestId,
    isNavigating,
    handleNavigate,
  };
}
