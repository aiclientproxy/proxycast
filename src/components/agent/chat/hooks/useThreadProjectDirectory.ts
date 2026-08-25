import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusSubscription,
} from "@/lib/api/appServerEventBus";
import {
  assignThreadProject,
  createProjectDirectoryEntry,
  readProjectDirectorySnapshot,
  type CreateProjectDirectoryRequest,
  type ProjectDirectorySnapshot,
} from "@/lib/api/projectDirectory";
import type { Project } from "@limecloud/app-server-client";

type ProjectDirectoryReader = (
  threadId: string,
) => Promise<ProjectDirectorySnapshot>;
type ProjectDirectoryAssigner = (
  threadId: string,
  projectId: string | null,
) => Promise<string | null>;
type ProjectDirectoryCreator = (
  request: CreateProjectDirectoryRequest,
) => Promise<Project>;
type ProjectNotificationSubscriber = (
  subscription: AppServerEventBusSubscription,
) => () => void;

export interface ThreadProjectDirectoryState {
  error: unknown;
  loading: boolean;
  mutating: boolean;
  projectId: string | null;
  projects: Project[];
}

export interface UseThreadProjectDirectoryOptions {
  createProject?: ProjectDirectoryCreator;
  readDirectory?: ProjectDirectoryReader;
  assignProject?: ProjectDirectoryAssigner;
  subscribeNotifications?: ProjectNotificationSubscriber;
  threadId?: string | null;
}

export function isThreadProjectDirectoryNotification(
  notification: AppServerJsonRpcNotification,
  threadId: string,
): boolean {
  const normalizedThreadId = threadId.trim();
  if (!normalizedThreadId) {
    return false;
  }
  if (notification.method === "project/changed") {
    return true;
  }
  if (notification.method !== "thread/project/updated") {
    return false;
  }
  const params = readRecord(notification.params);
  return params?.threadId === normalizedThreadId;
}

export function useThreadProjectDirectory(
  options: UseThreadProjectDirectoryOptions = {},
): ThreadProjectDirectoryState & {
  assign: (projectId: string | null) => Promise<void>;
  createAndAssign: (
    request: Omit<CreateProjectDirectoryRequest, "idempotencyKey">,
  ) => Promise<void>;
  refresh: () => Promise<void>;
} {
  const threadId = options.threadId?.trim() || null;
  const readDirectory = useMemo(
    () => options.readDirectory ?? readProjectDirectorySnapshot,
    [options.readDirectory],
  );
  const assignProject = useMemo(
    () => options.assignProject ?? assignThreadProject,
    [options.assignProject],
  );
  const createProject = useMemo(
    () => options.createProject ?? createProjectDirectoryEntry,
    [options.createProject],
  );
  const subscribeNotifications = useMemo(
    () => options.subscribeNotifications ?? subscribeAppServerNotifications,
    [options.subscribeNotifications],
  );
  const [state, setState] = useState<ThreadProjectDirectoryState>({
    error: null,
    loading: false,
    mutating: false,
    projectId: null,
    projects: [],
  });
  const readRevisionRef = useRef(0);

  const refresh = useCallback(async () => {
    if (!threadId) {
      return;
    }
    const revision = ++readRevisionRef.current;
    setState((current) => ({ ...current, error: null, loading: true }));
    try {
      const snapshot = await readDirectory(threadId);
      if (revision !== readRevisionRef.current) {
        return;
      }
      setState((current) => ({
        ...current,
        error: null,
        loading: false,
        projectId: snapshot.projectId,
        projects: snapshot.projects,
      }));
    } catch (error) {
      if (revision !== readRevisionRef.current) {
        return;
      }
      setState((current) => ({ ...current, error, loading: false }));
    }
  }, [readDirectory, threadId]);

  const assign = useCallback(
    async (projectId: string | null) => {
      if (!threadId) {
        throw new Error("threadId is required to assign a Project");
      }
      setState((current) => ({ ...current, error: null, mutating: true }));
      try {
        await assignProject(threadId, projectId);
        await refresh();
      } catch (error) {
        setState((current) => ({ ...current, error }));
        throw error;
      } finally {
        setState((current) => ({ ...current, mutating: false }));
      }
    },
    [assignProject, refresh, threadId],
  );

  const createAndAssign = useCallback(
    async (request: Omit<CreateProjectDirectoryRequest, "idempotencyKey">) => {
      if (!threadId) {
        throw new Error("threadId is required to create a Project");
      }
      const existing = state.projects.find((project) =>
        project.roots.some((root) => root.path === request.rootPath),
      );
      if (existing) {
        await assign(existing.id);
        return;
      }
      setState((current) => ({ ...current, error: null, mutating: true }));
      try {
        const project = await createProject({
          ...request,
          idempotencyKey: stableProjectIdempotencyKey(request.rootPath),
        });
        await assignProject(threadId, project.id);
        await refresh();
      } catch (error) {
        setState((current) => ({ ...current, error }));
        throw error;
      } finally {
        setState((current) => ({ ...current, mutating: false }));
      }
    },
    [assign, assignProject, createProject, refresh, state.projects, threadId],
  );

  useEffect(() => {
    readRevisionRef.current += 1;
    if (!threadId) {
      setState({
        error: null,
        loading: false,
        mutating: false,
        projectId: null,
        projects: [],
      });
      return;
    }
    void refresh();
    return () => {
      readRevisionRef.current += 1;
    };
  }, [refresh, threadId]);

  useEffect(() => {
    if (!threadId) {
      return;
    }
    let active = true;
    const unsubscribe = subscribeNotifications({
      onNotifications: (notifications) => {
        if (
          active &&
          notifications.some((notification) =>
            isThreadProjectDirectoryNotification(notification, threadId),
          )
        ) {
          void refresh();
        }
      },
      onError: (error) => {
        if (active) {
          setState((current) => ({ ...current, error }));
        }
      },
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [refresh, subscribeNotifications, threadId]);

  return { ...state, assign, createAndAssign, refresh };
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stableProjectIdempotencyKey(rootPath: string): string {
  let hash = 2_166_136_261;
  for (const character of rootPath) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16_777_619);
  }
  return `project-directory-${(hash >>> 0).toString(16)}`;
}
