import {
  METHOD_PROJECT_CREATE,
  METHOD_PROJECT_LIST,
  METHOD_THREAD_METADATA_UPDATE,
  METHOD_THREAD_READ,
  type Project,
  type ProjectCreateResponse,
  type ProjectListResponse,
  type ThreadMetadataUpdateResponse,
  type ThreadReadResponse,
} from "@limecloud/app-server-client";
import { AppServerClient } from "./appServer";

const PROJECT_PAGE_LIMIT = 100;

export type ProjectDirectoryAppServerClient = Pick<AppServerClient, "request">;

export interface ProjectDirectorySnapshot {
  projects: Project[];
  projectId: string | null;
}

export interface CreateProjectDirectoryRequest {
  name: string;
  rootPath: string;
  idempotencyKey: string;
}

export async function listProjectDirectory(
  appServerClient: ProjectDirectoryAppServerClient = new AppServerClient(),
): Promise<Project[]> {
  const projects: Project[] = [];
  const seenCursors = new Set<string>();
  let cursor: string | null = null;

  do {
    const response = await appServerClient.request<ProjectListResponse>(
      METHOD_PROJECT_LIST,
      {
        limit: PROJECT_PAGE_LIMIT,
        ...(cursor ? { cursor } : {}),
      },
    );
    const page = assertProjectListResponse(response.result);
    projects.push(...page.data);
    cursor = page.nextCursor ?? null;
    if (cursor) {
      if (seenCursors.has(cursor)) {
        throw new Error("project/list returned a repeated nextCursor");
      }
      seenCursors.add(cursor);
    }
  } while (cursor);

  return projects;
}

export async function readThreadProjectId(
  threadId: string,
  appServerClient: ProjectDirectoryAppServerClient = new AppServerClient(),
): Promise<string | null> {
  const normalizedThreadId = requireIdentifier(threadId, "threadId");
  const response = await appServerClient.request<ThreadReadResponse>(
    METHOD_THREAD_READ,
    { threadId: normalizedThreadId, includeTurns: false },
  );
  const thread = response.result?.thread;
  if (!thread || thread.id !== normalizedThreadId) {
    throw new Error("thread/read returned an invalid thread");
  }
  return thread.projectId?.trim() || null;
}

export async function readProjectDirectorySnapshot(
  threadId: string,
  appServerClient: ProjectDirectoryAppServerClient = new AppServerClient(),
): Promise<ProjectDirectorySnapshot> {
  const [projects, projectId] = await Promise.all([
    listProjectDirectory(appServerClient),
    readThreadProjectId(threadId, appServerClient),
  ]);
  return { projects, projectId };
}

export async function assignThreadProject(
  threadId: string,
  projectId: string | null,
  appServerClient: ProjectDirectoryAppServerClient = new AppServerClient(),
): Promise<string | null> {
  const normalizedThreadId = requireIdentifier(threadId, "threadId");
  const normalizedProjectId = projectId?.trim() || "";
  const response = await appServerClient.request<ThreadMetadataUpdateResponse>(
    METHOD_THREAD_METADATA_UPDATE,
    {
      threadId: normalizedThreadId,
      // The v2 contract uses an empty string for the explicit clear operation.
      projectId: normalizedProjectId,
    },
  );
  const thread = response.result?.thread;
  if (!thread || thread.id !== normalizedThreadId) {
    throw new Error("thread/metadata/update returned an invalid thread");
  }
  return thread.projectId?.trim() || null;
}

export async function createProjectDirectoryEntry(
  request: CreateProjectDirectoryRequest,
  appServerClient: ProjectDirectoryAppServerClient = new AppServerClient(),
): Promise<Project> {
  const name = requireIdentifier(request.name, "project name");
  const rootPath = requireIdentifier(request.rootPath, "project root path");
  const idempotencyKey = requireIdentifier(
    request.idempotencyKey,
    "idempotencyKey",
  );
  const response = await appServerClient.request<ProjectCreateResponse>(
    METHOD_PROJECT_CREATE,
    {
      name,
      roots: [{ path: rootPath }],
      idempotencyKey,
    },
  );
  const project = response.result?.project;
  if (!project || typeof project.id !== "string" || !project.name) {
    throw new Error("project/create returned an invalid project");
  }
  return project;
}

function assertProjectListResponse(
  value: ProjectListResponse,
): ProjectListResponse {
  if (!value || !Array.isArray(value.data)) {
    throw new Error("project/list returned an invalid page");
  }
  if (
    value.nextCursor !== undefined &&
    value.nextCursor !== null &&
    typeof value.nextCursor !== "string"
  ) {
    throw new Error("project/list returned an invalid nextCursor");
  }
  for (const project of value.data) {
    if (
      !project ||
      typeof project.id !== "string" ||
      !project.id.trim() ||
      typeof project.name !== "string" ||
      !project.name.trim() ||
      !Array.isArray(project.roots) ||
      project.roots.some(
        (root) => !root || typeof root.path !== "string" || !root.path.trim(),
      )
    ) {
      throw new Error("project/list returned an invalid project");
    }
  }
  return value;
}

function requireIdentifier(value: string, label: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${label} is required`);
  }
  return normalized;
}
