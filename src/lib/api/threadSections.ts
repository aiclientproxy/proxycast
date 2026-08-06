import {
  AppServerClient,
  type AppServerThreadSection,
  type AppServerThreadSectionCreateParams,
  type AppServerThreadSectionDeleteParams,
  type AppServerThreadSectionListParams,
  type AppServerThreadSectionMoveParams,
  type AppServerThreadSectionUpdateParams,
} from "@/lib/api/appServer";
import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";

export const PINNED_THREAD_SECTION = Object.freeze({
  id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
  name: "Pinned",
});

const THREAD_SECTION_PAGE_LIMIT = 100;

export type ThreadSectionClient = Pick<
  AppServerClient,
  | "moveThreadToSection"
  | "listThreadSections"
  | "createThreadSection"
  | "updateThreadSection"
  | "deleteThreadSection"
>;

export function isPinnedThreadSession(
  session: Pick<AgentSessionInfo, "section">,
): boolean {
  return session.section?.id === PINNED_THREAD_SECTION.id;
}

export function createThreadSectionClient(
  appServerClient: ThreadSectionClient = new AppServerClient(),
) {
  async function listThreadSections(
    params: AppServerThreadSectionListParams = {},
  ): Promise<AppServerThreadSection[]> {
    const sections: AppServerThreadSection[] = [];
    const seenCursors = new Set<string>();
    let cursor = params.cursor?.trim() || undefined;
    const { cursor: _cursor, ...requestParams } = params;
    do {
      const response = await appServerClient.listThreadSections({
        ...requestParams,
        ...(cursor ? { cursor } : {}),
        limit: params.limit ?? THREAD_SECTION_PAGE_LIMIT,
      });
      sections.push(...response.result.data);
      const nextCursor = response.result.nextCursor?.trim() || undefined;
      if (!nextCursor || seenCursors.has(nextCursor)) {
        break;
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
    } while (cursor);
    return sections;
  }

  async function createThreadSection(
    params: AppServerThreadSectionCreateParams,
  ): Promise<AppServerThreadSection> {
    const name = requireNonEmpty(params.name, "section name");
    const response = await appServerClient.createThreadSection({ name });
    return response.result.section;
  }

  async function updateThreadSection(
    params: AppServerThreadSectionUpdateParams,
  ): Promise<AppServerThreadSection> {
    const sectionId = requireNonEmpty(params.sectionId, "section id");
    const name = requireNonEmpty(params.name, "section name");
    const response = await appServerClient.updateThreadSection({
      sectionId,
      name,
    });
    return response.result.section;
  }

  async function deleteThreadSection(
    params: AppServerThreadSectionDeleteParams,
  ): Promise<void> {
    const sectionId = requireNonEmpty(params.sectionId, "section id");
    await appServerClient.deleteThreadSection({ sectionId });
  }

  async function moveThreadToSection(
    params: AppServerThreadSectionMoveParams,
  ): Promise<void> {
    const threadId = params.threadId.trim();
    const sectionId =
      params.sectionId === null ? null : params.sectionId.trim();
    const beforeThreadId = params.beforeThreadId?.trim() || undefined;
    if (!threadId) {
      throw new Error("threadId is required to move a thread to a section");
    }
    if (sectionId !== null && !sectionId) {
      throw new Error("sectionId must be null or a non-empty section id");
    }

    await appServerClient.moveThreadToSection({
      threadId,
      sectionId,
      ...(beforeThreadId ? { beforeThreadId } : {}),
    });
  }

  return {
    createThreadSection,
    deleteThreadSection,
    listThreadSections,
    moveThreadToSection,
    updateThreadSection,
  };
}

function requireNonEmpty(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${field} is required`);
  }
  return normalized;
}

const threadSectionClient = createThreadSectionClient();

export const moveThreadToSection =
  threadSectionClient.moveThreadToSection.bind(threadSectionClient);
export const listThreadSections =
  threadSectionClient.listThreadSections.bind(threadSectionClient);
export const createThreadSection =
  threadSectionClient.createThreadSection.bind(threadSectionClient);
export const updateThreadSection =
  threadSectionClient.updateThreadSection.bind(threadSectionClient);
export const deleteThreadSection =
  threadSectionClient.deleteThreadSection.bind(threadSectionClient);
