import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_MEMORY_STORE_ADD_NOTE,
  METHOD_MEMORY_STORE_CONSOLIDATE,
  METHOD_MEMORY_STORE_HEALTH,
  METHOD_MEMORY_STORE_INDEX_REBUILD,
  METHOD_MEMORY_STORE_LIST,
  METHOD_MEMORY_STORE_READ,
  METHOD_MEMORY_STORE_REVIEW_LIST,
  METHOD_MEMORY_STORE_REVIEW_RESOLVE,
  METHOD_MEMORY_RESET,
  METHOD_MEMORY_STORE_SEARCH,
  type MemoryResetResponse,
  type MemoryStoreAddNoteParams,
  type MemoryStoreAddNoteResponse,
  type MemoryStoreConsolidateParams,
  type MemoryStoreConsolidateResponse,
  type MemoryStoreCitation,
  type MemoryStoreEntry,
  type MemoryStoreHealthResponse,
  type MemoryStoreIndexRebuildResponse,
  type MemoryStoreListParams,
  type MemoryStoreListResponse,
  type MemoryStoreReadParams,
  type MemoryStoreReadResponse,
  type MemoryStoreReviewListParams,
  type MemoryStoreReviewListResponse,
  type MemoryStoreReviewNote,
  type MemoryStoreReviewResolveParams,
  type MemoryStoreReviewResolveResponse,
  type MemoryStoreRootParams,
  type MemoryStoreScope,
  type MemoryStoreSearchHit,
  type MemoryStoreSearchMatchMode,
  type MemoryStoreSearchParams,
  type MemoryStoreSearchResponse,
} from "../../../packages/app-server-client/src/protocol";

export type {
  MemoryStoreAddNoteParams,
  MemoryStoreAddNoteResponse,
  MemoryStoreConsolidateParams,
  MemoryStoreConsolidateResponse,
  MemoryStoreCitation,
  MemoryStoreEntry,
  MemoryStoreHealthResponse,
  MemoryStoreIndexRebuildResponse,
  MemoryStoreListParams,
  MemoryStoreListResponse,
  MemoryStoreReadParams,
  MemoryStoreReadResponse,
  MemoryStoreReviewListParams,
  MemoryStoreReviewListResponse,
  MemoryStoreReviewNote,
  MemoryStoreReviewResolveParams,
  MemoryStoreReviewResolveResponse,
  MemoryResetResponse,
  MemoryStoreRootParams,
  MemoryStoreScope,
  MemoryStoreSearchHit,
  MemoryStoreSearchMatchMode,
  MemoryStoreSearchParams,
  MemoryStoreSearchResponse,
};

type MemoryStoreAppServerClient = Pick<AppServerClient, "request">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isOptionalString(value: unknown): value is string | undefined | null {
  return value === undefined || value === null || typeof value === "string";
}

function isMemoryStoreScope(value: unknown): value is MemoryStoreScope {
  return value === "global" || value === "workspace";
}

function isMemoryStoreCitation(value: unknown): value is MemoryStoreCitation {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    typeof value.startLineNumber === "number" &&
    typeof value.endLineNumber === "number"
  );
}

function isMemoryStoreEntry(value: unknown): value is MemoryStoreEntry {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    typeof value.entryType === "string" &&
    typeof value.size === "number" &&
    typeof value.modifiedAt === "number"
  );
}

function isMemoryStoreSearchHit(value: unknown): value is MemoryStoreSearchHit {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    Array.isArray(value.matchedQueries) &&
    value.matchedQueries.every((query) => typeof query === "string") &&
    typeof value.matchLineNumber === "number" &&
    typeof value.contentStartLineNumber === "number" &&
    typeof value.content === "string" &&
    isMemoryStoreCitation(value.citation)
  );
}

function isMemoryStoreReviewAction(
  value: unknown,
): value is "accept" | "reject" {
  return value === "accept" || value === "reject";
}

function isMemoryStoreReviewNote(
  value: unknown,
): value is MemoryStoreReviewNote {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    typeof value.size === "number" &&
    typeof value.modifiedAt === "number" &&
    typeof value.preview === "string" &&
    isMemoryStoreCitation(value.citation)
  );
}

function assertMemoryStoreListResponse(
  value: unknown,
): MemoryStoreListResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.path !== "string" ||
    !Array.isArray(value.entries) ||
    !value.entries.every(isMemoryStoreEntry) ||
    typeof value.truncated !== "boolean" ||
    !isOptionalString(value.nextCursor)
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_LIST} returned an invalid memory store list response`,
    );
  }
  return value as unknown as MemoryStoreListResponse;
}

function assertMemoryStoreReadResponse(
  value: unknown,
): MemoryStoreReadResponse {
  if (
    !isRecord(value) ||
    typeof value.path !== "string" ||
    typeof value.startLineNumber !== "number" ||
    typeof value.content !== "string" ||
    typeof value.truncated !== "boolean" ||
    !isMemoryStoreCitation(value.citation)
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_READ} returned an invalid memory store read response`,
    );
  }
  return value as unknown as MemoryStoreReadResponse;
}

function assertMemoryStoreSearchResponse(
  value: unknown,
): MemoryStoreSearchResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.hits) ||
    !value.hits.every(isMemoryStoreSearchHit) ||
    typeof value.truncated !== "boolean" ||
    !isOptionalString(value.nextCursor)
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_SEARCH} returned an invalid memory store search response`,
    );
  }
  return value as unknown as MemoryStoreSearchResponse;
}

function assertMemoryStoreAddNoteResponse(
  value: unknown,
): MemoryStoreAddNoteResponse {
  if (
    !isRecord(value) ||
    typeof value.path !== "string" ||
    !isMemoryStoreCitation(value.citation)
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_ADD_NOTE} returned an invalid memory store add note response`,
    );
  }
  return value as unknown as MemoryStoreAddNoteResponse;
}

function assertMemoryStoreConsolidateResponse(
  value: unknown,
): MemoryStoreConsolidateResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.rootPath !== "string" ||
    typeof value.processedNotes !== "number" ||
    typeof value.skippedNotes !== "number" ||
    typeof value.archivedNotes !== "number" ||
    typeof value.memoryPath !== "string" ||
    typeof value.summaryPath !== "string" ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every((warning) => typeof warning === "string") ||
    typeof value.updated !== "boolean"
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_CONSOLIDATE} returned an invalid memory store consolidate response`,
    );
  }
  return value as unknown as MemoryStoreConsolidateResponse;
}

function assertMemoryStoreReviewListResponse(
  value: unknown,
): MemoryStoreReviewListResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.rootPath !== "string" ||
    !Array.isArray(value.notes) ||
    !value.notes.every(isMemoryStoreReviewNote) ||
    typeof value.truncated !== "boolean" ||
    !isOptionalString(value.nextCursor)
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_REVIEW_LIST} returned an invalid memory store review list response`,
    );
  }
  return value as unknown as MemoryStoreReviewListResponse;
}

function assertMemoryStoreReviewResolveResponse(
  value: unknown,
): MemoryStoreReviewResolveResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.rootPath !== "string" ||
    typeof value.sourcePath !== "string" ||
    typeof value.archivedPath !== "string" ||
    !isMemoryStoreReviewAction(value.action) ||
    typeof value.memoryPath !== "string" ||
    typeof value.summaryPath !== "string" ||
    typeof value.updated !== "boolean"
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_REVIEW_RESOLVE} returned an invalid memory store review resolve response`,
    );
  }
  return value as unknown as MemoryStoreReviewResolveResponse;
}

function assertMemoryStoreHealthResponse(
  value: unknown,
): MemoryStoreHealthResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.rootPath !== "string" ||
    typeof value.initialized !== "boolean" ||
    typeof value.fileCount !== "number" ||
    typeof value.totalBytes !== "number" ||
    typeof value.summaryExists !== "boolean" ||
    typeof value.summaryBytes !== "number" ||
    typeof value.memoryExists !== "boolean" ||
    typeof value.memoryBytes !== "number" ||
    typeof value.notesCount !== "number"
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_HEALTH} returned an invalid memory store health response`,
    );
  }
  return value as unknown as MemoryStoreHealthResponse;
}

function assertMemoryResetResponse(value: unknown): MemoryResetResponse {
  if (!isRecord(value) || Object.keys(value).length !== 0) {
    throw new Error(`${METHOD_MEMORY_RESET} returned an invalid response`);
  }
  return value;
}

function assertMemoryStoreIndexRebuildResponse(
  value: unknown,
): MemoryStoreIndexRebuildResponse {
  if (
    !isRecord(value) ||
    !isMemoryStoreScope(value.rootScope) ||
    typeof value.rootPath !== "string" ||
    typeof value.manifestPath !== "string" ||
    typeof value.schemaVersion !== "string" ||
    typeof value.sourceFileCount !== "number" ||
    typeof value.sourceTotalBytes !== "number" ||
    typeof value.sourceChecksum !== "string" ||
    typeof value.indexedAt !== "string" ||
    typeof value.rebuilt !== "boolean"
  ) {
    throw new Error(
      `${METHOD_MEMORY_STORE_INDEX_REBUILD} returned an invalid memory store index rebuild response`,
    );
  }
  return value as unknown as MemoryStoreIndexRebuildResponse;
}

export async function listMemoryStore(
  params: MemoryStoreListParams = {},
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreListResponse> {
  const response = await appServerClient.request<MemoryStoreListResponse>(
    METHOD_MEMORY_STORE_LIST,
    params,
  );
  return assertMemoryStoreListResponse(response.result);
}

export async function readMemoryStore(
  params: MemoryStoreReadParams,
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreReadResponse> {
  const response = await appServerClient.request<MemoryStoreReadResponse>(
    METHOD_MEMORY_STORE_READ,
    params,
  );
  return assertMemoryStoreReadResponse(response.result);
}

export async function searchMemoryStore(
  params: MemoryStoreSearchParams,
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreSearchResponse> {
  const response = await appServerClient.request<MemoryStoreSearchResponse>(
    METHOD_MEMORY_STORE_SEARCH,
    params,
  );
  return assertMemoryStoreSearchResponse(response.result);
}

export async function addMemoryStoreNote(
  params: MemoryStoreAddNoteParams,
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreAddNoteResponse> {
  const response = await appServerClient.request<MemoryStoreAddNoteResponse>(
    METHOD_MEMORY_STORE_ADD_NOTE,
    params,
  );
  return assertMemoryStoreAddNoteResponse(response.result);
}

export async function consolidateMemoryStore(
  params: MemoryStoreConsolidateParams = {},
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreConsolidateResponse> {
  const response =
    await appServerClient.request<MemoryStoreConsolidateResponse>(
      METHOD_MEMORY_STORE_CONSOLIDATE,
      params,
    );
  return assertMemoryStoreConsolidateResponse(response.result);
}

export async function listMemoryStoreReviewNotes(
  params: MemoryStoreReviewListParams = {},
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreReviewListResponse> {
  const response = await appServerClient.request<MemoryStoreReviewListResponse>(
    METHOD_MEMORY_STORE_REVIEW_LIST,
    params,
  );
  return assertMemoryStoreReviewListResponse(response.result);
}

export async function resolveMemoryStoreReviewNote(
  params: MemoryStoreReviewResolveParams,
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreReviewResolveResponse> {
  const response =
    await appServerClient.request<MemoryStoreReviewResolveResponse>(
      METHOD_MEMORY_STORE_REVIEW_RESOLVE,
      params,
    );
  return assertMemoryStoreReviewResolveResponse(response.result);
}

export async function getMemoryStoreHealth(
  params: MemoryStoreRootParams = {},
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreHealthResponse> {
  const response = await appServerClient.request<MemoryStoreHealthResponse>(
    METHOD_MEMORY_STORE_HEALTH,
    params,
  );
  return assertMemoryStoreHealthResponse(response.result);
}

export async function resetMemory(
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryResetResponse> {
  const response =
    await appServerClient.request<MemoryResetResponse>(METHOD_MEMORY_RESET);
  return assertMemoryResetResponse(response.result);
}

export async function rebuildMemoryStoreIndex(
  params: MemoryStoreRootParams = {},
  appServerClient: MemoryStoreAppServerClient = new AppServerClient(),
): Promise<MemoryStoreIndexRebuildResponse> {
  const response =
    await appServerClient.request<MemoryStoreIndexRebuildResponse>(
      METHOD_MEMORY_STORE_INDEX_REBUILD,
      params,
    );
  return assertMemoryStoreIndexRebuildResponse(response.result);
}
