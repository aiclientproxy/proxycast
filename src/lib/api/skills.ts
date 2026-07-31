import { safeInvoke } from "@/lib/dev-bridge";
import { AppServerClient } from "@/lib/api/appServer";
import type {
  SkillMarketplaceBundle,
  SkillMarketplaceInstallResult,
} from "./officialSkillMarketplace";
import { assertNotDiagnosticFacade } from "./diagnosticFacade";
import { revealPathInFinder } from "./fileSystem";
import { skillExecutionApi } from "./skill-execution";
import { projectRuntimeSkillCatalog } from "./runtimeSkillCatalog";
import {
  METHOD_SKILL_LOCAL_DETAIL_INSPECT,
  METHOD_SKILL_LOCAL_IMPORT,
  METHOD_SKILL_LOCAL_INSPECT,
  METHOD_SKILL_LOCAL_RENAME,
  METHOD_SKILL_LOCAL_SCAFFOLD_CREATE,
  METHOD_SKILL_MANAGEMENT_INSTALL,
  METHOD_SKILL_MANAGEMENT_LIST,
  METHOD_SKILL_MANAGEMENT_UNINSTALL,
  METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL,
  METHOD_SKILL_MARKETPLACE_INSTALL,
  METHOD_SKILL_PACKAGE_EXPORT,
  METHOD_SKILL_PACKAGE_LOCAL_INSPECT,
  METHOD_SKILL_PACKAGE_LOCAL_INSTALL,
  METHOD_SKILL_PACKAGE_LOCAL_REPLACE,
  METHOD_SKILL_REMOTE_INSPECT,
  METHOD_SKILL_REPOSITORY_DELETE,
  METHOD_SKILL_REPOSITORY_LIST,
  METHOD_SKILL_REPOSITORY_SAVE,
  METHOD_SKILL_CACHE_REFRESH,
  METHOD_SKILL_INSTALLED_DIRECTORIES_LIST,
  type SkillDownloadInstallParams,
  type SkillLocalImportParams,
  type SkillLocalInspectParams,
  type SkillLocalDetailInspectParams,
  type SkillLocalRenameParams,
  type SkillManagementInstallParams,
  type SkillManagementListParams,
  type SkillManagementUninstallParams,
  type SkillMarketplaceInstallParams,
  type SkillPackageExportParams,
  type SkillPackageLocalInspectParams,
  type SkillPackageLocalInstallParams,
  type SkillPackageLocalReplaceParams,
  type SkillRemoteInspectParams,
  type SkillRepositoryDeleteParams,
  type SkillRepositorySaveParams,
  type SkillScaffoldCreateParams,
} from "../../../packages/app-server-client/src/protocol";

export type SkillSourceKind = "builtin" | "other";
export type SkillCatalogSource = "project" | "user" | "remote";
export type SkillScaffoldTarget = "project" | "user";

export interface SkillResourceSummary {
  hasScripts: boolean;
  hasReferences: boolean;
  hasAssets: boolean;
}

export interface SkillStandardCompliance {
  isStandard: boolean;
  validationErrors: string[];
  deprecatedFields: string[];
}

export interface SkillInspection {
  content: string;
  license?: string;
  compatibility?: string;
  metadata: Record<string, string>;
  allowedTools: string[];
  resourceSummary: SkillResourceSummary;
  standardCompliance: SkillStandardCompliance;
}

export type LocalSkillInspection = SkillInspection;

export interface RemoteSkillLocator extends Record<string, unknown> {
  owner: string;
  name: string;
  branch: string;
  directory: string;
}

export interface CreateSkillScaffoldRequest extends Record<string, unknown> {
  target: SkillScaffoldTarget;
  directory: string;
  name: string;
  description: string;
  whenToUse?: string[];
  inputs?: string[];
  outputs?: string[];
  steps?: string[];
  fallbackStrategy?: string[];
}

export interface Skill {
  key: string;
  name: string;
  description: string;
  directory: string;
  localDirectoryPath?: string;
  readmeUrl?: string;
  installed: boolean;
  sourceKind: SkillSourceKind;
  catalogSource?: SkillCatalogSource;
  repoOwner?: string;
  repoName?: string;
  repoBranch?: string;
  license?: string;
  compatibility?: string;
  metadata?: Record<string, string>;
  allowedTools?: string[];
  resourceSummary?: SkillResourceSummary;
  standardCompliance?: SkillStandardCompliance;
}

export interface SkillRepo {
  owner: string;
  name: string;
  branch: string;
  enabled: boolean;
}

export interface ImportedSkillResult {
  directory: string;
}

export interface LocalSkillPackageFileEntry {
  path: string;
  isDirectory: boolean;
  size: number;
  content?: string;
}

export interface LocalSkillPackageInspectionResult {
  directory: string;
  inspection: SkillInspection;
  files: LocalSkillPackageFileEntry[];
}

export type LocalSkillDetailInspectionResult =
  LocalSkillPackageInspectionResult;

export interface SkillPackageExportResult {
  directory: string;
  outputPath: string;
  fileCount: number;
  bytesWritten: number;
}

export interface SkillDownloadInstallRequest extends Record<string, unknown> {
  skillName: string;
  downloadUrl: string;
}

export type AppType = "claude" | "codex" | "gemini" | "lime";

export const SKILL_PACKAGE_OPEN_EVENT = "skill-package://open";

async function invokeSkillPackageShellCommand<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const result = args
    ? await safeInvoke(command, args)
    : await safeInvoke(command);
  assertNotDiagnosticFacade(
    command,
    result,
    "真实 Skill package Desktop Host current 通道",
  );
  return result as T;
}

function normalizeStandardCompliance(
  compliance?: Partial<SkillStandardCompliance> | null,
): SkillStandardCompliance | undefined {
  if (!compliance) {
    return undefined;
  }

  return {
    isStandard: Boolean(compliance.isStandard),
    validationErrors: Array.isArray(compliance.validationErrors)
      ? compliance.validationErrors
      : [],
    deprecatedFields: Array.isArray(compliance.deprecatedFields)
      ? compliance.deprecatedFields
      : [],
  };
}

function normalizeSkill(skill: Skill): Skill {
  return {
    ...skill,
    localDirectoryPath: normalizeOptionalPath(skill.localDirectoryPath),
    standardCompliance: normalizeStandardCompliance(skill.standardCompliance),
  };
}

function normalizeInspection(inspection: SkillInspection): SkillInspection {
  return {
    ...inspection,
    standardCompliance: normalizeStandardCompliance(
      inspection.standardCompliance,
    ) ?? {
      isStandard: false,
      validationErrors: [],
      deprecatedFields: [],
    },
  };
}

function normalizeSkills(value: Skill[] | null | undefined): Skill[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map(normalizeSkill);
}

function normalizeSkillRepos(
  value: SkillRepo[] | null | undefined,
): SkillRepo[] {
  return Array.isArray(value) ? value : [];
}

function normalizeStringList(value: string[] | null | undefined): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function isLocalManagedSkill(skill: Skill): boolean {
  if (skill.catalogSource === "project") {
    return false;
  }
  if (skill.catalogSource === "remote") {
    return false;
  }
  return !(
    skill.catalogSource === undefined &&
    skill.repoOwner &&
    skill.repoName
  );
}

function normalizeOptionalPath(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isStringList(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

async function requestSkillAppServer<T>(
  method: string,
  params: Record<string, unknown>,
): Promise<T> {
  const response = await new AppServerClient().request<T>(method, params);
  return response.result;
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    isRecord(value) &&
    Object.values(value).every((item) => typeof item === "string")
  );
}

function isOptionalString(value: unknown): boolean {
  return value === undefined || value === null || typeof value === "string";
}

function isSkillResourceSummary(value: unknown): value is SkillResourceSummary {
  return (
    isRecord(value) &&
    typeof value.hasScripts === "boolean" &&
    typeof value.hasReferences === "boolean" &&
    typeof value.hasAssets === "boolean"
  );
}

function isSkillStandardComplianceLike(value: unknown): boolean {
  return (
    value === undefined ||
    value === null ||
    (isRecord(value) &&
      (value.isStandard === undefined ||
        typeof value.isStandard === "boolean") &&
      (value.validationErrors === undefined ||
        isStringList(value.validationErrors)) &&
      (value.deprecatedFields === undefined ||
        isStringList(value.deprecatedFields)))
  );
}

function isSkillInspection(value: unknown): value is SkillInspection {
  return (
    isRecord(value) &&
    typeof value.content === "string" &&
    isOptionalString(value.license) &&
    isOptionalString(value.compatibility) &&
    isStringRecord(value.metadata) &&
    isStringList(value.allowedTools) &&
    isSkillResourceSummary(value.resourceSummary) &&
    isSkillStandardComplianceLike(value.standardCompliance)
  );
}

function assertSkillManagementWriteResult(command: string, value: unknown) {
  if (!isRecord(value) || value.success !== true) {
    throw new Error(`${command} did not return success result`);
  }
}

function assertImportedSkillResult(
  command: string,
  value: unknown,
): asserts value is ImportedSkillResult {
  if (!isRecord(value) || typeof value.directory !== "string") {
    throw new Error(`${command} did not return imported skill result`);
  }
}

function isLocalSkillPackageFileEntry(
  value: unknown,
): value is LocalSkillPackageFileEntry {
  return (
    isRecord(value) &&
    typeof value.path === "string" &&
    typeof value.isDirectory === "boolean" &&
    typeof value.size === "number" &&
    Number.isFinite(value.size) &&
    isOptionalString(value.content)
  );
}

function assertLocalSkillPackageInspectionResult(
  command: string,
  value: unknown,
): asserts value is LocalSkillPackageInspectionResult {
  if (
    !isRecord(value) ||
    typeof value.directory !== "string" ||
    !isSkillInspection(value.inspection) ||
    !Array.isArray(value.files) ||
    !value.files.every(isLocalSkillPackageFileEntry)
  ) {
    throw new Error(`${command} did not return local skill package inspection`);
  }
}

function assertSkillMarketplaceInstallResult(
  command: string,
  value: unknown,
): asserts value is SkillMarketplaceInstallResult {
  if (
    !isRecord(value) ||
    typeof value.directory !== "string" ||
    !isSkillInspection(value.inspection)
  ) {
    throw new Error(`${command} did not return skill install result`);
  }
}

function assertSkillPackageExportResult(
  command: string,
  value: unknown,
): asserts value is SkillPackageExportResult {
  if (
    !isRecord(value) ||
    typeof value.directory !== "string" ||
    typeof value.outputPath !== "string" ||
    typeof value.fileCount !== "number" ||
    !Number.isFinite(value.fileCount) ||
    typeof value.bytesWritten !== "number" ||
    !Number.isFinite(value.bytesWritten)
  ) {
    throw new Error(`${command} did not return skill package export result`);
  }
}

function assertSkillRepositoryListResult(
  command: string,
  value: unknown,
): asserts value is { repos: SkillRepo[] } {
  if (!isRecord(value) || !Array.isArray(value.repos)) {
    throw new Error(`${command} did not return skill repositories`);
  }
}

function assertSkillListResult(
  command: string,
  value: unknown,
): asserts value is { skills: Skill[] } {
  if (!isRecord(value) || !Array.isArray(value.skills)) {
    throw new Error(`${command} did not return skills`);
  }
}

function assertInstalledSkillDirectoriesResult(
  command: string,
  value: unknown,
): asserts value is { directories: string[] } {
  if (!isRecord(value) || !isStringList(value.directories)) {
    throw new Error(`${command} did not return installed skill directories`);
  }
}

function assertSkillInspectionResult(
  command: string,
  value: unknown,
): asserts value is { inspection: SkillInspection } {
  if (!isRecord(value) || !isSkillInspection(value.inspection)) {
    throw new Error(`${command} did not return skill inspection result`);
  }
}

export const skillsApi = {
  async getRuntimeCatalog(): Promise<Skill[]> {
    return projectRuntimeSkillCatalog(
      await skillExecutionApi.listExecutableSkills(),
    );
  },

  async getLocal(app: AppType = "lime"): Promise<Skill[]> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_MANAGEMENT_LIST,
      {
        app,
        refreshRemote: false,
        scope: "user",
      } satisfies SkillManagementListParams,
    );
    assertSkillListResult(METHOD_SKILL_MANAGEMENT_LIST, result);
    return normalizeSkills(result.skills).filter(isLocalManagedSkill);
  },

  async getAll(
    app: AppType = "lime",
    options?: { refreshRemote?: boolean },
  ): Promise<Skill[]> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_MANAGEMENT_LIST,
      {
        app,
        refreshRemote: options?.refreshRemote ?? false,
      } satisfies SkillManagementListParams,
    );
    assertSkillListResult(METHOD_SKILL_MANAGEMENT_LIST, result);
    return normalizeSkills(result.skills);
  },

  async install(directory: string, app: AppType = "lime"): Promise<boolean> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_MANAGEMENT_INSTALL,
      { app, directory } satisfies SkillManagementInstallParams,
    );
    assertSkillManagementWriteResult(METHOD_SKILL_MANAGEMENT_INSTALL, result);
    return true;
  },

  async uninstall(directory: string, app: AppType = "lime"): Promise<boolean> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_MANAGEMENT_UNINSTALL,
      { app, directory } satisfies SkillManagementUninstallParams,
    );
    assertSkillManagementWriteResult(METHOD_SKILL_MANAGEMENT_UNINSTALL, result);
    return true;
  },

  async getRepos(): Promise<SkillRepo[]> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_REPOSITORY_LIST,
      {},
    );
    assertSkillRepositoryListResult(METHOD_SKILL_REPOSITORY_LIST, result);
    return normalizeSkillRepos(result.repos);
  },

  async addRepo(repo: SkillRepo): Promise<boolean> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_REPOSITORY_SAVE,
      { repo } satisfies SkillRepositorySaveParams,
    );
    assertSkillManagementWriteResult(METHOD_SKILL_REPOSITORY_SAVE, result);
    return true;
  },

  async removeRepo(owner: string, name: string): Promise<boolean> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_REPOSITORY_DELETE,
      { owner, name } satisfies SkillRepositoryDeleteParams,
    );
    assertSkillManagementWriteResult(METHOD_SKILL_REPOSITORY_DELETE, result);
    return true;
  },

  async refreshCache(): Promise<boolean> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_CACHE_REFRESH,
      {},
    );
    assertSkillManagementWriteResult(METHOD_SKILL_CACHE_REFRESH, result);
    return true;
  },

  /**
   * 获取已安装的 Lime Skills 目录列表
   *
   * 扫描 Lime 可发现的 provider Skills 根目录，返回包含 SKILL.md 的子目录名列表。
   * 这些 Skills 将被传递给 agent 用于 AI Agent 功能。
   *
   * @returns 已安装的 Skill 目录名列表
   */
  async getInstalledLimeSkills(): Promise<string[]> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_INSTALLED_DIRECTORIES_LIST,
      {},
    );
    assertInstalledSkillDirectoriesResult(
      METHOD_SKILL_INSTALLED_DIRECTORIES_LIST,
      result,
    );
    return normalizeStringList(result.directories);
  },

  /**
   * 获取本地已安装 Skill 的标准检查结果
   *
   * @param directory Skill 目录名
   * @param app 应用类型
   * @returns 标准检查结果与原始 SKILL.md 内容
   */
  async inspectLocalSkill(
    directory: string,
    app: AppType = "lime",
  ): Promise<SkillInspection> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_LOCAL_INSPECT,
      { app, directory } satisfies SkillLocalInspectParams,
    );
    assertSkillInspectionResult(METHOD_SKILL_LOCAL_INSPECT, result);
    return normalizeInspection(result.inspection);
  },

  async inspectLocalSkillDetail(
    directory: string,
    app: AppType = "lime",
  ): Promise<LocalSkillDetailInspectionResult> {
    const result =
      await requestSkillAppServer<LocalSkillDetailInspectionResult>(
        METHOD_SKILL_LOCAL_DETAIL_INSPECT,
        { app, directory } satisfies SkillLocalDetailInspectParams,
      );
    assertLocalSkillPackageInspectionResult(
      METHOD_SKILL_LOCAL_DETAIL_INSPECT,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async revealLocalSkill(
    directory: string,
    app: AppType = "lime",
  ): Promise<boolean> {
    const normalizedDirectory = directory.trim();
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_MANAGEMENT_LIST,
      {
        app,
        refreshRemote: false,
        scope: "user",
      } satisfies SkillManagementListParams,
    );
    assertSkillListResult(METHOD_SKILL_MANAGEMENT_LIST, result);
    const skill = normalizeSkills(result.skills).find(
      (item) => item.directory === normalizedDirectory,
    );
    const localDirectoryPath = normalizeOptionalPath(skill?.localDirectoryPath);
    if (!localDirectoryPath) {
      throw new Error(
        `skill/list did not return localDirectoryPath for ${normalizedDirectory}`,
      );
    }
    await revealPathInFinder(localDirectoryPath);
    return true;
  },

  async renameLocalSkill(
    directory: string,
    newDirectory: string,
    app: AppType = "lime",
  ): Promise<ImportedSkillResult> {
    const result = await requestSkillAppServer<ImportedSkillResult>(
      METHOD_SKILL_LOCAL_RENAME,
      {
        app,
        directory,
        newDirectory,
      } satisfies SkillLocalRenameParams,
    );
    assertImportedSkillResult(METHOD_SKILL_LOCAL_RENAME, result);
    return result;
  },

  async replaceLocalSkillPackage(
    directory: string,
    sourcePath: string,
    app: AppType = "lime",
  ): Promise<SkillMarketplaceInstallResult> {
    const result = await requestSkillAppServer<SkillMarketplaceInstallResult>(
      METHOD_SKILL_PACKAGE_LOCAL_REPLACE,
      {
        app,
        directory,
        sourcePath,
      } satisfies SkillPackageLocalReplaceParams,
    );
    assertSkillMarketplaceInstallResult(
      METHOD_SKILL_PACKAGE_LOCAL_REPLACE,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async createSkillScaffold(
    request: CreateSkillScaffoldRequest,
    app: AppType = "lime",
  ): Promise<SkillInspection> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_LOCAL_SCAFFOLD_CREATE,
      {
        app,
        request,
      } satisfies SkillScaffoldCreateParams,
    );
    assertSkillInspectionResult(METHOD_SKILL_LOCAL_SCAFFOLD_CREATE, result);
    return normalizeInspection(result.inspection);
  },

  async importLocalSkill(
    sourcePath: string,
    app: AppType = "lime",
  ): Promise<ImportedSkillResult> {
    const result = await requestSkillAppServer<ImportedSkillResult>(
      METHOD_SKILL_LOCAL_IMPORT,
      {
        app,
        sourcePath,
      } satisfies SkillLocalImportParams,
    );
    assertImportedSkillResult(METHOD_SKILL_LOCAL_IMPORT, result);
    return result;
  },

  async inspectLocalSkillPackage(
    sourcePath: string,
    app: AppType = "lime",
  ): Promise<LocalSkillPackageInspectionResult> {
    const result =
      await requestSkillAppServer<LocalSkillPackageInspectionResult>(
        METHOD_SKILL_PACKAGE_LOCAL_INSPECT,
        { app, sourcePath } satisfies SkillPackageLocalInspectParams,
      );
    assertLocalSkillPackageInspectionResult(
      METHOD_SKILL_PACKAGE_LOCAL_INSPECT,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async installLocalSkillPackage(
    sourcePath: string,
    app: AppType = "lime",
  ): Promise<SkillMarketplaceInstallResult> {
    const result = await requestSkillAppServer<SkillMarketplaceInstallResult>(
      METHOD_SKILL_PACKAGE_LOCAL_INSTALL,
      { app, sourcePath } satisfies SkillPackageLocalInstallParams,
    );
    assertSkillMarketplaceInstallResult(
      METHOD_SKILL_PACKAGE_LOCAL_INSTALL,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async takePendingSkillPackageOpenRequests(): Promise<string[]> {
    const paths = await invokeSkillPackageShellCommand<string[]>(
      "take_pending_skill_package_open_requests",
    );
    return normalizeStringList(paths);
  },

  async exportLocalSkillPackage(
    directory: string,
    targetPath: string,
    app: AppType = "lime",
  ): Promise<SkillPackageExportResult> {
    const result = await requestSkillAppServer<SkillPackageExportResult>(
      METHOD_SKILL_PACKAGE_EXPORT,
      { app, directory, targetPath } satisfies SkillPackageExportParams,
    );
    assertSkillPackageExportResult(METHOD_SKILL_PACKAGE_EXPORT, result);
    return result;
  },

  async installMarketplaceBundle(
    bundle: SkillMarketplaceBundle,
    app: AppType = "lime",
  ): Promise<SkillMarketplaceInstallResult> {
    const result = await requestSkillAppServer<SkillMarketplaceInstallResult>(
      METHOD_SKILL_MARKETPLACE_INSTALL,
      {
        app,
        manifestVersion: bundle.manifestVersion,
        name: bundle.name,
        aliases: bundle.aliases,
        version: bundle.version,
        contentHash: bundle.contentHash,
        fileCount: bundle.fileCount,
        files: bundle.files,
      } satisfies SkillMarketplaceInstallParams,
    );
    assertSkillMarketplaceInstallResult(
      METHOD_SKILL_MARKETPLACE_INSTALL,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async installFromDownloadUrl(
    request: SkillDownloadInstallRequest,
    app: AppType = "lime",
  ): Promise<SkillMarketplaceInstallResult> {
    const result = await requestSkillAppServer<SkillMarketplaceInstallResult>(
      METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL,
      {
        app,
        skillName: request.skillName,
        downloadUrl: request.downloadUrl,
      } satisfies SkillDownloadInstallParams,
    );
    assertSkillMarketplaceInstallResult(
      METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL,
      result,
    );
    return {
      ...result,
      inspection: normalizeInspection(result.inspection),
    };
  },

  async inspectRemoteSkill(
    locator: RemoteSkillLocator,
  ): Promise<SkillInspection> {
    const result = await requestSkillAppServer<unknown>(
      METHOD_SKILL_REMOTE_INSPECT,
      locator satisfies SkillRemoteInspectParams,
    );
    assertSkillInspectionResult(METHOD_SKILL_REMOTE_INSPECT, result);
    return normalizeInspection(result.inspection);
  },
};
