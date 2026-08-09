import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  collectRustFiles,
  collectTextFiles,
  REPO_ROOT,
  repoRelative,
} from "./appServerRuntimeBoundary.testSupport";

type UsageBaseline = Record<string, Record<string, number>>;

const RUST_CRATES_ROOT = join(REPO_ROOT, "lime-rs/crates");
const CORE_SRC_ROOT = join(REPO_ROOT, "lime-rs/crates/core/src");
const ELECTRON_ROOT = join(REPO_ROOT, "electron");
const APP_SERVER_CLIENT_ROOT = join(
  REPO_ROOT,
  "packages/app-server-client/src",
);
const SCRIPTS_ROOT = join(REPO_ROOT, "scripts");
const APP_SERVER_MAIN = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/main.rs",
);
const APP_SERVER_PROJECTION_STORE = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/runtime/projection_store.rs",
);
const LOCAL_APP_DATA_SOURCE = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source.rs",
);
const SESSION_FILE_STORAGE = join(
  REPO_ROOT,
  "lime-rs/crates/core/src/session_files/storage.rs",
);
const CORE_LOG_STORE = join(REPO_ROOT, "lime-rs/crates/core/src/logger.rs");
const CORE_STARTUP_MIGRATIONS = join(
  REPO_ROOT,
  "lime-rs/crates/core/src/database/startup_migrations.rs",
);
const CORE_API_KEY_MIGRATION = join(
  REPO_ROOT,
  "lime-rs/crates/core/src/database/migration/api_key_migration.rs",
);
const VOICE_CONFIG_SERVICE = join(
  REPO_ROOT,
  "lime-rs/crates/services/src/voice_config_service.rs",
);
const VOICE_ASR_SERVICE = join(
  REPO_ROOT,
  "lime-rs/crates/services/src/voice_asr_service.rs",
);
const VOICE_ASR_CREDENTIALS = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/voice_asr_credentials.rs",
);
const CONNECT_DATA_SOURCE = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/connect.rs",
);
const CONNECT_DATA_SOURCE_IMPL = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/impls/connect.rs",
);
const PLUGIN_DATA_SOURCE_IMPL = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/impls/plugins.rs",
);
const PLUGIN_CATALOG = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/plugin_catalog.rs",
);
const LOCAL_DIAGNOSTICS = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/diagnostics.rs",
);
const SUPPORT_BUNDLE = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/diagnostics/support_bundle.rs",
);
const STYLE_PACK_PATHS = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/runtime/soul/style_pack_paths.rs",
);
const MCP_OAUTH_STORE = join(
  REPO_ROOT,
  "lime-rs/crates/mcp/src/oauth_store.rs",
);
const STORAGE_ROOT_OWNER = "lime-rs/crates/core/src/app_paths.rs";
const ELECTRON_STORAGE_ROOT_OWNER = "electron/appDataPaths.ts";

const STORAGE_ROOT_BYPASS_PATTERN =
  /\b(?:[A-Za-z_][A-Za-z0-9_]*::)*(preferred_data_dir|resolve_sessions_dir|resolve_request_logs_dir|best_effort_[A-Za-z0-9_]+)\s*\(/gu;

// 叶子 writer 基线必须保持为空：任何 durable writer / 诊断 / 产品资产 owner
// 都只能接收组合根注入的 root，不得自行解析平台默认目录。
const KNOWN_STORAGE_ROOT_BYPASS_BASELINE: UsageBaseline = {};

// 组合根是唯一允许解析平台默认根的位置：App Server 二进制在启动时解析一次
// AppDataRoot，再把它注入 StorageRoots / LocalAppDataSource / 产品资产 owner。
// 这里只允许该一处，且计数只减不增。
const COMPOSITION_ROOT_ALLOWLIST: UsageBaseline = {
  "lime-rs/crates/app-server/src/main.rs": {
    preferred_data_dir: 1,
  },
};

const CURRENT_HOST_USER_DATA_CHILDREN: UsageBaseline = {
  "electron/appConfigHost.ts": { CONFIG_YAML_FILE: 1 },
  "electron/appServerHost.ts": { APP_SERVER_CONFIG_FILE_NAME: 1 },
  "electron/main.ts": { startup: 1 },
};

const CURRENT_APP_DATA_ROOT_CHILDREN: UsageBaseline = {
  "electron/systemUtilityHost.ts": { connectors: 2 },
  "electron/voiceModelHost.ts": { models: 2 },
};

const HOST_USER_DATA_CHILD_PATTERNS = [
  /(?:path|pathApi)\.(?:join|resolve)\(\s*(?:(?:this\.)?#?userDataDir|(?:options\.)?hostUserData)\s*,\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][A-Za-z0-9_$]*))/gu,
  /(?:path|pathApi)\.(?:join|resolve)\(\s*app\.getPath\(\s*["']userData["']\s*\)\s*,\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][A-Za-z0-9_$]*))/gu,
];

const APP_DATA_ROOT_CHILD_PATTERN =
  /(?:path|pathApi)\.(?:join|resolve)\(\s*(?:(?:this\.)?#?appDataRoot|appDataRoot)\s*,\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][A-Za-z0-9_$]*))/gu;

const RETIRED_PRODUCT_DB_STARTUP_CLEANUP_PATTERN =
  /APP_SERVER_PRODUCT_DB_MIGRATION_CLEANUP|--product-db-migration-cleanup|productDbMigrationCleanup|cleanup_migrated_product_db_source|ProductDbMigrationCleanupPolicy|resolve_database_path_with_migration|DatabasePathResolution|migration-manifest\.json|storage-migration\.v1|database-path-v1/gu;
const RETIRED_MANAGED_PROJECT_MIGRATION_PATTERN =
  /migration_v6|migrate_managed_workspace_paths|migrate_managed_project_path_to_preferred|migrated_managed_workspace_paths_to_lime_v1|copy_dir_contents_if_missing/gu;
const RETIRED_TIMESTAMP_MIGRATION_MARKER = ".migration_completed";

function isDedicatedTestFile(relativePath: string): boolean {
  return (
    relativePath.includes("/tests/") ||
    relativePath.endsWith("/tests.rs") ||
    relativePath.endsWith("_tests.rs") ||
    /\.(?:test|spec)\.[^/.]+$/u.test(relativePath)
  );
}

function incrementUsage(
  usages: UsageBaseline,
  relativePath: string,
  symbol: string,
): void {
  const fileUsages = (usages[relativePath] ??= {});
  fileUsages[symbol] = (fileUsages[symbol] ?? 0) + 1;
}

function collectRustStorageRootBypasses(): UsageBaseline {
  const usages: UsageBaseline = {};

  for (const filePath of collectRustFiles(RUST_CRATES_ROOT)) {
    const relativePath = repoRelative(filePath);
    if (
      relativePath === STORAGE_ROOT_OWNER ||
      isDedicatedTestFile(relativePath)
    ) {
      continue;
    }

    const source = readFileSync(filePath, "utf8");
    for (const match of source.matchAll(STORAGE_ROOT_BYPASS_PATTERN)) {
      incrementUsage(usages, relativePath, match[1]);
    }
  }

  return usages;
}

function collectHostUserDataChildren(): UsageBaseline {
  const usages: UsageBaseline = {};

  for (const filePath of collectTextFiles(ELECTRON_ROOT)) {
    const relativePath = repoRelative(filePath);
    if (
      isDedicatedTestFile(relativePath) ||
      !relativePath.endsWith(".ts") ||
      relativePath.endsWith(".d.ts") ||
      !statSync(filePath).isFile()
    ) {
      continue;
    }

    const source = readFileSync(filePath, "utf8");
    for (const pattern of HOST_USER_DATA_CHILD_PATTERNS) {
      for (const match of source.matchAll(pattern)) {
        const child = match[1] ?? match[2] ?? match[3];
        incrementUsage(usages, relativePath, child);
      }
    }
  }

  return usages;
}

function collectAppDataRootChildren(): UsageBaseline {
  const usages: UsageBaseline = {};

  for (const filePath of collectTextFiles(ELECTRON_ROOT)) {
    const relativePath = repoRelative(filePath);
    if (
      relativePath === ELECTRON_STORAGE_ROOT_OWNER ||
      isDedicatedTestFile(relativePath) ||
      !relativePath.endsWith(".ts") ||
      relativePath.endsWith(".d.ts") ||
      !statSync(filePath).isFile()
    ) {
      continue;
    }

    const source = readFileSync(filePath, "utf8");
    for (const match of source.matchAll(APP_DATA_ROOT_CHILD_PATTERN)) {
      const child = match[1] ?? match[2] ?? match[3];
      incrementUsage(usages, relativePath, child);
    }
  }

  return usages;
}

function collectRetiredProductDbStartupCleanupReferences(): string[] {
  const findings: string[] = [];

  for (const root of [ELECTRON_ROOT, APP_SERVER_CLIENT_ROOT, SCRIPTS_ROOT]) {
    for (const filePath of collectTextFiles(root)) {
      const relativePath = repoRelative(filePath);
      if (
        isDedicatedTestFile(relativePath) ||
        !/\.(?:ts|mjs|js)$/u.test(relativePath) ||
        !statSync(filePath).isFile()
      ) {
        continue;
      }

      const source = readFileSync(filePath, "utf8");
      for (const match of source.matchAll(
        RETIRED_PRODUCT_DB_STARTUP_CLEANUP_PATTERN,
      )) {
        findings.push(`${relativePath}: ${match[0]}`);
      }
    }
  }

  const appServerProductionSource = readFileSync(APP_SERVER_MAIN, "utf8").split(
    "\n#[cfg(test)]\nmod tests",
    1,
  )[0];
  for (const match of appServerProductionSource.matchAll(
    RETIRED_PRODUCT_DB_STARTUP_CLEANUP_PATTERN,
  )) {
    findings.push(`${repoRelative(APP_SERVER_MAIN)}: ${match[0]}`);
  }

  return findings;
}

function collectRetiredManagedProjectMigrationReferences(): string[] {
  const findings: string[] = [];

  for (const filePath of collectRustFiles(CORE_SRC_ROOT)) {
    const relativePath = repoRelative(filePath);
    const source = readFileSync(filePath, "utf8");
    for (const match of source.matchAll(
      RETIRED_MANAGED_PROJECT_MIGRATION_PATTERN,
    )) {
      findings.push(`${relativePath}: ${match[0]}`);
    }
  }

  return findings;
}

function collectRetiredTimestampMigrationMarkerReferences(): string[] {
  const findings: string[] = [];
  const storageOwnerPath = join(REPO_ROOT, STORAGE_ROOT_OWNER);
  if (
    readFileSync(storageOwnerPath, "utf8").includes(
      RETIRED_TIMESTAMP_MIGRATION_MARKER,
    )
  ) {
    findings.push(STORAGE_ROOT_OWNER);
  }

  for (const filePath of collectTextFiles(SCRIPTS_ROOT)) {
    const relativePath = repoRelative(filePath);
    if (
      isDedicatedTestFile(relativePath) ||
      !/\.(?:ts|mjs|js)$/u.test(relativePath) ||
      !statSync(filePath).isFile()
    ) {
      continue;
    }
    if (
      readFileSync(filePath, "utf8").includes(
        RETIRED_TIMESTAMP_MIGRATION_MARKER,
      )
    ) {
      findings.push(relativePath);
    }
  }
  return findings;
}

function findBaselineIncreases(
  actual: UsageBaseline,
  ...allowedBaselines: UsageBaseline[]
): string[] {
  return Object.entries(actual).flatMap(([relativePath, symbols]) =>
    Object.entries(symbols).flatMap(([symbol, count]) => {
      const maximum = allowedBaselines.reduce(
        (sum, baseline) => sum + (baseline[relativePath]?.[symbol] ?? 0),
        0,
      );
      return count > maximum
        ? [`${relativePath}: ${symbol} actual=${count} baseline=${maximum}`]
        : [];
    }),
  );
}

describe("storage root boundary", () => {
  it("Rust production storage 不应新增绕过显式 StorageRoots 的平台 root 解析", () => {
    const increases = findBaselineIncreases(
      collectRustStorageRootBypasses(),
      KNOWN_STORAGE_ROOT_BYPASS_BASELINE,
      COMPOSITION_ROOT_ALLOWLIST,
    );

    expect(
      increases,
      "preferred_data_dir / resolve_sessions_dir / resolve_request_logs_dir / best_effort_* 只允许出现在组合根；新增 durable writer 必须接收显式 StorageRoots 或领域 root",
    ).toEqual([]);
  });

  it("叶子 writer 的 root bypass 基线必须保持为空", () => {
    expect(
      KNOWN_STORAGE_ROOT_BYPASS_BASELINE,
      "叶子 writer 不得重新登记迁移负债；只允许组合根解析平台默认根",
    ).toEqual({});
  });

  it("退役的 root bypass 不得回流到叶子 owner", () => {
    const diagnostics = readFileSync(LOCAL_DIAGNOSTICS, "utf8");
    const supportBundle = readFileSync(SUPPORT_BUNDLE, "utf8");
    const stylePackPaths = readFileSync(STYLE_PACK_PATHS, "utf8");
    const mcpOAuthStore = readFileSync(MCP_OAUTH_STORE, "utf8");

    for (const source of [diagnostics, supportBundle, stylePackPaths]) {
      expect(source).not.toContain("preferred_data_dir");
    }
    expect(mcpOAuthStore).not.toContain("best_effort_runtime_subdir");
    expect(mcpOAuthStore).not.toContain("app_paths");

    // 诊断必须报告实际生效的根与实际打开的库，不得重新解析。
    expect(diagnostics).toContain("app_data_root: &Path");
    expect(diagnostics).not.toContain("database::get_db_path()");
    expect(supportBundle).not.toContain("database::get_db_path()");

    // 领域 root 只能由 AppDataRoot 派生。
    expect(stylePackPaths).toContain("style_pack_data_root_for_app_data_root");
    expect(mcpOAuthStore).toContain("oauth_store_root_for_app_data_root");
  });

  it("production canonical ThreadStore 必须注入 StorageRoots 与 AgentRoot rollout owner", () => {
    const mainProductionSource = readFileSync(APP_SERVER_MAIN, "utf8").split(
      "\n#[cfg(test)]\nmod tests",
      1,
    )[0];
    const projectionStore = readFileSync(APP_SERVER_PROJECTION_STORE, "utf8");

    expect(mainProductionSource).toContain(
      "ProjectionStore::initialize_with_storage_paths(",
    );
    expect(mainProductionSource).toContain("&storage_roots.projection_db_path");
    expect(mainProductionSource).toContain("&storage_roots.state_db_path");
    expect(mainProductionSource).toContain(
      "&storage_roots.thread_history_db_path",
    );
    expect(mainProductionSource).toContain("&storage_roots.data_root");
    expect(mainProductionSource).not.toContain("ProjectionStore::initialize(");
    expect(projectionStore).toContain("rollout_store: Option<RolloutStore>");
    expect(projectionStore).toContain("pub fn initialize_with_storage_paths(");
  });

  it("sessionFile current writer 只能使用注入的 AgentRoot artifact 目录", () => {
    const appPathsProductionSource = readFileSync(
      join(REPO_ROOT, STORAGE_ROOT_OWNER),
      "utf8",
    ).split("\n#[cfg(test)]\n", 1)[0];
    const localDataSource = readFileSync(LOCAL_APP_DATA_SOURCE, "utf8");
    const sessionFileStorage = readFileSync(SESSION_FILE_STORAGE, "utf8");

    expect(appPathsProductionSource).not.toContain("resolve_sessions_dir");
    expect(localDataSource).toContain(
      "SessionFileStorage::base_dir_for_agent_root(&data_root)",
    );
    expect(sessionFileStorage).toContain(
      ".join(ARTIFACTS_DIR_NAME)\n            .join(SESSION_FILES_DIR_NAME)",
    );
    expect(sessionFileStorage).not.toContain("crate::app_paths");
  });

  it("text diagnostics 只能使用注入的 AgentRoot observability/log", () => {
    const appPathsProductionSource = readFileSync(
      join(REPO_ROOT, STORAGE_ROOT_OWNER),
      "utf8",
    ).split("\n#[cfg(test)]\n", 1)[0];
    const localDataSource = readFileSync(LOCAL_APP_DATA_SOURCE, "utf8");
    const logStore = readFileSync(CORE_LOG_STORE, "utf8");
    const diagnostics = readFileSync(LOCAL_DIAGNOSTICS, "utf8");

    expect(appPathsProductionSource).not.toContain("resolve_logs_dir");
    expect(localDataSource).toContain(
      "logger::create_log_store_from_config(&data_root, &config.logging)?",
    );
    expect(logStore).toContain(
      ".join(OBSERVABILITY_DIR_NAME)\n            .join(LOG_DIR_NAME)\n            .join(LOG_FILE_NAME)",
    );
    expect(logStore).not.toContain("crate::app_paths");
    expect(logStore).not.toContain("impl Default for LogStore {");
    expect(logStore).not.toContain("log_raw_response");
    expect(diagnostics).not.toContain("resolve_logs_dir");
    expect(diagnostics).not.toContain("fn current_log_path");
  });

  it("Rust voice model consumer 只能使用显式绝对路径", () => {
    const voiceConfig = readFileSync(VOICE_CONFIG_SERVICE, "utf8");
    const voiceAsr = readFileSync(VOICE_ASR_SERVICE, "utf8");
    const voiceAsrCredentials = readFileSync(VOICE_ASR_CREDENTIALS, "utf8");

    for (const source of [voiceConfig, voiceAsr]) {
      expect(source).not.toContain("lime_core::app_paths");
      expect(source).not.toContain("preferred_data_dir");
      expect(source).not.toContain("dirs::data_dir");
    }
    expect(voiceConfig).toContain("fn resolve_absolute_model_path(");
    expect(voiceConfig).toContain("if !path.is_absolute()");
    expect(voiceAsr).toContain(
      "voice_config_service::resolve_whisper_model_path(whisper_config)?",
    );
    expect(voiceAsr).toContain(
      "voice_config_service::resolve_sensevoice_model_dir(config)?",
    );
    expect(voiceAsrCredentials).not.toContain("lime_core::app_paths");
    expect(voiceAsrCredentials).not.toContain("best_effort_data_dir");
    expect(voiceAsrCredentials).not.toContain(
      "default_voice_model_install_dir",
    );
    expect(voiceAsrCredentials).toContain(
      "validate_voice_model_install_dir(&params.install_dir)?",
    );
  });

  it("Connect registry cache 只能使用注入的 AgentRoot cache 目录", () => {
    const localDataSource = readFileSync(LOCAL_APP_DATA_SOURCE, "utf8");
    const connectDataSource = readFileSync(CONNECT_DATA_SOURCE, "utf8");
    const connectDataSourceImpl = readFileSync(
      CONNECT_DATA_SOURCE_IMPL,
      "utf8",
    );

    expect(connectDataSource).not.toContain("lime_core::app_paths");
    expect(connectDataSource).not.toContain("best_effort_data_dir");
    expect(connectDataSource).toContain(
      'agent_root\n        .join("cache")\n        .join("connect")\n        .join("registry.json")',
    );
    expect(localDataSource).toContain(
      "connect::connect_registry_cache_path_for_agent_root(&data_root)",
    );
    expect(connectDataSourceImpl).toContain(
      "&self.connect_registry_cache_path",
    );
  });

  it("Plugin catalog 与 runtime 只能使用 PluginDataSource 注入的 AgentRoot/plugins", () => {
    const localDataSource = readFileSync(LOCAL_APP_DATA_SOURCE, "utf8");
    const pluginDataSourceImpl = readFileSync(PLUGIN_DATA_SOURCE_IMPL, "utf8");
    const pluginCatalog = readFileSync(PLUGIN_CATALOG, "utf8");

    expect(localDataSource).toContain(
      'let plugin_data_root = data_root.join("plugins");',
    );
    expect(pluginDataSourceImpl).toContain(
      "plugin_catalog::enabled_plugin_turn_snapshots(\n            &self.plugin_data_root,",
    );
    expect(pluginCatalog).toContain('const STORE_DIR: &str = "v3";');
    expect(pluginCatalog).toContain(
      'plugin_data_root.join("data").join(plugin_id)',
    );
    expect(pluginCatalog).toContain(
      "plugin_data_root.join(STORE_DIR).join(PACKAGES_DIR)",
    );
    for (const source of [
      localDataSource,
      pluginDataSourceImpl,
      pluginCatalog,
    ]) {
      expect(source).not.toContain("lime_core::app_paths");
      expect(source).not.toContain("preferred_data_dir");
    }
  });

  it("HostUserData 只能派生 host config/profile，不能派生 machine asset", () => {
    const increases = findBaselineIncreases(
      collectHostUserDataChildren(),
      CURRENT_HOST_USER_DATA_CHILDREN,
    );

    expect(
      increases,
      "HostUserData 不得新增 models、connectors、plugins、sessions、runtime 等 machine asset；这些数据必须由 AppDataRoot/AgentRoot owner 提供显式路径",
    ).toEqual([]);
  });

  it("AppDataRoot machine asset 只能进入已登记的唯一 owner 目录", () => {
    const increases = findBaselineIncreases(
      collectAppDataRootChildren(),
      CURRENT_APP_DATA_ROOT_CHILDREN,
    );

    expect(
      increases,
      "AppDataRoot 不得新增 browser-connectors、第二 runtime root 或其他平行顶层目录；模型进入 models，连接器进入 connectors",
    ).toEqual([]);
  });

  it("正常启动链不得恢复 Product DB 迁移或 destructive cleanup", () => {
    expect(
      collectRetiredProductDbStartupCleanupReferences(),
      "Product DB 只使用 current AgentRoot；旧整库迁移、manifest 与 destructive cleanup 不得回到 Electron、client args、App Server startup 或 smoke env",
    ).toEqual([]);
  });

  it("core 不得恢复非模型 managed project 启动迁移", () => {
    expect(
      collectRetiredManagedProjectMigrationReferences(),
      "旧 projects、workspace 与 session path 不导入、不复制、不回填；migration_v6 及其复制 helper 均为 dead / forbidden-to-restore",
    ).toEqual([]);
  });

  it("正常数据库启动不得删除 deprecated 凭证 source", () => {
    const startupMigrations = readFileSync(CORE_STARTUP_MIGRATIONS, "utf8");
    const apiKeyMigration = readFileSync(CORE_API_KEY_MIGRATION, "utf8");

    expect(startupMigrations).not.toContain(
      "DELETE FROM provider_pool_credentials",
    );
    expect(startupMigrations).not.toContain(
      "remove_managed_provider_pool_credential_files",
    );
    expect(startupMigrations).not.toContain("remove_dir_all");
    expect(startupMigrations).not.toContain("preferred_data_dir");
    expect(apiKeyMigration).not.toContain("cleanup_legacy_api_key_credentials");
    expect(apiKeyMigration).not.toContain(
      "DELETE FROM provider_pool_credentials",
    );
  });

  it("current 存储迁移不得恢复时间戳 marker", () => {
    expect(
      collectRetiredTimestampMigrationMarkerReferences(),
      "旧 .migration_completed 已退役；Product DB current owner 不写任何旧整库迁移 marker 或 manifest",
    ).toEqual([]);
  });
});
