#[cfg(test)]
use rusqlite::Connection;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
use crate::database_path_migration::sqlite_sidecar_path;
use crate::database_path_migration::{
    database_snapshot, inspect_database_signal, inspect_migration_source_database_signal,
    migrate_legacy_database, DatabaseSignal,
};
#[cfg(test)]
use crate::migration_manifest::MIGRATION_MANIFEST_FILE_NAME;
use crate::migration_manifest::{
    completed_manifest_exists, now_timestamp, write_completed_manifest, DatabaseMigrationMode,
    DatabaseMigrationRecord,
};
use crate::product_db_migration_cleanup::remove_database_with_sidecars;

const APP_DATA_DIR_NAME: &str = "lime";
#[cfg(target_os = "windows")]
const WINDOWS_COMPANY_DIR_NAME: &str = "LimeCloud";
const LEGACY_APP_DATA_DIR_NAME: &str = "proxycast";
const LEGACY_HOME_DIR_NAME: &str = ".proxycast";
const USER_HOME_DIR_NAME: &str = ".lime";
const APP_SERVER_DATA_DIR_NAME: &str = "app-server";
const AGENT_RUNTIME_OVERRIDE_ENV: &str = "LIME_AGENT_RUNTIME_ROOT";
const DATABASE_FILE_NAME: &str = "lime.db";
const LEGACY_DATABASE_FILE_NAME: &str = "proxycast.db";
const LEGACY_PRODUCT_DATABASE_FILE_NAME: &str = "app.db";
const USER_MEMORY_FILE_NAME: &str = "AGENTS.md";
const CODEX_HOME_ENV: &str = "CODEX_HOME";
const CODEX_HOME_DIR_NAME: &str = ".codex";
const WORKSPACE_RUNTIME_DIR_NAME: &str = ".lime";
const SKILL_PROVIDER_DIRS: &[&str] = &[
    ".agents", ".warp", ".claude", ".codex", ".cursor", ".gemini", ".copilot", ".factory",
    ".github",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabasePathResolution {
    pub database_path: PathBuf,
    pub migrated_from: Option<PathBuf>,
}

pub fn preferred_data_dir() -> Result<PathBuf, String> {
    let dir = preferred_data_parent_dir()?.join(APP_DATA_DIR_NAME);
    fs::create_dir_all(&dir).map_err(|e| format!("无法创建应用数据目录 {}: {e}", dir.display()))?;
    Ok(dir)
}

pub fn legacy_home_dir() -> Result<PathBuf, String> {
    Ok(dirs::home_dir()
        .ok_or_else(|| "无法获取主目录".to_string())?
        .join(LEGACY_HOME_DIR_NAME))
}

fn legacy_app_data_dir() -> Result<PathBuf, String> {
    Ok(roaming_data_parent_dir()?.join(LEGACY_APP_DATA_DIR_NAME))
}

pub fn user_home_dir() -> Result<PathBuf, String> {
    Ok(dirs::home_dir()
        .ok_or_else(|| "无法获取主目录".to_string())?
        .join(USER_HOME_DIR_NAME))
}

pub fn preferred_agent_root() -> Result<PathBuf, String> {
    let app_data_root = preferred_data_parent_dir()?.join(APP_DATA_DIR_NAME);
    Ok(resolve_agent_root_from_app_data_root(
        &app_data_root,
        resolve_agent_dir_override(),
    ))
}

fn platform_default_agent_root() -> Result<PathBuf, String> {
    let app_data_root = preferred_data_parent_dir()?.join(APP_DATA_DIR_NAME);
    Ok(resolve_agent_root_from_app_data_root(&app_data_root, None))
}

pub fn preferred_database_path() -> Result<PathBuf, String> {
    Ok(preferred_agent_root()?.join(DATABASE_FILE_NAME))
}

pub fn legacy_database_path() -> Result<PathBuf, String> {
    Ok(legacy_home_dir()?.join(LEGACY_DATABASE_FILE_NAME))
}

/// 返回同一 Lime 产品边界内可用于模型控制面迁移的旧数据库候选。
///
/// 候选只负责定位；调用方仍需排除 current target，并按模型控制面信号选择 source。
/// 显式/E2E AgentRoot 只检查自身及 parent，不会扫描 ambient 用户目录。
pub fn model_control_migration_source_paths(data_root: &Path) -> Vec<PathBuf> {
    let mut roots = explicit_data_dir_migration_source_roots(data_root);
    push_unique_root(&mut roots, data_root.to_path_buf());

    let mut candidates = Vec::new();
    for root in roots {
        for file_name in [
            DATABASE_FILE_NAME,
            LEGACY_DATABASE_FILE_NAME,
            LEGACY_PRODUCT_DATABASE_FILE_NAME,
        ] {
            push_unique_root(&mut candidates, root.join(file_name));
        }
    }
    candidates
}

pub fn resolve_database_path() -> Result<PathBuf, String> {
    resolve_database_path_with_migration().map(|resolution| resolution.database_path)
}

pub fn resolve_database_path_for_data_dir(data_dir: impl AsRef<Path>) -> Result<PathBuf, String> {
    resolve_database_path_for_data_dir_with_migration(data_dir)
        .map(|resolution| resolution.database_path)
}

pub fn resolve_database_path_with_migration() -> Result<DatabasePathResolution, String> {
    let preferred_root = preferred_agent_root()?;
    let legacy_roots = database_migration_source_roots()?;
    resolve_database_path_from_source_roots(&preferred_root, &legacy_roots)
}

pub fn resolve_database_path_for_data_dir_with_migration(
    data_dir: impl AsRef<Path>,
) -> Result<DatabasePathResolution, String> {
    let data_dir = data_dir.as_ref();
    if let Some(resolved) = resolve_database_path_from_explicit_data_dir_parent(data_dir)? {
        return Ok(resolved);
    }
    let legacy_roots = explicit_data_dir_migration_source_roots(data_dir);
    resolve_database_path_from_source_roots(data_dir, &legacy_roots)
}

pub fn resolve_request_logs_dir() -> Result<PathBuf, String> {
    resolve_runtime_subdir("request_logs")
}

pub fn resolve_projects_dir() -> Result<PathBuf, String> {
    resolve_runtime_subdir("projects")
}

pub fn resolve_skills_dir() -> Result<PathBuf, String> {
    resolve_home_skills_dir()
}

pub fn resolve_project_skills_dir() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|cwd| resolve_project_skills_dir_from_cwd(&cwd))
}

pub fn resolve_lime_project_skill_roots() -> Vec<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|cwd| resolve_project_skill_roots(&cwd))
        .unwrap_or_default()
}

pub fn resolve_project_skill_roots(base: &Path) -> Vec<PathBuf> {
    resolve_provider_skill_roots_from_base(base)
}

pub fn resolve_user_agents_skills_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|home| resolve_user_agents_skills_dir_from_home(&home))
}

pub fn resolve_lime_user_skill_roots() -> Vec<PathBuf> {
    dirs::home_dir()
        .map(|home| resolve_user_skill_roots_from_home(&home))
        .unwrap_or_default()
}

pub fn resolve_lime_skill_roots() -> Result<Vec<PathBuf>, String> {
    let mut roots = Vec::new();
    for project_dir in resolve_lime_project_skill_roots() {
        push_unique_root(&mut roots, project_dir);
    }
    for user_dir in resolve_lime_user_skill_roots() {
        push_unique_root(&mut roots, user_dir);
    }
    push_unique_root(&mut roots, resolve_skills_dir()?);
    Ok(roots)
}

pub fn resolve_codex_home_dir() -> Option<PathBuf> {
    std::env::var_os(CODEX_HOME_ENV)
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(CODEX_HOME_DIR_NAME)))
}

pub fn resolve_workspace_runtime_agents_path(working_dir: &Path) -> PathBuf {
    working_dir
        .join(WORKSPACE_RUNTIME_DIR_NAME)
        .join(USER_MEMORY_FILE_NAME)
}

pub fn resolve_user_memory_path() -> Result<PathBuf, String> {
    Ok(user_home_dir()?.join(USER_MEMORY_FILE_NAME))
}

pub fn resolve_default_project_dir() -> Result<PathBuf, String> {
    with_app_roots(resolve_default_project_dir_from_source_roots)
}

pub fn migrate_managed_project_path_to_preferred(path: &Path) -> Result<Option<PathBuf>, String> {
    with_app_roots(|preferred_root, legacy_roots| {
        migrate_managed_project_path_to_preferred_from_source_roots(
            path,
            preferred_root,
            legacy_roots,
        )
    })
}

pub fn best_effort_runtime_subdir(subdir: &str) -> PathBuf {
    resolve_runtime_subdir(subdir).unwrap_or_else(|_| fallback_runtime_subdir(subdir))
}

pub fn best_effort_data_dir() -> PathBuf {
    preferred_data_dir().unwrap_or_else(|_| fallback_app_data_dir())
}

pub fn best_effort_app_data_file(file_name: &str) -> PathBuf {
    best_effort_data_dir().join(file_name)
}

pub fn migrate_legacy_install_data() -> Result<(), String> {
    let _ = resolve_database_path()?;

    for (label, action) in [
        (
            "request_logs",
            resolve_request_logs_dir as fn() -> Result<PathBuf, String>,
        ),
        (
            "projects",
            resolve_projects_dir as fn() -> Result<PathBuf, String>,
        ),
        (
            "skills",
            resolve_skills_dir as fn() -> Result<PathBuf, String>,
        ),
    ] {
        if let Err(error) = action() {
            tracing::warn!("[路径迁移] 启动迁移 {} 失败: {}", label, error);
        }
    }

    Ok(())
}

fn preferred_data_parent_dir() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    {
        return dirs::data_local_dir()
            .map(|root| root.join(WINDOWS_COMPANY_DIR_NAME))
            .ok_or_else(|| "无法获取本地应用数据目录".to_string());
    }

    #[cfg(not(target_os = "windows"))]
    {
        roaming_data_parent_dir()
    }
}

fn roaming_data_parent_dir() -> Result<PathBuf, String> {
    dirs::data_dir().ok_or_else(|| "无法获取应用数据目录".to_string())
}

#[cfg(target_os = "windows")]
fn legacy_windows_roaming_app_data_dir() -> Result<PathBuf, String> {
    Ok(roaming_data_parent_dir()?.join(APP_DATA_DIR_NAME))
}

#[cfg(target_os = "windows")]
fn windows_squirrel_install_root() -> Result<PathBuf, String> {
    Ok(dirs::data_local_dir()
        .ok_or_else(|| "无法获取本地应用数据目录".to_string())?
        .join(APP_DATA_DIR_NAME))
}

fn with_app_roots<T>(
    resolver: impl FnOnce(&Path, &[PathBuf]) -> Result<T, String>,
) -> Result<T, String> {
    let preferred_root = preferred_data_dir()?;
    let legacy_roots = migration_source_roots()?;
    resolver(&preferred_root, &legacy_roots)
}

fn resolve_runtime_subdir(subdir: &str) -> Result<PathBuf, String> {
    with_app_roots(|preferred_root, legacy_roots| {
        resolve_subdir_with_legacy_copy_from_source_roots(preferred_root, legacy_roots, subdir)
    })
}

fn resolve_home_skills_dir() -> Result<PathBuf, String> {
    let preferred_root = user_home_dir()?;
    let mut legacy_roots = Vec::new();

    #[cfg(target_os = "windows")]
    push_unique_root(&mut legacy_roots, legacy_windows_roaming_app_data_dir()?);

    push_unique_root(&mut legacy_roots, preferred_data_dir()?);
    push_unique_root(&mut legacy_roots, legacy_app_data_dir()?);
    push_unique_root(&mut legacy_roots, legacy_home_dir()?);

    resolve_subdir_with_legacy_copy_from_source_roots(&preferred_root, &legacy_roots, "skills")
}

#[cfg(test)]
fn resolve_home_skills_dir_from_roots(
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    resolve_subdir_with_legacy_copy_from_source_roots(preferred_root, legacy_roots, "skills")
}

fn fallback_runtime_subdir(subdir: &str) -> PathBuf {
    fallback_app_data_dir().join(subdir)
}

fn resolve_agent_dir_override() -> Option<PathBuf> {
    std::env::var(AGENT_RUNTIME_OVERRIDE_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn resolve_agent_root_from_app_data_root(
    app_data_root: &Path,
    override_root: Option<PathBuf>,
) -> PathBuf {
    override_root.unwrap_or_else(|| app_data_root.join(APP_SERVER_DATA_DIR_NAME))
}

fn resolve_project_skills_dir_from_cwd(cwd: &Path) -> PathBuf {
    cwd.join(".agents").join("skills")
}

fn resolve_user_agents_skills_dir_from_home(home: &Path) -> PathBuf {
    home.join(".agents").join("skills")
}

fn resolve_user_skill_roots_from_home(home: &Path) -> Vec<PathBuf> {
    resolve_provider_skill_roots_from_base(home)
}

fn resolve_provider_skill_roots_from_base(base: &Path) -> Vec<PathBuf> {
    SKILL_PROVIDER_DIRS
        .iter()
        .map(|provider_dir| base.join(provider_dir).join("skills"))
        .collect()
}

fn fallback_app_data_dir() -> PathBuf {
    std::env::temp_dir().join(APP_DATA_DIR_NAME)
}

fn migration_source_roots() -> Result<Vec<PathBuf>, String> {
    let mut roots = Vec::new();

    push_unique_root(&mut roots, preferred_data_dir()?);

    #[cfg(target_os = "windows")]
    push_unique_root(&mut roots, legacy_windows_roaming_app_data_dir()?);

    push_unique_root(&mut roots, legacy_app_data_dir()?);
    push_unique_root(&mut roots, legacy_home_dir()?);
    push_unique_root(&mut roots, user_home_dir()?);

    Ok(roots)
}

fn database_migration_source_roots() -> Result<Vec<PathBuf>, String> {
    let recursive_roots = migration_source_roots()?;
    #[cfg(target_os = "windows")]
    let exact_only_roots = vec![windows_squirrel_install_root()?];
    #[cfg(not(target_os = "windows"))]
    let exact_only_roots = Vec::new();

    Ok(expand_database_migration_source_roots(
        recursive_roots,
        &exact_only_roots,
    ))
}

fn expand_database_migration_source_roots(
    recursive_roots: Vec<PathBuf>,
    exact_only_roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut roots = recursive_roots;
    for root in exact_only_roots {
        push_unique_root(&mut roots, root.clone());
    }
    for root in roots.clone() {
        push_unique_root(&mut roots, root.join(APP_SERVER_DATA_DIR_NAME));
    }
    roots
}

fn explicit_data_dir_migration_source_roots(data_dir: &Path) -> Vec<PathBuf> {
    let platform_default_root = platform_default_agent_root().ok();
    let is_platform_default_root = platform_default_root
        .as_deref()
        .is_some_and(|default_root| default_root == data_dir);
    let mut platform_migration_roots = Vec::new();

    if is_platform_default_root {
        if let Ok(preferred_root) = preferred_data_dir() {
            push_unique_root(&mut platform_migration_roots, preferred_root);
        }

        if let Ok(source_roots) = database_migration_source_roots() {
            for root in source_roots {
                push_unique_root(&mut platform_migration_roots, root);
            }
        }
    }

    explicit_data_dir_migration_source_roots_from_roots(
        data_dir,
        platform_default_root.as_deref(),
        &platform_migration_roots,
    )
}

fn explicit_data_dir_migration_source_roots_from_roots(
    data_dir: &Path,
    platform_default_root: Option<&Path>,
    platform_migration_roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if data_dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == APP_SERVER_DATA_DIR_NAME)
    {
        if let Some(parent) = data_dir.parent() {
            push_unique_root_if_different(&mut roots, parent.to_path_buf(), data_dir);
        }
    }

    if platform_default_root.is_some_and(|default_root| default_root == data_dir) {
        for root in platform_migration_roots {
            push_unique_root_if_different(&mut roots, root.clone(), data_dir);
        }
    }

    roots
}

fn resolve_database_path_from_explicit_data_dir_parent(
    data_dir: &Path,
) -> Result<Option<DatabasePathResolution>, String> {
    if !data_dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == APP_SERVER_DATA_DIR_NAME)
    {
        return Ok(None);
    }
    if completed_manifest_exists(data_dir)? {
        return Ok(None);
    }

    let Some(parent) = data_dir.parent() else {
        return Ok(None);
    };
    let legacy_path = parent.join(DATABASE_FILE_NAME);
    let legacy_signal = inspect_migration_source_database_signal(&legacy_path)?;
    let preferred_path = data_dir.join(DATABASE_FILE_NAME);
    let preferred_signal = inspect_database_signal(&preferred_path);
    if !should_replace_preferred_with_legacy(
        &preferred_path,
        preferred_signal.as_ref(),
        &legacy_path,
        legacy_signal.as_ref(),
    ) {
        return Ok(None);
    }

    migrate_database_to_preferred(&legacy_path, &preferred_path, data_dir).map(Some)
}

fn push_unique_root(roots: &mut Vec<PathBuf>, root: PathBuf) {
    if !roots.iter().any(|existing| existing == &root) {
        roots.push(root);
    }
}

fn push_unique_root_if_different(roots: &mut Vec<PathBuf>, root: PathBuf, current_root: &Path) {
    if root != current_root {
        push_unique_root(roots, root);
    }
}

fn resolve_default_project_dir_from_source_roots(
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    let default_dir = resolve_subdir_with_legacy_copy_from_source_roots(
        preferred_root,
        legacy_roots,
        "projects",
    )?
    .join("default");
    fs::create_dir_all(&default_dir)
        .map_err(|e| format!("无法创建默认项目目录 {}: {e}", default_dir.display()))?;
    Ok(default_dir)
}

#[cfg(test)]
fn resolve_default_project_dir_from_roots(
    preferred_root: &Path,
    legacy_root: &Path,
) -> Result<PathBuf, String> {
    resolve_default_project_dir_from_source_roots(preferred_root, &[legacy_root.to_path_buf()])
}

fn resolve_database_path_from_source_roots(
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<DatabasePathResolution, String> {
    fs::create_dir_all(preferred_root)
        .map_err(|e| format!("无法创建数据库目录 {}: {e}", preferred_root.display()))?;

    let preferred_path = preferred_root.join(DATABASE_FILE_NAME);
    if completed_manifest_exists(preferred_root)? {
        return Ok(DatabasePathResolution {
            database_path: preferred_path,
            migrated_from: None,
        });
    }
    let preferred_signal = inspect_database_signal(&preferred_path);
    let legacy_path = select_best_legacy_database_candidate(preferred_root, legacy_roots)?;

    if let Some(legacy_path) = legacy_path.as_ref() {
        let legacy_signal = inspect_migration_source_database_signal(legacy_path)?;
        let should_migrate = should_replace_preferred_with_legacy(
            preferred_path.as_path(),
            preferred_signal.as_ref(),
            legacy_path.as_path(),
            legacy_signal.as_ref(),
        );

        if should_migrate {
            return migrate_database_to_preferred(legacy_path, &preferred_path, preferred_root);
        }

        write_database_migration_manifest(
            preferred_root,
            DatabaseMigrationMode::AdoptedExisting,
            Some(legacy_path),
            legacy_signal.as_ref(),
            &preferred_path,
            preferred_signal.as_ref(),
        )?;
        return Ok(DatabasePathResolution {
            database_path: preferred_path,
            migrated_from: None,
        });
    }

    let mode = if preferred_path.exists() {
        DatabaseMigrationMode::AdoptedExisting
    } else {
        DatabaseMigrationMode::FreshInstall
    };
    write_database_migration_manifest(
        preferred_root,
        mode,
        None,
        None,
        &preferred_path,
        preferred_signal.as_ref(),
    )?;
    Ok(DatabasePathResolution {
        database_path: preferred_path,
        migrated_from: None,
    })
}

#[cfg(test)]
fn resolve_database_path_from_roots(
    preferred_root: &Path,
    legacy_root: &Path,
) -> Result<PathBuf, String> {
    resolve_database_path_from_source_roots(preferred_root, &[legacy_root.to_path_buf()])
        .map(|resolution| resolution.database_path)
}

fn migrate_database_to_preferred(
    legacy_path: &Path,
    preferred_path: &Path,
    preferred_root: &Path,
) -> Result<DatabasePathResolution, String> {
    let outcome = migrate_legacy_database(legacy_path, preferred_path).map_err(|error| {
        format!(
            "数据库迁移失败，拒绝回退旧路径 {}: {error}",
            legacy_path.display()
        )
    })?;
    let record = DatabaseMigrationRecord {
        mode: DatabaseMigrationMode::Copied,
        source_path: Some(legacy_path.to_path_buf()),
        source: Some(outcome.source),
        target_path: preferred_path.to_path_buf(),
        target: outcome.target,
        started_at: outcome.started_at,
        verified_at: outcome.verified_at,
        cutover_at: outcome.cutover_at,
    };
    if let Err(manifest_error) = write_completed_manifest(preferred_root, record) {
        let rollback = remove_database_with_sidecars(preferred_path);
        return match rollback {
            Ok(_) => Err(format!(
                "数据库迁移 manifest 写入失败，已回滚本次目标 {}: {manifest_error}",
                preferred_path.display()
            )),
            Err(rollback_error) => Err(format!(
                "数据库迁移 manifest 写入失败且目标回滚失败 {}: manifest={manifest_error}; rollback={rollback_error}",
                preferred_path.display()
            )),
        };
    }
    tracing::info!(
        "[路径迁移] 数据库已从旧路径迁移到 {}",
        preferred_path.display()
    );
    Ok(DatabasePathResolution {
        database_path: preferred_path.to_path_buf(),
        migrated_from: Some(legacy_path.to_path_buf()),
    })
}

fn write_database_migration_manifest(
    preferred_root: &Path,
    mode: DatabaseMigrationMode,
    source_path: Option<&Path>,
    source_signal: Option<&DatabaseSignal>,
    target_path: &Path,
    target_signal: Option<&DatabaseSignal>,
) -> Result<(), String> {
    let timestamp = now_timestamp();
    write_completed_manifest(
        preferred_root,
        DatabaseMigrationRecord {
            mode,
            source_path: source_path.map(Path::to_path_buf),
            source: match source_path {
                Some(path) => Some(database_snapshot(path, source_signal)?),
                None => None,
            },
            target_path: target_path.to_path_buf(),
            target: database_snapshot(target_path, target_signal)?,
            started_at: timestamp.clone(),
            verified_at: timestamp.clone(),
            cutover_at: timestamp,
        },
    )
}

fn resolve_subdir_with_legacy_copy_from_source_roots(
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
    subdir: &str,
) -> Result<PathBuf, String> {
    let preferred_dir = preferred_root.join(subdir);
    fs::create_dir_all(&preferred_dir)
        .map_err(|e| format!("无法创建目录 {}: {e}", preferred_dir.display()))?;

    if completed_manifest_exists(preferred_root)? && dir_has_entries(&preferred_dir) {
        return Ok(preferred_dir);
    }

    for legacy_root in legacy_roots {
        if legacy_root == preferred_root {
            continue;
        }
        let legacy_dir = legacy_root.join(subdir);
        if legacy_dir.exists() {
            copy_dir_contents_if_missing(&legacy_dir, &preferred_dir)?;
        }
    }

    Ok(preferred_dir)
}

#[cfg(test)]
fn resolve_subdir_with_legacy_copy_from_roots(
    preferred_root: &Path,
    legacy_root: &Path,
    subdir: &str,
) -> Result<PathBuf, String> {
    resolve_subdir_with_legacy_copy_from_source_roots(
        preferred_root,
        &[legacy_root.to_path_buf()],
        subdir,
    )
}

fn migrate_managed_project_path_to_preferred_from_source_roots(
    path: &Path,
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<Option<PathBuf>, String> {
    let preferred_projects_root = preferred_root.join("projects");
    fs::create_dir_all(&preferred_projects_root).map_err(|e| {
        format!(
            "无法创建当前项目目录 {}: {e}",
            preferred_projects_root.display()
        )
    })?;

    if let Ok(relative_path) = path.strip_prefix(&preferred_projects_root) {
        return Ok(Some(preferred_projects_root.join(relative_path)));
    }

    for legacy_root in legacy_roots {
        let legacy_projects_root = legacy_root.join("projects");
        let Ok(relative_path) = path.strip_prefix(&legacy_projects_root) else {
            continue;
        };

        let target_path = preferred_projects_root.join(relative_path);
        fs::create_dir_all(&target_path)
            .map_err(|e| format!("无法创建项目目录 {}: {e}", target_path.display()))?;

        if path.exists() && path.is_dir() {
            copy_dir_contents_if_missing(path, &target_path)?;
        }

        return Ok(Some(target_path));
    }

    Ok(None)
}

#[cfg(test)]
pub(crate) fn migrate_managed_project_path_to_preferred_from_roots(
    path: &Path,
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<Option<PathBuf>, String> {
    migrate_managed_project_path_to_preferred_from_source_roots(path, preferred_root, legacy_roots)
}

fn dir_has_entries(path: &Path) -> bool {
    fs::read_dir(path)
        .ok()
        .and_then(|mut entries| entries.next())
        .is_some()
}

fn select_best_legacy_database_candidate(
    preferred_root: &Path,
    legacy_roots: &[PathBuf],
) -> Result<Option<PathBuf>, String> {
    let preferred_path = preferred_root.join(DATABASE_FILE_NAME);
    let mut candidates = vec![preferred_root.join(LEGACY_DATABASE_FILE_NAME)];

    for legacy_root in legacy_roots {
        candidates.push(legacy_root.join(LEGACY_DATABASE_FILE_NAME));
    }
    for legacy_root in legacy_roots {
        candidates.push(legacy_root.join(DATABASE_FILE_NAME));
    }

    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped
            .iter()
            .any(|existing: &PathBuf| existing == &candidate)
        {
            deduped.push(candidate);
        }
    }

    let mut best: Option<(usize, DatabaseSignal, PathBuf)> = None;
    for (priority, candidate) in deduped.into_iter().enumerate() {
        if candidate == preferred_path {
            continue;
        }
        let Some(signal) = inspect_migration_source_database_signal(&candidate)? else {
            continue;
        };
        if !signal.has_schema() && signal.user_signal == 0 {
            continue;
        }

        let should_replace = match best.as_ref() {
            None => true,
            Some((best_priority, best_signal, _)) => {
                signal.user_signal > best_signal.user_signal
                    || (signal.user_signal == best_signal.user_signal
                        && signal.has_schema()
                        && !best_signal.has_schema())
                    || (signal.user_signal == best_signal.user_signal
                        && signal.has_schema() == best_signal.has_schema()
                        && priority < *best_priority)
            }
        };

        if should_replace {
            best = Some((priority, signal, candidate));
        }
    }

    Ok(best.map(|(_, _, path)| path))
}

fn should_replace_preferred_with_legacy(
    preferred_path: &Path,
    preferred_signal: Option<&DatabaseSignal>,
    legacy_path: &Path,
    legacy_signal: Option<&DatabaseSignal>,
) -> bool {
    let Some(legacy_signal) = legacy_signal else {
        return false;
    };

    let Some(preferred_signal) = preferred_signal else {
        return true;
    };

    if !preferred_signal.has_schema() && legacy_signal.has_schema() {
        tracing::warn!(
            "[路径迁移] 当前数据库 {} 无有效 schema，准备回退旧库 {}",
            preferred_path.display(),
            legacy_path.display()
        );
        return true;
    }

    if preferred_signal.user_signal == 0 && legacy_signal.user_signal > 0 {
        tracing::warn!(
            "[路径迁移] 当前数据库 {} 缺少用户数据，检测到旧库 {} 含历史数据，准备自动恢复",
            preferred_path.display(),
            legacy_path.display()
        );
        return true;
    }

    false
}

fn copy_dir_contents_if_missing(from: &Path, to: &Path) -> Result<(), String> {
    let entries =
        fs::read_dir(from).map_err(|e| format!("读取目录失败 {}: {e}", from.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("读取目录项失败 {}: {e}", from.display()))?;
        let source_path = entry.path();
        let target_path = to.join(entry.file_name());

        if source_path.is_dir() {
            fs::create_dir_all(&target_path)
                .map_err(|e| format!("创建目录失败 {}: {e}", target_path.display()))?;
            copy_dir_contents_if_missing(&source_path, &target_path)?;
            continue;
        }

        if target_path.exists() {
            continue;
        }

        fs::copy(&source_path, &target_path).map_err(|e| {
            format!(
                "复制文件失败 {} -> {}: {e}",
                source_path.display(),
                target_path.display()
            )
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn preferred_agent_root_resolution_does_not_create_directories() {
        let temp = tempdir().unwrap();
        let app_data_root = temp
            .path()
            .join("application-support")
            .join(APP_DATA_DIR_NAME);

        let resolved = resolve_agent_root_from_app_data_root(&app_data_root, None);

        assert_eq!(resolved, app_data_root.join(APP_SERVER_DATA_DIR_NAME));
        assert!(!app_data_root.exists());
    }

    #[test]
    fn explicit_agent_root_override_wins_without_touching_default_root() {
        let temp = tempdir().unwrap();
        let app_data_root = temp
            .path()
            .join("application-support")
            .join(APP_DATA_DIR_NAME);
        let override_root = temp.path().join("e2e").join(APP_SERVER_DATA_DIR_NAME);

        let resolved =
            resolve_agent_root_from_app_data_root(&app_data_root, Some(override_root.clone()));

        assert_eq!(resolved, override_root);
        assert!(!app_data_root.exists());
        assert!(!override_root.exists());
    }

    #[test]
    fn explicit_agent_root_override_does_not_expand_platform_migration_sources() {
        let temp = tempdir().unwrap();
        let app_data_root = temp
            .path()
            .join("application-support")
            .join(APP_DATA_DIR_NAME);
        let platform_default_root = resolve_agent_root_from_app_data_root(&app_data_root, None);
        let override_root = temp.path().join("e2e").join(APP_SERVER_DATA_DIR_NAME);
        let preferred_root_with_override =
            resolve_agent_root_from_app_data_root(&app_data_root, Some(override_root.clone()));
        let global_legacy_root = temp.path().join("legacy-global");

        assert_eq!(preferred_root_with_override, override_root);
        let sources = explicit_data_dir_migration_source_roots_from_roots(
            &preferred_root_with_override,
            Some(&platform_default_root),
            std::slice::from_ref(&global_legacy_root),
        );

        assert_eq!(sources, vec![temp.path().join("e2e")]);
        assert!(!sources.contains(&global_legacy_root));
    }

    #[test]
    fn resolve_database_path_migrates_legacy_database() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        fs::create_dir_all(&legacy_root).unwrap();

        let legacy_db = legacy_root.join(DATABASE_FILE_NAME);
        let conn = Connection::open(&legacy_db).unwrap();
        conn.execute(
            "CREATE TABLE sample (id INTEGER PRIMARY KEY, name TEXT)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO sample (name) VALUES ('lime')", [])
            .unwrap();

        let resolved = resolve_database_path_from_roots(&preferred_root, &legacy_root).unwrap();
        assert_eq!(resolved, preferred_root.join(DATABASE_FILE_NAME));
        assert!(resolved.exists());

        let migrated = Connection::open(resolved).unwrap();
        let name: String = migrated
            .query_row("SELECT name FROM sample LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(name, "lime");
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn migrate_database_supports_uri_reserved_characters_in_source_path() {
        let temp = tempdir().unwrap();
        let source_root = temp.path().join("legacy path %#?");
        let target_root = temp.path().join("current");
        fs::create_dir_all(&source_root).unwrap();
        let source_path = source_root.join(DATABASE_FILE_NAME);
        let source = Connection::open(&source_path).unwrap();
        source
            .execute_batch(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'legacy');",
            )
            .unwrap();
        drop(source);

        let resolution = resolve_database_path_from_source_roots(
            &target_root,
            std::slice::from_ref(&source_root),
        )
        .unwrap();

        assert_eq!(resolution.migrated_from, Some(source_path));
        let migrated = Connection::open(resolution.database_path).unwrap();
        let value: String = migrated
            .query_row(
                "SELECT value FROM settings WHERE key = 'providers.active_tab'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(value, "legacy");
    }

    #[test]
    fn resolve_database_path_rejects_uncheckpointed_wal_without_touching_source() {
        let temp = tempdir().unwrap();
        let source_root = temp.path().join("legacy");
        let target_root = temp.path().join("current");
        fs::create_dir_all(&source_root).unwrap();
        let source_path = source_root.join(DATABASE_FILE_NAME);
        let target_path = target_root.join(DATABASE_FILE_NAME);
        let writer = Connection::open(&source_path).unwrap();
        writer
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA wal_autocheckpoint = 0;
                 CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'legacy');",
            )
            .unwrap();
        let wal_path = sqlite_sidecar_path(&source_path, "-wal");
        let shm_path = sqlite_sidecar_path(&source_path, "-shm");
        assert!(wal_path.is_file());
        assert!(shm_path.is_file());
        let source_before = fs::read(&source_path).unwrap();
        let wal_before = fs::read(&wal_path).unwrap();
        let shm_before = fs::read(&shm_path).unwrap();
        let source_mtime_before = fs::metadata(&source_path).unwrap().modified().unwrap();
        let wal_mtime_before = fs::metadata(&wal_path).unwrap().modified().unwrap();
        let shm_mtime_before = fs::metadata(&shm_path).unwrap().modified().unwrap();
        assert!(!wal_before.is_empty());

        let error = resolve_database_path_from_source_roots(
            &target_root,
            std::slice::from_ref(&source_root),
        )
        .expect_err("未 checkpoint 的 WAL 源必须交给显式维护流程");

        assert!(error.contains("迁移源数据库存在活动伴生文件"));
        assert!(!target_path.exists());
        assert!(!target_root.join(MIGRATION_MANIFEST_FILE_NAME).exists());
        assert_eq!(fs::read(&source_path).unwrap(), source_before);
        assert_eq!(fs::read(&wal_path).unwrap(), wal_before);
        assert_eq!(fs::read(&shm_path).unwrap(), shm_before);
        assert_eq!(
            fs::metadata(&source_path).unwrap().modified().unwrap(),
            source_mtime_before
        );
        assert_eq!(
            fs::metadata(&wal_path).unwrap().modified().unwrap(),
            wal_mtime_before
        );
        assert_eq!(
            fs::metadata(&shm_path).unwrap().modified().unwrap(),
            shm_mtime_before
        );
        let value: String = writer
            .query_row(
                "SELECT value FROM settings WHERE key = 'providers.active_tab'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(value, "legacy");
        drop(writer);
    }

    #[test]
    fn migrate_legacy_database_fails_closed_when_target_sidecars_exist() {
        let temp = tempdir().unwrap();
        let source_root = temp.path().join("legacy");
        let target_root = temp.path().join("current");
        fs::create_dir_all(&source_root).unwrap();
        fs::create_dir_all(&target_root).unwrap();
        let source_path = source_root.join(DATABASE_FILE_NAME);
        let target_path = target_root.join(DATABASE_FILE_NAME);
        let source = Connection::open(&source_path).unwrap();
        source
            .execute_batch(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'legacy');",
            )
            .unwrap();
        drop(source);
        let target_wal = sqlite_sidecar_path(&target_path, "-wal");
        let target_shm = sqlite_sidecar_path(&target_path, "-shm");
        fs::write(&target_wal, b"existing wal").unwrap();
        fs::write(&target_shm, b"existing shm").unwrap();

        let error = migrate_legacy_database(&source_path, &target_path)
            .expect_err("目标伴生文件存在时必须拒绝迁移");

        assert!(error.contains("目标数据库已存在，拒绝覆盖"));
        assert!(!target_path.exists());
        assert_eq!(fs::read(&target_wal).unwrap(), b"existing wal");
        assert_eq!(fs::read(&target_shm).unwrap(), b"existing shm");
        assert!(!fs::read_dir(&target_root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("migration-staging")
        }));
    }

    #[test]
    fn exact_database_source_does_not_modify_or_expand_install_root() {
        let temp = tempdir().unwrap();
        let recursive_root = temp.path().join("roaming").join(APP_DATA_DIR_NAME);
        let install_root = temp.path().join("local").join(APP_DATA_DIR_NAME);
        let package_root = install_root.join("packages").join("current");
        fs::create_dir_all(&package_root).unwrap();
        fs::write(package_root.join("Lime.exe"), b"installer payload").unwrap();

        let source_path = install_root.join(DATABASE_FILE_NAME);
        let source = Connection::open(&source_path).unwrap();
        source
            .execute_batch(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'legacy');",
            )
            .unwrap();
        drop(source);
        let source_before = fs::read(&source_path).unwrap();
        let install_mtime_before = fs::metadata(&install_root).unwrap().modified().unwrap();

        let recursive_roots = vec![recursive_root.clone()];
        let database_roots = expand_database_migration_source_roots(
            recursive_roots.clone(),
            std::slice::from_ref(&install_root),
        );
        assert_eq!(recursive_roots, vec![recursive_root]);
        assert!(database_roots.contains(&install_root));
        assert!(database_roots.contains(&install_root.join(APP_SERVER_DATA_DIR_NAME)));

        let target_root = temp.path().join("current").join(APP_SERVER_DATA_DIR_NAME);
        let resolution =
            resolve_database_path_from_source_roots(&target_root, &database_roots).unwrap();

        assert_eq!(resolution.migrated_from, Some(source_path.clone()));
        assert_eq!(fs::read(&source_path).unwrap(), source_before);
        assert_eq!(
            fs::metadata(&install_root).unwrap().modified().unwrap(),
            install_mtime_before
        );
        assert!(!install_root.join(MIGRATION_MANIFEST_FILE_NAME).exists());
        assert!(!install_root.join(APP_SERVER_DATA_DIR_NAME).exists());
        assert!(!target_root.join("packages").exists());
        assert_eq!(
            fs::read(package_root.join("Lime.exe")).unwrap(),
            b"installer payload"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_recursive_migration_sources_exclude_squirrel_install_root() {
        let install_root = windows_squirrel_install_root().unwrap();
        let recursive_roots = migration_source_roots().unwrap();
        let database_roots = database_migration_source_roots().unwrap();

        assert!(!recursive_roots.contains(&install_root));
        assert!(database_roots.contains(&install_root));
        assert!(database_roots.contains(&install_root.join(APP_SERVER_DATA_DIR_NAME)));
    }

    #[test]
    fn resolve_subdir_skips_same_root_as_legacy_source() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let nested = preferred_root.join("logs").join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("lime.log"), "current log").unwrap();

        let resolved = resolve_subdir_with_legacy_copy_from_source_roots(
            &preferred_root,
            std::slice::from_ref(&preferred_root),
            "logs",
        )
        .unwrap();

        assert_eq!(resolved, preferred_root.join("logs"));
        assert_eq!(
            fs::read_to_string(nested.join("lime.log")).unwrap(),
            "current log"
        );
    }

    #[test]
    fn resolve_projects_dir_copies_legacy_project_directories() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        let legacy_project_dir = legacy_root.join("projects").join("legacy-project");
        fs::create_dir_all(&legacy_project_dir).unwrap();
        fs::write(legacy_project_dir.join("note.md"), "legacy project").unwrap();

        let resolved =
            resolve_subdir_with_legacy_copy_from_roots(&preferred_root, &legacy_root, "projects")
                .unwrap();

        assert_eq!(resolved, preferred_root.join("projects"));
        assert_eq!(
            fs::read_to_string(resolved.join("legacy-project").join("note.md")).unwrap(),
            "legacy project"
        );
    }

    #[test]
    fn resolve_skills_dir_copies_legacy_skill_directories() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        let legacy_skill_dir = legacy_root.join("skills").join("legacy-skill");
        fs::create_dir_all(&legacy_skill_dir).unwrap();
        fs::write(legacy_skill_dir.join("SKILL.md"), "legacy skill").unwrap();

        let resolved =
            resolve_subdir_with_legacy_copy_from_roots(&preferred_root, &legacy_root, "skills")
                .unwrap();

        assert_eq!(resolved, preferred_root.join("skills"));
        assert_eq!(
            fs::read_to_string(resolved.join("legacy-skill").join("SKILL.md")).unwrap(),
            "legacy skill"
        );
    }

    #[test]
    fn resolve_home_skills_dir_prefers_lime_home_skills() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("home").join(".lime");
        let legacy_root = temp.path().join("appdata").join("lime");
        let legacy_skill_dir = legacy_root.join("skills").join("legacy-skill");
        fs::create_dir_all(&legacy_skill_dir).unwrap();
        fs::write(legacy_skill_dir.join("SKILL.md"), "legacy skill").unwrap();

        let resolved = resolve_home_skills_dir_from_roots(&preferred_root, &[legacy_root]).unwrap();

        assert_eq!(resolved, preferred_root.join("skills"));
        assert_eq!(
            fs::read_to_string(resolved.join("legacy-skill").join("SKILL.md")).unwrap(),
            "legacy skill"
        );
    }

    #[test]
    fn resolve_project_skills_dir_from_cwd_builds_agents_skills_path() {
        let cwd = Path::new("/tmp/workspace");
        let resolved = resolve_project_skills_dir_from_cwd(cwd);
        assert_eq!(resolved, cwd.join(".agents").join("skills"));
    }

    #[test]
    fn resolve_user_agents_skills_dir_from_home_builds_standard_user_path() {
        let home = Path::new("/tmp/home");
        let resolved = resolve_user_agents_skills_dir_from_home(home);
        assert_eq!(resolved, home.join(".agents").join("skills"));
    }

    #[test]
    fn resolve_project_skill_roots_builds_cross_provider_roots_in_precedence_order() {
        let cwd = Path::new("/tmp/workspace");
        let resolved = resolve_project_skill_roots(cwd);

        assert_eq!(resolved.first(), Some(&cwd.join(".agents").join("skills")));
        assert!(resolved.contains(&cwd.join(".claude").join("skills")));
        assert!(resolved.contains(&cwd.join(".codex").join("skills")));
        assert!(resolved.contains(&cwd.join(".gemini").join("skills")));
    }

    #[test]
    fn resolve_user_skill_roots_from_home_builds_cross_provider_roots_in_precedence_order() {
        let home = Path::new("/tmp/home");
        let resolved = resolve_user_skill_roots_from_home(home);

        assert_eq!(resolved.first(), Some(&home.join(".agents").join("skills")));
        assert!(resolved.contains(&home.join(".claude").join("skills")));
        assert!(resolved.contains(&home.join(".codex").join("skills")));
        assert!(resolved.contains(&home.join(".gemini").join("skills")));
    }

    #[test]
    fn resolve_workspace_runtime_agents_path_builds_workspace_file_path() {
        let workspace_root = Path::new("/tmp/workspace");
        let resolved = resolve_workspace_runtime_agents_path(workspace_root);
        assert_eq!(
            resolved,
            workspace_root
                .join(WORKSPACE_RUNTIME_DIR_NAME)
                .join(USER_MEMORY_FILE_NAME)
        );
    }

    #[test]
    fn resolve_default_project_dir_creates_default_subdirectory() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");

        let resolved =
            resolve_default_project_dir_from_roots(&preferred_root, &legacy_root).unwrap();

        assert_eq!(resolved, preferred_root.join("projects").join("default"));
        assert!(resolved.exists());
        assert!(resolved.is_dir());
    }

    #[test]
    fn migrate_managed_project_path_to_preferred_remaps_proxycast_home_project() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".proxycast");
        let legacy_project = legacy_root.join("projects").join("default");
        fs::create_dir_all(&legacy_project).unwrap();
        fs::write(legacy_project.join("index.md"), "# legacy").unwrap();

        let resolved = migrate_managed_project_path_to_preferred_from_roots(
            &legacy_project,
            &preferred_root,
            &[legacy_root],
        )
        .unwrap();

        let expected = preferred_root.join("projects").join("default");
        assert_eq!(resolved, Some(expected.clone()));
        assert_eq!(
            fs::read_to_string(expected.join("index.md")).unwrap(),
            "# legacy"
        );
    }

    #[test]
    fn migrate_managed_project_path_to_preferred_ignores_custom_project() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".proxycast");
        let custom_project = temp.path().join("workspace").join("demo");

        let resolved = migrate_managed_project_path_to_preferred_from_roots(
            &custom_project,
            &preferred_root,
            &[legacy_root],
        )
        .unwrap();

        assert_eq!(resolved, None);
    }

    #[test]
    fn fallback_runtime_subdir_uses_lime_temp_namespace() {
        let fallback = fallback_runtime_subdir("logs");
        assert!(fallback.ends_with(Path::new(APP_DATA_DIR_NAME).join("logs")));
    }

    #[test]
    fn resolve_database_path_fails_closed_when_bootstrap_db_exists() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        fs::create_dir_all(&preferred_root).unwrap();
        fs::create_dir_all(&legacy_root).unwrap();

        let preferred_db = preferred_root.join(DATABASE_FILE_NAME);
        let preferred_conn = Connection::open(&preferred_db).unwrap();
        preferred_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        drop(preferred_conn);

        let legacy_db = legacy_root.join(DATABASE_FILE_NAME);
        let legacy_conn = Connection::open(&legacy_db).unwrap();
        legacy_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        legacy_conn
            .execute(
                "CREATE TABLE agent_sessions (id INTEGER PRIMARY KEY, name TEXT)",
                [],
            )
            .unwrap();
        legacy_conn
            .execute("INSERT INTO contents (title) VALUES ('legacy')", [])
            .unwrap();

        let error = resolve_database_path_from_roots(&preferred_root, &legacy_root)
            .expect_err("已有目标数据库时必须拒绝覆盖");
        assert!(error.contains("目标数据库已存在，拒绝覆盖"));
        let conn = Connection::open(preferred_db).unwrap();
        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM contents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn resolve_database_path_keeps_preferred_when_it_has_user_data() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        fs::create_dir_all(&preferred_root).unwrap();
        fs::create_dir_all(&legacy_root).unwrap();

        let preferred_db = preferred_root.join(DATABASE_FILE_NAME);
        let preferred_conn = Connection::open(&preferred_db).unwrap();
        preferred_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        preferred_conn
            .execute("INSERT INTO contents (title) VALUES ('preferred')", [])
            .unwrap();

        let legacy_db = legacy_root.join(DATABASE_FILE_NAME);
        let legacy_conn = Connection::open(&legacy_db).unwrap();
        legacy_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        legacy_conn
            .execute("INSERT INTO contents (title) VALUES ('legacy')", [])
            .unwrap();

        let resolved = resolve_database_path_from_roots(&preferred_root, &legacy_root).unwrap();
        let conn = Connection::open(resolved).unwrap();
        let title: String = conn
            .query_row("SELECT title FROM contents LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(title, "preferred");
    }

    #[test]
    fn resolve_database_path_fails_closed_when_target_is_empty_and_legacy_has_data() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        fs::create_dir_all(&preferred_root).unwrap();
        fs::create_dir_all(&legacy_root).unwrap();

        // preferred 库为空（只有 schema）
        let preferred_db = preferred_root.join(DATABASE_FILE_NAME);
        let preferred_conn = Connection::open(&preferred_db).unwrap();
        preferred_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        drop(preferred_conn);

        // 旧库有数据
        let legacy_db = legacy_root.join(DATABASE_FILE_NAME);
        let legacy_conn = Connection::open(&legacy_db).unwrap();
        legacy_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        legacy_conn
            .execute("INSERT INTO contents (title) VALUES ('legacy')", [])
            .unwrap();
        drop(legacy_conn);

        // 目标文件已创建时，启动迁移不得覆盖目标。
        let error = resolve_database_path_from_roots(&preferred_root, &legacy_root)
            .expect_err("已有目标数据库时必须拒绝覆盖");
        assert!(error.contains("目标数据库已存在，拒绝覆盖"));

        let conn = Connection::open(&preferred_db).unwrap();
        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM contents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn resolve_database_path_for_explicit_data_dir_migrates_previous_product_db() {
        let temp = tempdir().unwrap();
        let electron_user_data_root = temp.path().join("user-data");
        let app_server_root = electron_user_data_root.join(APP_SERVER_DATA_DIR_NAME);
        let previous_product_root = temp.path().join("appdata").join("lime");
        fs::create_dir_all(&previous_product_root).unwrap();

        let previous_db = previous_product_root.join(DATABASE_FILE_NAME);
        let previous_conn = Connection::open(&previous_db).unwrap();
        previous_conn
            .execute(
                "CREATE TABLE api_keys (id INTEGER PRIMARY KEY, name TEXT)",
                [],
            )
            .unwrap();
        previous_conn
            .execute("INSERT INTO api_keys (name) VALUES ('provider-key')", [])
            .unwrap();
        drop(previous_conn);

        let resolution =
            resolve_database_path_from_source_roots(&app_server_root, &[previous_product_root])
                .unwrap();
        let resolved = resolution.database_path;

        assert_eq!(resolved, app_server_root.join(DATABASE_FILE_NAME));
        assert_eq!(resolution.migrated_from, Some(previous_db));
        let conn = Connection::open(&resolved).unwrap();
        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM api_keys", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn explicit_data_dir_migration_sources_include_electron_user_data_parent() {
        let temp = tempdir().unwrap();
        let electron_user_data_root = temp.path().join("user-data");
        let app_server_root = electron_user_data_root.join(APP_SERVER_DATA_DIR_NAME);

        let sources = explicit_data_dir_migration_source_roots(&app_server_root);

        assert!(sources.contains(&electron_user_data_root));
        assert!(!sources.contains(&app_server_root));
        assert_eq!(sources, vec![electron_user_data_root]);
    }

    #[test]
    fn resolve_database_path_for_explicit_data_dir_prefers_parent_product_db() {
        let temp = tempdir().unwrap();
        let electron_user_data_root = temp.path().join("user-data");
        let app_server_root = electron_user_data_root.join(APP_SERVER_DATA_DIR_NAME);
        let other_legacy_root = temp.path().join("appdata").join("lime");
        fs::create_dir_all(&electron_user_data_root).unwrap();
        fs::create_dir_all(&other_legacy_root).unwrap();

        let parent_db = electron_user_data_root.join(DATABASE_FILE_NAME);
        let parent_conn = Connection::open(&parent_db).unwrap();
        parent_conn
            .execute(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                [],
            )
            .unwrap();
        parent_conn
            .execute(
                "INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'parent')",
                [],
            )
            .unwrap();
        drop(parent_conn);

        let other_db = other_legacy_root.join(DATABASE_FILE_NAME);
        let other_conn = Connection::open(&other_db).unwrap();
        other_conn
            .execute(
                "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
                [],
            )
            .unwrap();
        other_conn
            .execute("INSERT INTO contents (title) VALUES ('other-legacy')", [])
            .unwrap();
        drop(other_conn);

        let resolution = resolve_database_path_from_explicit_data_dir_parent(&app_server_root)
            .unwrap()
            .expect("parent migration");
        let resolved = resolution.database_path;
        assert_eq!(resolved, app_server_root.join(DATABASE_FILE_NAME));
        assert_eq!(resolution.migrated_from, Some(parent_db));

        let conn = Connection::open(&resolved).unwrap();
        let value: String = conn
            .query_row(
                "SELECT value FROM settings WHERE key = 'providers.active_tab'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(value, "parent");
    }

    #[test]
    fn resolve_database_path_fails_closed_when_target_database_cannot_be_replaced() {
        let temp = tempdir().unwrap();
        let electron_user_data_root = temp.path().join("user-data");
        let app_server_root = electron_user_data_root.join(APP_SERVER_DATA_DIR_NAME);
        fs::create_dir_all(&app_server_root).unwrap();

        let source_db = electron_user_data_root.join(DATABASE_FILE_NAME);
        let source_conn = Connection::open(&source_db).unwrap();
        source_conn
            .execute(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                [],
            )
            .unwrap();
        source_conn
            .execute(
                "INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'source')",
                [],
            )
            .unwrap();
        drop(source_conn);

        let target_db = app_server_root.join(DATABASE_FILE_NAME);
        fs::create_dir_all(&target_db).unwrap();

        let error = resolve_database_path_from_explicit_data_dir_parent(&app_server_root)
            .expect_err("目标数据库不可替换时必须阻止旧路径回退");

        assert!(error.contains("数据库迁移失败，拒绝回退旧路径"));
        assert!(error.contains(source_db.to_string_lossy().as_ref()));
        assert!(target_db.is_dir());
        assert!(!app_server_root.join(MIGRATION_MANIFEST_FILE_NAME).exists());
        let source_conn = Connection::open(&source_db).unwrap();
        let value: String = source_conn
            .query_row(
                "SELECT value FROM settings WHERE key = 'providers.active_tab'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(value, "source");
    }

    #[test]
    fn resolve_database_path_migrates_proxycast_appdata_database_on_first_lime_launch() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_appdata_root = temp.path().join("appdata").join("proxycast");
        fs::create_dir_all(&legacy_appdata_root).unwrap();

        let legacy_db = legacy_appdata_root.join(LEGACY_DATABASE_FILE_NAME);
        let conn = Connection::open(&legacy_db).unwrap();
        conn.execute(
            "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO contents (title) VALUES ('proxycast legacy')",
            [],
        )
        .unwrap();
        drop(conn);

        let resolution =
            resolve_database_path_from_source_roots(&preferred_root, &[legacy_appdata_root])
                .unwrap();
        let resolved = resolution.database_path;
        assert_eq!(resolved, preferred_root.join(DATABASE_FILE_NAME));
        assert_eq!(resolution.migrated_from, Some(legacy_db));

        let conn = Connection::open(&resolved).unwrap();
        let title: String = conn
            .query_row("SELECT title FROM contents LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(title, "proxycast legacy");
    }

    #[test]
    fn resolve_database_path_writes_versioned_manifest_after_successful_migration() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        fs::create_dir_all(&legacy_root).unwrap();

        let legacy_db = legacy_root.join(DATABASE_FILE_NAME);
        let conn = Connection::open(&legacy_db).unwrap();
        conn.execute(
            "CREATE TABLE contents (id INTEGER PRIMARY KEY, title TEXT)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO contents (title) VALUES ('data')", [])
            .unwrap();
        drop(conn);

        let manifest_path = preferred_root.join(MIGRATION_MANIFEST_FILE_NAME);
        assert!(!manifest_path.exists());

        let _ = resolve_database_path_from_roots(&preferred_root, &legacy_root).unwrap();

        assert!(manifest_path.exists());
        let manifest = fs::read_to_string(manifest_path).unwrap();
        assert!(manifest.contains("\"schemaVersion\": \"storage-migration.v1\""));
        assert!(manifest.contains("\"mode\": \"copied\""));
        assert!(manifest.contains("\"manifestSha256\""));
        assert!(manifest.contains(legacy_db.to_string_lossy().as_ref()));
        let manifest: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        assert_eq!(manifest["target"]["relativePath"], "lime.db");
        assert_eq!(manifest["cleanupAuthorizedAt"], serde_json::Value::Null);
        assert_eq!(
            manifest["source"]["snapshot"]["fingerprint"]["sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            manifest["target"]["snapshot"]["fingerprint"]["sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            manifest["source"]["snapshot"]["userSignal"],
            manifest["target"]["snapshot"]["userSignal"]
        );
        assert_eq!(
            manifest["source"]["snapshot"]["schemaObjectCount"],
            manifest["target"]["snapshot"]["schemaObjectCount"]
        );
    }

    #[test]
    fn resolve_database_path_writes_versioned_manifest_for_fresh_install() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        // 不创建 legacy_root → 模拟全新安装

        let manifest_path = preferred_root.join(MIGRATION_MANIFEST_FILE_NAME);
        assert!(!manifest_path.exists());

        let resolved = resolve_database_path_from_roots(&preferred_root, &legacy_root).unwrap();
        assert_eq!(resolved, preferred_root.join(DATABASE_FILE_NAME));

        assert!(manifest_path.exists());
        let manifest = fs::read_to_string(manifest_path).unwrap();
        assert!(manifest.contains("\"mode\": \"fresh-install\""));
        assert!(manifest.contains("\"cleanupAuthorizedAt\": null"));
    }

    #[test]
    fn model_control_candidates_for_explicit_root_stay_within_root_and_parent() {
        let temp = tempdir().unwrap();
        let data_root = temp.path().join("portable").join(APP_SERVER_DATA_DIR_NAME);
        let parent = data_root.parent().unwrap();

        let candidates = model_control_migration_source_paths(&data_root);

        assert!(candidates.contains(&parent.join(DATABASE_FILE_NAME)));
        assert!(candidates.contains(&parent.join(LEGACY_DATABASE_FILE_NAME)));
        assert!(candidates.contains(&parent.join(LEGACY_PRODUCT_DATABASE_FILE_NAME)));
        assert!(candidates.contains(&data_root.join(DATABASE_FILE_NAME)));
        assert!(candidates.contains(&data_root.join(LEGACY_DATABASE_FILE_NAME)));
        assert!(candidates.contains(&data_root.join(LEGACY_PRODUCT_DATABASE_FILE_NAME)));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.starts_with(temp.path())));
    }
}
