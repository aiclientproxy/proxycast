use std::fs;
use std::path::{Path, PathBuf};

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

pub fn preferred_data_dir() -> Result<PathBuf, String> {
    Ok(preferred_data_parent_dir()?.join(APP_DATA_DIR_NAME))
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

/// Product DB 只能落在 current AgentRoot。旧库不迁移、不复制、不写 migration manifest；
/// 旧路径只进入 exact-path cleanup inventory。
pub fn resolve_database_path() -> Result<PathBuf, String> {
    resolve_database_path_for_data_dir(preferred_agent_root()?)
}

pub fn resolve_database_path_for_data_dir(data_dir: impl AsRef<Path>) -> Result<PathBuf, String> {
    Ok(data_dir.as_ref().join(DATABASE_FILE_NAME))
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

pub fn resolve_codex_agents_path() -> Option<PathBuf> {
    resolve_codex_home_dir().map(|home| resolve_codex_agents_path_from_home(&home))
}

fn resolve_codex_agents_path_from_home(home: &Path) -> PathBuf {
    home.join(USER_MEMORY_FILE_NAME)
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
    resolve_default_project_dir_from_source_roots(&preferred_data_dir()?)
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

fn resolve_runtime_subdir(subdir: &str) -> Result<PathBuf, String> {
    resolve_subdir_under_root(&preferred_data_dir()?, subdir)
}

fn resolve_home_skills_dir() -> Result<PathBuf, String> {
    resolve_subdir_under_root(&user_home_dir()?, "skills")
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

fn resolve_default_project_dir_from_source_roots(preferred_root: &Path) -> Result<PathBuf, String> {
    let default_dir = resolve_subdir_under_root(preferred_root, "projects")?.join("default");
    fs::create_dir_all(&default_dir)
        .map_err(|e| format!("无法创建默认项目目录 {}: {e}", default_dir.display()))?;
    Ok(default_dir)
}

/// 只解析并创建 current 目录。旧根内容不迁移：非模型数据零 import/copy/backfill，
/// 旧目录只进入 exact-path cleanup inventory。
fn resolve_subdir_under_root(preferred_root: &Path, subdir: &str) -> Result<PathBuf, String> {
    let preferred_dir = preferred_root.join(subdir);
    fs::create_dir_all(&preferred_dir)
        .map_err(|e| format!("无法创建目录 {}: {e}", preferred_dir.display()))?;
    Ok(preferred_dir)
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
    fn database_path_resolution_does_not_create_directories() {
        let temp = tempdir().unwrap();
        let data_root = temp.path().join("portable").join(APP_SERVER_DATA_DIR_NAME);

        let resolved = resolve_database_path_for_data_dir(&data_root).expect("database path");

        assert_eq!(resolved, data_root.join(DATABASE_FILE_NAME));
        assert!(!data_root.exists());
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
    fn resolve_subdir_preserves_current_root_contents() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let nested = preferred_root.join("logs").join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("lime.log"), "current log").unwrap();

        let resolved = resolve_subdir_under_root(&preferred_root, "logs").unwrap();

        assert_eq!(resolved, preferred_root.join("logs"));
        assert_eq!(
            fs::read_to_string(nested.join("lime.log")).unwrap(),
            "current log"
        );
    }

    #[test]
    fn resolve_projects_dir_does_not_copy_legacy_project_directories() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        let legacy_project_dir = legacy_root.join("projects").join("legacy-project");
        fs::create_dir_all(&legacy_project_dir).unwrap();
        fs::write(legacy_project_dir.join("note.md"), "legacy project").unwrap();

        let resolved = resolve_subdir_under_root(&preferred_root, "projects").unwrap();

        assert_eq!(resolved, preferred_root.join("projects"));
        assert!(!resolved.join("legacy-project").exists());
        assert!(legacy_project_dir.join("note.md").is_file());
    }

    #[test]
    fn resolve_skills_dir_does_not_copy_legacy_skill_directories() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");
        let legacy_root = temp.path().join("home").join(".lime");
        let legacy_skill_dir = legacy_root.join("skills").join("legacy-skill");
        fs::create_dir_all(&legacy_skill_dir).unwrap();
        fs::write(legacy_skill_dir.join("SKILL.md"), "legacy skill").unwrap();

        let resolved = resolve_subdir_under_root(&preferred_root, "skills").unwrap();

        assert_eq!(resolved, preferred_root.join("skills"));
        assert!(!resolved.join("legacy-skill").exists());
        assert!(legacy_skill_dir.join("SKILL.md").is_file());
    }

    #[test]
    fn resolve_home_skills_dir_uses_current_home_without_legacy_copy() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("home").join(".lime");
        let legacy_root = temp.path().join("appdata").join("lime");
        let legacy_skill_dir = legacy_root.join("skills").join("legacy-skill");
        fs::create_dir_all(&legacy_skill_dir).unwrap();
        fs::write(legacy_skill_dir.join("SKILL.md"), "legacy skill").unwrap();

        let resolved = resolve_subdir_under_root(&preferred_root, "skills").unwrap();

        assert_eq!(resolved, preferred_root.join("skills"));
        assert!(!resolved.join("legacy-skill").exists());
        assert!(legacy_skill_dir.join("SKILL.md").is_file());
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
    fn resolve_codex_agents_path_uses_codex_home_namespace() {
        let home = Path::new("/tmp/home/.codex");
        assert_eq!(
            resolve_codex_agents_path_from_home(home),
            home.join(USER_MEMORY_FILE_NAME)
        );
    }

    #[test]
    fn resolve_default_project_dir_creates_default_subdirectory() {
        let temp = tempdir().unwrap();
        let preferred_root = temp.path().join("appdata").join("lime");

        let resolved = resolve_default_project_dir_from_source_roots(&preferred_root).unwrap();

        assert_eq!(resolved, preferred_root.join("projects").join("default"));
        assert!(resolved.exists());
        assert!(resolved.is_dir());
    }

    #[test]
    fn fallback_runtime_subdir_uses_lime_temp_namespace() {
        let fallback = fallback_runtime_subdir("logs");
        assert!(fallback.ends_with(Path::new(APP_DATA_DIR_NAME).join("logs")));
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
