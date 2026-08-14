//! Agent turn 可用技能轻量快照。
//!
//! 这里只保留 metadata 和 `SKILL.md` locator，不把技能正文注入普通回合。

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::skill_summary::{
    invalidate_skill_summary_cache, load_skill_summaries_from_directory, LoadedSkillSummary,
};
use lime_core::app_paths;
use lime_core::config::SkillConfig;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillMetadata {
    pub skill_id: String,
    pub name: String,
    pub description: String,
    pub scope: AgentSkillScope,
    pub source: AgentSkillSource,
    pub authority: AgentSkillAuthority,
    pub enabled: bool,
    pub interface: AgentSkillInterface,
    pub dependencies: AgentSkillDependencies,
    pub policy: AgentSkillPolicy,
    pub capabilities: Vec<String>,
    pub directory: PathBuf,
    pub skill_file_path: PathBuf,
}

impl AgentSkillMetadata {
    /// 返回不依赖本地目录的稳定 skill identity。
    ///
    /// 本地路径只是正文读取 locator，不能作为跨 workspace/session 的 skill
    /// identity。scope 由 snapshot root 提供，name 由 skill 目录名提供。
    pub fn stable_id(&self) -> &str {
        &self.skill_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSkillSource {
    Project,
    User,
    App,
    Orchestrator,
    Other,
}

impl AgentSkillSource {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::User => "user",
            Self::App => "app",
            Self::Orchestrator => "orchestrator",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSkillAuthority {
    Workspace,
    User,
    Application,
    Orchestrator,
    External,
}

impl AgentSkillAuthority {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::User => "user",
            Self::Application => "application",
            Self::Orchestrator => "orchestrator",
            Self::External => "external",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillInterface {
    pub display_name: String,
    pub execution_mode: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub argument_hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentSkillDependencies {
    pub tools: Vec<AgentSkillToolDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillToolDependency {
    pub dependency_type: String,
    pub value: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillPolicy {
    pub allow_implicit_invocation: bool,
    pub when_to_use: Option<String>,
}

pub(crate) fn agent_skill_stable_id(scope: AgentSkillScope, name: &str) -> String {
    format!("{}:{}", scope.as_label(), name.trim().to_ascii_lowercase())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSkillScope {
    Project,
    User,
    App,
    Orchestrator,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSkillRoot {
    pub path: PathBuf,
    pub scope: AgentSkillScope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentSkillSnapshot {
    pub roots: Vec<AgentSkillRoot>,
    pub skills: Vec<AgentSkillMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentSkillSnapshotOptions {
    pub roots: Vec<AgentSkillRoot>,
}

pub fn build_agent_skill_snapshot() -> AgentSkillSnapshot {
    build_agent_skill_snapshot_from_roots(default_agent_skill_roots())
}

pub fn build_agent_skill_snapshot_from_workspace(
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> AgentSkillSnapshot {
    build_agent_skill_snapshot_from_roots(agent_skill_roots_for_workspace(
        working_dir,
        project_root,
    ))
}

pub fn build_agent_skill_snapshot_from_roots(
    roots: impl IntoIterator<Item = AgentSkillRoot>,
) -> AgentSkillSnapshot {
    let roots = roots.into_iter().collect::<Vec<_>>();
    let cache_key = snapshot_cache_key(&roots);
    if let Some(cached) = cached_snapshot(&cache_key) {
        return cached;
    }
    let snapshot = build_agent_skill_snapshot_uncached(roots);
    store_cached_snapshot(cache_key, &snapshot);
    snapshot
}

fn build_agent_skill_snapshot_uncached(
    roots: impl IntoIterator<Item = AgentSkillRoot>,
) -> AgentSkillSnapshot {
    let mut seen = HashSet::<(String, PathBuf)>::new();
    let mut snapshot = AgentSkillSnapshot::default();

    for root in roots {
        if !root.path.exists() {
            continue;
        }
        let root_path = normalize_path(&root.path);
        if snapshot
            .roots
            .iter()
            .any(|existing| normalize_path(&existing.path) == root_path)
        {
            continue;
        }

        snapshot.roots.push(AgentSkillRoot {
            path: root_path.clone(),
            scope: root.scope,
        });

        for skill in load_skill_summaries_from_directory(&root_path) {
            let metadata = metadata_from_skill_summary(skill, root.scope);
            let dedupe_key = (metadata.name.clone(), normalize_path(&metadata.directory));
            if seen.insert(dedupe_key) {
                snapshot.skills.push(metadata);
            }
        }
    }

    snapshot
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotCacheEntry {
    key: String,
    snapshot: AgentSkillSnapshot,
}

fn snapshot_cache() -> &'static Mutex<Option<SnapshotCacheEntry>> {
    static CACHE: OnceLock<Mutex<Option<SnapshotCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

pub fn invalidate_agent_skill_snapshot_cache() {
    let mut guard = snapshot_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = None;
    drop(guard);
    invalidate_skill_summary_cache();
}

fn runtime_extra_roots() -> &'static Mutex<Vec<PathBuf>> {
    static ROOTS: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    ROOTS.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn set_runtime_extra_skill_roots(
    roots: impl IntoIterator<Item = PathBuf>,
) -> Result<(), String> {
    let mut normalized = Vec::new();
    for root in roots {
        if !root.is_absolute() {
            return Err(format!(
                "skills/extraRoots/set requires absolute paths: {}",
                root.display()
            ));
        }
        let root = normalize_path(&root);
        if !normalized.contains(&root) {
            normalized.push(root);
        }
    }
    let mut guard = runtime_extra_roots()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = normalized;
    drop(guard);
    invalidate_agent_skill_snapshot_cache();
    Ok(())
}

pub fn apply_agent_skill_config(snapshot: &mut AgentSkillSnapshot, config: &[SkillConfig]) {
    for skill in &mut snapshot.skills {
        let effective_enabled = config
            .iter()
            .filter(|entry| skill_config_matches(entry, skill))
            .map(|entry| entry.enabled)
            .next_back();
        if effective_enabled == Some(false) {
            skill.enabled = false;
            skill.policy.allow_implicit_invocation = false;
        }
    }
}

fn skill_config_matches(entry: &SkillConfig, skill: &AgentSkillMetadata) -> bool {
    match (entry.path.as_ref(), entry.name.as_deref()) {
        (Some(path), None) => normalize_path(path) == normalize_path(&skill.skill_file_path),
        (None, Some(name)) => {
            let name = name.trim();
            !name.is_empty()
                && (skill.name.eq_ignore_ascii_case(name)
                    || skill.stable_id().eq_ignore_ascii_case(name))
        }
        _ => false,
    }
}

fn cached_snapshot(key: &str) -> Option<AgentSkillSnapshot> {
    let guard = snapshot_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let entry = guard.as_ref()?;
    (entry.key == key).then(|| entry.snapshot.clone())
}

fn store_cached_snapshot(key: String, snapshot: &AgentSkillSnapshot) {
    let mut guard = snapshot_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = Some(SnapshotCacheEntry {
        key,
        snapshot: snapshot.clone(),
    });
}

fn snapshot_cache_key(roots: &[AgentSkillRoot]) -> String {
    roots
        .iter()
        .map(|root| {
            let normalized = normalize_path(&root.path);
            format!(
                "{}:{}:{}",
                root.scope.as_label(),
                normalized.display(),
                root_signature(&normalized)
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn root_signature(root: &Path) -> String {
    if !root.exists() {
        return "missing".to_string();
    }
    let mut parts = vec![format!("root={}", path_mtime(root).unwrap_or_default())];
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut child_signatures = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .filter_map(|path| {
                let skill_file = path.join("SKILL.md");
                let name = path.file_name()?.to_string_lossy().to_string();
                Some(format!(
                    "{name}:{}",
                    path_mtime(&skill_file).unwrap_or_default()
                ))
            })
            .collect::<Vec<_>>();
        child_signatures.sort();
        parts.extend(child_signatures);
    }
    parts.join(",")
}

fn path_mtime(path: &Path) -> Option<u128> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

pub fn agent_skill_roots_for_workspace(
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Vec<AgentSkillRoot> {
    let mut roots = Vec::new();

    if let Some(project_root) = project_root {
        for path in app_paths::resolve_project_skill_roots(project_root) {
            push_unique_root(&mut roots, path, AgentSkillScope::Project);
        }
    }
    if let Some(working_dir) = working_dir {
        for path in app_paths::resolve_project_skill_roots(working_dir) {
            push_unique_root(&mut roots, path, AgentSkillScope::Project);
        }
    }

    for root in default_agent_skill_roots() {
        if root.scope == AgentSkillScope::Project
            && (working_dir.is_some() || project_root.is_some())
        {
            continue;
        }
        push_unique_root(&mut roots, root.path, root.scope);
    }
    for path in runtime_extra_roots()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .cloned()
    {
        push_unique_root(&mut roots, path, AgentSkillScope::Other);
    }

    roots
}

pub fn default_agent_skill_roots() -> Vec<AgentSkillRoot> {
    scoped_default_agent_skill_roots(
        app_paths::resolve_lime_project_skill_roots(),
        app_paths::resolve_lime_user_skill_roots(),
        app_paths::resolve_skills_dir().ok(),
    )
}

fn scoped_default_agent_skill_roots(
    project_roots: impl IntoIterator<Item = PathBuf>,
    user_roots: impl IntoIterator<Item = PathBuf>,
    app_root: Option<PathBuf>,
) -> Vec<AgentSkillRoot> {
    let mut roots = Vec::new();
    for path in project_roots {
        push_unique_root(&mut roots, path, AgentSkillScope::Project);
    }
    for path in user_roots {
        push_unique_root(&mut roots, path, AgentSkillScope::User);
    }
    if let Some(path) = app_root {
        push_unique_root(&mut roots, path, AgentSkillScope::App);
    }
    roots
}

impl AgentSkillScope {
    pub fn as_label(self) -> &'static str {
        match self {
            AgentSkillScope::Project => "project",
            AgentSkillScope::User => "user",
            AgentSkillScope::App => "app",
            AgentSkillScope::Orchestrator => "orchestrator",
            AgentSkillScope::Other => "other",
        }
    }

    pub fn source(self) -> AgentSkillSource {
        match self {
            Self::Project => AgentSkillSource::Project,
            Self::User => AgentSkillSource::User,
            Self::App => AgentSkillSource::App,
            Self::Orchestrator => AgentSkillSource::Orchestrator,
            Self::Other => AgentSkillSource::Other,
        }
    }

    pub fn authority(self) -> AgentSkillAuthority {
        match self {
            Self::Project => AgentSkillAuthority::Workspace,
            Self::User => AgentSkillAuthority::User,
            Self::App => AgentSkillAuthority::Application,
            Self::Orchestrator => AgentSkillAuthority::Orchestrator,
            Self::Other => AgentSkillAuthority::External,
        }
    }
}

fn metadata_from_skill_summary(
    skill: LoadedSkillSummary,
    scope: AgentSkillScope,
) -> AgentSkillMetadata {
    let directory = normalize_path(&skill.local_directory_path);
    let capabilities = skill.allowed_tools.unwrap_or_default();
    let dependencies = AgentSkillDependencies {
        tools: capabilities
            .iter()
            .map(|value| AgentSkillToolDependency {
                dependency_type: "runtime_tool".to_string(),
                value: value.clone(),
                required: true,
            })
            .collect(),
    };
    AgentSkillMetadata {
        skill_id: agent_skill_stable_id(scope, &skill.skill_name),
        name: skill.skill_name,
        description: normalize_inline_text(&skill.description),
        scope,
        source: scope.source(),
        authority: scope.authority(),
        enabled: !skill.disable_model_invocation,
        interface: AgentSkillInterface {
            display_name: skill.display_name,
            execution_mode: skill.execution_mode,
            provider: skill.provider,
            model: skill.model,
            argument_hint: skill.argument_hint.and_then(normalize_optional_text),
        },
        dependencies,
        capabilities,
        skill_file_path: directory.join("SKILL.md"),
        directory,
        policy: AgentSkillPolicy {
            allow_implicit_invocation: !skill.disable_model_invocation,
            when_to_use: skill.when_to_use.and_then(normalize_optional_text),
        },
    }
}

fn push_unique_root(roots: &mut Vec<AgentSkillRoot>, path: PathBuf, scope: AgentSkillScope) {
    let normalized = normalize_path(&path);
    if roots
        .iter()
        .any(|root| normalize_path(&root.path) == normalized)
    {
        return;
    }
    roots.push(AgentSkillRoot {
        path: normalized,
        scope,
    });
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn normalize_optional_text(value: String) -> Option<String> {
    let normalized = normalize_inline_text(&value);
    (!normalized.is_empty()).then_some(normalized)
}

fn normalize_inline_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn snapshot_loads_metadata_without_body_and_preserves_root_order() {
        let project_root = TempDir::new().expect("project root");
        let user_root = TempDir::new().expect("user root");
        write_skill(
            project_root.path(),
            "research",
            "Research",
            "Search and cite sources.",
            Some("Use for source-backed research."),
        );
        write_skill(
            user_root.path(),
            "summary",
            "Summary",
            "Summarize text.",
            None,
        );

        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: project_root.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: user_root.path().to_path_buf(),
                scope: AgentSkillScope::User,
            },
        ]);

        assert_eq!(snapshot.roots.len(), 2);
        assert_eq!(snapshot.skills.len(), 2);
        assert_eq!(snapshot.skills[0].name, "research");
        assert_eq!(snapshot.skills[0].interface.display_name, "Research");
        assert_eq!(snapshot.skills[0].scope, AgentSkillScope::Project);
        assert_eq!(
            snapshot.skills[0].policy.when_to_use.as_deref(),
            Some("Use for source-backed research.")
        );
        assert!(snapshot.skills[0]
            .skill_file_path
            .ends_with("research/SKILL.md"));
        assert_eq!(snapshot.skills[0].skill_id, "project:research");
        assert_eq!(snapshot.skills[0].source, AgentSkillSource::Project);
        assert_eq!(snapshot.skills[0].authority, AgentSkillAuthority::Workspace);
        assert!(snapshot.skills[0].enabled);
    }

    #[test]
    fn snapshot_deduplicates_same_root_and_directory() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "analysis", "Analysis", "Analyze inputs.", None);

        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: root.path().to_path_buf(),
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: root.path().to_path_buf(),
                scope: AgentSkillScope::User,
            },
        ]);

        assert_eq!(snapshot.roots.len(), 1);
        assert_eq!(snapshot.skills.len(), 1);
        assert_eq!(snapshot.skills[0].scope, AgentSkillScope::Project);
    }

    #[test]
    fn stable_skill_id_uses_scope_and_name_without_local_path() {
        let first_root = TempDir::new().expect("first root");
        let second_root = TempDir::new().expect("second root");
        write_skill(
            first_root.path(),
            "research",
            "Research",
            "Search sources.",
            None,
        );
        write_skill(
            second_root.path(),
            "research",
            "Research",
            "Search sources.",
            None,
        );

        let first = build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: first_root.path().to_path_buf(),
            scope: AgentSkillScope::Project,
        }]);
        let second = build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: second_root.path().to_path_buf(),
            scope: AgentSkillScope::Project,
        }]);

        assert_ne!(first.skills[0].directory, second.skills[0].directory);
        assert_eq!(first.skills[0].stable_id(), "project:research");
        assert_eq!(first.skills[0].stable_id(), second.skills[0].stable_id());
        assert!(!first.skills[0]
            .stable_id()
            .contains(&first_root.path().display().to_string()));
    }

    #[test]
    fn workspace_roots_prefer_project_before_defaults() {
        let workspace = TempDir::new().expect("workspace");
        let working = workspace.path().join("nested");
        std::fs::create_dir_all(&working).expect("working dir");

        let roots = agent_skill_roots_for_workspace(Some(&working), Some(workspace.path()));

        assert!(roots[0].path.ends_with(".agents/skills"));
        assert_eq!(roots[0].scope, AgentSkillScope::Project);
        assert!(roots
            .iter()
            .any(|root| root.path == working.join(".agents/skills")));
        assert!(roots
            .iter()
            .any(|root| root.path == workspace.path().join(".codex/skills")));
        assert!(roots
            .iter()
            .filter(|root| root.scope == AgentSkillScope::Project)
            .all(|root| root.path.starts_with(workspace.path())));
    }

    #[test]
    fn default_roots_preserve_scope_across_all_provider_roots() {
        let project_roots = (0..9)
            .map(|index| PathBuf::from(format!("/workspace/provider-{index}/skills")))
            .collect::<Vec<_>>();
        let user_roots = (0..9)
            .map(|index| PathBuf::from(format!("/home/provider-{index}/skills")))
            .collect::<Vec<_>>();
        let app_root = PathBuf::from("/app/skills");

        let roots = scoped_default_agent_skill_roots(
            project_roots.clone(),
            user_roots.clone(),
            Some(app_root.clone()),
        );

        assert_eq!(roots.len(), 19);
        assert!(roots[..9]
            .iter()
            .zip(project_roots)
            .all(|(root, path)| root.path == path && root.scope == AgentSkillScope::Project));
        assert!(roots[9..18]
            .iter()
            .zip(user_roots)
            .all(|(root, path)| root.path == path && root.scope == AgentSkillScope::User));
        assert_eq!(roots[18].path, app_root);
        assert_eq!(roots[18].scope, AgentSkillScope::App);
    }

    #[test]
    fn snapshot_cache_invalidates_when_root_contents_change() {
        let root = TempDir::new().expect("root");
        write_skill(root.path(), "writer", "Writer", "Write clearly.", None);
        let roots = [AgentSkillRoot {
            path: root.path().to_path_buf(),
            scope: AgentSkillScope::Project,
        }];

        let first = build_agent_skill_snapshot_from_roots(roots.clone());
        write_skill(root.path(), "summary", "Summary", "Summarize text.", None);
        let second = build_agent_skill_snapshot_from_roots(roots);

        assert_eq!(first.skills.len(), 1);
        assert_eq!(second.skills.len(), 2);
        assert!(second.skills.iter().any(|skill| skill.name == "summary"));
    }

    fn write_skill(
        root: &Path,
        name: &str,
        display_name: &str,
        description: &str,
        when_to_use: Option<&str>,
    ) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        let when_to_use = when_to_use
            .map(|value| format!("when-to-use: {value}\n"))
            .unwrap_or_default();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {display_name}\ndescription: {description}\n{when_to_use}---\n\n# {display_name}\n\nFull body should not be rendered by snapshot."
            ),
        )
        .expect("skill file");
    }
}
