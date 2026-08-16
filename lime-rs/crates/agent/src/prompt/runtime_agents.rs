//! Lime 运行时 AGENTS 指令加载
//!
//! 仅用于 Lime 应用运行时会话：
//! - 全局：优先 `CODEX_HOME/AGENTS.md`，缺失时回退 Lime 的 `~/.lime/AGENTS.md`
//! - 工作区：从项目根到当前工作目录，优先标准 `AGENTS.override.md` / `AGENTS.md`，
//!   缺失时回退 `.lime/AGENTS.override.md` / `.lime/AGENTS.md`

use lime_core::app_paths;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::prompt_assets::HIERARCHICAL_AGENTS_MESSAGE;

pub const RUNTIME_AGENTS_PROMPT_MARKER: &str = "【Lime Runtime AGENTS 指令】";
const RUNTIME_AGENTS_MAX_BYTES: usize = 32 * 1024;
const PROJECT_ROOT_MARKERS: &[&str] = &[".git"];
const WORKSPACE_AGENTS_FILE_NAME: &str = "AGENTS.md";
const WORKSPACE_OVERRIDE_AGENTS_FILE_NAME: &str = "AGENTS.override.md";

pub fn merge_system_prompt_with_runtime_agents(
    base_prompt: Option<String>,
    working_dir: Option<&Path>,
) -> Option<String> {
    merge_system_prompt_with_runtime_agents_for_project(base_prompt, working_dir, None)
}

pub fn merge_system_prompt_with_runtime_agents_for_project(
    base_prompt: Option<String>,
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Option<String> {
    let runtime_prompt = build_runtime_agents_prompt_for_project(working_dir, project_root);
    match (base_prompt, runtime_prompt) {
        (Some(base), Some(runtime)) => {
            if base.contains(RUNTIME_AGENTS_PROMPT_MARKER) {
                Some(base)
            } else if base.trim().is_empty() {
                Some(runtime)
            } else {
                Some(format!("{base}\n\n{runtime}"))
            }
        }
        (Some(base), None) => Some(base),
        (None, Some(runtime)) => Some(runtime),
        (None, None) => None,
    }
}

pub fn build_runtime_agents_prompt(working_dir: Option<&Path>) -> Option<String> {
    build_runtime_agents_prompt_for_project(working_dir, None)
}

pub fn build_runtime_agents_prompt_for_project(
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Option<String> {
    let global_path = app_paths::resolve_codex_agents_path()
        .filter(|path| path.is_file())
        .or_else(|| match app_paths::resolve_user_memory_path() {
            Ok(path) => Some(path),
            Err(error) => {
                tracing::warn!(
                    "[AgentRuntime] 解析全局运行时 AGENTS 路径失败，跳过全局指令层: {}",
                    error
                );
                None
            }
        });
    let workspace_paths = working_dir
        .map(|working_dir| discover_workspace_runtime_agents_paths(working_dir, project_root))
        .unwrap_or_default();
    build_runtime_agents_prompt_with_paths(
        global_path.as_deref(),
        workspace_paths.iter().map(PathBuf::as_path),
    )
}

fn build_runtime_agents_prompt_with_paths<P>(
    global_path: Option<&Path>,
    workspace_paths: impl IntoIterator<Item = P>,
) -> Option<String>
where
    P: AsRef<Path>,
{
    let mut sections = Vec::new();
    let mut seen = HashSet::<PathBuf>::new();
    let mut remaining = RUNTIME_AGENTS_MAX_BYTES;

    if let Some((path, content)) = load_runtime_agents_layer(global_path, &mut seen, &mut remaining)
    {
        sections.push(format_runtime_agents_section(
            "全局运行时指令",
            &path,
            None,
            &content,
        ));
    }

    for workspace_path in workspace_paths {
        if let Some((path, content)) =
            load_runtime_agents_layer(Some(workspace_path.as_ref()), &mut seen, &mut remaining)
        {
            let scope = workspace_scope_from_agents_path(&path);
            sections.push(format_runtime_agents_section(
                "Workspace 运行时指令",
                &path,
                scope.as_deref(),
                &content,
            ));
        }
        if remaining == 0 {
            break;
        }
    }

    if sections.is_empty() {
        None
    } else {
        Some(format!(
            "{RUNTIME_AGENTS_PROMPT_MARKER}\n{}\n\n以下内容来自 Lime 运行时 AGENTS 文件，请按作用域优先遵循：\n\n{}",
            HIERARCHICAL_AGENTS_MESSAGE.trim(),
            sections.join("\n\n")
        ))
    }
}

fn format_runtime_agents_section(
    title: &str,
    path: &Path,
    scope: Option<&Path>,
    content: &str,
) -> String {
    let scope_label = scope
        .map(|path| format!(" for {}", path.display()))
        .unwrap_or_default();
    format!(
        "### {title} ({})\n# AGENTS.md instructions{scope_label}\n\n<INSTRUCTIONS>\n{}\n</INSTRUCTIONS>",
        path.display(),
        content.trim()
    )
}

fn discover_workspace_runtime_agents_paths(
    working_dir: &Path,
    project_root: Option<&Path>,
) -> Vec<PathBuf> {
    let explicit_root = project_root.map(Path::to_path_buf);
    let Some(root) = explicit_root.or_else(|| discover_project_root(working_dir)) else {
        return workspace_runtime_agents_candidates(working_dir);
    };
    let working_dir = working_dir.to_path_buf();
    if !working_dir.starts_with(&root) {
        return workspace_runtime_agents_candidates(&working_dir);
    }

    let mut search_dirs = Vec::new();
    let mut cursor = working_dir;
    loop {
        search_dirs.push(cursor.clone());
        if cursor == root {
            break;
        }
        let Some(parent) = cursor.parent() else {
            break;
        };
        cursor = parent.to_path_buf();
    }
    search_dirs.reverse();
    search_dirs
        .into_iter()
        .flat_map(|dir| workspace_runtime_agents_candidates(&dir))
        .collect()
}

fn discover_project_root(working_dir: &Path) -> Option<PathBuf> {
    working_dir.ancestors().find_map(|ancestor| {
        PROJECT_ROOT_MARKERS
            .iter()
            .any(|marker| ancestor.join(marker).exists())
            .then(|| ancestor.to_path_buf())
    })
}

fn workspace_runtime_agents_candidates(dir: &Path) -> Vec<PathBuf> {
    let standard_override = dir.join(WORKSPACE_OVERRIDE_AGENTS_FILE_NAME);
    if standard_override.is_file() {
        vec![standard_override]
    } else if dir.join(WORKSPACE_AGENTS_FILE_NAME).is_file() {
        vec![dir.join(WORKSPACE_AGENTS_FILE_NAME)]
    } else {
        let runtime_dir = dir.join(".lime");
        let override_path = runtime_dir.join(WORKSPACE_OVERRIDE_AGENTS_FILE_NAME);
        if override_path.is_file() {
            vec![override_path]
        } else {
            vec![runtime_dir.join(WORKSPACE_AGENTS_FILE_NAME)]
        }
    }
}

fn workspace_scope_from_agents_path(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    if parent.file_name().and_then(|name| name.to_str()) == Some(".lime") {
        parent.parent().map(PathBuf::from)
    } else {
        Some(parent.to_path_buf())
    }
}

fn load_runtime_agents_layer(
    path: Option<&Path>,
    seen: &mut HashSet<PathBuf>,
    remaining: &mut usize,
) -> Option<(PathBuf, String)> {
    let path = path?;
    if *remaining == 0 {
        return None;
    }
    let dedup_key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(dedup_key) || !path.is_file() {
        return None;
    }

    let mut bytes = std::fs::read(path).ok()?;
    if bytes.len() > *remaining {
        bytes.truncate(*remaining);
    }
    let content = String::from_utf8_lossy(&bytes);
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    let consumed = bytes.len();
    *remaining = remaining.saturating_sub(consumed);

    Some((path.to_path_buf(), trimmed.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn should_build_prompt_with_global_and_workspace_layers() {
        let tmp = TempDir::new().expect("create temp dir");
        let global_path = tmp.path().join("global").join("AGENTS.md");
        let workspace_root = tmp.path().join("workspace");
        let workspace_path = app_paths::resolve_workspace_runtime_agents_path(&workspace_root);
        fs::create_dir_all(global_path.parent().expect("global parent")).expect("create global");
        fs::create_dir_all(workspace_path.parent().expect("workspace parent"))
            .expect("create workspace");
        fs::write(&global_path, "- 全局偏好").expect("write global agents");
        fs::write(&workspace_path, "- 工作区偏好").expect("write workspace agents");

        let prompt = build_runtime_agents_prompt_with_paths(
            Some(global_path.as_path()),
            [workspace_path.as_path()],
        )
        .expect("prompt should exist");

        assert!(prompt.contains(RUNTIME_AGENTS_PROMPT_MARKER));
        assert!(prompt.contains("Each AGENTS.md governs the entire directory"));
        assert!(prompt.contains("全局偏好"));
        assert!(prompt.contains("工作区偏好"));
        assert!(prompt.contains("# AGENTS.md instructions"));
        assert!(prompt.contains("<INSTRUCTIONS>"));
        assert!(prompt.contains("</INSTRUCTIONS>"));
    }

    #[test]
    fn should_build_workspace_prompt_without_global_layer() {
        let tmp = TempDir::new().expect("create temp dir");
        let workspace_path = tmp.path().join("workspace").join(".lime").join("AGENTS.md");
        fs::create_dir_all(workspace_path.parent().expect("workspace parent"))
            .expect("create workspace");
        fs::write(&workspace_path, "- 工作区指令").expect("write workspace agents");

        let prompt = build_runtime_agents_prompt_with_paths(None, [workspace_path.as_path()])
            .expect("workspace prompt should exist");

        assert!(prompt.contains("工作区指令"));
        assert!(!prompt.contains("全局运行时指令"));
    }

    #[test]
    fn should_skip_duplicate_paths() {
        let tmp = TempDir::new().expect("create temp dir");
        let path = tmp.path().join("shared").join("AGENTS.md");
        fs::create_dir_all(path.parent().expect("shared parent")).expect("create dir");
        fs::write(&path, "- 同一路径").expect("write agents");

        let prompt = build_runtime_agents_prompt_with_paths(Some(path.as_path()), [path.as_path()])
            .expect("prompt should exist");

        assert_eq!(prompt.matches("### ").count(), 1);
    }

    #[test]
    fn should_discover_workspace_layers_from_project_root_to_working_dir() {
        let tmp = TempDir::new().expect("create temp dir");
        let repo = tmp.path().join("repo");
        let nested = repo.join("apps").join("writer");
        fs::create_dir_all(&nested).expect("create nested");
        fs::write(repo.join(".git"), "gitdir: /tmp/git").expect("write marker");
        fs::create_dir_all(repo.join(".lime")).expect("create root runtime dir");
        fs::create_dir_all(nested.join(".lime")).expect("create nested runtime dir");
        fs::write(repo.join(".lime").join("AGENTS.md"), "root shared").expect("write root shared");
        fs::write(nested.join(".lime").join("AGENTS.md"), "nested shared")
            .expect("write nested shared");

        let prompt = build_runtime_agents_prompt(Some(&nested)).expect("prompt should exist");
        let root_shared = prompt.find("root shared").expect("root shared");
        let nested_shared = prompt.find("nested shared").expect("nested shared");
        let nested_scope = nested.clone();

        assert!(root_shared < nested_shared);
        assert!(prompt.contains(&format!(
            "# AGENTS.md instructions for {}",
            nested_scope.display()
        )));
    }

    #[test]
    fn should_stop_workspace_discovery_at_project_root() {
        let tmp = TempDir::new().expect("create temp dir");
        let parent = tmp.path().join("parent");
        let repo = parent.join("repo");
        let nested = repo.join("pkg");
        fs::create_dir_all(&nested).expect("create nested");
        fs::create_dir_all(parent.join(".lime")).expect("create parent runtime dir");
        fs::create_dir_all(repo.join(".lime")).expect("create repo runtime dir");
        fs::write(repo.join(".git"), "").expect("write marker");
        fs::write(parent.join(".lime").join("AGENTS.md"), "parent shared")
            .expect("write parent shared");
        fs::write(repo.join(".lime").join("AGENTS.md"), "repo shared").expect("write repo shared");

        let paths = discover_workspace_runtime_agents_paths(&nested, None);
        let parent_runtime_agents = parent.join(".lime").join("AGENTS.md");
        let repo_runtime_agents = repo.join(".lime").join("AGENTS.md");

        assert!(
            paths.contains(&repo_runtime_agents),
            "expected repo runtime AGENTS path in {paths:?}",
        );
        assert!(
            !paths.contains(&parent_runtime_agents),
            "parent path must not be searched after project root is found: {paths:?}",
        );
    }

    #[test]
    fn should_prefer_override_over_shared_agents_in_same_scope() {
        let tmp = TempDir::new().expect("create temp dir");
        let repo = tmp.path().join("repo");
        let nested = repo.join("pkg");
        fs::create_dir_all(nested.join(".lime")).expect("create nested runtime dir");
        fs::write(
            nested.join(".lime").join("AGENTS.md"),
            "shared should not appear",
        )
        .expect("write shared agents");
        fs::write(
            nested.join(".lime").join("AGENTS.override.md"),
            "override should appear",
        )
        .expect("write override agents");

        let prompt = build_runtime_agents_prompt(Some(&nested)).expect("prompt should exist");

        assert!(prompt.contains("override should appear"));
        assert!(!prompt.contains("shared should not appear"));
    }

    #[test]
    fn should_prefer_standard_agents_files_over_lime_fallback() {
        let tmp = TempDir::new().expect("create temp dir");
        let repo = tmp.path().join("repo");
        let nested = repo.join("pkg");
        fs::create_dir_all(repo.join(".lime")).expect("create root runtime dir");
        fs::create_dir_all(nested.join(".lime")).expect("create nested runtime dir");
        fs::write(repo.join("AGENTS.md"), "standard root").expect("write standard root");
        fs::write(repo.join(".lime").join("AGENTS.md"), "lime root").expect("write lime root");
        fs::write(nested.join("AGENTS.override.md"), "standard override")
            .expect("write standard override");
        fs::write(nested.join(".lime").join("AGENTS.md"), "lime nested")
            .expect("write lime nested");

        let paths = discover_workspace_runtime_agents_paths(&nested, Some(&repo));
        assert_eq!(
            paths,
            vec![repo.join("AGENTS.md"), nested.join("AGENTS.override.md")]
        );
        let prompt =
            build_runtime_agents_prompt_with_paths(None, paths.iter().map(PathBuf::as_path))
                .expect("standard prompt");
        assert!(prompt.contains("standard root"));
        assert!(prompt.contains("standard override"));
        assert!(!prompt.contains("lime root"));
        assert!(!prompt.contains("lime nested"));
    }

    #[test]
    fn should_use_lime_agents_when_standard_file_is_absent() {
        let tmp = TempDir::new().expect("create temp dir");
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".lime")).expect("create runtime dir");
        fs::write(repo.join(".lime").join("AGENTS.md"), "lime fallback")
            .expect("write lime fallback");

        let paths = discover_workspace_runtime_agents_paths(&repo, Some(&repo));
        assert_eq!(paths, vec![repo.join(".lime").join("AGENTS.md")]);
        let prompt =
            build_runtime_agents_prompt_with_paths(None, paths.iter().map(PathBuf::as_path))
                .expect("fallback prompt");
        assert!(prompt.contains("lime fallback"));
    }

    #[test]
    fn empty_override_still_replaces_shared_agents_in_same_scope() {
        let tmp = TempDir::new().expect("create temp dir");
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".lime")).expect("create runtime dir");
        fs::write(
            repo.join(".lime").join("AGENTS.md"),
            "shared should not appear",
        )
        .expect("write shared agents");
        fs::write(repo.join(".lime").join("AGENTS.override.md"), "  \n")
            .expect("write empty override agents");

        let paths = discover_workspace_runtime_agents_paths(&repo, Some(&repo));
        assert!(build_runtime_agents_prompt_with_paths(None, paths).is_none());
    }

    #[test]
    fn workspace_discovery_preserves_symlinked_working_directory() {
        let tmp = TempDir::new().expect("create temp dir");
        let target = tmp.path().join("target");
        fs::create_dir_all(target.join(".lime")).expect("create target runtime dir");
        fs::write(target.join(".lime").join("AGENTS.md"), "symlink rule")
            .expect("write target agents");
        let linked = tmp.path().join("linked");
        if !create_directory_symlink(&target, &linked) {
            return;
        }

        let paths = discover_workspace_runtime_agents_paths(&linked, None);
        assert_eq!(paths, vec![linked.join(".lime").join("AGENTS.md")]);

        let prompt =
            build_runtime_agents_prompt_with_paths(None, paths.iter().map(PathBuf::as_path))
                .expect("symlinked agents prompt");
        assert!(prompt.contains("symlink rule"));
        assert!(prompt.contains(&linked.display().to_string()));
        assert!(!prompt.contains(&target.display().to_string()));
    }

    #[test]
    fn should_honor_explicit_project_root_without_git_marker() {
        let tmp = TempDir::new().expect("create temp dir");
        let parent = tmp.path().join("parent");
        let repo = parent.join("repo");
        let nested = repo.join("packages").join("app");
        fs::create_dir_all(parent.join(".lime")).expect("create parent runtime dir");
        fs::create_dir_all(repo.join(".lime")).expect("create repo runtime dir");
        fs::create_dir_all(nested.as_path()).expect("create nested dir");
        fs::write(
            parent.join(".lime").join("AGENTS.md"),
            "parent should not appear",
        )
        .expect("write parent agents");
        fs::write(repo.join(".lime").join("AGENTS.md"), "package root applies")
            .expect("write repo agents");

        let prompt = build_runtime_agents_prompt_for_project(Some(&nested), Some(&repo))
            .expect("prompt should exist");

        assert!(prompt.contains("package root applies"));
        assert!(!prompt.contains("parent should not appear"));
    }

    #[test]
    fn explicit_project_root_outside_working_dir_does_not_scan_parent_tree() {
        let tmp = TempDir::new().expect("create temp dir");
        let parent = tmp.path().join("parent");
        let repo = parent.join("repo");
        let outside = parent.join("outside");
        fs::create_dir_all(parent.join(".lime")).expect("create parent runtime dir");
        fs::create_dir_all(outside.join(".lime")).expect("create outside runtime dir");
        fs::create_dir_all(repo.as_path()).expect("create repo dir");
        fs::write(
            parent.join(".lime").join("AGENTS.md"),
            "parent should not appear",
        )
        .expect("write parent agents");
        fs::write(outside.join(".lime").join("AGENTS.md"), "outside applies")
            .expect("write outside agents");

        let prompt = build_runtime_agents_prompt_for_project(Some(&outside), Some(&repo))
            .expect("prompt should exist");

        assert!(prompt.contains("outside applies"));
        assert!(!prompt.contains("parent should not appear"));
    }

    #[test]
    fn should_skip_empty_layers_and_cap_runtime_agents_bytes() {
        let tmp = TempDir::new().expect("create temp dir");
        let first = tmp.path().join("first").join("AGENTS.md");
        let second = tmp.path().join("second").join("AGENTS.md");
        let third = tmp.path().join("third").join("AGENTS.md");
        for path in [&first, &second, &third] {
            fs::create_dir_all(path.parent().expect("parent")).expect("create dir");
        }
        fs::write(&first, "  \n").expect("write empty");
        fs::write(&second, "A".repeat(RUNTIME_AGENTS_MAX_BYTES + 20)).expect("write large");
        fs::write(&third, "should not appear").expect("write third");

        let mut seen = HashSet::<PathBuf>::new();
        let mut remaining = RUNTIME_AGENTS_MAX_BYTES;
        assert!(load_runtime_agents_layer(Some(&first), &mut seen, &mut remaining).is_none());
        let (_path, content) = load_runtime_agents_layer(Some(&second), &mut seen, &mut remaining)
            .expect("large layer should load");

        assert_eq!(content.len(), RUNTIME_AGENTS_MAX_BYTES);
        assert_eq!(remaining, 0);
        assert!(load_runtime_agents_layer(Some(&third), &mut seen, &mut remaining).is_none());
    }

    #[test]
    fn merge_should_append_runtime_agents_once() {
        let merged = merge_system_prompt_with_runtime_agents(
            Some(format!("{RUNTIME_AGENTS_PROMPT_MARKER}\n已有内容")),
            None,
        )
        .expect("merged prompt");

        assert_eq!(merged.matches(RUNTIME_AGENTS_PROMPT_MARKER).count(), 1);
    }

    #[cfg(unix)]
    fn create_directory_symlink(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[cfg(windows)]
    fn create_directory_symlink(target: &Path, link: &Path) -> bool {
        std::os::windows::fs::symlink_dir(target, link).is_ok()
    }
}
