use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    FuzzyFileSearchMatchType, FuzzyFileSearchParams, FuzzyFileSearchResponse, FuzzyFileSearchResult,
};
use app_server_protocol::JsonRpcError;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use tokio::sync::Mutex;

const MATCH_LIMIT: usize = 50;

#[derive(Clone, Default)]
pub(crate) struct FuzzyFileSearchServer {
    pending: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl FuzzyFileSearchServer {
    pub(crate) async fn search(
        &self,
        params: FuzzyFileSearchParams,
    ) -> Result<FuzzyFileSearchResponse, JsonRpcError> {
        let roots = validate_roots(&params.roots).await?;
        if params.query.is_empty() || roots.is_empty() {
            return Ok(FuzzyFileSearchResponse { files: Vec::new() });
        }

        let cancellation_flag = self
            .replace_cancellation_flag(params.cancellation_token.as_deref())
            .await;
        let query = params.query;
        let search_flag = cancellation_flag.clone();
        let search =
            tokio::task::spawn_blocking(move || run_search(&query, roots, search_flag.as_ref()))
                .await
                .map_err(|error| runtime_error(format!("fuzzy file search task failed: {error}")));

        self.remove_cancellation_flag(params.cancellation_token.as_deref(), &cancellation_flag)
            .await;

        Ok(FuzzyFileSearchResponse { files: search? })
    }

    async fn replace_cancellation_flag(&self, token: Option<&str>) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        let Some(token) = token else {
            return flag;
        };
        if let Some(previous) = self
            .pending
            .lock()
            .await
            .insert(token.to_string(), flag.clone())
        {
            previous.store(true, AtomicOrdering::Relaxed);
        }
        flag
    }

    async fn remove_cancellation_flag(&self, token: Option<&str>, flag: &Arc<AtomicBool>) {
        let Some(token) = token else {
            return;
        };
        let mut pending = self.pending.lock().await;
        if pending
            .get(token)
            .is_some_and(|current| Arc::ptr_eq(current, flag))
        {
            pending.remove(token);
        }
    }
}

async fn validate_roots(roots: &[String]) -> Result<Vec<SearchRoot>, JsonRpcError> {
    let mut validated = Vec::with_capacity(roots.len());
    for root in roots {
        let path = PathBuf::from(root);
        if !path.is_absolute() {
            return Err(invalid_params(
                "fuzzyFileSearch.roots must be absolute paths",
            ));
        }
        let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
            invalid_params(format!(
                "fuzzyFileSearch root is not readable: {}: {error}",
                path.display()
            ))
        })?;
        if !metadata.is_dir() {
            return Err(invalid_params(format!(
                "fuzzyFileSearch root must be a directory: {}",
                path.display()
            )));
        }
        validated.push(SearchRoot {
            wire: root.clone(),
            path,
        });
    }
    Ok(validated)
}

#[derive(Clone)]
struct SearchRoot {
    wire: String,
    path: PathBuf,
}

fn run_search(
    query: &str,
    roots: Vec<SearchRoot>,
    canceled: &AtomicBool,
) -> Vec<FuzzyFileSearchResult> {
    let mut results = Vec::new();
    for root in roots {
        if canceled.load(AtomicOrdering::Relaxed) {
            break;
        }
        let mut pending_directories = vec![root.path.clone()];
        while let Some(directory) = pending_directories.pop() {
            if canceled.load(AtomicOrdering::Relaxed) {
                break;
            }
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.filter_map(Result::ok) {
                if canceled.load(AtomicOrdering::Relaxed) {
                    break;
                }
                let path = entry.path();
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                let Ok(relative) = path.strip_prefix(&root.path) else {
                    continue;
                };
                if file_type.is_dir() && !file_type.is_symlink() && !should_skip_directory(relative)
                {
                    pending_directories.push(path.clone());
                }
                push_match(&mut results, query, &root, relative, file_type.is_dir());
            }
        }
    }
    results
}

fn push_match(
    results: &mut Vec<FuzzyFileSearchResult>,
    query: &str,
    root: &SearchRoot,
    relative: &Path,
    is_directory: bool,
) {
    let relative_path = relative.to_string_lossy().into_owned();
    let Some((score, indices)) = fuzzy_score(&relative_path, query) else {
        return;
    };
    results.push(FuzzyFileSearchResult {
        root: root.wire.clone(),
        file_name: file_name(relative),
        path: relative_path,
        match_type: if is_directory {
            FuzzyFileSearchMatchType::Directory
        } else {
            FuzzyFileSearchMatchType::File
        },
        score,
        indices: Some(indices),
    });
    results.sort_by(compare_results);
    results.truncate(MATCH_LIMIT);
}

fn should_skip_directory(relative: &Path) -> bool {
    relative
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            matches!(
                name,
                ".git" | ".hg" | ".svn" | "node_modules" | "target" | "dist" | "build" | "coverage"
            )
        })
}

fn fuzzy_score(path: &str, query: &str) -> Option<(u32, Vec<u32>)> {
    let query = query.chars().collect::<Vec<_>>();
    if query.is_empty() {
        return None;
    }
    let path_chars = path.chars().collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(query.len());
    let mut cursor = 0;
    for needle in query {
        let offset = path_chars[cursor..]
            .iter()
            .position(|candidate| chars_equal(*candidate, needle))?;
        let index = cursor + offset;
        indices.push(u32::try_from(index).ok()?);
        cursor = index + 1;
    }

    let filename_start = path_chars
        .iter()
        .rposition(|character| matches!(character, '/' | '\\'))
        .map_or(0, |index| index + 1);
    let mut score = (indices.len() as u32) * 16;
    for (position, index) in indices.iter().copied().enumerate() {
        let index = index as usize;
        if position > 0 && index == indices[position - 1] as usize + 1 {
            score += 12;
        }
        if index == 0 || is_boundary(path_chars[index - 1]) {
            score += 8;
        }
    }
    if indices
        .first()
        .is_some_and(|index| *index as usize >= filename_start)
    {
        score += 16;
    }
    let first = indices.first().copied().unwrap_or_default();
    score += 20_u32.saturating_sub(first.min(20));
    score = score.saturating_sub(
        u32::try_from(path_chars.len().saturating_sub(indices.len()))
            .unwrap_or(u32::MAX)
            .min(30),
    );
    Some((score, indices))
}

fn chars_equal(left: char, right: char) -> bool {
    left == right || left.eq_ignore_ascii_case(&right)
}

fn is_boundary(character: char) -> bool {
    matches!(character, '/' | '\\' | '-' | '_' | ' ' | '.')
}

fn compare_results(left: &FuzzyFileSearchResult, right: &FuzzyFileSearchResult) -> Ordering {
    right
        .score
        .cmp(&left.score)
        .then_with(|| left.path.cmp(&right.path))
        .then_with(|| left.root.cmp(&right.root))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}

fn runtime_error(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_score_returns_character_indices_and_prefers_filename_matches() {
        let (nested_score, nested_indices) =
            fuzzy_score("src/deep/agent_runtime.rs", "art").expect("nested match");
        let (filename_score, filename_indices) =
            fuzzy_score("src/artifact.rs", "art").expect("filename match");

        assert_eq!(nested_indices, vec![9, 15, 18]);
        assert_eq!(filename_indices, vec![4, 5, 6]);
        assert!(filename_score > nested_score);
    }

    #[test]
    fn search_sorts_results_and_preserves_relative_paths() {
        let temp = tempfile::tempdir().expect("fuzzy search temp dir");
        std::fs::create_dir_all(temp.path().join("src")).expect("create src");
        std::fs::write(temp.path().join("src/app.rs"), "app").expect("write app");
        std::fs::write(temp.path().join("src/apple.rs"), "apple").expect("write apple");
        let root = SearchRoot {
            wire: temp.path().to_string_lossy().into_owned(),
            path: temp.path().to_path_buf(),
        };

        let results = run_search("app", vec![root], &AtomicBool::new(false));

        assert_eq!(results[0].path, "src/app.rs");
        assert_eq!(results[0].file_name, "app.rs");
        assert_eq!(results[0].match_type, FuzzyFileSearchMatchType::File);
        assert_eq!(results[0].indices, Some(vec![4, 5, 6]));
        assert!(results.iter().any(|result| result.path == "src/apple.rs"));
    }

    #[test]
    fn canceled_search_returns_no_results() {
        let temp = tempfile::tempdir().expect("fuzzy search temp dir");
        std::fs::write(temp.path().join("app.rs"), "app").expect("write app");
        let root = SearchRoot {
            wire: temp.path().to_string_lossy().into_owned(),
            path: temp.path().to_path_buf(),
        };
        let canceled = AtomicBool::new(true);

        assert!(run_search("app", vec![root], &canceled).is_empty());
    }

    #[tokio::test]
    async fn cancellation_token_replacement_only_removes_the_current_request() {
        let server = FuzzyFileSearchServer::default();
        let first = server.replace_cancellation_flag(Some("composer")).await;
        let second = server.replace_cancellation_flag(Some("composer")).await;

        assert!(first.load(AtomicOrdering::Relaxed));
        assert!(!second.load(AtomicOrdering::Relaxed));

        server
            .remove_cancellation_flag(Some("composer"), &first)
            .await;
        assert!(server
            .pending
            .lock()
            .await
            .get("composer")
            .is_some_and(|current| Arc::ptr_eq(current, &second)));

        server
            .remove_cancellation_flag(Some("composer"), &second)
            .await;
        assert!(!server.pending.lock().await.contains_key("composer"));
    }
}
