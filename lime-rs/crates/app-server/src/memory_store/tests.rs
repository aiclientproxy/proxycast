use super::*;

fn backend(temp: &tempfile::TempDir) -> LocalMemoryBackend {
    LocalMemoryBackend::new(temp.path().join("data"))
}

fn global_root() -> MemoryStoreRootParams {
    MemoryStoreRootParams {
        scope: MemoryStoreScope::Global,
        workspace_root: None,
    }
}

#[tokio::test]
async fn list_initializes_stable_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);

    let response = backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("list");

    let paths = response
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<Vec<_>>();
    assert!(paths.contains(&MEMORY_FILE));
    assert!(paths.contains(&SUMMARY_FILE));
    assert!(paths.contains(&"extensions"));
    assert!(paths.contains(&"rollout_summaries"));
    assert!(response
        .entries
        .iter()
        .all(|entry| !entry.path.starts_with('/')));
}

#[tokio::test]
async fn read_rejects_traversal_and_hidden_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);

    let traversal = backend
        .read(MemoryStoreReadParams {
            root: global_root(),
            path: "../secret.md".to_string(),
            ..Default::default()
        })
        .await;
    assert!(traversal.is_err());

    let hidden = backend
        .read(MemoryStoreReadParams {
            root: global_root(),
            path: ".hidden/secret.md".to_string(),
            ..Default::default()
        })
        .await;
    assert!(hidden.is_err());
}

#[tokio::test]
async fn add_note_only_writes_ad_hoc_note() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let response = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Remember Context".to_string()),
            slug: Some("Remember Context".to_string()),
            content: "Prefer file-backed memory.".to_string(),
        })
        .await
        .expect("add note");

    assert!(response.path.starts_with("extensions/ad_hoc/notes/"));
    assert!(response.path.ends_with(".md"));
    let read = backend
        .read(MemoryStoreReadParams {
            root: global_root(),
            path: response.path,
            ..Default::default()
        })
        .await
        .expect("read note");
    assert!(read.content.contains("Prefer file-backed memory."));
    assert!(!read.path.starts_with('/'));
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(!summary.contains("Prefer file-backed memory."));
}

#[tokio::test]
async fn search_skips_non_utf8_files_and_returns_citations() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    fs::write(root.join(MEMORY_FILE), "Alpha memory\nBeta detail\n").expect("write memory");
    fs::write(root.join("binary.bin"), [0xff, 0xfe, 0xfd]).expect("write binary");

    let response = backend
        .search(MemoryStoreSearchParams {
            root: global_root(),
            queries: vec!["beta".to_string()],
            ..Default::default()
        })
        .await
        .expect("search");

    assert_eq!(response.hits.len(), 1);
    assert_eq!(response.hits[0].path, MEMORY_FILE);
    assert_eq!(response.hits[0].match_line_number, 2);
    assert_eq!(response.hits[0].citation.start_line_number, 2);
}

#[tokio::test]
async fn consolidate_accepts_notes_updates_summary_and_archives_processed_notes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Tone".to_string()),
            slug: Some("tone".to_string()),
            content: "Prefer concise recommendation first.".to_string(),
        })
        .await
        .expect("add note");

    let response = backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate");

    assert_eq!(response.root_scope, MemoryStoreScope::Global);
    assert_eq!(response.processed_notes, 1);
    assert_eq!(response.skipped_notes, 0);
    assert_eq!(response.archived_notes, 1);
    assert!(response.updated);
    assert!(response.warnings.is_empty());
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let memory = fs::read_to_string(root.join(MEMORY_FILE)).expect("memory");
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(memory.contains("## Consolidated notes"));
    assert!(memory.contains("Prefer concise recommendation first."));
    assert!(summary.contains("## Consolidated memory"));
    assert!(summary.contains("Prefer concise recommendation first."));
    assert!(!root.join(&note.path).exists());
    let note_file = Path::new(&note.path)
        .file_name()
        .expect("note file name")
        .to_owned();
    assert!(root
        .join("extensions/ad_hoc/processed")
        .join(note_file)
        .is_file());
    let audit_log = fs::read_to_string(root.join("audit/memory_events.jsonl")).expect("audit log");
    assert!(audit_log.contains("\"schemaVersion\":\"memory-audit-event/v1\""));
    assert!(audit_log.contains("\"operation\":\"consolidate\""));
    assert!(audit_log.contains("\"processedNotes\":1"));
    assert!(audit_log.contains("\"updated\":true"));
}

#[tokio::test]
async fn consolidate_reviews_secret_or_conflicting_notes_without_summary_update() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let secret_note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Secret".to_string()),
            slug: Some("secret".to_string()),
            content: "api_key should never be stored.".to_string(),
        })
        .await
        .expect("secret note");
    let conflict_note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Conflict".to_string()),
            slug: Some("conflict".to_string()),
            content: "Forget this preference.".to_string(),
        })
        .await
        .expect("conflict note");

    let response = backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate");

    assert_eq!(response.processed_notes, 0);
    assert_eq!(response.skipped_notes, 2);
    assert_eq!(response.archived_notes, 2);
    assert!(!response.updated);
    assert_eq!(response.warnings.len(), 2);
    assert!(response
        .warnings
        .iter()
        .any(|warning| warning.contains("secret-like content requires review")));
    assert!(response
        .warnings
        .iter()
        .any(|warning| warning.contains("conflicting memory intent requires review")));
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(!summary.contains("api_key"));
    assert!(!summary.contains("Forget this preference"));
    assert!(!root.join(&secret_note.path).exists());
    assert!(!root.join(&conflict_note.path).exists());
    let secret_file = Path::new(&secret_note.path)
        .file_name()
        .expect("secret file name")
        .to_owned();
    let conflict_file = Path::new(&conflict_note.path)
        .file_name()
        .expect("conflict file name")
        .to_owned();
    assert!(root
        .join("extensions/ad_hoc/review")
        .join(secret_file)
        .is_file());
    assert!(root
        .join("extensions/ad_hoc/review")
        .join(conflict_file)
        .is_file());
    let audit_log = fs::read_to_string(root.join("audit/memory_events.jsonl")).expect("audit log");
    assert!(audit_log.contains("\"operation\":\"consolidate\""));
    assert!(audit_log.contains("\"processedNotes\":0"));
    assert!(audit_log.contains("\"skippedNotes\":2"));
    assert!(audit_log.contains("\"updated\":false"));
}

#[tokio::test]
async fn consolidate_accepts_rollout_summaries_only_on_explicit_consolidate() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let rollout_path = root.join("rollout_summaries/thread-1.md");
    fs::write(
        &rollout_path,
        "# Thread rollout\n\nUser prefers architecture decisions with tradeoffs first.\n",
    )
    .expect("write rollout summary");
    let before_summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(!before_summary.contains("architecture decisions"));

    let response = backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate rollout summary");

    assert_eq!(response.processed_notes, 1);
    assert_eq!(response.skipped_notes, 0);
    assert_eq!(response.archived_notes, 1);
    assert!(response.updated);
    let memory = fs::read_to_string(root.join(MEMORY_FILE)).expect("memory");
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(memory.contains("architecture decisions with tradeoffs first"));
    assert!(memory.contains("source: rollout_summaries/processed/thread-1.md"));
    assert!(summary.contains("architecture decisions with tradeoffs first"));
    assert!(!rollout_path.exists());
    assert!(root
        .join("rollout_summaries/processed/thread-1.md")
        .is_file());
}

#[tokio::test]
async fn consolidate_reviews_sensitive_rollout_summaries_without_summary_update() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let rollout_path = root.join("rollout_summaries/thread-secret.md");
    fs::write(&rollout_path, "api_key appeared in a rollout summary.\n")
        .expect("write rollout summary");

    let response = backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate rollout summary");

    assert_eq!(response.processed_notes, 0);
    assert_eq!(response.skipped_notes, 1);
    assert_eq!(response.archived_notes, 1);
    assert!(!response.updated);
    assert!(response
        .warnings
        .iter()
        .any(|warning| warning.contains("secret-like content requires review")));
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(!summary.contains("api_key"));
    assert!(!rollout_path.exists());
    assert!(root
        .join("extensions/ad_hoc/review/thread-secret.md")
        .is_file());
}

#[tokio::test]
async fn review_list_reads_only_review_markdown_notes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let review_note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Secret".to_string()),
            slug: Some("secret".to_string()),
            content: "api_key should be checked by a human.".to_string(),
        })
        .await
        .expect("review note");
    backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    fs::write(
        root.join("extensions/ad_hoc/review/ignored.txt"),
        "not markdown",
    )
    .expect("ignored txt");

    let response = backend
        .list_review(MemoryStoreReviewListParams {
            root: global_root(),
            max_results: Some(20),
            ..Default::default()
        })
        .await
        .expect("list review notes");

    assert_eq!(response.root_scope, MemoryStoreScope::Global);
    assert_eq!(response.root_path, root.to_string_lossy());
    assert_eq!(response.notes.len(), 1);
    assert!(response.notes[0]
        .path
        .starts_with("extensions/ad_hoc/review/"));
    assert!(response.notes[0].path.ends_with(".md"));
    assert!(response.notes[0].preview.contains("api_key"));
    assert_eq!(response.notes[0].citation.start_line_number, 1);
    assert!(!root.join(&review_note.path).exists());
}

#[tokio::test]
async fn review_accept_consolidates_review_note_and_archives_processed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Conflict".to_string()),
            slug: Some("conflict".to_string()),
            content: "Forget this wording, but prefer direct status updates.".to_string(),
        })
        .await
        .expect("review note");
    backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate");
    let file_name = Path::new(&note.path)
        .file_name()
        .expect("note file name")
        .to_string_lossy();
    let review_path = format!("extensions/ad_hoc/review/{file_name}");

    let response = backend
        .resolve_review(MemoryStoreReviewResolveParams {
            root: global_root(),
            path: review_path.clone(),
            action: MemoryStoreReviewResolveAction::Accept,
        })
        .await
        .expect("accept review note");

    assert_eq!(response.source_path, review_path);
    assert_eq!(response.action, MemoryStoreReviewResolveAction::Accept);
    assert!(response
        .archived_path
        .starts_with("extensions/ad_hoc/processed/"));
    assert!(response.updated);
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let memory = fs::read_to_string(root.join(MEMORY_FILE)).expect("memory");
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(memory.contains("prefer direct status updates"));
    assert!(summary.contains("prefer direct status updates"));
    assert!(!root.join(response.source_path).exists());
    assert!(root.join(response.archived_path).is_file());
    let audit_log = fs::read_to_string(root.join("audit/memory_events.jsonl")).expect("audit log");
    assert!(audit_log.contains("\"operation\":\"reviewResolve\""));
    assert!(audit_log.contains("\"action\":\"accept\""));
    assert!(audit_log.contains("\"updated\":true"));
}

#[tokio::test]
async fn review_reject_archives_without_summary_update() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Secret".to_string()),
            slug: Some("secret".to_string()),
            content: "api_key should not enter memory.".to_string(),
        })
        .await
        .expect("review note");
    backend
        .consolidate(MemoryStoreConsolidateParams {
            root: global_root(),
            max_notes: Some(20),
        })
        .await
        .expect("consolidate");
    let file_name = Path::new(&note.path)
        .file_name()
        .expect("note file name")
        .to_string_lossy();
    let review_path = format!("extensions/ad_hoc/review/{file_name}");

    let response = backend
        .resolve_review(MemoryStoreReviewResolveParams {
            root: global_root(),
            path: review_path.clone(),
            action: MemoryStoreReviewResolveAction::Reject,
        })
        .await
        .expect("reject review note");

    assert_eq!(response.source_path, review_path);
    assert_eq!(response.action, MemoryStoreReviewResolveAction::Reject);
    assert!(response
        .archived_path
        .starts_with("extensions/ad_hoc/rejected/"));
    assert!(!response.updated);
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let summary = fs::read_to_string(root.join(SUMMARY_FILE)).expect("summary");
    assert!(!summary.contains("api_key"));
    assert!(!root.join(response.source_path).exists());
    assert!(root.join(response.archived_path).is_file());
    let audit_log = fs::read_to_string(root.join("audit/memory_events.jsonl")).expect("audit log");
    assert!(audit_log.contains("\"operation\":\"reviewResolve\""));
    assert!(audit_log.contains("\"action\":\"reject\""));
    assert!(audit_log.contains("\"updated\":false"));
}

#[tokio::test]
async fn health_initializes_layout_and_reports_store_stats() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);

    let response = backend
        .health(global_root())
        .await
        .expect("memory store health");

    assert_eq!(response.root_scope, MemoryStoreScope::Global);
    assert_eq!(response.root_path, root.to_string_lossy());
    assert!(response.initialized);
    assert!(response.summary_exists);
    assert!(response.memory_exists);
    assert_eq!(response.notes_count, 0);
    assert!(root.join(SUMMARY_FILE).is_file());
    assert!(root.join(MEMORY_FILE).is_file());
    assert!(root.join(NOTES_DIR).is_dir());
    assert!(root.join("audit").is_dir());
}

#[tokio::test]
async fn audit_events_are_file_backed_but_excluded_from_search_and_index_sources() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    fs::write(
        root.join("audit").join("memory_events.jsonl"),
        "{\"operation\":\"consolidate\",\"sentinel\":\"audit-only-memory\"}\n",
    )
    .expect("write audit");
    fs::write(root.join(MEMORY_FILE), "Indexable memory\n").expect("write memory");

    let search = backend
        .search(MemoryStoreSearchParams {
            root: global_root(),
            queries: vec!["audit-only-memory".to_string()],
            ..Default::default()
        })
        .await
        .expect("search skips audit");
    assert!(search.hits.is_empty());

    let rebuilt = backend
        .rebuild_index(global_root())
        .await
        .expect("rebuild index");
    assert_eq!(rebuilt.source_file_count, 2);
}

#[tokio::test]
async fn reset_clears_store_contents_preserves_layout_and_soul_boundary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    let outside_file = temp.path().join("data").join("memory.soul.json");
    fs::create_dir_all(outside_file.parent().expect("outside parent")).expect("outside parent");
    fs::write(&outside_file, "soul config").expect("outside soul");
    let note = backend
        .add_note(MemoryStoreAddNoteParams {
            root: global_root(),
            title: Some("Reset note".to_string()),
            slug: Some("reset-note".to_string()),
            content: "Remove this note only from the file store.".to_string(),
        })
        .await
        .expect("add note");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    fs::write(root.join(SUMMARY_FILE), "summary content").expect("summary");
    fs::write(root.join(INDEX_DIR).join("scratch.json"), "{}").expect("index");

    backend.reset().await.expect("reset");

    assert!(outside_file.is_file());
    assert!(root.join(SUMMARY_FILE).is_file());
    assert!(root.join(MEMORY_FILE).is_file());
    assert!(root.join(NOTES_DIR).is_dir());
    assert!(!root.join(note.path).exists());
    assert_eq!(
        fs::read_to_string(root.join(SUMMARY_FILE)).expect("empty summary"),
        ""
    );
}

#[tokio::test]
async fn rebuild_index_writes_deletable_manifest_without_becoming_search_truth() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    fs::write(root.join(MEMORY_FILE), "Indexable memory\n").expect("write memory");

    let response = backend
        .rebuild_index(global_root())
        .await
        .expect("rebuild index");

    assert_eq!(response.root_scope, MemoryStoreScope::Global);
    assert_eq!(response.manifest_path, "index/manifest.json");
    assert_eq!(response.schema_version, "memory-index-manifest/v1");
    assert!(response.source_file_count >= 2);
    assert!(response.source_total_bytes > 0);
    assert!(!response.source_checksum.is_empty());
    assert!(response.rebuilt);
    let manifest = root.join("index/manifest.json");
    assert!(manifest.is_file());
    let manifest_content = fs::read_to_string(&manifest).expect("manifest");
    assert!(manifest_content.contains("memory-index-manifest/v1"));

    fs::remove_file(&manifest).expect("delete manifest");
    let rebuilt = backend
        .rebuild_index(global_root())
        .await
        .expect("rebuild after delete");
    assert_eq!(rebuilt.manifest_path, "index/manifest.json");
    assert!(manifest.is_file());

    fs::write(&manifest, "{not-json").expect("corrupt manifest");
    let search = backend
        .search(MemoryStoreSearchParams {
            root: global_root(),
            queries: vec!["indexable".to_string()],
            ..Default::default()
        })
        .await
        .expect("search falls back to text scan");
    assert_eq!(search.hits.len(), 1);
    assert_eq!(search.hits[0].path, MEMORY_FILE);
}

#[tokio::test]
async fn list_and_search_skip_symlinks() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend(&temp);
    backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("init");
    let root = temp.path().join("data").join(MEMORY_ROOT_DIR);
    let target = root.join("target.md");
    fs::write(&target, "target memory").expect("write target");
    let link = root.join("link.md");
    if !create_symlink(&target, &link) {
        return;
    }

    let list = backend
        .list(MemoryStoreListParams {
            root: global_root(),
            ..Default::default()
        })
        .await
        .expect("list");
    assert!(list.entries.iter().any(|entry| entry.path == "target.md"));
    assert!(!list.entries.iter().any(|entry| entry.path == "link.md"));

    let search = backend
        .search(MemoryStoreSearchParams {
            root: global_root(),
            queries: vec!["target".to_string()],
            ..Default::default()
        })
        .await
        .expect("search");
    assert!(search.hits.iter().any(|hit| hit.path == "target.md"));
    assert!(!search.hits.iter().any(|hit| hit.path == "link.md"));
}

#[cfg(unix)]
fn create_symlink(target: &Path, link: &Path) -> bool {
    std::os::unix::fs::symlink(target, link).is_ok()
}

#[cfg(windows)]
fn create_symlink(target: &Path, link: &Path) -> bool {
    std::os::windows::fs::symlink_file(target, link).is_ok()
}
