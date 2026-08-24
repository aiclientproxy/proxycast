use super::*;
use agent_protocol::{SessionId, SortDirection, Thread, ThreadId, ThreadStatus, ThreadTurnsView};
use futures::executor::block_on;
use serde_json::json;
use std::collections::BTreeMap;
use thread_store::{
    ArchiveThreadParams, CreateThreadParams, ListThreadsParams, PageRequest, ReadThreadParams,
    ThreadStore,
};

fn store() -> (tempfile::TempDir, ProjectionStore) {
    let temp = tempfile::tempdir().expect("project store tempdir");
    let store =
        ProjectionStore::initialize(temp.path().join("projection.sqlite")).expect("project store");
    (temp, store)
}

fn project_params(name: &str, key: &str, thread_ids: Vec<String>) -> CreateProjectParams {
    CreateProjectParams {
        name: name.to_string(),
        roots: vec![StoredProjectRoot {
            path: "/workspace".to_string(),
        }],
        metadata: BTreeMap::from([("color".to_string(), "blue".to_string())]),
        thread_ids,
        idempotency_key: key.to_string(),
    }
}

fn thread(id: &str) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{id}")),
        thread_id: ThreadId::new(id),
        status: ThreadStatus::Idle,
        created_at_ms: 1_700_000_000_000,
        updated_at_ms: 1_700_000_000_000,
        archived: false,
        recency_at_ms: None,
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: format!("preview-{id}"),
        model_provider: "fixture-provider".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn create_thread(store: &ProjectionStore, id: &str) -> Thread {
    let thread = thread(id);
    block_on(store.create_thread(CreateThreadParams {
        thread: thread.clone(),
    }))
    .expect("create canonical thread");
    thread
}

fn read_thread(store: &ProjectionStore, id: &ThreadId, include_archived: bool) -> Thread {
    block_on(store.read_thread(ReadThreadParams {
        thread_id: id.clone(),
        include_archived,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read canonical thread")
    .expect("canonical thread exists")
}

fn list(store: &ProjectionStore, cursor: Option<String>, limit: u32) -> StoredProjectsPage {
    list_projects(store, ListProjectsParams { cursor, limit }).expect("list projects")
}

#[test]
fn project_crud_cursor_idempotency_and_move_are_durable() {
    let (_temp, store) = store();
    let first = create_project(&store, project_params("First", "project-first", Vec::new()))
        .expect("create first project");
    assert!(first.created);
    assert_eq!(
        uuid::Uuid::parse_str(&first.project.id)
            .expect("project UUID")
            .get_version_num(),
        7
    );
    assert_eq!(first.project.position, 0);

    let mut replay_params = project_params("Changed", "project-first", Vec::new());
    replay_params.roots.clear();
    replay_params.metadata.clear();
    let replay = create_project(&store, replay_params).expect("replay first project");
    assert!(!replay.created);
    assert_eq!(replay.project, first.project);

    let second = create_project(
        &store,
        project_params("Second", "project-second", Vec::new()),
    )
    .expect("create second project");
    let third = create_project(&store, project_params("Third", "project-third", Vec::new()))
        .expect("create third project");

    let first_page = list(&store, None, 2);
    assert_eq!(
        first_page
            .projects
            .iter()
            .map(|project| project.name.as_str())
            .collect::<Vec<_>>(),
        ["First", "Second"]
    );
    let cursor = first_page.next_cursor.expect("project cursor");
    assert!(!cursor.contains(&second.project.id));
    let second_page = list(&store, Some(cursor), 2);
    assert_eq!(second_page.projects, vec![third.project.clone()]);
    assert!(second_page.next_cursor.is_none());
    assert!(list_projects(
        &store,
        ListProjectsParams {
            cursor: Some("not-a-project-cursor".to_string()),
            limit: 10,
        }
    )
    .is_err());

    let updated = update_project(
        &store,
        UpdateProjectParams {
            project_id: first.project.id.clone(),
            name: Some("Renamed".to_string()),
            roots: Some(Vec::new()),
            metadata: Some(BTreeMap::new()),
        },
    )
    .expect("update project")
    .expect("project exists");
    assert!(updated.changed);
    assert_eq!(updated.project.name, "Renamed");
    assert!(updated.project.roots.is_empty());
    assert!(updated.project.metadata.is_empty());
    let unchanged = update_project(
        &store,
        UpdateProjectParams {
            project_id: updated.project.id.clone(),
            name: Some(updated.project.name.clone()),
            roots: Some(updated.project.roots.clone()),
            metadata: Some(updated.project.metadata.clone()),
        },
    )
    .expect("repeat project update")
    .expect("project exists");
    assert!(!unchanged.changed);

    assert_eq!(
        move_project(
            &store,
            MoveProjectParams {
                project_id: third.project.id.clone(),
                before_project_id: Some(first.project.id.clone()),
            },
        )
        .expect("move third project"),
        Some(ProjectMoveOutcome::Moved)
    );
    assert_eq!(
        list(&store, None, 10)
            .projects
            .iter()
            .map(|project| project.name.as_str())
            .collect::<Vec<_>>(),
        ["Third", "Renamed", "Second"]
    );
    assert_eq!(
        move_project(
            &store,
            MoveProjectParams {
                project_id: third.project.id.clone(),
                before_project_id: Some(first.project.id.clone()),
            },
        )
        .expect("repeat third project move"),
        Some(ProjectMoveOutcome::Unchanged)
    );
    assert!(move_project(
        &store,
        MoveProjectParams {
            project_id: first.project.id.clone(),
            before_project_id: Some(first.project.id.clone()),
        },
    )
    .is_err());
    assert!(move_project(
        &store,
        MoveProjectParams {
            project_id: first.project.id.clone(),
            before_project_id: Some("missing-project".to_string()),
        },
    )
    .is_err());

    delete_project(&store, first.project.id.clone())
        .expect("delete first project")
        .expect("first project exists");
    assert!(read_project(&store, first.project.id.clone())
        .expect("read deleted project")
        .is_none());
    assert!(create_project(
        &store,
        project_params("Replayed", "project-first", Vec::new()),
    )
    .is_err());
}

#[test]
fn project_import_and_delete_update_active_and_archived_threads_atomically() {
    let (_temp, store) = store();
    let active = create_thread(&store, "thread-active");
    let archived = create_thread(&store, "thread-archived");
    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: archived.thread_id.clone(),
    }))
    .expect("archive imported thread");

    let missing = create_project(
        &store,
        project_params(
            "Atomic failure",
            "project-atomic-failure",
            vec![
                active.thread_id.as_str().to_string(),
                "missing-thread".to_string(),
            ],
        ),
    );
    assert!(missing.is_err());
    assert!(list(&store, None, 10).projects.is_empty());
    assert_eq!(
        read_thread(&store, &active.thread_id, false)
            .metadata
            .get("projectId"),
        None
    );

    let imported = create_project(
        &store,
        project_params(
            "Imported",
            "project-imported",
            vec![
                active.thread_id.as_str().to_string(),
                archived.thread_id.as_str().to_string(),
            ],
        ),
    )
    .expect("import project");
    let project_id = imported.project.id.as_str();
    assert_eq!(
        read_thread(&store, &active.thread_id, false).metadata["projectId"],
        project_id
    );
    assert_eq!(
        read_thread(&store, &archived.thread_id, true).metadata["projectId"],
        project_id
    );

    let assigned = block_on(store.list_threads(ListThreadsParams {
        include_archived: false,
        page: PageRequest {
            cursor: None,
            limit: 10,
            sort_direction: SortDirection::Asc,
        },
        section: None,
        project: Some(Some(project_id.to_string())),
        sort_by_section_position: false,
    }))
    .expect("list project threads");
    assert_eq!(assigned.data.len(), 1);
    assert_eq!(assigned.data[0].thread_id, active.thread_id);

    let deleted = delete_project(&store, project_id.to_string())
        .expect("delete imported project")
        .expect("imported project exists");
    assert_eq!(
        deleted.affected_active_thread_ids,
        vec![active.thread_id.as_str().to_string()]
    );
    assert_eq!(
        deleted.affected_archived_thread_ids,
        vec![archived.thread_id.as_str().to_string()]
    );
    assert_eq!(
        read_thread(&store, &active.thread_id, false)
            .metadata
            .get("projectId"),
        None
    );
    assert_eq!(
        read_thread(&store, &archived.thread_id, true)
            .metadata
            .get("projectId"),
        None
    );

    let unassigned = block_on(store.list_threads(ListThreadsParams {
        include_archived: false,
        page: PageRequest {
            cursor: None,
            limit: 10,
            sort_direction: SortDirection::Asc,
        },
        section: None,
        project: Some(None),
        sort_by_section_position: false,
    }))
    .expect("list unassigned threads");
    assert_eq!(unassigned.data.len(), 1);
    assert_eq!(unassigned.data[0].thread_id, active.thread_id);
}

#[test]
fn project_assignment_fails_closed_on_corrupt_thread_metadata() {
    let (_temp, store) = store();
    let source = create_thread(&store, "thread-corrupt-metadata");
    let conn = store.open_thread_store().expect("open canonical store");
    let mut value = serde_json::to_value(&source).expect("encode canonical thread");
    value["metadata"] = json!("not-an-object");
    conn.execute(
        "UPDATE canonical_threads SET thread_json = ?1 WHERE thread_id = ?2",
        params![
            serde_json::to_string(&value).expect("encode corrupt thread"),
            source.thread_id.as_str()
        ],
    )
    .expect("corrupt canonical thread metadata");
    drop(conn);

    let result = create_project(
        &store,
        project_params(
            "Corrupt",
            "project-corrupt",
            vec![source.thread_id.as_str().to_string()],
        ),
    );
    assert!(result.is_err());
    assert!(list(&store, None, 10).projects.is_empty());
}
