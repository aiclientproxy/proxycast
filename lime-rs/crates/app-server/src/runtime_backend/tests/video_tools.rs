use super::*;
use crate::NoopAppDataSource;
use lime_agent::agent_tools::catalog::VIDEO_GENERATE_TOOL_NAME;
use lime_core::models::{parse_skill_manifest_from_content, split_skill_frontmatter};

const VIDEO_GENERATE_DEFAULT_SKILL: &str =
    include_str!("../../../../../resources/default-skills/video_generate/SKILL.md");

#[test]
fn default_video_generate_skill_uses_current_native_video_task_tool() {
    let manifest = parse_skill_manifest_from_content(VIDEO_GENERATE_DEFAULT_SKILL)
        .expect("video_generate default skill manifest should parse");
    assert!(
        manifest.compliance.validation_errors.is_empty(),
        "video_generate default skill manifest should be standard: {:?}",
        manifest.compliance.validation_errors
    );
    assert_eq!(manifest.metadata.name.as_deref(), Some("video_generate"));
    assert_eq!(
        manifest.metadata.allowed_tools,
        vec![VIDEO_GENERATE_TOOL_NAME.to_string()],
        "video_generate must only expose the current native video task tool"
    );

    let (_frontmatter, body) =
        split_skill_frontmatter(VIDEO_GENERATE_DEFAULT_SKILL).expect("frontmatter");
    assert!(
        body.contains(&format!("调用 `{VIDEO_GENERATE_TOOL_NAME}` 创建真实任务")),
        "video_generate prompt must route through the current native task tool"
    );
    assert!(
        body.contains("不得经 CLI") && body.contains("Bash"),
        "video_generate prompt must forbid CLI/Bash task-creation detours"
    );
}

#[tokio::test]
async fn runtime_backend_registers_video_generation_task_native_tool() {
    let db: lime_core::database::DbConnection = std::sync::Arc::new(std::sync::Mutex::new(
        rusqlite::Connection::open_in_memory().expect("db"),
    ));
    {
        let conn = db.lock().expect("db lock");
        lime_core::database::schema::create_tables(&conn).expect("schema");
    }
    let backend = RuntimeBackend::with_db(db.clone());
    ExecutionBackend::set_app_data_source(&backend, std::sync::Arc::new(NoopAppDataSource))
        .expect("app data source should be accepted");
    backend
        .agent_state
        .init_agent_with_db(&db)
        .await
        .expect("agent should initialize");
    backend
        .register_current_native_tools_if_available()
        .await
        .expect("current native tools should register");

    assert!(
        backend
            .agent_state
            .contains_native_tool(VIDEO_GENERATE_TOOL_NAME)
            .await,
        "{VIDEO_GENERATE_TOOL_NAME} should be registered as the current video task native tool"
    );
}
