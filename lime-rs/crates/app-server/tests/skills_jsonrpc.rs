use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_SKILLS_CONFIG_WRITE, METHOD_SKILLS_EXTRA_ROOTS_SET, METHOD_SKILLS_LIST,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED};
use lime_core::config::ConfigManager;
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn skills_list_preserves_cwd_order_errors_and_exact_metadata_shape() {
    let first = TempDir::new().expect("first cwd");
    let second = TempDir::new().expect("second cwd");
    write_skill(first.path(), "writer", "First description", true);
    write_skill(second.path(), "reviewer", "Second description", false);
    write_invalid_skill(second.path(), "broken");

    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let response = request(
        &server,
        2,
        METHOD_SKILLS_LIST,
        json!({
            "cwds": [first.path(), second.path()],
            "forceReload": true
        }),
    )
    .await;
    let data = response["result"]["data"].as_array().expect("skills data");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["cwd"], first.path().to_string_lossy().as_ref());
    assert_eq!(data[1]["cwd"], second.path().to_string_lossy().as_ref());

    let writer = data[0]["skills"]
        .as_array()
        .expect("first skills")
        .iter()
        .find(|skill| skill["name"] == "writer")
        .expect("writer skill");
    assert_eq!(writer["description"], "First description");
    assert_eq!(writer["scope"], "repo");
    assert_eq!(writer["enabled"], true);
    assert_eq!(writer["interface"]["displayName"], "writer");
    assert_eq!(writer["dependencies"]["tools"][0]["type"], "runtime_tool");
    assert_eq!(writer["dependencies"]["tools"][0]["value"], "Read");
    assert_eq!(
        writer["path"],
        first
            .path()
            .join(".agents/skills/writer/SKILL.md")
            .canonicalize()
            .expect("canonical writer path")
            .to_string_lossy()
            .as_ref()
    );

    let reviewer = data[1]["skills"]
        .as_array()
        .expect("second skills")
        .iter()
        .find(|skill| skill["name"] == "reviewer")
        .expect("reviewer skill");
    assert_eq!(reviewer["enabled"], false);
    let errors = data[1]["errors"].as_array().expect("second errors");
    assert!(errors.iter().any(|error| {
        error["path"]
            .as_str()
            .is_some_and(|path| path.ends_with("broken/SKILL.md"))
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("frontmatter"))
    }));
}

#[tokio::test]
async fn skills_list_force_reload_reloads_changed_metadata() {
    let cwd = TempDir::new().expect("cwd");
    write_skill(cwd.path(), "writer", "Before", true);
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let before = request(
        &server,
        2,
        METHOD_SKILLS_LIST,
        json!({"cwds":[cwd.path()],"forceReload":true}),
    )
    .await;
    assert_eq!(find_skill(&before, "writer")["description"], "Before");

    write_skill(cwd.path(), "writer", "After", true);
    let after = request(
        &server,
        3,
        METHOD_SKILLS_LIST,
        json!({"cwds":[cwd.path()],"forceReload":true}),
    )
    .await;
    assert_eq!(find_skill(&after, "writer")["description"], "After");
}

#[tokio::test]
async fn skills_extra_roots_replace_clear_and_notify_over_public_jsonrpc() {
    let cwd = TempDir::new().expect("cwd");
    let first_root = TempDir::new().expect("first extra root");
    let second_root = TempDir::new().expect("second extra root");
    write_skill_root(first_root.path(), "first-extra", "First extra");
    write_skill_root(second_root.path(), "second-extra", "Second extra");
    let _clear_extra_roots = scopeguard::guard((), |_| {
        let _ = lime_skills::set_runtime_extra_skill_roots(Vec::new());
    });

    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let messages = request_messages(
        &server,
        2,
        METHOD_SKILLS_EXTRA_ROOTS_SET,
        json!({"extraRoots": [first_root.path()]}),
    )
    .await;
    assert_eq!(response_for(&messages, 2)["result"], json!({}));
    assert!(messages.iter().any(|message| {
        message["method"] == "skills/changed" && message["params"] == json!({})
    }));
    let first_list = request(
        &server,
        3,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert_eq!(find_skill(&first_list, "first-extra")["enabled"], true);

    request(
        &server,
        4,
        METHOD_SKILLS_EXTRA_ROOTS_SET,
        json!({"extraRoots": [second_root.path()]}),
    )
    .await;
    let second_list = request(
        &server,
        5,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert!(try_find_skill(&second_list, "first-extra").is_none());
    assert_eq!(find_skill(&second_list, "second-extra")["enabled"], true);

    let missing_root = cwd.path().join("missing-extra-root");
    request(
        &server,
        6,
        METHOD_SKILLS_EXTRA_ROOTS_SET,
        json!({"extraRoots": [missing_root]}),
    )
    .await;
    let missing_list = request(
        &server,
        7,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert!(try_find_skill(&missing_list, "first-extra").is_none());
    assert!(try_find_skill(&missing_list, "second-extra").is_none());

    request(
        &server,
        8,
        METHOD_SKILLS_EXTRA_ROOTS_SET,
        json!({"extraRoots": []}),
    )
    .await;
    let error = request_error(
        &server,
        9,
        METHOD_SKILLS_EXTRA_ROOTS_SET,
        json!({"extraRoots": ["relative/skills"]}),
    )
    .await;
    assert_eq!(error["error"]["code"], -32602);
}

#[tokio::test]
async fn skills_config_write_validates_selector_and_changes_effective_state() {
    let _config_env_lock = config_env_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let temp = TempDir::new().expect("skills config temp");
    let config_path = temp.path().join("config.yaml");
    let previous_config_path = std::env::var_os("LIME_CONFIG_PATH");
    let _restore_config_path = scopeguard::guard(previous_config_path, |value| {
        if let Some(value) = value {
            std::env::set_var("LIME_CONFIG_PATH", value);
        } else {
            std::env::remove_var("LIME_CONFIG_PATH");
        }
    });
    std::env::set_var("LIME_CONFIG_PATH", &config_path);

    let cwd = TempDir::new().expect("cwd");
    write_skill(cwd.path(), "writer", "Writer", true);
    let skill_path = cwd.path().join(".agents/skills/writer/SKILL.md");
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    for (id, params) in [
        (2, json!({"enabled": false})),
        (
            3,
            json!({"path": skill_path, "name": "writer", "enabled": false}),
        ),
        (4, json!({"path": "relative/SKILL.md", "enabled": false})),
    ] {
        let error = request_error(&server, id, METHOD_SKILLS_CONFIG_WRITE, params).await;
        assert_eq!(error["error"]["code"], -32602);
    }

    let disabled = request(
        &server,
        5,
        METHOD_SKILLS_CONFIG_WRITE,
        json!({"name": "writer", "enabled": false}),
    )
    .await;
    assert_eq!(disabled["result"], json!({"effectiveEnabled": false}));
    let listed = request(
        &server,
        6,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert_eq!(find_skill(&listed, "writer")["enabled"], false);

    request(
        &server,
        7,
        METHOD_SKILLS_CONFIG_WRITE,
        json!({"name": "writer", "enabled": true}),
    )
    .await;
    let path_disabled = request(
        &server,
        8,
        METHOD_SKILLS_CONFIG_WRITE,
        json!({"path": skill_path, "enabled": false}),
    )
    .await;
    assert_eq!(path_disabled["result"]["effectiveEnabled"], false);
    let listed = request(
        &server,
        9,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert_eq!(find_skill(&listed, "writer")["enabled"], false);

    request(
        &server,
        10,
        METHOD_SKILLS_CONFIG_WRITE,
        json!({"path": skill_path, "enabled": true}),
    )
    .await;
    let listed = request(
        &server,
        11,
        METHOD_SKILLS_LIST,
        json!({"cwds": [cwd.path()], "forceReload": true}),
    )
    .await;
    assert_eq!(find_skill(&listed, "writer")["enabled"], true);
    let persisted = ConfigManager::load(&config_path).expect("persisted skills config");
    assert!(persisted.config().skills.config.is_empty());
}

fn write_skill(cwd: &Path, name: &str, description: &str, enabled: bool) {
    let skill_dir = cwd.join(".agents/skills").join(name);
    fs::create_dir_all(&skill_dir).expect("skill directory");
    fs::write(
        skill_dir.join("SKILL.md"),
        format!(
            "---\nname: {name}\ndescription: {description}\nallowed-tools: Read\ndisable-model-invocation: {}\n---\n\n# Body\n",
            !enabled
        ),
    )
    .expect("skill file");
}

fn write_invalid_skill(cwd: &Path, name: &str) {
    let skill_dir = cwd.join(".agents/skills").join(name);
    fs::create_dir_all(&skill_dir).expect("invalid skill directory");
    fs::write(skill_dir.join("SKILL.md"), "# Missing frontmatter\n").expect("invalid skill file");
}

fn write_skill_root(root: &Path, name: &str, description: &str) {
    let skill_dir = root.join(name);
    fs::create_dir_all(&skill_dir).expect("extra root skill directory");
    fs::write(
        skill_dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: {description}\n---\n\n# Body\n"),
    )
    .expect("extra root skill file");
}

fn config_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn find_skill<'a>(response: &'a Value, name: &str) -> &'a Value {
    try_find_skill(response, name).unwrap_or_else(|| panic!("missing skill {name}: {response:#}"))
}

fn try_find_skill<'a>(response: &'a Value, name: &str) -> Option<&'a Value> {
    response["result"]["data"][0]["skills"]
        .as_array()
        .expect("skills")
        .iter()
        .find(|skill| skill["name"] == name)
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo":{"name":"skills-jsonrpc-test","version":"1"}}),
    )
    .await;
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc":"2.0","method":METHOD_INITIALIZED,"params":{}}).to_string(),
        )
        .await
        .expect("initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = request_messages(server, id, method, params).await;
    let response = response_for(&messages, id).clone();
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
    );
    response
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = request_messages(server, id, method, params).await;
    let response = response_for(&messages, id).clone();
    assert!(
        response.get("error").is_some(),
        "request succeeded: {response:#}"
    );
    response
}

async fn request_messages(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    server
        .handle_json_line(
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}).to_string(),
        )
        .await
        .expect("JSON-RPC request")
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .collect()
}

fn response_for(messages: &[Value], id: u64) -> &Value {
    messages
        .iter()
        .find(|value| value.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing response id {id}: {messages:#?}"))
}
