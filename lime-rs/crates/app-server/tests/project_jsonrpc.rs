use std::collections::VecDeque;
use std::sync::Arc;

use app_server::{run_json_lines, AppServer, AppServerError, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_PROJECT_CHANGED, METHOD_PROJECT_CREATE, METHOD_PROJECT_DELETE, METHOD_PROJECT_IMPORT,
    METHOD_PROJECT_LIST, METHOD_PROJECT_MOVE, METHOD_PROJECT_READ, METHOD_PROJECT_UPDATE,
    METHOD_THREAD_LIST, METHOD_THREAD_METADATA_UPDATE, METHOD_THREAD_PROJECT_UPDATED,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

const METHOD_THREAD_STARTED: &str = "thread/started";

#[tokio::test]
async fn project_methods_and_thread_fields_require_experimental_api() {
    let temp = TempDir::new().expect("project gate tempdir");
    let mut client = TransportClient::start(project_server(&temp), "project-gate", false).await;

    for (id, method) in [
        (2, METHOD_PROJECT_LIST),
        (3, METHOD_PROJECT_READ),
        (4, METHOD_PROJECT_CREATE),
        (5, METHOD_PROJECT_IMPORT),
        (6, METHOD_PROJECT_UPDATE),
        (7, METHOD_PROJECT_MOVE),
        (8, METHOD_PROJECT_DELETE),
    ] {
        let response = client.request_raw(id, method, json!({})).await;
        assert_eq!(response["error"]["code"], error_codes::INVALID_REQUEST);
        assert_eq!(
            response["error"]["message"],
            "experimental method requires initialize capabilities.experimentalApi"
        );
    }

    for (id, method, params, reason) in [
        (
            9,
            METHOD_THREAD_START,
            json!({"projectId": "project-1"}),
            "thread/start.projectId",
        ),
        (
            10,
            METHOD_THREAD_METADATA_UPDATE,
            json!({"threadId": "thread-1", "projectId": ""}),
            "thread/metadata/update.projectId",
        ),
        (
            11,
            METHOD_THREAD_LIST,
            json!({"projectId": null}),
            "thread/list.projectId",
        ),
    ] {
        let response = client.request_raw(id, method, params).await;
        assert_eq!(response["error"]["code"], error_codes::INVALID_REQUEST);
        assert_eq!(
            response["error"]["message"],
            format!("{reason} requires experimentalApi capability")
        );
    }

    client.shutdown().await;
}

#[tokio::test]
async fn project_crud_pagination_notifications_and_thread_assignment_use_public_jsonrpc() {
    let temp = TempDir::new().expect("project CRUD tempdir");
    let server = project_server(&temp);
    let mut stable = TransportClient::start(server.clone(), "project-stable", false).await;
    let mut client = TransportClient::start(server, "project-crud", true).await;
    let root = temp.path().join("workspace");
    std::fs::create_dir_all(&root).expect("project root");

    let created = client
        .request_ok(
            2,
            METHOD_PROJECT_CREATE,
            json!({
                "name": "  Work  ",
                "roots": [{"path": root}],
                "metadata": {"color": "blue"},
                "idempotencyKey": "project-crud-primary",
            }),
        )
        .await;
    let project = created["result"]["project"].clone();
    let project_id = required_string(&created, "/result/project/id", "project id");
    assert_eq!(project["name"], "Work");
    assert_eq!(
        uuid::Uuid::parse_str(&project_id)
            .expect("project UUID")
            .get_version_num(),
        7
    );
    assert_eq!(project["position"], 0);
    let created_notification = client.take_notification(METHOD_PROJECT_CHANGED).await;
    assert_eq!(
        created_notification["params"],
        json!({"projectId": project_id, "changeType": "created"})
    );
    stable
        .assert_no_wire_message("project notification gate")
        .await;

    let replayed = client
        .request_ok(
            3,
            METHOD_PROJECT_CREATE,
            json!({
                "name": "Changed payload",
                "roots": [],
                "idempotencyKey": "project-crud-primary",
            }),
        )
        .await;
    assert_eq!(replayed["result"]["project"], project);
    client
        .assert_no_wire_message("idempotent create notification")
        .await;

    let assigned_thread = start_thread(&mut client, 4, Some(&project_id)).await;
    let assigned_started = client.take_notification(METHOD_THREAD_STARTED).await;
    assert_eq!(
        assigned_started["params"]["thread"]["projectId"],
        project_id
    );
    let unassigned_thread = start_thread(&mut client, 5, None).await;
    client.take_notification(METHOD_THREAD_STARTED).await;

    let assigned = client
        .request_ok(
            6,
            METHOD_THREAD_LIST,
            json!({"projectId": project_id, "limit": 10}),
        )
        .await;
    assert_eq!(assigned["result"]["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(assigned["result"]["data"][0]["id"], assigned_thread);
    let unassigned = client
        .request_ok(
            7,
            METHOD_THREAD_LIST,
            json!({"projectId": null, "limit": 10}),
        )
        .await;
    assert_eq!(
        unassigned["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(unassigned["result"]["data"][0]["id"], unassigned_thread);

    let cleared = client
        .request_ok(
            8,
            METHOD_THREAD_METADATA_UPDATE,
            json!({"threadId": assigned_thread, "projectId": ""}),
        )
        .await;
    assert_eq!(cleared["result"]["thread"]["projectId"], Value::Null);
    let cleared_notification = client
        .take_notification(METHOD_THREAD_PROJECT_UPDATED)
        .await;
    assert_eq!(
        cleared_notification["params"],
        json!({"threadId": assigned_thread, "projectId": null})
    );
    let now_unassigned = client
        .request_ok(
            9,
            METHOD_THREAD_LIST,
            json!({"projectId": null, "limit": 10}),
        )
        .await;
    assert_eq!(
        now_unassigned["result"]["data"].as_array().map(Vec::len),
        Some(2)
    );

    let reassigned = client
        .request_ok(
            10,
            METHOD_THREAD_METADATA_UPDATE,
            json!({"threadId": assigned_thread, "projectId": project_id}),
        )
        .await;
    assert_eq!(reassigned["result"]["thread"]["projectId"], project_id);
    let reassigned_notification = client
        .take_notification(METHOD_THREAD_PROJECT_UPDATED)
        .await;
    assert_eq!(reassigned_notification["params"]["projectId"], project_id);

    let updated = client
        .request_ok(
            11,
            METHOD_PROJECT_UPDATE,
            json!({
                "projectId": project_id,
                "name": "Renamed",
                "roots": [],
                "metadata": {},
            }),
        )
        .await;
    assert_eq!(updated["result"]["project"]["name"], "Renamed");
    assert_eq!(updated["result"]["project"]["roots"], json!([]));
    client.take_notification(METHOD_PROJECT_CHANGED).await;

    let second = client
        .request_ok(
            12,
            METHOD_PROJECT_CREATE,
            json!({
                "name": "Second",
                "roots": [],
                "idempotencyKey": "project-crud-second",
            }),
        )
        .await;
    let second_id = required_string(&second, "/result/project/id", "second project id");
    client.take_notification(METHOD_PROJECT_CHANGED).await;

    let first_page = client
        .request_ok(13, METHOD_PROJECT_LIST, json!({"limit": 1}))
        .await;
    assert_eq!(
        first_page["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
    let cursor = required_string(&first_page, "/result/nextCursor", "project cursor");
    assert!(!cursor.contains(&project_id));
    let second_page = client
        .request_ok(
            14,
            METHOD_PROJECT_LIST,
            json!({"cursor": cursor, "limit": 1}),
        )
        .await;
    assert_eq!(
        second_page["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(second_page["result"]["nextCursor"], Value::Null);

    client
        .request_ok(
            15,
            METHOD_PROJECT_MOVE,
            json!({"projectId": second_id, "beforeProjectId": project_id}),
        )
        .await;
    client.take_notification(METHOD_PROJECT_CHANGED).await;
    let reordered = client
        .request_ok(16, METHOD_PROJECT_LIST, json!({"limit": 10}))
        .await;
    assert_eq!(reordered["result"]["data"][0]["id"], second_id);
    assert_eq!(reordered["result"]["data"][1]["id"], project_id);

    client
        .request_ok(17, METHOD_PROJECT_DELETE, json!({"projectId": project_id}))
        .await;
    let deleted_project = client.take_notification(METHOD_PROJECT_CHANGED).await;
    assert_eq!(deleted_project["params"]["changeType"], "deleted");
    let unassigned_thread_notification = client
        .take_notification(METHOD_THREAD_PROJECT_UPDATED)
        .await;
    assert_eq!(
        unassigned_thread_notification["params"],
        json!({"threadId": assigned_thread, "projectId": null})
    );
    let deleted_read = client
        .request_raw(18, METHOD_PROJECT_READ, json!({"projectId": project_id}))
        .await;
    assert_eq!(deleted_read["error"]["code"], error_codes::INVALID_PARAMS);
    let thread_read = client
        .request_ok(
            19,
            METHOD_THREAD_READ,
            json!({"threadId": assigned_thread, "includeTurns": false}),
        )
        .await;
    assert_eq!(thread_read["result"]["thread"]["projectId"], Value::Null);
    let deleted_replay = client
        .request_raw(
            20,
            METHOD_PROJECT_CREATE,
            json!({
                "name": "Cannot replay",
                "roots": [],
                "idempotencyKey": "project-crud-primary",
            }),
        )
        .await;
    assert_eq!(deleted_replay["error"]["code"], error_codes::INVALID_PARAMS);

    stable.shutdown().await;
    client.shutdown().await;
}

#[tokio::test]
async fn project_import_is_atomic_and_notifies_before_response_in_order() {
    let temp = TempDir::new().expect("project import tempdir");
    let mut client = TransportClient::start(project_server(&temp), "project-import", true).await;
    let thread_id = start_thread(&mut client, 2, None).await;
    client.take_notification(METHOD_THREAD_STARTED).await;

    let missing = client
        .request_raw(
            3,
            METHOD_PROJECT_IMPORT,
            json!({
                "name": "Missing",
                "roots": [],
                "threads": [thread_id, "missing-thread"],
                "idempotencyKey": "project-import-missing",
            }),
        )
        .await;
    assert_eq!(missing["error"]["code"], error_codes::INVALID_PARAMS);
    let empty = client
        .request_ok(4, METHOD_PROJECT_LIST, json!({"limit": 10}))
        .await;
    assert_eq!(empty["result"], json!({"data": [], "nextCursor": null}));

    let duplicate = client
        .request_raw(
            5,
            METHOD_PROJECT_IMPORT,
            json!({
                "name": "Duplicate",
                "roots": [],
                "threads": [thread_id, thread_id],
                "idempotencyKey": "project-import-duplicate",
            }),
        )
        .await;
    assert_eq!(duplicate["error"]["code"], error_codes::INVALID_PARAMS);

    client
        .write_request(
            6,
            METHOD_PROJECT_IMPORT,
            json!({
                "name": "Imported",
                "roots": [],
                "threads": [thread_id],
                "idempotencyKey": "project-import-success",
            }),
        )
        .await;
    let project_changed = client
        .next_wire_message("project/import project notification")
        .await;
    assert_eq!(project_changed["method"], METHOD_PROJECT_CHANGED);
    let imported_project_id =
        required_string(&project_changed, "/params/projectId", "imported project id");
    let thread_updated = client
        .next_wire_message("project/import thread notification")
        .await;
    assert_eq!(thread_updated["method"], METHOD_THREAD_PROJECT_UPDATED);
    assert_eq!(thread_updated["params"]["threadId"], thread_id);
    assert_eq!(thread_updated["params"]["projectId"], imported_project_id);
    let response = client.next_wire_message("project/import response").await;
    assert_eq!(response["id"], 6);
    assert_eq!(response["result"]["project"]["id"], imported_project_id);

    let replayed = client
        .request_ok(
            7,
            METHOD_PROJECT_IMPORT,
            json!({
                "name": "Changed payload",
                "roots": [],
                "threads": [thread_id],
                "idempotencyKey": "project-import-success",
            }),
        )
        .await;
    assert_eq!(replayed["result"]["project"]["id"], imported_project_id);
    client
        .assert_no_wire_message("idempotent import notification")
        .await;

    client.shutdown().await;
}

#[tokio::test]
async fn project_requests_reject_invalid_names_roots_keys_threads_moves_and_cursors() {
    let temp = TempDir::new().expect("project validation tempdir");
    let mut client =
        TransportClient::start(project_server(&temp), "project-validation", true).await;
    let root = temp.path().join("root");
    std::fs::create_dir_all(&root).expect("validation root");

    for (id, params) in [
        (
            2,
            json!({"name": "   ", "roots": [], "idempotencyKey": "name"}),
        ),
        (
            3,
            json!({
                "name": "Relative",
                "roots": [{"path": "relative/path"}],
                "idempotencyKey": "relative",
            }),
        ),
        (
            4,
            json!({
                "name": "Duplicate roots",
                "roots": [{"path": root}, {"path": root}],
                "idempotencyKey": "duplicate-roots",
            }),
        ),
        (
            5,
            json!({"name": "Empty key", "roots": [], "idempotencyKey": " "}),
        ),
        (
            6,
            json!({
                "name": "Long key",
                "roots": [],
                "idempotencyKey": "x".repeat(513),
            }),
        ),
    ] {
        let response = client.request_raw(id, METHOD_PROJECT_CREATE, params).await;
        assert_eq!(response["error"]["code"], error_codes::INVALID_PARAMS);
    }

    let invalid_cursor = client
        .request_raw(
            7,
            METHOD_PROJECT_LIST,
            json!({"cursor": "not-a-cursor", "limit": 10}),
        )
        .await;
    assert_eq!(invalid_cursor["error"]["code"], error_codes::INVALID_PARAMS);

    let created = client
        .request_ok(
            8,
            METHOD_PROJECT_CREATE,
            json!({"name": "Valid", "roots": [], "idempotencyKey": "valid"}),
        )
        .await;
    let project_id = required_string(&created, "/result/project/id", "valid project id");
    client.take_notification(METHOD_PROJECT_CHANGED).await;
    let invalid_move = client
        .request_raw(
            9,
            METHOD_PROJECT_MOVE,
            json!({"projectId": project_id, "beforeProjectId": "missing-project"}),
        )
        .await;
    assert_eq!(invalid_move["error"]["code"], error_codes::INVALID_PARAMS);

    client.shutdown().await;
}

fn project_server(temp: &TempDir) -> AppServer {
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("project projection store"),
    );
    let runtime = RuntimeCore::default()
        .with_app_config_path(temp.path().join("config.yaml"))
        .with_projection_store(store);
    AppServer::with_runtime(runtime)
}

async fn start_thread(client: &mut TransportClient, id: u64, project_id: Option<&str>) -> String {
    let mut params = json!({
        "model": "fixture-model",
        "modelProvider": "fixture-provider",
    });
    if let Some(project_id) = project_id {
        params["projectId"] = json!(project_id);
    }
    let response = client.request_ok(id, METHOD_THREAD_START, params).await;
    let thread_id = required_string(&response, "/result/thread/id", "thread id");
    assert_eq!(response["result"]["thread"]["projectId"], json!(project_id));
    thread_id
}

fn required_string(value: &Value, pointer: &str, label: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{label} missing at {pointer}: {value:#}"))
        .to_string()
}

struct TransportClient {
    input: DuplexStream,
    lines: Lines<BufReader<DuplexStream>>,
    pending_messages: VecDeque<Value>,
    runner: JoinHandle<Result<(), AppServerError>>,
}

impl TransportClient {
    async fn start(server: AppServer, name: &str, experimental_api: bool) -> Self {
        let (input, input_server) = tokio::io::duplex(128 * 1024);
        let (output_server, output) = tokio::io::duplex(128 * 1024);
        let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
        let mut client = Self {
            input,
            lines: BufReader::new(output).lines(),
            pending_messages: VecDeque::new(),
            runner,
        };
        let mut params = json!({"clientInfo": {"name": name, "version": "1.0.0"}});
        if experimental_api {
            params["capabilities"] = json!({"experimentalApi": true});
        }
        client.request_ok(1, METHOD_INITIALIZE, params).await;
        client
            .write(json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {},
            }))
            .await;
        client
    }

    async fn request_ok(&mut self, id: u64, method: &str, params: Value) -> Value {
        let response = self.request_raw(id, method, params).await;
        assert!(
            response.get("error").is_none(),
            "{method} returned an error: {response:#}"
        );
        assert!(
            response.get("result").is_some(),
            "{method} returned no result"
        );
        response
    }

    async fn request_raw(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.write_request(id, method, params).await;
        loop {
            let message = self.next_wire_message(method).await;
            if message.get("id") == Some(&json!(id)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn write_request(&mut self, id: u64, method: &str, params: Value) {
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await;
    }

    async fn take_notification(&mut self, method: &str) -> Value {
        if let Some(index) = self
            .pending_messages
            .iter()
            .position(|message| message.get("method") == Some(&json!(method)))
        {
            return self
                .pending_messages
                .remove(index)
                .expect("pending notification index");
        }
        loop {
            let message = self.next_wire_message(method).await;
            if message.get("method") == Some(&json!(method)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn assert_no_wire_message(&mut self, scenario: &str) {
        assert!(
            self.pending_messages.is_empty(),
            "unexpected pending JSON-RPC message for {scenario}: {:#?}",
            self.pending_messages
        );
        assert!(
            timeout(Duration::from_millis(50), self.lines.next_line())
                .await
                .is_err(),
            "unexpected JSON-RPC wire message for {scenario}"
        );
    }

    async fn next_wire_message(&mut self, scenario: &str) -> Value {
        let line = timeout(Duration::from_secs(5), self.lines.next_line())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for JSON-RPC message: {scenario}"))
            .expect("read JSON-RPC message")
            .expect("JSON-RPC output closed");
        serde_json::from_str(&line).expect("decode JSON-RPC message")
    }

    async fn write(&mut self, message: Value) {
        self.input
            .write_all(format!("{message}\n").as_bytes())
            .await
            .expect("write JSON-RPC message");
    }

    async fn shutdown(self) {
        drop(self.input);
        timeout(Duration::from_secs(2), self.runner)
            .await
            .expect("JSON lines runner should stop after input closes")
            .expect("JSON lines runner task")
            .expect("JSON lines runner result");
    }
}
