mod support;

use app_server::{AppServer, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN, METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
    METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE, METHOD_THREAD_SHELL_COMMAND,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn background_terminals_are_thread_scoped_and_control_real_processes() {
    let temp = TempDir::new().expect("background terminal JSON-RPC temp dir");
    let core = support::runtime_core_with_chat_provider(&temp, "provider-test", "model-test");
    let server = AppServer::with_runtime(core.clone());
    initialize_server(&server).await;

    let first = start_thread(&server, 2, &temp).await;
    let second = start_thread(&server, 3, &temp).await;
    let command = long_running_command();

    request(
        &server,
        4,
        METHOD_THREAD_SHELL_COMMAND,
        json!({"threadId": first.thread_id, "command": command}),
    )
    .await;
    let listed = wait_for_terminal(&server, &first.thread_id, 10).await;
    let terminal = &listed["result"]["data"][0];
    assert_eq!(terminal["command"], command);
    assert_eq!(
        terminal["cwd"].as_str(),
        Some(
            temp.path()
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .as_ref()
        )
    );
    assert_eq!(terminal["osPid"], Value::Null);
    assert_eq!(terminal["cpuPercent"], Value::Null);
    assert_eq!(terminal["rssKb"], Value::Null);
    let process_id = terminal["processId"]
        .as_str()
        .expect("public process id")
        .to_string();
    process_id
        .parse::<u64>()
        .expect("Codex background process id is numeric");

    let wrong_thread = request(
        &server,
        11,
        METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE,
        json!({"threadId": second.thread_id, "processId": process_id}),
    )
    .await;
    assert_eq!(wrong_thread["result"]["terminated"], false);

    let terminated = request(
        &server,
        12,
        METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE,
        json!({"threadId": first.thread_id, "processId": process_id}),
    )
    .await;
    assert_eq!(terminated["result"]["terminated"], true);
    assert_eq!(
        list(&server, 13, &first.thread_id).await["result"]["data"],
        json!([])
    );
    wait_for_shell_terminal(&core, &first.session_id, 1).await;

    request(
        &server,
        14,
        METHOD_THREAD_SHELL_COMMAND,
        json!({"threadId": first.thread_id, "command": command}),
    )
    .await;
    wait_for_terminal(&server, &first.thread_id, 15).await;
    let cleaned = request(
        &server,
        16,
        METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN,
        json!({"threadId": first.thread_id}),
    )
    .await;
    assert_eq!(cleaned["result"], json!({}));
    assert_eq!(
        list(&server, 17, &first.thread_id).await["result"]["data"],
        json!([])
    );
    wait_for_shell_terminal(&core, &first.session_id, 2).await;

    let invalid_cursor = request_raw(
        &server,
        18,
        METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
        json!({"threadId": first.thread_id, "cursor": "not-a-process-id"}),
    )
    .await;
    assert_eq!(
        invalid_cursor.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
}

struct StartedThread {
    thread_id: String,
    session_id: String,
}

async fn start_thread(server: &AppServer, id: u64, temp: &TempDir) -> StartedThread {
    let response = request(
        server,
        id,
        METHOD_THREAD_START,
        json!({
            "model": "model-test",
            "modelProvider": "provider-test",
            "cwd": temp.path().to_string_lossy(),
            "historyMode": "paginated"
        }),
    )
    .await;
    StartedThread {
        thread_id: response["result"]["thread"]["id"]
            .as_str()
            .expect("thread id")
            .to_string(),
        session_id: response["result"]["thread"]["sessionId"]
            .as_str()
            .expect("session id")
            .to_string(),
    }
}

async fn wait_for_terminal(server: &AppServer, thread_id: &str, first_id: u64) -> Value {
    timeout(Duration::from_secs(5), async {
        let mut id = first_id;
        loop {
            let response = list(server, id, thread_id).await;
            if response["result"]["data"]
                .as_array()
                .is_some_and(|data| !data.is_empty())
            {
                return response;
            }
            id += 1;
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background terminal registration timeout")
}

async fn wait_for_shell_terminal(core: &RuntimeCore, session_id: &str, expected: usize) {
    timeout(Duration::from_secs(5), async {
        loop {
            let terminal_count = core
                .events_for_session(session_id)
                .expect("shell events")
                .iter()
                .filter(|event| {
                    matches!(
                        event.event_type.as_str(),
                        "turn.completed" | "turn.canceled" | "turn.failed"
                    )
                })
                .count();
            if terminal_count >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("shell terminal event timeout");
}

async fn list(server: &AppServer, id: u64, thread_id: &str) -> Value {
    request(
        server,
        id,
        METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
        json!({"threadId": thread_id}),
    )
    .await
}

fn long_running_command() -> &'static str {
    if cfg!(windows) {
        "ping -n 30 127.0.0.1 >NUL"
    } else {
        "sleep 30"
    }
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-background-terminals-jsonrpc-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            })
            .to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request")
        .iter()
        .map(|line| serde_json::from_str(line).expect("decode JSON-RPC response"))
        .find(|value: &Value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}
