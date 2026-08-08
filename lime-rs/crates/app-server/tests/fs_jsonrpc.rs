use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

use app_server::{run_json_lines, AppServer, AppServerError, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_FS_CHANGED, METHOD_FS_COPY, METHOD_FS_CREATE_DIRECTORY, METHOD_FS_GET_METADATA,
    METHOD_FS_READ_DIRECTORY, METHOD_FS_READ_FILE, METHOD_FS_REMOVE, METHOD_FS_UNWATCH,
    METHOD_FS_WATCH, METHOD_FS_WRITE_FILE,
};
use app_server_protocol::{error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::task::JoinHandle;
use tokio::time::timeout;

#[tokio::test]
async fn exact_fs_methods_round_trip_over_public_jsonrpc_and_retire_file_system_wire() {
    let temp = TempDir::new().expect("fs JSON-RPC temp dir");
    let root = temp.path().join("workspace");
    let source = root.join("source.bin");
    let copied = root.join("copied.bin");
    let copied_directory = temp.path().join("workspace-copy");
    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::default()),
        "fs-jsonrpc-test",
    )
    .await;

    client
        .request_ok(
            2,
            METHOD_FS_CREATE_DIRECTORY,
            json!({"path": path_string(&root)}),
        )
        .await;
    client
        .request_ok(
            3,
            METHOD_FS_WRITE_FILE,
            json!({
                "path": path_string(&source),
                "dataBase64": STANDARD.encode([0_u8, 1, 2, 255]),
            }),
        )
        .await;

    let read = client
        .request_ok(
            4,
            METHOD_FS_READ_FILE,
            json!({"path": path_string(&source)}),
        )
        .await;
    assert_eq!(
        read.pointer("/result/dataBase64"),
        Some(&json!(STANDARD.encode([0_u8, 1, 2, 255])))
    );

    let metadata = client
        .request_ok(
            5,
            METHOD_FS_GET_METADATA,
            json!({"path": path_string(&source)}),
        )
        .await;
    assert_eq!(metadata.pointer("/result/isFile"), Some(&json!(true)));
    assert_eq!(metadata.pointer("/result/isDirectory"), Some(&json!(false)));

    let listing = client
        .request_ok(
            6,
            METHOD_FS_READ_DIRECTORY,
            json!({"path": path_string(&root)}),
        )
        .await;
    assert_eq!(
        listing.pointer("/result/entries"),
        Some(&json!([{
            "fileName": "source.bin",
            "isDirectory": false,
            "isFile": true,
        }]))
    );

    client
        .request_ok(
            7,
            METHOD_FS_COPY,
            json!({
                "sourcePath": path_string(&source),
                "destinationPath": path_string(&copied),
            }),
        )
        .await;
    assert_eq!(
        std::fs::read(&copied).expect("copied file"),
        [0_u8, 1, 2, 255]
    );

    client
        .request_ok(
            8,
            METHOD_FS_COPY,
            json!({
                "sourcePath": path_string(&root),
                "destinationPath": path_string(&copied_directory),
                "recursive": true,
            }),
        )
        .await;
    assert!(copied_directory.join("source.bin").is_file());

    client
        .request_ok(9, METHOD_FS_REMOVE, json!({"path": path_string(&root)}))
        .await;
    client
        .request_ok(
            10,
            METHOD_FS_REMOVE,
            json!({"path": path_string(&copied_directory)}),
        )
        .await;
    assert!(!root.exists());
    assert!(!copied_directory.exists());

    let relative = client
        .request_raw(11, METHOD_FS_READ_FILE, json!({"path": "relative.txt"}))
        .await;
    assert_eq!(
        relative.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_PARAMS))
    );

    let retired = client
        .request_raw(
            12,
            "fileSystem/readFilePreview",
            json!({"path": path_string(&source), "maxSize": 1024}),
        )
        .await;
    assert_eq!(
        retired.pointer("/error/code"),
        Some(&json!(error_codes::METHOD_NOT_FOUND))
    );

    client.shutdown().await;
}

#[tokio::test]
async fn fs_watch_and_unwatch_round_trip_over_public_transport() {
    let root = TempDir::new().expect("watch root");
    let watched_file = root.path().join("watched.txt");
    std::fs::write(&watched_file, "before").expect("seed watched file");
    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::default()),
        "fs-watch-jsonrpc-test",
    )
    .await;

    client
        .request_ok(
            2,
            METHOD_FS_WATCH,
            json!({"watchId": "workspace", "path": path_string(root.path())}),
        )
        .await;

    client
        .request_ok(
            3,
            METHOD_FS_WRITE_FILE,
            json!({
                "path": path_string(&watched_file),
                "dataBase64": STANDARD.encode(b"after"),
            }),
        )
        .await;
    let changed = client
        .next_notification(METHOD_FS_CHANGED, "watched file update")
        .await;
    assert_eq!(
        changed.pointer("/params/watchId"),
        Some(&json!("workspace"))
    );
    assert_changed_path(&changed, &watched_file);

    client
        .request_ok(4, METHOD_FS_UNWATCH, json!({"watchId": "workspace"}))
        .await;
    client
        .request_ok(
            5,
            METHOD_FS_WRITE_FILE,
            json!({
                "path": path_string(&watched_file),
                "dataBase64": STANDARD.encode(b"done"),
            }),
        )
        .await;
    assert!(
        client.pending_messages.is_empty(),
        "fs/unwatch left a queued notification: {:#?}",
        client.pending_messages
    );
    assert!(
        timeout(Duration::from_millis(500), client.lines.next_line())
            .await
            .is_err(),
        "fs/unwatch returned before its watcher stopped"
    );

    client.shutdown().await;
}

struct TransportClient {
    input: DuplexStream,
    lines: Lines<BufReader<DuplexStream>>,
    pending_messages: VecDeque<Value>,
    runner: JoinHandle<Result<(), AppServerError>>,
}

impl TransportClient {
    async fn start(server: AppServer, name: &str) -> Self {
        let (input, input_server) = tokio::io::duplex(64 * 1024);
        let (output_server, output) = tokio::io::duplex(64 * 1024);
        let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
        let mut client = Self {
            input,
            lines: BufReader::new(output).lines(),
            pending_messages: VecDeque::new(),
            runner,
        };
        client
            .request_ok(
                1,
                METHOD_INITIALIZE,
                json!({"clientInfo": {"name": name, "version": "1.0.0"}}),
            )
            .await;
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
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await;
        loop {
            let message = self.next_wire_message(method).await;
            if message.get("id") == Some(&json!(id)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn next_notification(&mut self, method: &str, scenario: &str) -> Value {
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
            let message = self.next_wire_message(scenario).await;
            if message.get("method") == Some(&json!(method)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
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

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn assert_changed_path(notification: &Value, expected: &Path) {
    let expected = canonical_path_string(expected);
    assert!(
        notification
            .pointer("/params/changedPaths")
            .and_then(Value::as_array)
            .is_some_and(|paths| paths.contains(&json!(expected))),
        "fs/changed did not include {expected}: {notification:#}"
    );
}

fn canonical_path_string(path: &Path) -> String {
    path.canonicalize()
        .expect("canonical fs test path")
        .to_string_lossy()
        .into_owned()
}
