use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tool_runtime::apply_patch::runtime_apply_patch_executor_handle;
use tool_runtime::file_read_execution::{execute_runtime_file_read_tool, RuntimeFileReadRequest};
use tool_runtime::file_search_execution::{
    execute_runtime_file_search_tool, RuntimeFileSearchRequest,
};
use tool_runtime::filesystem_gateway::{
    RuntimeFileEntry, RuntimeFileMetadata, RuntimeFilePatchResult, RuntimeFileSystemGateway,
    RuntimeFileWalkOptions,
};
use tool_runtime::tool_call::{ToolCall, ToolEnvironment};
use tool_runtime::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
use tool_runtime::tool_lifecycle::{
    ToolLifecycleEmissionFuture, ToolLifecycleEmitter, ToolLifecycleEvent,
};

#[derive(Debug, Default)]
struct RecordingGateway {
    files: Mutex<HashMap<PathBuf, Vec<u8>>>,
    walks: Mutex<Vec<Vec<RuntimeFileEntry>>>,
    calls: Mutex<Vec<String>>,
    patch_error: Option<String>,
}

impl RecordingGateway {
    fn with_file(path: impl Into<PathBuf>, contents: impl AsRef<[u8]>) -> Arc<Self> {
        Arc::new(Self {
            files: Mutex::new(HashMap::from([(path.into(), contents.as_ref().to_vec())])),
            ..Self::default()
        })
    }

    fn with_walk(self: Arc<Self>, entries: Vec<RuntimeFileEntry>) -> Arc<Self> {
        self.walks.lock().expect("walk lock").push(entries);
        self
    }

    fn record(&self, operation: &str, environment_id: &str) {
        self.calls
            .lock()
            .expect("call lock")
            .push(format!("{operation}:{environment_id}"));
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("call lock").clone()
    }
}

#[async_trait]
impl RuntimeFileSystemGateway for RecordingGateway {
    async fn read_file(
        &self,
        environment_id: &str,
        path: &Path,
        _sandbox_policy: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        self.record("read", environment_id);
        self.files
            .lock()
            .expect("file lock")
            .get(path)
            .cloned()
            .ok_or_else(|| format!("fixture file not found: {}", path.display()))
    }

    async fn write_file(
        &self,
        environment_id: &str,
        path: &Path,
        data: &[u8],
        _sandbox_policy: Option<&str>,
    ) -> Result<(), String> {
        self.record("write", environment_id);
        self.files
            .lock()
            .expect("file lock")
            .insert(path.to_path_buf(), data.to_vec());
        Ok(())
    }

    async fn metadata(
        &self,
        environment_id: &str,
        _path: &Path,
        _sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFileMetadata, String> {
        self.record("metadata", environment_id);
        Err("metadata is not used by this fixture".to_string())
    }

    async fn canonicalize(
        &self,
        environment_id: &str,
        path: &Path,
        _sandbox_policy: Option<&str>,
    ) -> Result<PathBuf, String> {
        self.record("canonicalize", environment_id);
        Ok(path.to_path_buf())
    }

    async fn read_directory(
        &self,
        environment_id: &str,
        _path: &Path,
        _sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String> {
        self.record("read_directory", environment_id);
        Err("read_directory is not used by this fixture".to_string())
    }

    async fn walk(
        &self,
        environment_id: &str,
        _path: &Path,
        _options: RuntimeFileWalkOptions,
        _sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String> {
        self.record("walk", environment_id);
        self.walks
            .lock()
            .expect("walk lock")
            .first()
            .cloned()
            .ok_or_else(|| "fixture walk is not configured".to_string())
    }

    async fn apply_patch(
        &self,
        environment_id: &str,
        _working_directory: &Path,
        _patch: &str,
        _sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFilePatchResult, String> {
        self.record("apply_patch", environment_id);
        if let Some(error) = &self.patch_error {
            return Err(error.clone());
        }
        Ok(RuntimeFilePatchResult {
            modified_paths: vec![PathBuf::from("remote/changed.txt")],
        })
    }
}

#[derive(Debug, Default)]
struct NoopLifecycleEmitter;

impl ToolLifecycleEmitter for NoopLifecycleEmitter {
    fn emit<'a>(&'a self, _event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async {})
    }
}

fn result_text(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|content| content.as_text().map(|text| text.text.clone()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn remote_read_uses_gateway_instead_of_host_filesystem() {
    let remote_path = PathBuf::from("/remote/workspace/note.txt");
    let gateway = RecordingGateway::with_file(&remote_path, "remote content");
    let host_path = tempfile::tempdir()
        .expect("host temp directory")
        .path()
        .join("note.txt");
    let result = execute_runtime_file_read_tool(RuntimeFileReadRequest {
        tool_name: "Read",
        params: &json!({"path": remote_path}),
        working_directory: PathBuf::from("/remote/workspace"),
        cancel_token: None,
        environment_id: Some("remote".to_string()),
        filesystem_gateway: Some(gateway.clone()),
        sandbox_policy: None,
    })
    .await
    .expect("Read should be recognized")
    .expect("remote Read should succeed");

    assert!(result_text(&result).contains("remote content"));
    assert!(
        !host_path.exists(),
        "remote Read must not create/read host files"
    );
    assert_eq!(gateway.calls(), vec!["read:remote"]);
}

#[tokio::test]
async fn remote_filesystem_tools_fail_closed_without_gateway() {
    let read = execute_runtime_file_read_tool(RuntimeFileReadRequest {
        tool_name: "Read",
        params: &json!({"path": "note.txt"}),
        working_directory: PathBuf::from("/remote/workspace"),
        cancel_token: None,
        environment_id: Some("remote".to_string()),
        filesystem_gateway: None,
        sandbox_policy: None,
    })
    .await
    .expect("Read should be recognized")
    .expect_err("remote Read must fail without gateway");
    assert!(read.message.contains("filesystem gateway is unavailable"));

    let search = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
        tool_name: "Glob",
        params: &json!({"pattern": "*.rs"}),
        working_directory: PathBuf::from("/remote/workspace"),
        cancel_token: None,
        environment_id: Some("remote".to_string()),
        filesystem_gateway: None,
        sandbox_policy: None,
    })
    .await
    .expect("Glob should be recognized")
    .expect_err("remote Glob must fail without gateway");
    assert!(search.message.contains("filesystem gateway is unavailable"));
}

#[tokio::test]
async fn remote_glob_and_grep_use_gateway_walk_and_read() {
    let remote_path = PathBuf::from("/remote/workspace/src/lib.rs");
    let gateway =
        RecordingGateway::with_file(&remote_path, "needle in remote file").with_walk(vec![
            RuntimeFileEntry {
                path: remote_path.clone(),
                is_directory: false,
                is_file: true,
            },
        ]);

    let glob = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
        tool_name: "Glob",
        params: &json!({"pattern": "src/*.rs"}),
        working_directory: PathBuf::from("/remote/workspace"),
        cancel_token: None,
        environment_id: Some("remote".to_string()),
        filesystem_gateway: Some(gateway.clone()),
        sandbox_policy: None,
    })
    .await
    .expect("Glob should be recognized")
    .expect("remote Glob should succeed");
    assert!(result_text(&glob).contains("lib.rs"));

    let grep = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
        tool_name: "Grep",
        params: &json!({"pattern": "needle"}),
        working_directory: PathBuf::from("/remote/workspace"),
        cancel_token: None,
        environment_id: Some("remote".to_string()),
        filesystem_gateway: Some(gateway.clone()),
        sandbox_policy: None,
    })
    .await
    .expect("Grep should be recognized")
    .expect("remote Grep should succeed");
    assert!(result_text(&grep).contains("needle in remote file"));
    assert_eq!(
        gateway.calls(),
        vec!["walk:remote", "walk:remote", "read:remote"]
    );
}

#[tokio::test]
async fn remote_apply_patch_uses_gateway_and_forwards_errors() {
    let gateway = RecordingGateway::with_file("/remote/workspace/unused.txt", "unused");
    let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
        working_directory: PathBuf::from("/remote/workspace"),
        session_id: "filesystem-gateway-test".to_string(),
        cancel_token: None,
        workspace_sandbox: None,
    })
    .with_filesystem_gateway(gateway.clone());
    let emitter = Arc::new(NoopLifecycleEmitter);
    let call = ToolCall::new(
        "turn-1",
        "call-1",
        "apply_patch",
        json!({
            "patch": "*** Begin Patch\n*** Add File: changed.txt\n+remote\n*** End Patch"
        }),
        vec![ToolEnvironment::new(
            "remote",
            PathBuf::from("/remote/workspace"),
        )],
        emitter,
    );

    let result = runtime_apply_patch_executor_handle()
        .execute_call(&call, &context, None)
        .await
        .expect("remote apply_patch should succeed");
    assert!(result.success);
    assert_eq!(result.metadata["environment_id"], "remote");
    assert_eq!(gateway.calls(), vec!["apply_patch:remote"]);

    let failing = Arc::new(RecordingGateway {
        patch_error: Some("remote disconnected".to_string()),
        ..RecordingGateway::default()
    });
    let failing_context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
        working_directory: PathBuf::from("/remote/workspace"),
        session_id: "filesystem-gateway-test-failing".to_string(),
        cancel_token: None,
        workspace_sandbox: None,
    })
    .with_filesystem_gateway(failing);
    let error = runtime_apply_patch_executor_handle()
        .execute_call(&call, &failing_context, None)
        .await
        .expect_err("gateway errors must not fall back to host apply_patch");
    assert!(error.message().contains("remote disconnected"));
}
