use super::environment_exec::{RemoteEnvironmentStatus, RemoteExecClient, RemoteFsWalkResponse};
use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use crate::runtime::{RuntimeCoreError, RuntimeEvent};
use agent_protocol::world_state::{
    RuntimeWorldEnvironmentSelection, RuntimeWorldEnvironmentStatus,
};
use app_server_protocol::protocol::v2::{
    EnvironmentConnectionNotification, EnvironmentInfoResponse, EnvironmentShellInfo,
    EnvironmentStatusKind, EnvironmentStatusResponse, PathUri, ServerNotification,
    TurnEnvironmentParams,
};
use app_server_protocol::{error_codes, JsonRpcError, JsonRpcNotification};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, Notify, RwLock};
use tokio::time::{timeout, Duration};
use tool_runtime::filesystem_gateway::{
    RuntimeFileEntry, RuntimeFileMetadata, RuntimeFilePatchResult, RuntimeFileSystemGateway,
    RuntimeFileWalkOptions,
};
use url::Url;

const LOCAL_ENVIRONMENT_ID: &str = "local";
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const RECONNECT_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EnvironmentConnectionStatus {
    Pending,
    Ready,
    Disconnected(String),
}

#[derive(Clone, Debug)]
pub(crate) struct EnvironmentStatusEvent {
    pub(crate) environment_id: String,
    pub(crate) status: EnvironmentConnectionStatus,
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PersistedEnvironment {
    environment_id: String,
    exec_server_url: String,
    connect_timeout_ms: Option<u64>,
}

#[derive(Debug)]
struct RemoteEnvironment {
    environment_id: String,
    exec_server_url: String,
    connect_timeout: Duration,
    retired: AtomicBool,
    status: Mutex<EnvironmentConnectionStatus>,
    info: Mutex<Option<EnvironmentInfoResponse>>,
    client: Mutex<Option<Arc<RemoteExecClient>>>,
    ready: Notify,
}

#[derive(Debug)]
pub(crate) struct EnvironmentRegistry {
    environments: RwLock<HashMap<String, Arc<RemoteEnvironment>>>,
    status_events: broadcast::Sender<EnvironmentStatusEvent>,
    storage_path: Option<PathBuf>,
    started: AtomicBool,
}

impl EnvironmentRegistry {
    pub(crate) fn new() -> Self {
        let (status_events, _) = broadcast::channel(64);
        Self {
            environments: RwLock::new(HashMap::new()),
            status_events,
            storage_path: None,
            started: AtomicBool::new(false),
        }
    }

    pub(crate) fn new_with_storage(storage_path: PathBuf) -> Self {
        let mut registry = Self::new();
        registry.storage_path = Some(storage_path.clone());
        match std::fs::read_to_string(&storage_path) {
            Ok(content) => match serde_json::from_str::<Vec<PersistedEnvironment>>(&content) {
                Ok(entries) => {
                    let mut environments = registry
                        .environments
                        .try_write()
                        .expect("new registry write lock must be available");
                    for entry in entries {
                        if entry.environment_id.trim().is_empty()
                            || entry.environment_id == LOCAL_ENVIRONMENT_ID
                            || validate_remote_url(&entry.exec_server_url).is_err()
                        {
                            tracing::warn!(
                                environment_id = %entry.environment_id,
                                "ignoring invalid persisted Environment registry entry"
                            );
                            continue;
                        }
                        let id = entry.environment_id.trim().to_string();
                        environments.insert(
                            id.clone(),
                            Arc::new(RemoteEnvironment {
                                environment_id: id,
                                exec_server_url: entry.exec_server_url,
                                connect_timeout: entry
                                    .connect_timeout_ms
                                    .map(Duration::from_millis)
                                    .unwrap_or(DEFAULT_CONNECT_TIMEOUT),
                                retired: AtomicBool::new(false),
                                status: Mutex::new(EnvironmentConnectionStatus::Pending),
                                info: Mutex::new(None),
                                client: Mutex::new(None),
                                ready: Notify::new(),
                            }),
                        );
                    }
                }
                Err(error) => tracing::warn!(
                    path = %storage_path.display(),
                    %error,
                    "ignoring malformed persisted Environment registry"
                ),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                path = %storage_path.display(),
                %error,
                "unable to read persisted Environment registry"
            ),
        }
        registry
    }

    pub(crate) fn start(self: &Arc<Self>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let registry = Arc::clone(self);
        tokio::spawn(async move {
            let environments = registry
                .environments
                .read()
                .await
                .values()
                .cloned()
                .collect::<Vec<_>>();
            for environment in environments {
                let registry = Arc::clone(&registry);
                tokio::spawn(async move {
                    registry.connect(Arc::clone(&environment)).await;
                    registry.monitor(environment).await;
                });
            }
        });
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<EnvironmentStatusEvent> {
        self.status_events.subscribe()
    }

    pub(crate) async fn upsert(
        self: &Arc<Self>,
        environment_id: String,
        exec_server_url: String,
        connect_timeout: Option<Duration>,
    ) -> Result<(), String> {
        validate_remote_url(&exec_server_url)?;
        let environment = Arc::new(RemoteEnvironment {
            environment_id: environment_id.clone(),
            exec_server_url,
            connect_timeout: connect_timeout.unwrap_or(DEFAULT_CONNECT_TIMEOUT),
            retired: AtomicBool::new(false),
            status: Mutex::new(EnvironmentConnectionStatus::Pending),
            info: Mutex::new(None),
            client: Mutex::new(None),
            ready: Notify::new(),
        });
        let previous = self
            .environments
            .write()
            .await
            .insert(environment_id.clone(), Arc::clone(&environment));
        if let Some(previous) = previous {
            previous.retired.store(true, Ordering::Release);
        }
        self.persist().await?;
        if self.started.load(Ordering::Acquire) {
            let registry = Arc::clone(self);
            tokio::spawn(async move {
                registry.connect(Arc::clone(&environment)).await;
                registry.monitor(environment).await;
            });
        }
        Ok(())
    }

    async fn monitor(&self, environment: Arc<RemoteEnvironment>) {
        loop {
            tokio::time::sleep(RECONNECT_INTERVAL).await;
            if environment.retired.load(Ordering::Acquire) {
                return;
            }
            match self.status(&environment.environment_id).await {
                Some(EnvironmentConnectionStatus::Disconnected(_)) => {
                    self.connect(Arc::clone(&environment)).await;
                }
                Some(EnvironmentConnectionStatus::Ready)
                | Some(EnvironmentConnectionStatus::Pending) => {}
                None => return,
            }
        }
    }

    async fn connect(&self, environment: Arc<RemoteEnvironment>) {
        if environment.retired.load(Ordering::Acquire) {
            return;
        }
        if !matches!(
            environment.status.lock().await.clone(),
            EnvironmentConnectionStatus::Disconnected(_)
        ) {
            self.set_status(&environment, EnvironmentConnectionStatus::Pending)
                .await;
        }
        *environment.client.lock().await = None;
        let result = RemoteExecClient::connect(
            environment.exec_server_url.as_str(),
            environment.connect_timeout,
        )
        .await;

        let (client, info) = match result {
            Ok(value) => value,
            Err(error) => {
                self.set_status(
                    &environment,
                    EnvironmentConnectionStatus::Disconnected(error),
                )
                .await;
                environment.ready.notify_waiters();
                return;
            }
        };

        {
            if environment.retired.load(Ordering::Acquire) {
                return;
            }
            let cwd = match info.cwd {
                Some(cwd) => match PathUri::parse(&cwd) {
                    Ok(cwd) => Some(cwd),
                    Err(error) => {
                        self.set_status(
                            &environment,
                            EnvironmentConnectionStatus::Disconnected(error.clone()),
                        )
                        .await;
                        environment.ready.notify_waiters();
                        return;
                    }
                },
                None => None,
            };
            let translated = EnvironmentInfoResponse {
                shell: EnvironmentShellInfo {
                    name: info.shell.name,
                    path: info.shell.path,
                },
                cwd,
            };
            *environment.info.lock().await = Some(translated);
            *environment.client.lock().await = Some(client);
            self.set_status(&environment, EnvironmentConnectionStatus::Ready)
                .await;
            environment.ready.notify_waiters();
        }
    }

    async fn status(&self, environment_id: &str) -> Option<EnvironmentConnectionStatus> {
        let environment = self
            .environments
            .read()
            .await
            .get(environment_id)
            .cloned()?;
        let current = environment.status.lock().await.clone();
        if !matches!(current, EnvironmentConnectionStatus::Ready) {
            return Some(current);
        }
        let probe = environment
            .client
            .lock()
            .await
            .clone()
            .map(|client| async move {
                client
                    .request::<RemoteEnvironmentStatus>("environment/status", json!({}))
                    .await
            });
        let probe = match probe {
            Some(probe) => Some(probe.await),
            None => None,
        };
        let Some(probe) = probe else {
            let error = "exec-server connection is unavailable".to_string();
            self.set_status(
                &environment,
                EnvironmentConnectionStatus::Disconnected(error.clone()),
            )
            .await;
            return Some(EnvironmentConnectionStatus::Disconnected(error));
        };
        match probe {
            Ok(status) if status.status == "ready" => Some(EnvironmentConnectionStatus::Ready),
            Ok(status) => {
                let error = format!(
                    "exec-server reported environment status `{}`",
                    status.status
                );
                *environment.client.lock().await = None;
                self.set_status(
                    &environment,
                    EnvironmentConnectionStatus::Disconnected(error.clone()),
                )
                .await;
                Some(EnvironmentConnectionStatus::Disconnected(error))
            }
            Err(error) => {
                *environment.client.lock().await = None;
                self.set_status(
                    &environment,
                    EnvironmentConnectionStatus::Disconnected(error.clone()),
                )
                .await;
                Some(EnvironmentConnectionStatus::Disconnected(error))
            }
        }
    }

    async fn info(&self, environment_id: &str) -> Result<EnvironmentInfoResponse, String> {
        let environment = self
            .environments
            .read()
            .await
            .get(environment_id)
            .cloned()
            .ok_or_else(|| format!("environment '{environment_id}' is not configured"))?;
        let ready_notification = environment.ready.notified();
        if matches!(
            environment.status.lock().await.clone(),
            EnvironmentConnectionStatus::Pending
        ) {
            let _ = timeout(DEFAULT_CONNECT_TIMEOUT, ready_notification).await;
        }
        let info = environment
            .info
            .lock()
            .await
            .clone()
            .ok_or_else(|| "environment is not ready".to_string());
        info
    }

    async fn set_status(
        &self,
        environment: &RemoteEnvironment,
        status: EnvironmentConnectionStatus,
    ) {
        let notify = {
            let mut current = environment.status.lock().await;
            if *current == status {
                return;
            } else {
                let notify = !same_connection_status_kind(&current, &status);
                *current = status.clone();
                notify
            }
        };
        if notify && !environment.retired.load(Ordering::Acquire) {
            let _ = self.status_events.send(EnvironmentStatusEvent {
                environment_id: environment.environment_id.clone(),
                status,
            });
        }
    }

    async fn persist(&self) -> Result<(), String> {
        let Some(path) = self.storage_path.as_ref() else {
            return Ok(());
        };
        let entries = self
            .environments
            .read()
            .await
            .values()
            .map(|environment| PersistedEnvironment {
                environment_id: environment.environment_id.clone(),
                exec_server_url: environment.exec_server_url.clone(),
                connect_timeout_ms: Some(environment.connect_timeout.as_millis() as u64),
            })
            .collect::<Vec<_>>();
        let content = serde_json::to_vec_pretty(&entries)
            .map_err(|error| format!("failed to serialize Environment registry: {error}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!("failed to create Environment registry directory: {error}")
            })?;
        }
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, content)
            .map_err(|error| format!("failed to write Environment registry: {error}"))?;
        std::fs::rename(&temp, path)
            .map_err(|error| format!("failed to replace Environment registry: {error}"))
    }

    pub(crate) async fn normalize_selections(
        &self,
        selections: Option<Vec<TurnEnvironmentParams>>,
    ) -> Result<Option<Vec<TurnEnvironmentParams>>, String> {
        let Some(selections) = selections else {
            return Ok(None);
        };
        let mut seen_ids = HashMap::<String, ()>::with_capacity(selections.len());
        let mut normalized = Vec::with_capacity(selections.len());
        for selection in selections {
            let environment_id = selection.environment_id.trim().to_string();
            if environment_id.is_empty() {
                return Err("environmentId must not be empty".to_string());
            }
            if seen_ids.insert(environment_id.clone(), ()).is_some() {
                return Err(format!("duplicate turn environment id `{environment_id}`"));
            }
            if environment_id != LOCAL_ENVIRONMENT_ID
                && !self.environments.read().await.contains_key(&environment_id)
            {
                return Err(format!("unknown turn environment id `{environment_id}`"));
            }
            let cwd = normalize_environment_path(&selection.cwd, "cwd", &environment_id)?;
            let mut roots = selection
                .runtime_workspace_roots
                .unwrap_or_else(|| vec![cwd.clone()])
                .into_iter()
                .map(|root| {
                    normalize_environment_path(&root, "runtimeWorkspaceRoots", &environment_id)
                })
                .collect::<Result<Vec<_>, _>>()?;
            roots.dedup();
            if roots.is_empty() {
                roots.push(cwd.clone());
            }
            normalized.push(TurnEnvironmentParams {
                environment_id,
                cwd,
                runtime_workspace_roots: Some(roots),
            });
        }
        Ok(Some(normalized))
    }

    async fn selection_status(&self, environment_id: &str) -> EnvironmentConnectionStatus {
        if environment_id == LOCAL_ENVIRONMENT_ID {
            EnvironmentConnectionStatus::Ready
        } else {
            self.status(environment_id).await.unwrap_or_else(|| {
                EnvironmentConnectionStatus::Disconnected(
                    "environment is not configured".to_string(),
                )
            })
        }
    }

    pub(crate) async fn execution_client(
        &self,
        environment_id: &str,
    ) -> Result<Arc<RemoteExecClient>, String> {
        let environment = self
            .environments
            .read()
            .await
            .get(environment_id)
            .cloned()
            .ok_or_else(|| format!("environment '{environment_id}' is not configured"))?;
        if !matches!(
            environment.status.lock().await.clone(),
            EnvironmentConnectionStatus::Ready
        ) {
            return Err(format!("environment '{environment_id}' is not ready"));
        }
        let client = environment
            .client
            .lock()
            .await
            .clone()
            .ok_or_else(|| format!("environment '{environment_id}' connection is unavailable"));
        client
    }
}

#[async_trait]
impl RuntimeFileSystemGateway for EnvironmentRegistry {
    async fn read_file(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        let client = self.execution_client(environment_id).await?;
        let path = remote_path_uri(path)?;
        client
            .fs_read_file(
                &path,
                Some(remote_fs_sandbox(
                    &path,
                    sandbox_policy,
                    RemoteFsAccess::Read,
                )?),
            )
            .await
    }

    async fn write_file(
        &self,
        environment_id: &str,
        path: &Path,
        data: &[u8],
        sandbox_policy: Option<&str>,
    ) -> Result<(), String> {
        let client = self.execution_client(environment_id).await?;
        let path = remote_path_uri(path)?;
        client
            .fs_write_file(
                &path,
                data,
                Some(remote_fs_sandbox(
                    &path,
                    sandbox_policy,
                    RemoteFsAccess::Write,
                )?),
            )
            .await
    }

    async fn metadata(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFileMetadata, String> {
        let client = self.execution_client(environment_id).await?;
        let path_uri = remote_path_uri(path)?;
        let metadata = client
            .fs_get_metadata(
                &path_uri,
                Some(remote_fs_sandbox(
                    &path_uri,
                    sandbox_policy,
                    RemoteFsAccess::Read,
                )?),
            )
            .await?;
        Ok(RuntimeFileMetadata {
            is_directory: metadata.is_directory,
            is_file: metadata.is_file,
            is_symlink: metadata.is_symlink,
            size: metadata.size,
            created_at_ms: metadata.created_at_ms,
            modified_at_ms: metadata.modified_at_ms,
        })
    }

    async fn canonicalize(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<PathBuf, String> {
        let client = self.execution_client(environment_id).await?;
        let path_uri = remote_path_uri(path)?;
        let response = client
            .fs_canonicalize(
                &path_uri,
                Some(remote_fs_sandbox(
                    &path_uri,
                    sandbox_policy,
                    RemoteFsAccess::Read,
                )?),
            )
            .await?;
        remote_path_to_host_path(&response.path)
    }

    async fn read_directory(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String> {
        let client = self.execution_client(environment_id).await?;
        let path_uri = remote_path_uri(path)?;
        let entries = client
            .fs_read_directory(
                &path_uri,
                Some(remote_fs_sandbox(
                    &path_uri,
                    sandbox_policy,
                    RemoteFsAccess::Read,
                )?),
            )
            .await?
            .entries;
        Ok(entries
            .into_iter()
            .map(|entry| RuntimeFileEntry {
                path: path.join(entry.file_name),
                is_directory: entry.is_directory,
                is_file: entry.is_file,
            })
            .collect())
    }

    async fn walk(
        &self,
        environment_id: &str,
        path: &Path,
        options: RuntimeFileWalkOptions,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String> {
        let client = self.execution_client(environment_id).await?;
        let path_uri = remote_path_uri(path)?;
        let response: RemoteFsWalkResponse = client
            .fs_walk(
                &path_uri,
                json!({
                    "maxDepth": options.max_depth,
                    "maxDirectories": options.max_directories,
                    "maxEntries": options.max_entries,
                    "followDirectorySymlinks": options.follow_directory_symlinks,
                    "pruneHiddenDirectories": options.prune_hidden_directories,
                }),
                Some(remote_fs_sandbox(
                    &path_uri,
                    sandbox_policy,
                    RemoteFsAccess::Read,
                )?),
            )
            .await?;
        if !response.errors.is_empty() {
            return Err(format!(
                "remote filesystem walk encountered {} errors",
                response.errors.len()
            ));
        }
        if response.truncated {
            tracing::debug!(environment_id, "remote filesystem walk was truncated");
        }
        response
            .entries
            .into_iter()
            .map(|entry| {
                Ok(RuntimeFileEntry {
                    path: remote_path_to_host_path(&entry.path)?,
                    is_directory: entry.kind.eq_ignore_ascii_case("directory"),
                    is_file: entry.kind.eq_ignore_ascii_case("file"),
                })
            })
            .collect()
    }

    async fn apply_patch(
        &self,
        environment_id: &str,
        working_directory: &Path,
        patch: &str,
        sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFilePatchResult, String> {
        let parsed = patch_apply::parse_patch(patch).map_err(|error| error.to_string())?;
        if let Some(patch_environment_id) = parsed.environment_id.as_deref() {
            if patch_environment_id != environment_id {
                return Err(format!(
                    "apply_patch environment id `{patch_environment_id}` does not match selected environment `{environment_id}`"
                ));
            }
        }
        let client = self.execution_client(environment_id).await?;
        let mut modified_paths = Vec::with_capacity(parsed.hunks.len());
        for hunk in parsed.hunks {
            match hunk {
                patch_apply::Hunk::AddFile { path, contents } => {
                    let remote_path = join_remote_path(working_directory, &path)?;
                    let uri = remote_path_uri(&remote_path)?;
                    if let Some(parent) = remote_path.parent() {
                        let parent_uri = remote_path_uri(parent)?;
                        client
                            .fs_create_directory(
                                &parent_uri,
                                Some(remote_fs_sandbox(
                                    &parent_uri,
                                    sandbox_policy,
                                    RemoteFsAccess::Write,
                                )?),
                            )
                            .await?;
                    }
                    client
                        .fs_write_file(
                            &uri,
                            contents.as_bytes(),
                            Some(remote_fs_sandbox(
                                &uri,
                                sandbox_policy,
                                RemoteFsAccess::Write,
                            )?),
                        )
                        .await?;
                    modified_paths.push(remote_path);
                }
                patch_apply::Hunk::DeleteFile { path } => {
                    let remote_path = join_remote_path(working_directory, &path)?;
                    let uri = remote_path_uri(&remote_path)?;
                    client
                        .fs_remove(
                            &uri,
                            Some(remote_fs_sandbox(
                                &uri,
                                sandbox_policy,
                                RemoteFsAccess::Write,
                            )?),
                        )
                        .await?;
                    modified_paths.push(remote_path);
                }
                patch_apply::Hunk::UpdateFile {
                    path,
                    move_path,
                    chunks,
                } => {
                    let remote_path = join_remote_path(working_directory, &path)?;
                    let source_uri = remote_path_uri(&remote_path)?;
                    let original = client
                        .fs_read_file(
                            &source_uri,
                            Some(remote_fs_sandbox(
                                &source_uri,
                                sandbox_policy,
                                RemoteFsAccess::Read,
                            )?),
                        )
                        .await?;
                    let original = String::from_utf8(original).map_err(|_| {
                        format!(
                            "remote patch target is not UTF-8: {}",
                            remote_path.display()
                        )
                    })?;
                    let updated = apply_remote_chunks(&original, &chunks)?;
                    let destination = move_path
                        .as_ref()
                        .map(|path| join_remote_path(working_directory, path))
                        .transpose()?
                        .unwrap_or_else(|| remote_path.clone());
                    let destination_uri = remote_path_uri(&destination)?;
                    if let Some(parent) = destination.parent() {
                        let parent_uri = remote_path_uri(parent)?;
                        client
                            .fs_create_directory(
                                &parent_uri,
                                Some(remote_fs_sandbox(
                                    &parent_uri,
                                    sandbox_policy,
                                    RemoteFsAccess::Write,
                                )?),
                            )
                            .await?;
                    }
                    client
                        .fs_write_file(
                            &destination_uri,
                            updated.as_bytes(),
                            Some(remote_fs_sandbox(
                                &destination_uri,
                                sandbox_policy,
                                RemoteFsAccess::Write,
                            )?),
                        )
                        .await?;
                    if destination != remote_path {
                        client
                            .fs_remove(
                                &source_uri,
                                Some(remote_fs_sandbox(
                                    &source_uri,
                                    sandbox_policy,
                                    RemoteFsAccess::Write,
                                )?),
                            )
                            .await?;
                    }
                    modified_paths.push(destination);
                }
            }
        }
        Ok(RuntimeFilePatchResult { modified_paths })
    }
}

fn join_remote_path(working_directory: &Path, path: &Path) -> Result<PathBuf, String> {
    let path = path.to_string_lossy();
    let is_absolute = path.starts_with('/')
        || path.starts_with('\\')
        || (path.len() >= 3
            && path.as_bytes()[1] == b':'
            && matches!(path.as_bytes()[2], b'/' | b'\\'));
    if is_absolute {
        return Ok(PathBuf::from(path.as_ref()));
    }
    let cwd = working_directory.to_string_lossy();
    if cwd.trim().is_empty() {
        return Err("remote Environment cwd must not be empty".to_string());
    }
    Ok(PathBuf::from(format!(
        "{}/{}",
        cwd.trim_end_matches(['/', '\\']),
        path.replace('\\', "/")
    )))
}

fn apply_remote_chunks(
    original: &str,
    chunks: &[patch_apply::UpdateFileChunk],
) -> Result<String, String> {
    let temp = tempfile::tempdir().map_err(|error| error.to_string())?;
    let file = temp.path().join("remote-target.txt");
    std::fs::write(&file, original).map_err(|error| error.to_string())?;
    patch_apply::apply_hunks_to_workdir(
        &[patch_apply::Hunk::UpdateFile {
            path: PathBuf::from("remote-target.txt"),
            move_path: None,
            chunks: chunks.to_vec(),
        }],
        temp.path(),
    )
    .map_err(|error| error.to_string())?;
    std::fs::read_to_string(file).map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteFsAccess {
    Read,
    Write,
}

fn remote_fs_sandbox(
    path: &PathUri,
    sandbox_policy: Option<&str>,
    default_access: RemoteFsAccess,
) -> Result<Value, String> {
    let policy = sandbox_policy
        .map(str::trim)
        .filter(|policy| !policy.is_empty())
        .unwrap_or(match default_access {
            RemoteFsAccess::Read => "read-only",
            RemoteFsAccess::Write => "workspace-write",
        })
        .to_ascii_lowercase()
        .replace('_', "-");
    let (access, network) = match policy.as_str() {
        "read-only" => ("read", "restricted"),
        "workspace-write" => ("write", "restricted"),
        "danger-full-access" => {
            return Ok(json!({
                "permissions": {
                    "type": "managed",
                    "fileSystem": {"type": "unrestricted"},
                    "network": "enabled"
                }
            }));
        }
        _ => {
            return Err(format!(
                "unsupported remote filesystem sandbox policy `{policy}`"
            ));
        }
    };
    Ok(json!({
        "permissions": {
            "type": "managed",
            "fileSystem": {
                "type": "restricted",
                "entries": [{
                    "path": {"type": "path", "path": path.as_str()},
                    "access": access
                }]
            },
            "network": network
        }
    }))
}

fn remote_path_uri(path: &Path) -> Result<PathUri, String> {
    let rendered = path.to_string_lossy();
    if rendered.len() >= 3
        && rendered.as_bytes()[0].is_ascii_alphabetic()
        && rendered.as_bytes()[1] == b':'
        && matches!(rendered.as_bytes()[2], b'/' | b'\\')
    {
        return PathUri::parse(&format!("file:///{}", rendered.replace('\\', "/")));
    }
    let normalized = rendered.replace('\\', "/");
    if let Some(unc) = normalized.strip_prefix("//") {
        let mut segments = unc.splitn(2, '/');
        let host = segments.next().unwrap_or_default();
        let share = segments.next().unwrap_or_default();
        if !host.is_empty() && !share.is_empty() {
            return PathUri::parse(&format!("file://{unc}"));
        }
    }
    PathUri::from_host_path(path)
}

fn remote_path_to_host_path(value: &str) -> Result<PathBuf, String> {
    let url =
        Url::parse(value).map_err(|error| format!("invalid remote filesystem path: {error}"))?;
    if url.scheme() != "file" {
        return Err(format!(
            "unsupported remote filesystem path scheme `{}`",
            url.scheme()
        ));
    }
    let decoded = urlencoding::decode(url.path())
        .map_err(|error| format!("invalid remote filesystem path encoding: {error}"))?;
    let decoded = decoded.as_ref();
    if let Some(host) = url.host_str().filter(|host| *host != "localhost") {
        return Ok(PathBuf::from(format!("//{host}{decoded}")));
    }
    if decoded.len() >= 3 && decoded.as_bytes()[0] == b'/' && decoded.as_bytes()[2] == b':' {
        let drive_path = &decoded[1..];
        #[cfg(windows)]
        return Ok(PathBuf::from(drive_path.replace('/', "\\")));
        #[cfg(not(windows))]
        return Ok(PathBuf::from(drive_path));
    }
    Ok(PathBuf::from(decoded))
}

fn same_connection_status_kind(
    left: &EnvironmentConnectionStatus,
    right: &EnvironmentConnectionStatus,
) -> bool {
    matches!(
        (left, right),
        (
            EnvironmentConnectionStatus::Pending,
            EnvironmentConnectionStatus::Pending
        ) | (
            EnvironmentConnectionStatus::Ready,
            EnvironmentConnectionStatus::Ready
        ) | (
            EnvironmentConnectionStatus::Disconnected(_),
            EnvironmentConnectionStatus::Disconnected(_)
        )
    )
}

fn normalize_environment_path(
    value: &str,
    field: &str,
    environment_id: &str,
) -> Result<String, String> {
    let value = value.trim();
    let absolute = value.starts_with('/')
        || value.starts_with('\\')
        || (value.len() >= 3
            && value.as_bytes()[1] == b':'
            && matches!(value.as_bytes()[2], b'\\' | b'/'));
    if value.is_empty() || value.contains('\0') || !absolute {
        return Err(format!(
            "invalid {field} for environment `{environment_id}`: path `{value}` must be absolute"
        ));
    }
    Ok(value.to_string())
}

impl RequestProcessor {
    pub(crate) fn start_environment_runtime(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        self.environment_registry.start();
        let mut events = self.environment_registry.subscribe();
        let processor = self.clone();
        tokio::spawn(async move {
            while let Ok(event) = events.recv().await {
                let thread_ids = processor
                    .selected_environment_threads
                    .lock()
                    .expect("selected Environment mutex poisoned")
                    .get(&event.environment_id)
                    .cloned()
                    .unwrap_or_default();
                let notification = match event.status {
                    EnvironmentConnectionStatus::Ready => ServerNotification::EnvironmentConnected(
                        EnvironmentConnectionNotification {
                            thread_id: String::new(),
                            environment_id: event.environment_id.clone(),
                        },
                    ),
                    EnvironmentConnectionStatus::Disconnected(_) => {
                        ServerNotification::EnvironmentDisconnected(
                            EnvironmentConnectionNotification {
                                thread_id: String::new(),
                                environment_id: event.environment_id.clone(),
                            },
                        )
                    }
                    EnvironmentConnectionStatus::Pending => continue,
                };
                for thread_id in thread_ids {
                    let notification = match notification.clone() {
                        ServerNotification::EnvironmentConnected(mut params) => {
                            params.thread_id = thread_id;
                            ServerNotification::EnvironmentConnected(params)
                        }
                        ServerNotification::EnvironmentDisconnected(mut params) => {
                            params.thread_id = thread_id;
                            ServerNotification::EnvironmentDisconnected(params)
                        }
                        _ => unreachable!(),
                    };
                    processor.publish_server_notification(notification).await;
                }
            }
        });
    }

    pub(super) fn record_environment_selections(
        &self,
        thread_id: &str,
        selections: Option<&[TurnEnvironmentParams]>,
    ) {
        let Some(selections) = selections else {
            return;
        };
        let mut selected = self
            .selected_environment_threads
            .lock()
            .expect("selected Environment mutex poisoned");
        for threads in selected.values_mut() {
            threads.remove(thread_id);
        }
        for selection in selections {
            selected
                .entry(selection.environment_id.clone())
                .or_default()
                .insert(thread_id.to_string());
        }
        selected.retain(|_, threads| !threads.is_empty());
    }

    pub(super) fn forget_environment_selections(&self, thread_id: &str) {
        let mut selected = self
            .selected_environment_threads
            .lock()
            .expect("selected Environment mutex poisoned");
        for threads in selected.values_mut() {
            threads.remove(thread_id);
        }
        selected.retain(|_, threads| !threads.is_empty());
    }

    pub(super) fn ensure_environment_execution_lowering(
        &self,
        selections: Option<&[TurnEnvironmentParams]>,
    ) -> Result<(), JsonRpcError> {
        let Some(remote) = selections
            .unwrap_or_default()
            .iter()
            .find(|selection| selection.environment_id != LOCAL_ENVIRONMENT_ID)
        else {
            return Ok(());
        };
        if self.environment_execution_lowering {
            return Ok(());
        }
        Err(JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            format!(
                "remote Environment '{}' is selected but RuntimeCore remote process/filesystem execution is not available",
                remote.environment_id
            ),
        ))
    }

    pub(super) fn append_environment_world_state(
        &self,
        session_id: &str,
        selections: Option<&[TurnEnvironmentParams]>,
    ) -> Result<Vec<app_server_protocol::AgentEvent>, RuntimeCoreError> {
        let Some(selections) = selections else {
            return Ok(Vec::new());
        };
        let current = environment_world_state(selections);
        let previous = self
            .runtime
            .events_for_session(session_id)?
            .into_iter()
            .rev()
            .find(|event| {
                event.event_type == "world_state"
                    && event.payload.get("section").and_then(Value::as_str) == Some("environments")
            })
            .and_then(|event| event.payload.get("state").cloned());
        let Some(previous) = previous else {
            return self.runtime.append_external_runtime_events(
                session_id,
                None,
                vec![RuntimeEvent::new(
                    "world_state",
                    json!({
                        "schema": "world_state.v1",
                        "kind": "full",
                        "section": "environments",
                        "state": current,
                    }),
                )],
            );
        };
        if previous == current {
            return Ok(Vec::new());
        }
        let patch = environment_world_state_patch(&previous, &current);
        self.runtime.append_external_runtime_events(
            session_id,
            None,
            vec![RuntimeEvent::new(
                "world_state",
                json!({
                    "schema": "world_state.v1",
                    "kind": "patch",
                    "section": "environments",
                    "patch": patch,
                    "state": current,
                }),
            )],
        )
    }

    pub(super) async fn normalize_environment_selections(
        &self,
        selections: Option<Vec<TurnEnvironmentParams>>,
    ) -> Result<Option<Vec<TurnEnvironmentParams>>, JsonRpcError> {
        self.environment_registry
            .normalize_selections(selections)
            .await
            .map_err(|error| JsonRpcError::new(error_codes::INVALID_REQUEST, error))
    }

    pub(super) async fn environment_world_state_snapshot(
        &self,
        selections: Option<&[TurnEnvironmentParams]>,
    ) -> Vec<RuntimeWorldEnvironmentSelection> {
        let Some(selections) = selections else {
            return Vec::new();
        };
        let primary_environment_id = selections
            .first()
            .map(|selection| selection.environment_id.as_str());
        let mut snapshot = Vec::with_capacity(selections.len());
        for selection in selections {
            let (status, shell) = if selection.environment_id == LOCAL_ENVIRONMENT_ID {
                (
                    RuntimeWorldEnvironmentStatus::Ready,
                    Some(local_shell_info().name),
                )
            } else {
                let environment = self
                    .environment_registry
                    .environments
                    .read()
                    .await
                    .get(&selection.environment_id)
                    .cloned();
                match environment {
                    Some(environment) => {
                        let status = match environment.status.lock().await.clone() {
                            EnvironmentConnectionStatus::Pending => {
                                RuntimeWorldEnvironmentStatus::Pending
                            }
                            EnvironmentConnectionStatus::Ready => {
                                RuntimeWorldEnvironmentStatus::Ready
                            }
                            EnvironmentConnectionStatus::Disconnected(_) => {
                                RuntimeWorldEnvironmentStatus::Disconnected
                            }
                        };
                        let shell = environment
                            .info
                            .lock()
                            .await
                            .as_ref()
                            .map(|info| info.shell.name.clone());
                        (status, shell)
                    }
                    None => (RuntimeWorldEnvironmentStatus::Unknown, None),
                }
            };
            snapshot.push(RuntimeWorldEnvironmentSelection {
                environment_id: selection.environment_id.clone(),
                cwd: selection.cwd.clone(),
                runtime_workspace_roots: selection
                    .runtime_workspace_roots
                    .clone()
                    .unwrap_or_default(),
                primary: primary_environment_id == Some(selection.environment_id.as_str()),
                status: Some(status),
                shell,
            });
        }
        snapshot.sort_by(|left, right| left.environment_id.cmp(&right.environment_id));
        snapshot
    }

    pub(super) async fn environment_selection_notifications(
        &self,
        thread_id: &str,
        selections: Option<&[TurnEnvironmentParams]>,
    ) -> Vec<JsonRpcNotification> {
        let Some(selections) = selections else {
            return Vec::new();
        };
        let mut notifications = Vec::with_capacity(selections.len());
        for selection in selections {
            let notification = match self
                .environment_registry
                .selection_status(&selection.environment_id)
                .await
            {
                EnvironmentConnectionStatus::Ready => {
                    ServerNotification::EnvironmentConnected(EnvironmentConnectionNotification {
                        thread_id: thread_id.to_string(),
                        environment_id: selection.environment_id.clone(),
                    })
                }
                EnvironmentConnectionStatus::Disconnected(_) => {
                    ServerNotification::EnvironmentDisconnected(EnvironmentConnectionNotification {
                        thread_id: thread_id.to_string(),
                        environment_id: selection.environment_id.clone(),
                    })
                }
                EnvironmentConnectionStatus::Pending => continue,
            };
            notifications.push(notification.into());
        }
        notifications
    }

    pub(super) async fn handle_environment_add_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::protocol::v2::EnvironmentAddParams = parse_params(params)?;
        let environment_id = params.environment_id.trim().to_string();
        let exec_server_url = params.exec_server_url.trim().to_string();
        if environment_id.is_empty() || exec_server_url.is_empty() {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "environment/add requires environmentId and execServerUrl",
            ));
        }
        if environment_id == LOCAL_ENVIRONMENT_ID {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "environmentId 'local' is reserved",
            ));
        }
        self.environment_registry
            .upsert(
                environment_id,
                exec_server_url,
                params.connect_timeout_ms.map(Duration::from_millis),
            )
            .await
            .map_err(|error| JsonRpcError::new(error_codes::INVALID_PARAMS, error))?;
        dispatch_result(app_server_protocol::protocol::v2::EnvironmentAddResponse {})
    }

    pub(super) async fn handle_environment_info_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::protocol::v2::EnvironmentInfoParams =
            parse_params(params)?;
        if params.environment_id == LOCAL_ENVIRONMENT_ID {
            let shell = local_shell_info();
            let cwd = std::env::current_dir()
                .ok()
                .and_then(|path| PathUri::from_host_path(path).ok());
            return dispatch_result(EnvironmentInfoResponse { shell, cwd });
        }
        let response = self
            .environment_registry
            .info(&params.environment_id)
            .await
            .map_err(|error| JsonRpcError::new(error_codes::INVALID_REQUEST, error))?;
        dispatch_result(response)
    }

    pub(super) async fn handle_environment_status_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::protocol::v2::EnvironmentStatusParams =
            parse_params(params)?;
        let response = if params.environment_id == LOCAL_ENVIRONMENT_ID {
            EnvironmentStatusResponse {
                status: EnvironmentStatusKind::Ready,
                error: None,
            }
        } else {
            match self
                .environment_registry
                .status(&params.environment_id)
                .await
            {
                Some(EnvironmentConnectionStatus::Pending) => EnvironmentStatusResponse {
                    status: EnvironmentStatusKind::Pending,
                    error: None,
                },
                Some(EnvironmentConnectionStatus::Ready) => EnvironmentStatusResponse {
                    status: EnvironmentStatusKind::Ready,
                    error: None,
                },
                Some(EnvironmentConnectionStatus::Disconnected(error)) => {
                    EnvironmentStatusResponse {
                        status: EnvironmentStatusKind::Disconnected,
                        error: Some(error),
                    }
                }
                None => EnvironmentStatusResponse {
                    status: EnvironmentStatusKind::Unknown,
                    error: Some(format!(
                        "environment '{}' is not configured",
                        params.environment_id
                    )),
                },
            }
        };
        dispatch_result(response)
    }
}

fn environment_world_state(selections: &[TurnEnvironmentParams]) -> Value {
    let environments = selections
        .iter()
        .map(|selection| {
            (
                selection.environment_id.clone(),
                json!({
                    "cwd": selection.cwd,
                    "runtimeWorkspaceRoots": selection.runtime_workspace_roots,
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();
    json!({"environments": environments})
}

fn environment_world_state_patch(previous: &Value, current: &Value) -> Value {
    let previous = previous
        .get("environments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let current = current
        .get("environments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let upsert = current
        .iter()
        .filter(|(id, value)| previous.get(*id) != Some(*value))
        .map(|(id, value)| (id.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let removed = previous
        .keys()
        .filter(|id| !current.contains_key(*id))
        .cloned()
        .collect::<Vec<_>>();
    json!({"upsert": upsert, "removed": removed})
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use futures::{SinkExt, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;
    use tokio_tungstenite::tungstenite::Message;

    #[tokio::test]
    async fn normalizes_local_selection_and_defaults_workspace_root() {
        let registry = EnvironmentRegistry::new();
        let selections = registry
            .normalize_selections(Some(vec![TurnEnvironmentParams {
                environment_id: " local ".to_string(),
                cwd: "/workspace".to_string(),
                runtime_workspace_roots: None,
            }]))
            .await
            .expect("local selection should be valid")
            .expect("selections should be present");

        assert_eq!(selections[0].environment_id, "local");
        assert_eq!(
            selections[0].runtime_workspace_roots,
            Some(vec!["/workspace".to_string()])
        );
    }

    #[tokio::test]
    async fn rejects_duplicate_environment_ids_and_relative_paths() {
        let registry = EnvironmentRegistry::new();
        let duplicate = registry
            .normalize_selections(Some(vec![
                TurnEnvironmentParams {
                    environment_id: "local".to_string(),
                    cwd: "/one".to_string(),
                    runtime_workspace_roots: None,
                },
                TurnEnvironmentParams {
                    environment_id: "local".to_string(),
                    cwd: "/two".to_string(),
                    runtime_workspace_roots: None,
                },
            ]))
            .await
            .expect_err("duplicate IDs must fail");
        assert!(duplicate.contains("duplicate turn environment id"));

        let relative = registry
            .normalize_selections(Some(vec![TurnEnvironmentParams {
                environment_id: "local".to_string(),
                cwd: "relative".to_string(),
                runtime_workspace_roots: None,
            }]))
            .await
            .expect_err("relative cwd must fail");
        assert!(relative.contains("must be absolute"));

        assert_eq!(
            normalize_environment_path("C:/workspace", "cwd", "local")
                .expect("Windows drive paths should be accepted"),
            "C:/workspace"
        );
        assert!(normalize_environment_path("/bad\0path", "cwd", "local").is_err());
    }

    #[tokio::test]
    async fn persisted_registry_restores_remote_environment_selection() {
        let temp = tempfile::tempdir().expect("temporary registry directory");
        let path = temp.path().join("environments.json");
        let registry = Arc::new(EnvironmentRegistry::new_with_storage(path.clone()));
        registry
            .upsert(
                "remote".to_string(),
                "ws://127.0.0.1:3210".to_string(),
                Some(Duration::from_millis(750)),
            )
            .await
            .expect("persist remote Environment");

        let persisted: Value = serde_json::from_slice(
            &std::fs::read(&path).expect("read persisted Environment registry"),
        )
        .expect("parse persisted Environment registry");
        assert_eq!(persisted[0]["environmentId"], "remote");
        assert_eq!(persisted[0]["connectTimeoutMs"], 750);

        let restored = EnvironmentRegistry::new_with_storage(path);
        let selections = restored
            .normalize_selections(Some(vec![TurnEnvironmentParams {
                environment_id: "remote".to_string(),
                cwd: "/workspace".to_string(),
                runtime_workspace_roots: None,
            }]))
            .await
            .expect("restored registry should accept remote selection")
            .expect("selection should be present");
        assert_eq!(selections[0].environment_id, "remote");
    }

    #[tokio::test]
    async fn remote_filesystem_gateway_lowers_fs_requests_over_one_websocket() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind filesystem fixture");
        let address = listener.local_addr().expect("filesystem fixture address");
        let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let server_calls = Arc::clone(&calls);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept filesystem fixture");
            let mut socket = accept_async(stream)
                .await
                .expect("filesystem fixture websocket handshake");
            while let Some(message) = socket.next().await {
                let Message::Text(text) = message.expect("filesystem fixture message") else {
                    continue;
                };
                let request: Value = serde_json::from_str(&text).expect("filesystem fixture JSON");
                let method = request["method"]
                    .as_str()
                    .expect("filesystem fixture method");
                server_calls
                    .lock()
                    .expect("filesystem fixture call lock")
                    .push(method.to_string());
                let Some(id) = request.get("id") else {
                    continue;
                };
                let result = match method {
                    "initialize" => json!({"sessionId": "filesystem-fixture-session"}),
                    "environment/info" => json!({
                        "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                        "cwd": "file:///remote/workspace"
                    }),
                    "environment/status" => json!({"status": "ready"}),
                    "fs/readFile" => json!({
                        "dataBase64": base64::engine::general_purpose::STANDARD
                            .encode("remote file contents")
                    }),
                    "fs/writeFile" | "fs/createDirectory" | "fs/remove" => json!({}),
                    "fs/getMetadata" => json!({
                        "isDirectory": false,
                        "isFile": true,
                        "isSymlink": false,
                        "size": 20,
                        "createdAtMs": 10,
                        "modifiedAtMs": 20
                    }),
                    "fs/canonicalize" => json!({
                        "path": "file:///remote/workspace/note.txt"
                    }),
                    "fs/readDirectory" => json!({
                        "entries": [{
                            "fileName": "entry.txt",
                            "isDirectory": false,
                            "isFile": true
                        }]
                    }),
                    "fs/walk" => json!({
                        "entries": [{
                            "path": "file:///remote/workspace/src/lib.rs",
                            "kind": "file"
                        }],
                        "errors": [],
                        "truncated": false
                    }),
                    method => panic!("unexpected filesystem fixture method: {method}"),
                };
                socket
                    .send(Message::Text(
                        json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                    ))
                    .await
                    .expect("send filesystem fixture response");
            }
        });

        let registry = Arc::new(EnvironmentRegistry::new());
        registry.start();
        registry
            .upsert(
                "fixture".to_string(),
                format!("ws://{address}"),
                Some(Duration::from_secs(2)),
            )
            .await
            .expect("register filesystem fixture");
        for _ in 0..40 {
            if matches!(
                registry.status("fixture").await,
                Some(EnvironmentConnectionStatus::Ready)
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(matches!(
            registry.status("fixture").await,
            Some(EnvironmentConnectionStatus::Ready)
        ));

        let cwd = PathBuf::from("/remote/workspace");
        assert_eq!(
            registry
                .read_file("fixture", &cwd.join("note.txt"), None)
                .await
                .expect("remote read"),
            b"remote file contents"
        );
        registry
            .write_file("fixture", &cwd.join("note.txt"), b"updated", None)
            .await
            .expect("remote write");
        assert_eq!(
            registry
                .metadata("fixture", &cwd.join("note.txt"), None)
                .await
                .expect("remote metadata")
                .size,
            20
        );
        assert_eq!(
            registry
                .canonicalize("fixture", &cwd.join("note.txt"), None)
                .await
                .expect("remote canonicalize"),
            cwd.join("note.txt")
        );
        assert_eq!(
            registry
                .read_directory("fixture", &cwd, None)
                .await
                .expect("remote directory")
                .first()
                .expect("directory entry")
                .path,
            cwd.join("entry.txt")
        );
        assert_eq!(
            registry
                .walk(
                    "fixture",
                    &cwd,
                    RuntimeFileWalkOptions {
                        max_depth: 8,
                        max_directories: 100,
                        max_entries: 100,
                        follow_directory_symlinks: false,
                        prune_hidden_directories: true,
                    },
                    None,
                )
                .await
                .expect("remote walk")
                .first()
                .expect("walk entry")
                .path,
            PathBuf::from("/remote/workspace/src/lib.rs")
        );
        registry
            .apply_patch(
                "fixture",
                &cwd,
                "*** Begin Patch\n*** Add File: added.txt\n+added\n*** End Patch",
                None,
            )
            .await
            .expect("remote apply patch");
        registry
            .apply_patch(
                "fixture",
                &cwd,
                "*** Begin Patch\n*** Delete File: added.txt\n*** End Patch",
                None,
            )
            .await
            .expect("remote delete patch");

        let calls = calls.lock().expect("filesystem fixture call lock").clone();
        for method in [
            "fs/readFile",
            "fs/writeFile",
            "fs/getMetadata",
            "fs/canonicalize",
            "fs/readDirectory",
            "fs/walk",
            "fs/createDirectory",
            "fs/remove",
        ] {
            assert!(calls.iter().any(|call| call == method), "missing {method}");
        }
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn remote_environment_reconnects_after_socket_loss() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind reconnect fixture");
        let address = listener.local_addr().expect("reconnect fixture address");
        let connections = Arc::new(AtomicUsize::new(0));
        let fixture_connections = Arc::clone(&connections);
        let fixture = tokio::spawn(async move {
            for connection_index in 0..2 {
                let (stream, _) = listener.accept().await.expect("accept reconnect fixture");
                fixture_connections.fetch_add(1, Ordering::SeqCst);
                let mut socket = accept_async(stream)
                    .await
                    .expect("reconnect fixture websocket");
                while let Some(message) = socket.next().await {
                    let Message::Text(text) = message.expect("reconnect fixture message") else {
                        continue;
                    };
                    let request: Value =
                        serde_json::from_str(&text).expect("reconnect fixture JSON");
                    let Some(id) = request.get("id") else {
                        continue;
                    };
                    let method = request["method"]
                        .as_str()
                        .expect("reconnect fixture method");
                    let result = match method {
                        "initialize" => json!({
                            "sessionId": format!("reconnect-fixture-{connection_index}")
                        }),
                        "environment/info" => json!({
                            "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                            "cwd": "file:///remote/workspace"
                        }),
                        "environment/status" => json!({"status": "ready"}),
                        method => panic!("unexpected reconnect fixture method: {method}"),
                    };
                    socket
                        .send(Message::Text(
                            json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                        ))
                        .await
                        .expect("send reconnect fixture response");
                    if connection_index == 0 && method == "environment/status" {
                        break;
                    }
                }
            }
        });

        let registry = Arc::new(EnvironmentRegistry::new());
        registry.start();
        registry
            .upsert(
                "reconnect-fixture".to_string(),
                format!("ws://{address}"),
                Some(Duration::from_millis(500)),
            )
            .await
            .expect("register reconnect fixture");

        for _ in 0..80 {
            if matches!(
                registry.status("reconnect-fixture").await,
                Some(EnvironmentConnectionStatus::Ready)
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(matches!(
            registry.status("reconnect-fixture").await,
            Some(EnvironmentConnectionStatus::Ready)
        ));

        for _ in 0..120 {
            if connections.load(Ordering::SeqCst) >= 2
                && matches!(
                    registry.status("reconnect-fixture").await,
                    Some(EnvironmentConnectionStatus::Ready)
                )
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(connections.load(Ordering::SeqCst) >= 2);
        assert!(matches!(
            registry.status("reconnect-fixture").await,
            Some(EnvironmentConnectionStatus::Ready)
        ));

        fixture.abort();
        let _ = fixture.await;
    }

    #[tokio::test]
    async fn persisted_environment_registry_connects_after_cold_start() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind cold-start fixture");
        let address = listener.local_addr().expect("cold-start fixture address");
        let fixture = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept cold-start fixture");
            let mut socket = accept_async(stream)
                .await
                .expect("cold-start fixture websocket");
            while let Some(message) = socket.next().await {
                let Message::Text(text) = message.expect("cold-start fixture message") else {
                    continue;
                };
                let request: Value = serde_json::from_str(&text).expect("cold-start fixture JSON");
                let Some(id) = request.get("id") else {
                    continue;
                };
                let method = request["method"]
                    .as_str()
                    .expect("cold-start fixture method");
                let result = match method {
                    "initialize" => json!({"sessionId": "cold-start-fixture"}),
                    "environment/info" => json!({
                        "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                        "cwd": "file:///remote/workspace"
                    }),
                    "environment/status" => json!({"status": "ready"}),
                    method => panic!("unexpected cold-start fixture method: {method}"),
                };
                socket
                    .send(Message::Text(
                        json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                    ))
                    .await
                    .expect("send cold-start fixture response");
            }
        });

        let temp = tempfile::tempdir().expect("cold-start registry directory");
        let storage_path = temp.path().join("environments.json");
        std::fs::write(
            &storage_path,
            serde_json::to_vec(&vec![json!({
                "environmentId": "cold-start-fixture",
                "execServerUrl": format!("ws://{address}"),
                "connectTimeoutMs": 500
            })])
            .expect("serialize cold-start registry"),
        )
        .expect("write cold-start registry");

        let registry = Arc::new(EnvironmentRegistry::new_with_storage(storage_path));
        registry.start();
        for _ in 0..80 {
            if matches!(
                registry.status("cold-start-fixture").await,
                Some(EnvironmentConnectionStatus::Ready)
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(matches!(
            registry.status("cold-start-fixture").await,
            Some(EnvironmentConnectionStatus::Ready)
        ));
        assert_eq!(
            registry
                .info("cold-start-fixture")
                .await
                .expect("cold-start environment info")
                .cwd
                .expect("cold-start cwd")
                .as_str(),
            "file:///remote/workspace"
        );

        fixture.abort();
        let _ = fixture.await;
    }

    #[test]
    fn environment_world_state_patch_is_deterministic_and_minimal() {
        let previous = environment_world_state(&[
            TurnEnvironmentParams {
                environment_id: "local".to_string(),
                cwd: "/old".to_string(),
                runtime_workspace_roots: Some(vec!["/old".to_string()]),
            },
            TurnEnvironmentParams {
                environment_id: "removed".to_string(),
                cwd: "/removed".to_string(),
                runtime_workspace_roots: Some(vec!["/removed".to_string()]),
            },
        ]);
        let current = environment_world_state(&[TurnEnvironmentParams {
            environment_id: "local".to_string(),
            cwd: "/new".to_string(),
            runtime_workspace_roots: Some(vec!["/new".to_string()]),
        }]);

        assert_eq!(
            environment_world_state_patch(&previous, &current),
            json!({
                "upsert": {
                    "local": {
                        "cwd": "/new",
                        "runtimeWorkspaceRoots": ["/new"]
                    }
                },
                "removed": ["removed"]
            })
        );
        assert!(same_connection_status_kind(
            &EnvironmentConnectionStatus::Disconnected("first".to_string()),
            &EnvironmentConnectionStatus::Disconnected("second".to_string())
        ));
    }

    #[test]
    fn remote_filesystem_sandbox_defaults_by_operation_and_rejects_unknown_policy() {
        let path = PathUri::parse("file:///workspace/src/main.rs").expect("valid path URI");

        let read_default = remote_fs_sandbox(&path, None, RemoteFsAccess::Read)
            .expect("read operations should receive a default sandbox");
        assert_eq!(
            read_default["permissions"]["fileSystem"]["entries"][0]["access"],
            "read"
        );

        let write_default = remote_fs_sandbox(&path, None, RemoteFsAccess::Write)
            .expect("write operations should receive a default sandbox");
        assert_eq!(
            write_default["permissions"]["fileSystem"]["entries"][0]["access"],
            "write"
        );
        assert_eq!(
            remote_fs_sandbox(&path, Some(" "), RemoteFsAccess::Read)
                .expect("blank policy should use the read default")["permissions"]["fileSystem"]
                ["entries"][0]["access"],
            "read"
        );

        let unrestricted =
            remote_fs_sandbox(&path, Some("danger-full-access"), RemoteFsAccess::Read)
                .expect("danger-full-access should be supported");
        assert_eq!(
            unrestricted["permissions"]["fileSystem"]["type"],
            "unrestricted"
        );

        let unknown = remote_fs_sandbox(&path, Some("unknown"), RemoteFsAccess::Read)
            .expect_err("unknown policy must fail closed");
        assert!(unknown.contains("unsupported remote filesystem sandbox policy"));
    }

    #[test]
    fn remote_path_uri_preserves_posix_drive_and_unc_forms() {
        assert_eq!(
            remote_path_uri(Path::new("/workspace/project/main.rs"))
                .expect("POSIX path URI")
                .as_str(),
            "file:///workspace/project/main.rs"
        );
        assert_eq!(
            remote_path_uri(Path::new("C:/Users/Alice/project/main.rs"))
                .expect("drive path URI")
                .as_str(),
            "file:///C:/Users/Alice/project/main.rs"
        );
        assert_eq!(
            remote_path_uri(Path::new("\\\\server\\share\\main.rs"))
                .expect("UNC path URI")
                .as_str(),
            "file://server/share/main.rs"
        );

        assert_eq!(
            remote_path_to_host_path("file:///workspace/project/main%20file.rs")
                .expect("decoded POSIX path"),
            PathBuf::from("/workspace/project/main file.rs")
        );
        assert_eq!(
            remote_path_to_host_path("file://server/share/main.rs").expect("decoded UNC path"),
            PathBuf::from("//server/share/main.rs")
        );
        #[cfg(windows)]
        assert_eq!(
            remote_path_to_host_path("file:///C:/Users/Alice/main.rs").expect("decoded drive path"),
            PathBuf::from(r"C:\Users\Alice\main.rs")
        );
        #[cfg(not(windows))]
        assert_eq!(
            remote_path_to_host_path("file:///C:/Users/Alice/main.rs").expect("decoded drive path"),
            PathBuf::from("C:/Users/Alice/main.rs")
        );
    }
}

fn validate_remote_url(value: &str) -> Result<(), String> {
    let url = Url::parse(value).map_err(|error| format!("invalid execServerUrl: {error}"))?;
    if !matches!(url.scheme(), "ws" | "wss") || url.host_str().is_none() {
        return Err("execServerUrl must be a ws:// or wss:// URL with a host".to_string());
    }
    Ok(())
}

fn local_shell_info() -> EnvironmentShellInfo {
    #[cfg(windows)]
    let path = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string());
    #[cfg(not(windows))]
    let path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let name = std::path::Path::new(&path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("sh")
        .to_string();
    EnvironmentShellInfo { name, path }
}
