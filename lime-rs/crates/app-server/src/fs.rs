use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    FsChangedNotification, FsCopyParams, FsCopyResponse, FsCreateDirectoryParams,
    FsCreateDirectoryResponse, FsGetMetadataParams, FsGetMetadataResponse, FsReadDirectoryEntry,
    FsReadDirectoryParams, FsReadDirectoryResponse, FsReadFileParams, FsReadFileResponse,
    FsRemoveParams, FsRemoveResponse, FsUnwatchParams, FsUnwatchResponse, FsWatchParams,
    FsWatchResponse, FsWriteFileParams, FsWriteFileResponse, ServerNotification,
};
use app_server_protocol::{JsonRpcError, JsonRpcNotification};
use app_server_transport::ConnectionId;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::future::BoxFuture;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const MAX_READ_FILE_BYTES: u64 = 512 * 1024 * 1024;
const FS_CHANGED_DEBOUNCE: Duration = Duration::from_millis(200);

pub(crate) type FsNotificationHook =
    Arc<dyn Fn(ConnectionId, JsonRpcNotification) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone, Default)]
pub(crate) struct FsServer {
    watches: Arc<Mutex<HashMap<FsWatchKey, FsWatchSession>>>,
    notification_hook: Arc<Mutex<Option<FsNotificationHook>>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FsWatchKey {
    connection_id: ConnectionId,
    watch_id: String,
}

struct FsWatchSession {
    _watcher: RecommendedWatcher,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl FsWatchSession {
    async fn stop(self) {
        let Self {
            _watcher,
            cancel,
            task,
        } = self;
        drop(_watcher);
        cancel.cancel();
        let _ = task.await;
    }
}

impl FsServer {
    pub(crate) fn with_notification_hook(self, hook: FsNotificationHook) -> Self {
        *self
            .notification_hook
            .try_lock()
            .expect("fs notification hook mutex poisoned") = Some(hook);
        self
    }

    pub(crate) async fn read_file(
        &self,
        params: FsReadFileParams,
    ) -> Result<FsReadFileResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/readFile.path")?;
        let file = tokio::fs::File::open(&path).await.map_err(map_io_error)?;
        let metadata = file.metadata().await.map_err(map_io_error)?;
        if metadata.len() > MAX_READ_FILE_BYTES {
            return Err(file_too_large());
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_READ_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(map_io_error)?;
        if bytes.len() as u64 > MAX_READ_FILE_BYTES {
            return Err(file_too_large());
        }
        Ok(FsReadFileResponse {
            data_base64: STANDARD.encode(bytes),
        })
    }

    pub(crate) async fn write_file(
        &self,
        params: FsWriteFileParams,
    ) -> Result<FsWriteFileResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/writeFile.path")?;
        let bytes = STANDARD.decode(params.data_base64).map_err(|error| {
            invalid_params(format!(
                "fs/writeFile requires valid base64 dataBase64: {error}"
            ))
        })?;
        tokio::fs::write(path, bytes).await.map_err(map_io_error)?;
        Ok(FsWriteFileResponse {})
    }

    pub(crate) async fn create_directory(
        &self,
        params: FsCreateDirectoryParams,
    ) -> Result<FsCreateDirectoryResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/createDirectory.path")?;
        if params.recursive.unwrap_or(true) {
            tokio::fs::create_dir_all(path)
                .await
                .map_err(map_io_error)?;
        } else {
            tokio::fs::create_dir(path).await.map_err(map_io_error)?;
        }
        Ok(FsCreateDirectoryResponse {})
    }

    pub(crate) async fn get_metadata(
        &self,
        params: FsGetMetadataParams,
    ) -> Result<FsGetMetadataResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/getMetadata.path")?;
        let symlink_metadata = tokio::fs::symlink_metadata(&path)
            .await
            .map_err(map_io_error)?;
        let is_symlink = symlink_metadata.is_symlink();
        let metadata = if is_symlink {
            tokio::fs::metadata(&path).await.map_err(map_io_error)?
        } else {
            symlink_metadata
        };
        Ok(FsGetMetadataResponse {
            is_directory: metadata.is_dir(),
            is_file: metadata.is_file(),
            is_symlink,
            created_at_ms: metadata.created().ok().map_or(0, system_time_to_unix_ms),
            modified_at_ms: metadata.modified().ok().map_or(0, system_time_to_unix_ms),
        })
    }

    pub(crate) async fn read_directory(
        &self,
        params: FsReadDirectoryParams,
    ) -> Result<FsReadDirectoryResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/readDirectory.path")?;
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(path).await.map_err(map_io_error)?;
        while let Some(entry) = read_dir.next_entry().await.map_err(map_io_error)? {
            let Ok(mut file_type) = entry.file_type().await else {
                continue;
            };
            if file_type.is_symlink() {
                let Ok(metadata) = tokio::fs::metadata(entry.path()).await else {
                    continue;
                };
                file_type = metadata.file_type();
            }
            entries.push(FsReadDirectoryEntry {
                file_name: entry.file_name().to_string_lossy().into_owned(),
                is_directory: file_type.is_dir(),
                is_file: file_type.is_file(),
            });
        }
        Ok(FsReadDirectoryResponse { entries })
    }

    pub(crate) async fn remove(
        &self,
        params: FsRemoveParams,
    ) -> Result<FsRemoveResponse, JsonRpcError> {
        let path = absolute_path(&params.path, "fs/remove.path")?;
        let recursive = params.recursive.unwrap_or(true);
        let force = params.force.unwrap_or(true);
        match tokio::fs::symlink_metadata(&path).await {
            Ok(metadata) if metadata.file_type().is_dir() && recursive => {
                tokio::fs::remove_dir_all(path)
                    .await
                    .map_err(map_io_error)?;
            }
            Ok(metadata) if metadata.file_type().is_dir() => {
                tokio::fs::remove_dir(path).await.map_err(map_io_error)?;
            }
            Ok(_) => tokio::fs::remove_file(path).await.map_err(map_io_error)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound && force => {}
            Err(error) => return Err(map_io_error(error)),
        }
        Ok(FsRemoveResponse {})
    }

    pub(crate) async fn copy(&self, params: FsCopyParams) -> Result<FsCopyResponse, JsonRpcError> {
        let source = absolute_path(&params.source_path, "fs/copy.sourcePath")?;
        let destination = absolute_path(&params.destination_path, "fs/copy.destinationPath")?;
        tokio::task::spawn_blocking(move || copy_path(&source, &destination, params.recursive))
            .await
            .map_err(|error| runtime_error(format!("filesystem task failed: {error}")))?
            .map_err(map_io_error)?;
        Ok(FsCopyResponse {})
    }

    pub(crate) async fn watch(
        &self,
        connection_id: ConnectionId,
        params: FsWatchParams,
    ) -> Result<FsWatchResponse, JsonRpcError> {
        if params.watch_id.trim().is_empty() {
            return Err(invalid_params("fs/watch.watchId must not be empty"));
        }
        let path = absolute_path(&params.path, "fs/watch.path")?;
        let path = tokio::fs::canonicalize(path).await.map_err(map_io_error)?;
        let key = FsWatchKey {
            connection_id,
            watch_id: params.watch_id.clone(),
        };
        let mut watches = self.watches.lock().await;
        if watches.contains_key(&key) {
            return Err(invalid_params(format!(
                "watchId already exists: {}",
                params.watch_id
            )));
        }

        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(event) = event {
                    if !event.paths.is_empty() {
                        let _ = events_tx.send(event.paths);
                    }
                }
            })
            .map_err(|error| {
                runtime_error(format!("failed to create filesystem watcher: {error}"))
            })?;
        watcher
            .watch(&path, RecursiveMode::NonRecursive)
            .map_err(|error| {
                runtime_error(format!("failed to watch {}: {error}", path.display()))
            })?;

        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_watch_events(
            connection_id,
            params.watch_id.clone(),
            path.clone(),
            events_rx,
            cancel.clone(),
            self.notification_hook.clone(),
        ));
        watches.insert(
            key,
            FsWatchSession {
                _watcher: watcher,
                cancel,
                task,
            },
        );
        Ok(FsWatchResponse {
            path: path.to_string_lossy().into_owned(),
        })
    }

    pub(crate) async fn unwatch(
        &self,
        connection_id: ConnectionId,
        params: FsUnwatchParams,
    ) -> Result<FsUnwatchResponse, JsonRpcError> {
        let session = self.watches.lock().await.remove(&FsWatchKey {
            connection_id,
            watch_id: params.watch_id,
        });
        if let Some(session) = session {
            session.stop().await;
        }
        Ok(FsUnwatchResponse {})
    }

    pub(crate) async fn connection_closed(&self, connection_id: ConnectionId) {
        let sessions = {
            let mut watches = self.watches.lock().await;
            let keys = watches
                .keys()
                .filter(|key| key.connection_id == connection_id)
                .cloned()
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|key| watches.remove(&key))
                .collect::<Vec<_>>()
        };
        for session in sessions {
            session.stop().await;
        }
    }
}

async fn run_watch_events(
    connection_id: ConnectionId,
    watch_id: String,
    watch_root: PathBuf,
    mut events_rx: mpsc::UnboundedReceiver<Vec<PathBuf>>,
    cancel: CancellationToken,
    notification_hook: Arc<Mutex<Option<FsNotificationHook>>>,
) {
    loop {
        let mut paths = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            paths = events_rx.recv() => match paths {
                Some(paths) => paths,
                None => break,
            },
        };
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            _ = tokio::time::sleep(FS_CHANGED_DEBOUNCE) => {}
        }
        while let Ok(next) = events_rx.try_recv() {
            paths.extend(next);
        }
        let changed_paths = normalize_changed_paths(&watch_root, paths);
        if changed_paths.is_empty() {
            continue;
        }
        let hook = notification_hook.lock().await.clone();
        if let Some(hook) = hook {
            let notification: JsonRpcNotification =
                ServerNotification::FsChanged(FsChangedNotification {
                    watch_id: watch_id.clone(),
                    changed_paths,
                })
                .into();
            hook(connection_id, notification).await;
        }
    }
}

fn normalize_changed_paths(watch_root: &Path, paths: Vec<PathBuf>) -> Vec<String> {
    let base = if watch_root.is_dir() {
        watch_root
    } else {
        watch_root.parent().unwrap_or(watch_root)
    };
    let mut paths = paths
        .into_iter()
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                base.join(path)
            }
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    paths
}

fn absolute_path(path: &str, field: &str) -> Result<PathBuf, JsonRpcError> {
    let path = PathBuf::from(path);
    if !path.is_absolute() {
        return Err(invalid_params(format!("{field} must be an absolute path")));
    }
    Ok(path)
}

fn file_too_large() -> JsonRpcError {
    invalid_params(format!(
        "file is too large to read: limit is {MAX_READ_FILE_BYTES} bytes"
    ))
}

fn system_time_to_unix_ms(time: SystemTime) -> i64 {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    millis.min(i64::MAX as u128) as i64
}

fn copy_path(source: &Path, destination: &Path, recursive: bool) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(source)?;
    let file_type = metadata.file_type();
    if file_type.is_dir() {
        if !recursive {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fs/copy requires recursive: true when sourcePath is a directory",
            ));
        }
        if destination_is_same_or_descendant_of_source(source, destination)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fs/copy cannot copy a directory to itself or one of its descendants",
            ));
        }
        return copy_directory(source, destination);
    }
    if file_type.is_symlink() {
        return copy_symlink(source, destination);
    }
    if file_type.is_file() {
        std::fs::copy(source, destination)?;
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "fs/copy only supports regular files, directories, and symlinks",
    ))
}

fn copy_directory(source: &Path, destination: &Path) -> io::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            std::fs::copy(source_path, destination_path)?;
        } else if file_type.is_symlink() {
            copy_symlink(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn destination_is_same_or_descendant_of_source(
    source: &Path,
    destination: &Path,
) -> io::Result<bool> {
    let source = std::fs::canonicalize(source)?;
    let destination = resolve_existing_path(destination)?;
    Ok(destination.starts_with(source))
}

fn resolve_existing_path(path: &Path) -> io::Result<PathBuf> {
    let mut suffix = Vec::new();
    let mut existing = path;
    while !existing.exists() {
        let Some(name) = existing.file_name() else {
            break;
        };
        suffix.push(name.to_os_string());
        let Some(parent) = existing.parent() else {
            break;
        };
        existing = parent;
    }
    let mut resolved = std::fs::canonicalize(existing)?;
    for name in suffix.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn copy_symlink(source: &Path, destination: &Path) -> io::Result<()> {
    let target = std::fs::read_link(source)?;
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, destination)
    }
    #[cfg(windows)]
    {
        if std::fs::metadata(source)?.is_dir() {
            std::os::windows::fs::symlink_dir(target, destination)
        } else {
            std::os::windows::fs::symlink_file(target, destination)
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = target;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "copying symlinks is unsupported on this platform",
        ))
    }
}

fn map_io_error(error: io::Error) -> JsonRpcError {
    if error.kind() == io::ErrorKind::InvalidInput {
        invalid_params(error.to_string())
    } else {
        runtime_error(error.to_string())
    }
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}

fn runtime_error(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
}

#[cfg(test)]
mod tests;
