use super::TransportEvent;
#[cfg(unix)]
use crate::transport::websocket::run_websocket_connection;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::io::Result as IoResult;
use std::path::Path;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
#[cfg(unix)]
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;
#[cfg(unix)]
use tracing::error;
#[cfg(unix)]
use tracing::info;
#[cfg(unix)]
use tracing::warn;

#[cfg(unix)]
const CONTROL_SOCKET_MODE: u32 = 0o600;
#[cfg(unix)]
const CONTROL_SOCKET_DIRECTORY_MODE: u32 = 0o700;

#[cfg(unix)]
pub async fn start_control_socket_acceptor(
    socket_path: PathBuf,
    transport_event_tx: mpsc::Sender<TransportEvent>,
    shutdown_token: CancellationToken,
) -> IoResult<JoinHandle<()>> {
    prepare_control_socket_path(&socket_path).await?;
    let listener = tokio::net::UnixListener::bind(&socket_path)?;
    let socket_guard = ControlSocketFileGuard { socket_path };
    set_control_socket_permissions(&socket_guard.socket_path).await?;
    info!(
        socket_path = %socket_guard.socket_path.display(),
        "app-server control socket listening"
    );

    Ok(tokio::spawn(run_control_socket_acceptor(
        listener,
        transport_event_tx,
        shutdown_token,
        socket_guard,
    )))
}

#[cfg(not(unix))]
pub async fn start_control_socket_acceptor(
    _socket_path: PathBuf,
    _transport_event_tx: mpsc::Sender<TransportEvent>,
    _shutdown_token: CancellationToken,
) -> IoResult<JoinHandle<()>> {
    Err(std::io::Error::new(
        ErrorKind::Unsupported,
        "unix socket transport is unavailable on this platform",
    ))
}

#[cfg(unix)]
async fn run_control_socket_acceptor(
    listener: tokio::net::UnixListener,
    transport_event_tx: mpsc::Sender<TransportEvent>,
    shutdown_token: CancellationToken,
    socket_guard: ControlSocketFileGuard,
) {
    let _socket_guard = socket_guard;
    loop {
        let stream = tokio::select! {
            _ = shutdown_token.cancelled() => break,
            result = listener.accept() => {
                match result {
                    Ok((stream, _)) => stream,
                    Err(error) => {
                        if matches!(error.kind(), ErrorKind::ConnectionAborted | ErrorKind::ConnectionReset | ErrorKind::Interrupted) {
                            warn!(%error, "recoverable app-server control socket accept error");
                            continue;
                        }
                        error!(%error, "app-server control socket accept error");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                }
            }
        };

        let transport_event_tx = transport_event_tx.clone();
        tokio::spawn(async move {
            let websocket_stream = match tokio_tungstenite::accept_async(stream).await {
                Ok(websocket_stream) => websocket_stream,
                Err(error) => {
                    warn!(%error, "failed to upgrade control socket websocket connection");
                    return;
                }
            };
            let (websocket_writer, websocket_reader) = futures::StreamExt::split(websocket_stream);
            run_websocket_connection(websocket_writer, websocket_reader, transport_event_tx).await;
        });
    }
    info!("app-server control socket acceptor stopped");
}

pub async fn prepare_control_socket_path(socket_path: &Path) -> IoResult<()> {
    #[cfg(not(unix))]
    {
        let _ = socket_path;
        return Err(std::io::Error::new(
            ErrorKind::Unsupported,
            "unix socket transport is unavailable on this platform",
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;

        if let Some(parent) = socket_path.parent() {
            prepare_private_socket_directory(parent).await?;
        }

        match tokio::net::UnixStream::connect(socket_path).await {
            Ok(_stream) => {
                return Err(std::io::Error::new(
                    ErrorKind::AddrInUse,
                    format!(
                        "app-server control socket is already in use at {}",
                        socket_path.display()
                    ),
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) if error.kind() == ErrorKind::ConnectionRefused => {}
            Err(error) => {
                if !socket_path.exists() {
                    return Ok(());
                }
                let metadata = tokio::fs::symlink_metadata(socket_path).await?;
                if !metadata.file_type().is_socket() {
                    return Err(std::io::Error::new(
                        ErrorKind::AlreadyExists,
                        format!(
                            "app-server control socket path exists and is not a socket: {}",
                            socket_path.display()
                        ),
                    ));
                }
                return Err(error);
            }
        }

        let metadata = tokio::fs::symlink_metadata(socket_path).await?;
        if !metadata.file_type().is_socket() {
            return Err(std::io::Error::new(
                ErrorKind::AlreadyExists,
                format!(
                    "app-server control socket path exists and is not a socket: {}",
                    socket_path.display()
                ),
            ));
        }
        tokio::fs::remove_file(socket_path).await
    }
}

pub struct AppServerStartupLock {
    _file: std::fs::File,
}

pub async fn acquire_app_server_startup_lock(
    startup_lock_path: PathBuf,
) -> IoResult<AppServerStartupLock> {
    if let Some(parent) = startup_lock_path.parent() {
        #[cfg(unix)]
        prepare_private_socket_directory(parent).await?;
        #[cfg(not(unix))]
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::task::spawn_blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&startup_lock_path)?;
        fs2::FileExt::lock_exclusive(&file)?;
        Ok(AppServerStartupLock { _file: file })
    })
    .await
    .map_err(|error| std::io::Error::other(format!("startup lock task failed: {error}")))?
}

#[cfg(unix)]
async fn prepare_private_socket_directory(socket_dir: &Path) -> IoResult<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut dir_builder = tokio::fs::DirBuilder::new();
    dir_builder.mode(CONTROL_SOCKET_DIRECTORY_MODE);
    match dir_builder.create(socket_dir).await {
        Ok(()) => return Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let metadata = tokio::fs::symlink_metadata(socket_dir).await?;
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "app-server control socket parent is not a directory: {}",
                socket_dir.display()
            ),
        ));
    }
    let mode = metadata.permissions().mode() & 0o777;
    if mode != CONTROL_SOCKET_DIRECTORY_MODE {
        tokio::fs::set_permissions(
            socket_dir,
            std::fs::Permissions::from_mode(CONTROL_SOCKET_DIRECTORY_MODE),
        )
        .await?;
    }
    Ok(())
}

#[cfg(unix)]
async fn set_control_socket_permissions(socket_path: &Path) -> IoResult<()> {
    use std::os::unix::fs::PermissionsExt;

    tokio::fs::set_permissions(
        socket_path,
        std::fs::Permissions::from_mode(CONTROL_SOCKET_MODE),
    )
    .await
}

#[cfg(unix)]
struct ControlSocketFileGuard {
    socket_path: PathBuf,
}

#[cfg(unix)]
impl Drop for ControlSocketFileGuard {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.socket_path) {
            if error.kind() != ErrorKind::NotFound {
                warn!(
                    socket_path = %self.socket_path.display(),
                    %error,
                    "failed to remove app-server control socket file"
                );
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::transport::ConnectionOrigin;
    use crate::CHANNEL_CAPACITY;
    use app_server_protocol::JsonRpcMessage;
    use app_server_protocol::JsonRpcNotification;
    use futures::SinkExt;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;
    use tokio_tungstenite::client_async;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn control_socket_acceptor_forwards_websocket_messages() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let socket_path = temp_dir.path().join("app-server.sock");
        let (transport_event_tx, mut transport_event_rx) =
            mpsc::channel::<TransportEvent>(CHANNEL_CAPACITY);
        let shutdown_token = CancellationToken::new();
        let accept_handle = start_control_socket_acceptor(
            socket_path.clone(),
            transport_event_tx,
            shutdown_token.clone(),
        )
        .await
        .expect("control socket acceptor should start");

        let stream = tokio::net::UnixStream::connect(&socket_path)
            .await
            .expect("client should connect");
        let (mut websocket, response) = client_async("ws://localhost/rpc", stream)
            .await
            .expect("websocket upgrade should complete");
        assert_eq!(response.status().as_u16(), 101);

        let connection_id = match timeout(Duration::from_secs(1), transport_event_rx.recv())
            .await
            .expect("connection opened event should arrive")
            .expect("connection opened event")
        {
            TransportEvent::ConnectionOpened {
                connection_id,
                origin,
                disconnect_sender,
                ..
            } => {
                assert_eq!(origin, ConnectionOrigin::WebSocket);
                assert!(disconnect_sender.is_some());
                connection_id
            }
            other => panic!("expected connection opened event, got {other:?}"),
        };

        let notification =
            JsonRpcMessage::Notification(JsonRpcNotification::new("initialized", None));
        websocket
            .send(Message::Text(
                serde_json::to_string(&notification)
                    .expect("notification should serialize")
                    .into(),
            ))
            .await
            .expect("notification should send");
        let incoming = timeout(Duration::from_secs(1), transport_event_rx.recv())
            .await
            .expect("incoming message event should arrive")
            .expect("incoming message event");
        assert!(matches!(
            incoming,
            TransportEvent::IncomingMessage {
                connection_id: incoming_connection_id,
                message,
            } if incoming_connection_id == connection_id && message == notification
        ));

        websocket
            .send(Message::Ping(b"check".to_vec()))
            .await
            .expect("ping should send");
        let pong = timeout(Duration::from_secs(1), websocket.next())
            .await
            .expect("pong should arrive")
            .expect("pong frame")
            .expect("pong should be valid");
        assert_eq!(pong, Message::Pong(b"check".to_vec()));

        websocket.close(None).await.expect("close should send");
        let closed = timeout(Duration::from_secs(1), transport_event_rx.recv())
            .await
            .expect("connection closed event should arrive")
            .expect("connection closed event");
        assert!(matches!(
            closed,
            TransportEvent::ConnectionClosed { connection_id: closed_id }
                if closed_id == connection_id
        ));

        shutdown_token.cancel();
        accept_handle.await.expect("acceptor should join");
        assert!(!socket_path.exists());
    }

    #[tokio::test]
    async fn control_socket_path_rejects_non_socket_path() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let socket_path = temp_dir.path().join("app-server.sock");
        tokio::fs::write(&socket_path, b"not a socket")
            .await
            .expect("fixture file");
        let error = prepare_control_socket_path(&socket_path)
            .await
            .expect_err("regular file must not be removed");
        assert_eq!(error.kind(), ErrorKind::AlreadyExists);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn control_socket_parent_directory_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let socket_dir = temp_dir.path().join("control");
        let socket_path = socket_dir.join("app-server.sock");

        prepare_control_socket_path(&socket_path)
            .await
            .expect("socket parent should be prepared");

        let mode = std::fs::metadata(&socket_dir)
            .expect("socket parent metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, CONTROL_SOCKET_DIRECTORY_MODE);
    }

    #[tokio::test]
    async fn app_server_startup_lock_serializes_waiters() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let lock_path = temp_dir.path().join("app-server-startup.lock");
        let first_lock = acquire_app_server_startup_lock(lock_path.clone())
            .await
            .expect("first startup lock should succeed");
        let mut second_lock = tokio::spawn(acquire_app_server_startup_lock(lock_path));

        assert!(timeout(Duration::from_millis(100), &mut second_lock)
            .await
            .is_err());

        drop(first_lock);
        second_lock
            .await
            .expect("second startup lock task should join")
            .expect("second startup lock should succeed");
    }
}
