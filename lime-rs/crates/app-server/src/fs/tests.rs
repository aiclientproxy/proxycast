use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[tokio::test]
async fn exact_fs_round_trip_covers_binary_metadata_directory_copy_and_remove() {
    let root = tempfile::tempdir().expect("temp dir");
    let source_dir = root.path().join("source");
    let source_file = source_dir.join("data.bin");
    let copied_dir = root.path().join("copied");
    let server = FsServer::default();

    server
        .create_directory(FsCreateDirectoryParams {
            path: path_string(&source_dir),
            recursive: None,
        })
        .await
        .expect("create directory");
    server
        .write_file(FsWriteFileParams {
            path: path_string(&source_file),
            data_base64: STANDARD.encode([0, 1, 2, 255]),
        })
        .await
        .expect("write file");

    let read = server
        .read_file(FsReadFileParams {
            path: path_string(&source_file),
        })
        .await
        .expect("read file");
    assert_eq!(
        STANDARD.decode(read.data_base64).expect("base64"),
        [0, 1, 2, 255]
    );

    let metadata = server
        .get_metadata(FsGetMetadataParams {
            path: path_string(&source_file),
        })
        .await
        .expect("metadata");
    assert!(metadata.is_file);
    assert!(!metadata.is_directory);

    let listing = server
        .read_directory(FsReadDirectoryParams {
            path: path_string(&source_dir),
        })
        .await
        .expect("directory");
    assert_eq!(listing.entries.len(), 1);
    assert_eq!(listing.entries[0].file_name, "data.bin");
    assert!(listing.entries[0].is_file);

    server
        .copy(FsCopyParams {
            source_path: path_string(&source_dir),
            destination_path: path_string(&copied_dir),
            recursive: true,
        })
        .await
        .expect("copy directory");
    assert_eq!(
        std::fs::read(copied_dir.join("data.bin")).expect("copied file"),
        [0, 1, 2, 255]
    );

    server
        .remove(FsRemoveParams {
            path: path_string(&source_dir),
            recursive: None,
            force: None,
        })
        .await
        .expect("remove directory");
    assert!(!source_dir.exists());
}

#[tokio::test]
async fn exact_fs_rejects_relative_paths_invalid_base64_and_recursive_copy_omission() {
    let server = FsServer::default();
    let relative = server
        .read_file(FsReadFileParams {
            path: "relative.txt".to_string(),
        })
        .await
        .expect_err("relative path must fail");
    assert_eq!(relative.code, error_codes::INVALID_PARAMS);

    let root = tempfile::tempdir().expect("temp dir");
    let invalid_base64 = server
        .write_file(FsWriteFileParams {
            path: path_string(&root.path().join("invalid.bin")),
            data_base64: "***".to_string(),
        })
        .await
        .expect_err("invalid base64 must fail");
    assert_eq!(invalid_base64.code, error_codes::INVALID_PARAMS);

    let copy_error = server
        .copy(FsCopyParams {
            source_path: path_string(root.path()),
            destination_path: path_string(&root.path().with_extension("copy")),
            recursive: false,
        })
        .await
        .expect_err("directory copy must opt into recursion");
    assert_eq!(copy_error.code, error_codes::INVALID_PARAMS);
}

#[tokio::test]
async fn watch_ids_are_connection_scoped_and_disconnect_cleans_only_the_owner() {
    let root = tempfile::tempdir().expect("temp dir");
    let path = path_string(root.path());
    let server = FsServer::default();

    for connection_id in [ConnectionId(1), ConnectionId(2)] {
        server
            .watch(
                connection_id,
                FsWatchParams {
                    watch_id: "workspace".to_string(),
                    path: path.clone(),
                },
            )
            .await
            .expect("connection scoped watch");
    }
    let duplicate = server
        .watch(
            ConnectionId(1),
            FsWatchParams {
                watch_id: "workspace".to_string(),
                path,
            },
        )
        .await
        .expect_err("duplicate owner watch must fail");
    assert_eq!(duplicate.code, error_codes::INVALID_PARAMS);

    server.connection_closed(ConnectionId(1)).await;
    let watches = server.watches.lock().await;
    let by_connection = watches.keys().fold(HashMap::new(), |mut counts, key| {
        *counts.entry(key.connection_id).or_insert(0usize) += 1;
        counts
    });
    assert_eq!(by_connection.get(&ConnectionId(1)), None);
    assert_eq!(by_connection.get(&ConnectionId(2)), Some(&1));
    drop(watches);
    server.connection_closed(ConnectionId(2)).await;
}

#[tokio::test]
async fn watch_notifications_keep_connection_owner_and_survive_other_disconnects() {
    let first_root = tempfile::tempdir().expect("first watch root");
    let second_root = tempfile::tempdir().expect("second watch root");
    let first_file = first_root.path().join("first.txt");
    let second_file = second_root.path().join("second.txt");
    std::fs::write(&first_file, "before").expect("seed first file");
    std::fs::write(&second_file, "before").expect("seed second file");
    let (notifications_tx, mut notifications_rx) = mpsc::unbounded_channel();
    let hook: FsNotificationHook = Arc::new(move |connection_id, notification| {
        let notifications_tx = notifications_tx.clone();
        Box::pin(async move {
            let _ = notifications_tx.send((connection_id, notification));
        })
    });
    let server = FsServer::default().with_notification_hook(hook);

    for (connection_id, root) in [
        (ConnectionId(11), first_root.path()),
        (ConnectionId(22), second_root.path()),
    ] {
        server
            .watch(
                connection_id,
                FsWatchParams {
                    watch_id: "workspace".to_string(),
                    path: path_string(root),
                },
            )
            .await
            .expect("watch should be connection scoped");
    }

    std::fs::write(&first_file, "first").expect("update first file");
    let (first_connection, first_notification) =
        next_watch_notification(&mut notifications_rx).await;
    assert_eq!(first_connection, ConnectionId(11));
    assert_eq!(first_notification.method, "fs/changed");

    std::fs::write(&second_file, "second").expect("update second file");
    let (second_connection, second_notification) =
        next_watch_notification(&mut notifications_rx).await;
    assert_eq!(second_connection, ConnectionId(22));
    assert_eq!(second_notification.method, "fs/changed");

    server.connection_closed(ConnectionId(11)).await;
    std::fs::write(&second_file, "second-again").expect("update surviving file");
    let (surviving_connection, _) = next_watch_notification(&mut notifications_rx).await;
    assert_eq!(surviving_connection, ConnectionId(22));

    server.connection_closed(ConnectionId(22)).await;
}

async fn next_watch_notification(
    notifications: &mut mpsc::UnboundedReceiver<(ConnectionId, JsonRpcNotification)>,
) -> (ConnectionId, JsonRpcNotification) {
    timeout(Duration::from_secs(3), notifications.recv())
        .await
        .expect("watch notification timeout")
        .expect("watch notification channel closed")
}
