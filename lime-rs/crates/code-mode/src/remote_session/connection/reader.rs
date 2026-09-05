//! Stdio frame reader for the process-host connection.

use super::driver::DriverEvent;
use code_mode_protocol::host::{FramedReader, HostToClient};
use tokio::sync::mpsc;

pub(super) fn spawn<R>(mut reader: FramedReader<R>, events: mpsc::Sender<DriverEvent>)
where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        loop {
            match reader.read::<HostToClient>().await {
                Ok(Some(message)) => {
                    if events
                        .send(DriverEvent::HostMessage(message))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(None) => {
                    let _ = events
                        .send(DriverEvent::Failed(
                            "code mode host closed stdout".to_string(),
                        ))
                        .await;
                    return;
                }
                Err(error) => {
                    let _ = events
                        .send(DriverEvent::Failed(format!(
                            "code mode host reader failed: {error}"
                        )))
                        .await;
                    return;
                }
            }
        }
    });
}
