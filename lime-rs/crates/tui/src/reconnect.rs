use std::time::Duration;

use anyhow::{bail, Result};
use app_server_client::StdioTransportConfig;

use crate::app::App;
use crate::app_server_session::AppServerSession;

const RECONNECT_DELAYS: [Duration; 4] = [
    Duration::ZERO,
    Duration::from_millis(100),
    Duration::from_millis(300),
    Duration::from_secs(1),
];

pub(crate) async fn reconnect_session(
    config: &StdioTransportConfig,
    app: &mut App,
    thread_id: &str,
    model: Option<String>,
    model_provider: Option<String>,
    effort: Option<String>,
    permissions: Option<String>,
) -> Result<AppServerSession> {
    let mut last_error = None;
    for delay in RECONNECT_DELAYS {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut candidate = match AppServerSession::connect(config.clone()).await {
            Ok(session) => session,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        match candidate.resume_thread(thread_id.to_string()).await {
            Ok(response) => {
                if let Err(error) = candidate
                    .update_settings(
                        model.clone(),
                        model_provider.clone(),
                        effort.clone(),
                        permissions.clone(),
                    )
                    .await
                {
                    let _ = candidate.shutdown().await;
                    last_error = Some(error);
                    continue;
                }
                app.projection.hydrate_thread(response.thread);
                return Ok(candidate);
            }
            Err(error) => {
                let _ = candidate.shutdown().await;
                last_error = Some(error);
            }
        }
    }
    let error = last_error
        .map(|error| format!("{error:#}"))
        .unwrap_or_else(|| "unknown reconnect failure".to_string());
    bail!("failed to reconnect App Server and resume thread {thread_id}: {error}")
}
