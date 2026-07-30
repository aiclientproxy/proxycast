use agent_protocol::ThreadId;
use app_server_protocol::protocol::v2::{
    CurrentTimeReadParams, CurrentTimeReadResponse, METHOD_CURRENT_TIME_READ,
};
use chrono::{DateTime, Utc};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{timeout_at, Instant};

use crate::server_request::{ServerRequestOwner, ServerRequestRouter};
use crate::{
    AppServer, AppServerError, JsonRpcMessage, OutgoingMessage, QueuedOutgoingMessage,
    TransportDisconnects, TransportWriters,
};

pub(crate) const CURRENT_TIME_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Error)]
pub enum CurrentTimeReadError {
    #[error(transparent)]
    AppServer(#[from] AppServerError),
    #[error("current-time request timed out after {seconds}s")]
    TimedOut { seconds: u64 },
    #[error("invalid current-time response: {source}")]
    InvalidResponse {
        #[source]
        source: serde_json::Error,
    },
    #[error("current-time response is outside the supported range: {current_time_at}")]
    OutOfRange { current_time_at: i64 },
}

#[derive(Clone)]
pub(crate) struct CurrentTimeRequestRouter {
    thread_states: crate::thread_state::ThreadStateManager,
    transport_writers: TransportWriters,
    transport_disconnects: TransportDisconnects,
    server_requests: ServerRequestRouter,
}

impl CurrentTimeRequestRouter {
    pub(crate) fn new(
        thread_states: crate::thread_state::ThreadStateManager,
        transport_writers: TransportWriters,
        transport_disconnects: TransportDisconnects,
        server_requests: ServerRequestRouter,
    ) -> Self {
        Self {
            thread_states,
            transport_writers,
            transport_disconnects,
            server_requests,
        }
    }

    async fn read_current_time(
        &self,
        thread_id: String,
        timeout: Duration,
    ) -> Result<DateTime<Utc>, CurrentTimeReadError> {
        let deadline = Instant::now() + timeout;
        let response = timeout_at(deadline, async {
            let thread_id = ThreadId::new(thread_id);
            self.thread_states
                .wait_for_thread_subscriber(&thread_id)
                .await;
            let connection_ids = self
                .thread_states
                .subscribed_connection_ids(&thread_id)
                .await;
            let connection_id = match connection_ids.as_slice() {
                [connection_id] => *connection_id,
                [] => {
                    return Err(AppServerError::from(
                        crate::ServerRequestError::ClientUnavailable,
                    ))
                }
                _ => {
                    return Err(AppServerError::from(
                        crate::ServerRequestError::ClientAmbiguous {
                            client_count: connection_ids.len(),
                        },
                    ));
                }
            };
            let writer = self
                .transport_writers
                .lock()
                .expect("app-server transport writer mutex poisoned")
                .get(&connection_id)
                .cloned()
                .ok_or(AppServerError::ConnectionUnavailable { connection_id })?;
            let pending = self.server_requests.register_for_owner(
                ServerRequestOwner::Transport(connection_id),
                METHOD_CURRENT_TIME_READ,
                Some(serde_json::to_value(CurrentTimeReadParams {
                    thread_id: thread_id.to_string(),
                })?),
            );
            let queued = QueuedOutgoingMessage::new(OutgoingMessage::from(
                JsonRpcMessage::Request(pending.request().clone()),
            ));
            let disconnect = self
                .transport_disconnects
                .lock()
                .expect("app-server transport disconnect mutex poisoned")
                .get(&connection_id)
                .cloned();
            if let Some(disconnect) = disconnect {
                match writer.try_send(queued) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => disconnect.cancel(),
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        disconnect.cancel();
                        return Err(AppServerError::ConnectionWriterClosed { connection_id });
                    }
                }
            } else {
                writer
                    .send(queued)
                    .await
                    .map_err(|_| AppServerError::ConnectionWriterClosed { connection_id })?;
            }
            pending.wait().await.map_err(AppServerError::from)
        })
        .await
        .map_err(|_| CurrentTimeReadError::TimedOut {
            seconds: timeout.as_secs(),
        })??;
        let response: CurrentTimeReadResponse = serde_json::from_value(response)
            .map_err(|source| CurrentTimeReadError::InvalidResponse { source })?;

        DateTime::from_timestamp(response.current_time_at, 0).ok_or(
            CurrentTimeReadError::OutOfRange {
                current_time_at: response.current_time_at,
            },
        )
    }
}

#[async_trait::async_trait]
impl tool_runtime::current_time::CurrentTimeGateway for CurrentTimeRequestRouter {
    async fn read_current_time(&self, thread_id: &str) -> Result<i64, String> {
        self.read_current_time(thread_id.to_string(), CURRENT_TIME_REQUEST_TIMEOUT)
            .await
            .map(|current_time| current_time.timestamp())
            .map_err(|error| error.to_string())
    }
}

impl AppServer {
    pub async fn read_current_time(
        &self,
        thread_id: impl Into<String>,
    ) -> Result<DateTime<Utc>, CurrentTimeReadError> {
        self.current_time_requests
            .read_current_time(thread_id.into(), CURRENT_TIME_REQUEST_TIMEOUT)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn read_current_time_with_timeout(
        &self,
        thread_id: String,
        timeout: Duration,
    ) -> Result<DateTime<Utc>, CurrentTimeReadError> {
        self.current_time_requests
            .read_current_time(thread_id, timeout)
            .await
    }
}
