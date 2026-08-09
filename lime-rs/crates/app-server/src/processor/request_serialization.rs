use crate::RuntimeCore;
use app_server_protocol::protocol::v2::Method as V2Method;
use app_server_protocol::{
    app_server_request_access, app_server_request_serialization_scope, error_codes,
    AppServerRequestAccess, AppServerRequestSerializationScope, JsonRpcError, JsonRpcErrorResponse,
    JsonRpcMessage, JsonRpcRequest,
};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

const MISSING_SCOPE_KEY: &str = "<missing>";

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum RequestSerializationQueueKey {
    Thread(String),
    McpOauth(String),
    McpResourceSubscription(String),
    BrowserSession(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RequestSerializationAccess {
    Exclusive,
    SharedRead,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RequestSerializationScope {
    pub(super) key: RequestSerializationQueueKey,
    pub(super) access: RequestSerializationAccess,
}

pub(super) async fn request_serialization_scope(
    runtime: &RuntimeCore,
    request: &JsonRpcRequest,
) -> Result<Option<RequestSerializationScope>, JsonRpcError> {
    let Some(scope) = app_server_request_serialization_scope(&request.method) else {
        return Ok(None);
    };
    let params = request.params.as_ref();
    let is_v2_method = V2Method::parse(&request.method).is_some();
    let key = match scope {
        AppServerRequestSerializationScope::Thread => {
            let key = if is_v2_method {
                scope_param(params, &["threadId"])?
            } else {
                thread_scope_param(runtime, params).await?
            };
            let Some(key) = key else {
                return Ok(None);
            };
            RequestSerializationQueueKey::Thread(key)
        }
        AppServerRequestSerializationScope::McpOauth => {
            let Some(key) = scope_param(params, &["name"])? else {
                return Ok(None);
            };
            RequestSerializationQueueKey::McpOauth(key)
        }
        AppServerRequestSerializationScope::McpResourceSubscription => {
            let Some(key) = scope_param(params, &["uri"])? else {
                return Ok(None);
            };
            RequestSerializationQueueKey::McpResourceSubscription(key)
        }
        AppServerRequestSerializationScope::BrowserSession => {
            let Some(key) = scope_param(
                params,
                &["sessionId", "session_id", "profileKey", "profile_key"],
            )?
            else {
                return Ok(None);
            };
            RequestSerializationQueueKey::BrowserSession(key)
        }
    };
    let access = match app_server_request_access(&request.method) {
        AppServerRequestAccess::Exclusive => RequestSerializationAccess::Exclusive,
        AppServerRequestAccess::SharedRead => RequestSerializationAccess::SharedRead,
    };

    Ok(Some(RequestSerializationScope { key, access }))
}

pub(super) async fn resolve_request_serialization_scope(
    runtime: &RuntimeCore,
    request: &JsonRpcRequest,
) -> Result<Option<RequestSerializationScope>, JsonRpcMessage> {
    request_serialization_scope(runtime, request)
        .await
        .map_err(|error| {
            JsonRpcMessage::Error(JsonRpcErrorResponse {
                id: request.id.clone(),
                error,
            })
        })
}

fn scope_param(
    params: Option<&serde_json::Value>,
    names: &[&str],
) -> Result<Option<String>, JsonRpcError> {
    let Some(params) = params.and_then(serde_json::Value::as_object) else {
        return Ok(None);
    };
    let mut resolved: Option<(&str, String)> = None;

    for name in names {
        let Some(raw_value) = params.get(*name) else {
            continue;
        };
        let Some(value) = raw_value.as_str().map(str::trim) else {
            return Err(invalid_scope_param(name, "must be a string"));
        };
        if value.is_empty() || value == MISSING_SCOPE_KEY {
            return Err(invalid_scope_param(name, "must not be empty"));
        }
        if let Some((resolved_name, resolved_value)) = resolved.as_ref() {
            if resolved_value != value {
                return Err(JsonRpcError::new(
                    error_codes::INVALID_PARAMS,
                    format!(
                        "invalid params: conflicting serialization scope fields {resolved_name} and {name}"
                    ),
                ));
            }
            continue;
        }
        resolved = Some((name, value.to_string()));
    }

    Ok(resolved.map(|(_, value)| value))
}

async fn thread_scope_param(
    runtime: &RuntimeCore,
    params: Option<&serde_json::Value>,
) -> Result<Option<String>, JsonRpcError> {
    let thread_id = scope_param(params, &["threadId", "thread_id"])?;
    let session_id = scope_param(params, &["sessionId", "session_id"])?;
    let Some(session_id) = session_id else {
        return Ok(thread_id);
    };

    let resolved_thread_id = runtime
        .resolve_session_thread_id_current(&session_id)
        .await
        .map_err(|error| error.into_jsonrpc_error())?;
    if let Some(thread_id) = thread_id {
        if thread_id != resolved_thread_id {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                format!(
                    "invalid params: sessionId {session_id} belongs to threadId {resolved_thread_id}, not {thread_id}"
                ),
            ));
        }
    }

    Ok(Some(resolved_thread_id))
}

fn invalid_scope_param(name: &str, reason: &str) -> JsonRpcError {
    JsonRpcError::new(
        error_codes::INVALID_PARAMS,
        format!("invalid params: serialization scope field {name} {reason}"),
    )
}

#[derive(Clone, Default)]
pub(super) struct RequestSerializationQueues {
    locks: Arc<Mutex<HashMap<RequestSerializationQueueKey, Arc<RwLock<()>>>>>,
}

enum RequestSerializationGuard {
    Exclusive { _guard: OwnedRwLockWriteGuard<()> },
    SharedRead { _guard: OwnedRwLockReadGuard<()> },
}

struct RequestSerializationLease {
    key: RequestSerializationQueueKey,
    lock: Arc<RwLock<()>>,
    guard: Option<RequestSerializationGuard>,
    locks: Arc<Mutex<HashMap<RequestSerializationQueueKey, Arc<RwLock<()>>>>>,
}

impl Drop for RequestSerializationLease {
    fn drop(&mut self) {
        // Dropping the owned guard is the cancellation-safe release point. The registry entry is
        // removed synchronously so an aborted request cannot leave an idle lock behind.
        self.guard.take();
        if Arc::strong_count(&self.lock) != 2 {
            return;
        }

        let mut locks = self.locks.lock().unwrap_or_else(|error| error.into_inner());
        if locks.get(&self.key).is_some_and(|current| {
            Arc::ptr_eq(current, &self.lock) && Arc::strong_count(&self.lock) == 2
        }) {
            locks.remove(&self.key);
        }
    }
}

impl RequestSerializationQueues {
    pub(super) async fn run<F>(
        &self,
        scope: Option<RequestSerializationScope>,
        future: F,
    ) -> F::Output
    where
        F: Future,
    {
        let Some(scope) = scope else {
            return future.await;
        };

        let _lease = self.acquire(&scope).await;
        let output = future.await;
        output
    }

    async fn acquire(&self, scope: &RequestSerializationScope) -> RequestSerializationLease {
        let lock = self.lock_for(&scope.key);
        let guard = match scope.access {
            RequestSerializationAccess::Exclusive => RequestSerializationGuard::Exclusive {
                _guard: lock.clone().write_owned().await,
            },
            RequestSerializationAccess::SharedRead => RequestSerializationGuard::SharedRead {
                _guard: lock.clone().read_owned().await,
            },
        };
        RequestSerializationLease {
            key: scope.key.clone(),
            lock,
            guard: Some(guard),
            locks: self.locks.clone(),
        }
    }

    fn lock_for(&self, key: &RequestSerializationQueueKey) -> Arc<RwLock<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(|error| error.into_inner());
        locks
            .entry(key.clone())
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }

    #[cfg(test)]
    pub(super) async fn active_scope_count(&self) -> usize {
        self.locks
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }
}
