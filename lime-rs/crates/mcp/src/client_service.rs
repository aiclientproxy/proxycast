//! MCP client service with lossless elicitation response metadata.
//!
//! Adapted from Codex `codex-rs/rmcp-client/src/elicitation_client_service.rs`
//! custom service boundary introduced at `7b6486a145e` and inspected at
//! `5c19155cbd93bfa099016e7487259f61669823ff` (Apache-2.0).

use crate::client::LimeMcpClient;
use crate::elicitation::ElicitationResponse;
use lime_core::DynEmitter;
use rmcp::model::{ClientInfo, ClientResult, CustomResult, ServerNotification, ServerRequest};
use rmcp::service::{NotificationContext, RequestContext, Service};
use rmcp::RoleClient;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::broadcast;

/// The one client service used by every production MCP transport.
pub struct LimeMcpClientService {
    handler: LimeMcpClient,
}

impl LimeMcpClientService {
    pub fn new(server_name: String, emitter: Option<DynEmitter>) -> Self {
        Self {
            handler: LimeMcpClient::new(server_name, emitter),
        }
    }

    pub fn with_elicitation_router(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: crate::elicitation::ElicitationRequestRouter,
    ) -> Self {
        Self {
            handler: LimeMcpClient::with_elicitation_router(
                server_name,
                emitter,
                elicitation_router,
            ),
        }
    }

    pub fn with_runtime_elicitation_router(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: crate::elicitation::ElicitationRequestRouter,
        session_id: String,
        thread_id: String,
    ) -> Self {
        Self::with_runtime_elicitation_router_and_notifications(
            server_name,
            emitter,
            elicitation_router,
            session_id,
            thread_id,
            None,
        )
    }

    pub fn with_runtime_elicitation_router_and_notifications(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: crate::elicitation::ElicitationRequestRouter,
        session_id: String,
        thread_id: String,
        notification_sender: Option<broadcast::Sender<crate::McpServerNotification>>,
    ) -> Self {
        Self {
            handler: LimeMcpClient::with_runtime_elicitation_router_and_notifications(
                server_name,
                emitter,
                elicitation_router,
                crate::McpRuntimeOwner {
                    session_id,
                    thread_id,
                },
                notification_sender,
            ),
        }
    }

    pub(crate) fn handler(&self) -> &LimeMcpClient {
        &self.handler
    }
}

impl Service<RoleClient> for LimeMcpClientService {
    async fn handle_request(
        &self,
        request: ServerRequest,
        context: RequestContext<RoleClient>,
    ) -> Result<ClientResult, rmcp::ErrorData> {
        match request {
            ServerRequest::CreateElicitationRequest(request) => {
                let (scope, meta) = self.handler.resolve_elicitation_request_meta(context.meta);
                let response = match scope {
                    Some(scope) => self
                        .handler
                        .handle_form_elicitation(request.params, scope, meta, context.ct)
                        .await
                        .map_err(|error| {
                            rmcp::ErrorData::internal_error(error.to_string(), None)
                        })?,
                    None => ElicitationResponse::Decline,
                };
                Ok(ClientResult::CustomResult(elicitation_response_result(
                    response,
                )?))
            }
            request => {
                <LimeMcpClient as Service<RoleClient>>::handle_request(
                    &self.handler,
                    request,
                    context,
                )
                .await
            }
        }
    }

    async fn handle_notification(
        &self,
        notification: ServerNotification,
        context: NotificationContext<RoleClient>,
    ) -> Result<(), rmcp::ErrorData> {
        <LimeMcpClient as Service<RoleClient>>::handle_notification(
            &self.handler,
            notification,
            context,
        )
        .await
    }

    fn get_info(&self) -> ClientInfo {
        <LimeMcpClient as Service<RoleClient>>::get_info(&self.handler)
    }
}

fn elicitation_response_result(
    response: ElicitationResponse,
) -> Result<CustomResult, rmcp::ErrorData> {
    let (action, content, meta) = response.into_wire_parts();
    serde_json::to_value(CreateElicitationResultWithMeta {
        action,
        content,
        meta,
    })
    .map(CustomResult)
    .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateElicitationResultWithMeta {
    action: rmcp::model::ElicitationAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Value>,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    meta: Option<Value>,
}
