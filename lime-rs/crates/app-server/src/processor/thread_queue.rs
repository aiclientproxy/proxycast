use super::{dispatch_result, parse_params, to_jsonrpc_error};
use super::{RequestProcessor, RpcDispatch};
use crate::runtime::ThreadQueuedSubmission;
use agent_protocol::AgentInput;
use app_server_protocol::protocol::v2::{
    QueuedSubmission, ServerNotification, ThreadQueueAddParams, ThreadQueueAddResponse,
    ThreadQueueChangedNotification, ThreadQueueDeleteParams, ThreadQueueDeleteResponse,
    ThreadQueueListParams, ThreadQueueListResponse, ThreadQueueReorderParams,
    ThreadQueueReorderResponse, ThreadQueueStartParams, ThreadQueueStartResponse,
    ThreadQueueUpdateParams, ThreadQueueUpdateResponse, UserInput,
};
use app_server_protocol::{error_codes, JsonRpcError, JsonRpcMessage};
use serde_json::Value;

const THREAD_QUEUE_DEFAULT_LIMIT: usize = 25;
const THREAD_QUEUE_MAX_LIMIT: usize = 100;
const REMOTE_IMAGE_URL_ERROR: &str =
    "remote image URLs are not supported; use an inline data URL instead";

impl RequestProcessor {
    pub(super) async fn handle_thread_queue_add_impl(
        &self,
        params: Option<Value>,
        _event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueAddParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let was_loaded = self
            .runtime
            .loaded_session_id_for_thread(&params.thread_id)
            .is_some();
        let queued = self
            .runtime
            .add_thread_queue_submission(
                &params.thread_id,
                lower_user_input(params.input)?,
                params.client_user_message_id,
            )
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_queue_changed(&params.thread_id).await;
        if was_loaded {
            self.runtime
                .wake_thread_queue_if_idle(&params.thread_id, self.runtime_host_context());
        }
        dispatch_result(ThreadQueueAddResponse {
            queued_submission: to_protocol_submission(queued),
        })
    }

    pub(super) async fn handle_thread_queue_list_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueListParams = parse_params(params)?;
        let cursor = parse_cursor(params.cursor.as_deref())?;
        let limit = params
            .limit
            .map(|limit| limit as usize)
            .unwrap_or(THREAD_QUEUE_DEFAULT_LIMIT)
            .clamp(1, THREAD_QUEUE_MAX_LIMIT);
        let data = self
            .runtime
            .list_thread_queue_submissions(&params.thread_id)
            .await
            .map_err(to_jsonrpc_error)?;
        let next_offset = cursor.saturating_add(limit);
        let next_cursor = (next_offset < data.len()).then(|| next_offset.to_string());
        let data = data
            .into_iter()
            .skip(cursor)
            .take(limit)
            .map(to_protocol_submission)
            .collect();
        dispatch_result(ThreadQueueListResponse { data, next_cursor })
    }

    pub(super) async fn handle_thread_queue_update_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueUpdateParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let queued = self
            .runtime
            .update_thread_queue_submission(
                &params.thread_id,
                &params.queued_submission_id,
                lower_user_input(params.input)?,
            )
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_queue_changed(&params.thread_id).await;
        dispatch_result(ThreadQueueUpdateResponse {
            queued_submission: to_protocol_submission(queued),
        })
    }

    pub(super) async fn handle_thread_queue_delete_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueDeleteParams = parse_params(params)?;
        let deleted = self
            .runtime
            .delete_thread_queue_submission(&params.thread_id, &params.queued_submission_id)
            .await
            .map_err(to_jsonrpc_error)?;
        if deleted {
            self.publish_queue_changed(&params.thread_id).await;
        }
        dispatch_result(ThreadQueueDeleteResponse { deleted })
    }

    pub(super) async fn handle_thread_queue_reorder_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueReorderParams = parse_params(params)?;
        self.runtime
            .reorder_thread_queue_submissions(&params.thread_id, params.queued_submission_ids)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_queue_changed(&params.thread_id).await;
        dispatch_result(ThreadQueueReorderResponse {})
    }

    pub(super) async fn handle_thread_queue_start_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadQueueStartParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        self.resolve_loaded_v2_thread_session(&params.thread_id)?;
        let turn = self
            .runtime
            .start_thread_queue_submission(
                &params.thread_id,
                params.queued_submission_id.as_deref(),
                self.runtime_host_context(),
            )
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(ThreadQueueStartResponse {
            turn: super::turn::v2_turn_from_agent_turn(turn),
        })
    }

    async fn publish_queue_changed(&self, thread_id: &str) {
        self.publish_server_notification(ServerNotification::ThreadQueueChanged(
            ThreadQueueChangedNotification {
                thread_id: thread_id.to_string(),
            },
        ))
        .await;
    }
}

fn parse_cursor(cursor: Option<&str>) -> Result<usize, JsonRpcError> {
    cursor
        .map(|cursor| {
            cursor.parse::<usize>().map_err(|error| {
                invalid_request(format!("invalid queue pagination cursor: {error}"))
            })
        })
        .transpose()
        .map(|cursor| cursor.unwrap_or_default())
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}

fn invalid_request(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_REQUEST, message)
}

fn lower_user_input(items: Vec<UserInput>) -> Result<Vec<AgentInput>, JsonRpcError> {
    items
        .into_iter()
        .map(|item| match item {
            UserInput::Text {
                text,
                text_elements,
            } => Ok(AgentInput::Text {
                text,
                text_elements,
            }),
            UserInput::Image { detail: _, url } if is_remote_image_url(&url) => {
                Err(invalid_request(REMOTE_IMAGE_URL_ERROR))
            }
            UserInput::Image { detail, url } => Ok(AgentInput::Image { uri: url, detail }),
            UserInput::LocalImage { detail, path } => Ok(AgentInput::LocalImage { path, detail }),
            UserInput::Skill { name, path } => Ok(AgentInput::Skill { name, path }),
            UserInput::Mention { name, path } => Ok(AgentInput::Mention { name, path }),
        })
        .collect()
}

fn is_remote_image_url(image_url: &str) -> bool {
    image_url.split_once(':').is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    })
}

fn to_protocol_submission(submission: ThreadQueuedSubmission) -> QueuedSubmission {
    QueuedSubmission {
        id: submission.id,
        input: submission
            .input
            .into_iter()
            .map(|input| match input {
                AgentInput::Text {
                    text,
                    text_elements,
                } => UserInput::Text {
                    text,
                    text_elements,
                },
                AgentInput::Image { uri, detail } => UserInput::Image { detail, url: uri },
                AgentInput::LocalImage { path, detail } => UserInput::LocalImage { detail, path },
                AgentInput::Skill { name, path } => UserInput::Skill { name, path },
                AgentInput::Mention { name, path } => UserInput::Mention { name, path },
            })
            .collect(),
        client_user_message_id: submission.client_user_message_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::ImageDetail;

    #[test]
    fn remote_queue_images_fail_closed_with_codex_error() {
        for url in [
            "http://example.test/image.png",
            "HTTPS://example.test/image.png",
        ] {
            let error = lower_user_input(vec![UserInput::Image {
                detail: Some(ImageDetail::High),
                url: url.to_string(),
            }])
            .expect_err("remote queue image must fail closed");

            assert_eq!(error.code, error_codes::INVALID_REQUEST);
            assert_eq!(error.message, REMOTE_IMAGE_URL_ERROR);
        }
    }

    #[test]
    fn inline_queue_images_keep_typed_detail() {
        let input = lower_user_input(vec![UserInput::Image {
            detail: Some(ImageDetail::Low),
            url: "data:image/png;base64,AA==".to_string(),
        }])
        .expect("inline image");

        assert_eq!(
            input,
            vec![AgentInput::Image {
                uri: "data:image/png;base64,AA==".to_string(),
                detail: Some(ImageDetail::Low),
            }]
        );
    }

    #[test]
    fn queue_cursor_uses_usize_and_rejects_non_numeric_values() {
        assert_eq!(
            parse_cursor(Some(&usize::MAX.to_string())).expect("maximum cursor"),
            usize::MAX
        );

        let error = parse_cursor(Some("not-a-cursor")).expect_err("invalid cursor");
        assert_eq!(error.code, error_codes::INVALID_REQUEST);
        assert!(error
            .message
            .starts_with("invalid queue pagination cursor:"));
    }
}
