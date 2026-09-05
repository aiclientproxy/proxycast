mod approval_overlay;
mod chat_composer;
mod render;
mod request_user_input;

use std::collections::VecDeque;

use app_server_protocol::RequestId;
use app_server_protocol::protocol::v2::{
    CommandExecutionApprovalDecision, CommandExecutionRequestApprovalResponse,
    FileChangeApprovalDecision, FileChangeRequestApprovalResponse, GrantedPermissionProfile,
    PermissionGrantScope, PermissionsRequestApprovalResponse, ServerRequest,
    ToolRequestUserInputResponse,
};
use crossterm::event::{Event, KeyEvent};

use approval_overlay::ApprovalOverlay;
pub(crate) use chat_composer::{ChatComposer, InputResult};
use request_user_input::RequestUserInputOverlay;

pub(crate) use render::{desired_height_with_locale, render_with_locale};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AppServerResponse {
    Command {
        id: RequestId,
        response: CommandExecutionRequestApprovalResponse,
    },
    FileChange {
        id: RequestId,
        response: FileChangeRequestApprovalResponse,
    },
    Permissions {
        id: RequestId,
        response: PermissionsRequestApprovalResponse,
    },
    UserInput {
        id: RequestId,
        response: ToolRequestUserInputResponse,
    },
}

impl AppServerResponse {
    pub(crate) fn fail_closed(request: ServerRequest) -> Result<Self, Box<ServerRequest>> {
        match request {
            ServerRequest::ItemCommandExecutionRequestApproval { id, .. } => Ok(Self::Command {
                id,
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Cancel,
                },
            }),
            ServerRequest::ItemFileChangeRequestApproval { id, .. } => Ok(Self::FileChange {
                id,
                response: FileChangeRequestApprovalResponse {
                    decision: FileChangeApprovalDecision::Cancel,
                },
            }),
            ServerRequest::ItemPermissionsRequestApproval { id, .. } => Ok(Self::Permissions {
                id,
                response: PermissionsRequestApprovalResponse {
                    permissions: GrantedPermissionProfile::default(),
                    scope: PermissionGrantScope::Turn,
                    strict_auto_review: None,
                },
            }),
            ServerRequest::ItemToolRequestUserInput { id, .. } => Ok(Self::UserInput {
                id,
                response: ToolRequestUserInputResponse {
                    answers: Default::default(),
                },
            }),
            request => Err(Box::new(request)),
        }
    }
}

#[derive(Debug)]
enum PendingInteraction {
    Approval(ApprovalOverlay),
    UserInput(RequestUserInputOverlay),
}

impl PendingInteraction {
    fn from_server_request(request: ServerRequest) -> Result<Self, Box<ServerRequest>> {
        match request {
            request @ (ServerRequest::ItemCommandExecutionRequestApproval { .. }
            | ServerRequest::ItemFileChangeRequestApproval { .. }
            | ServerRequest::ItemPermissionsRequestApproval { .. }) => Ok(Self::Approval(
                ApprovalOverlay::from_server_request(request),
            )),
            ServerRequest::ItemToolRequestUserInput { id, params } => {
                Ok(Self::UserInput(RequestUserInputOverlay::new(id, params)))
            }
            request => Err(Box::new(request)),
        }
    }

    fn handle_key_event(&mut self, key: KeyEvent) -> Option<AppServerResponse> {
        match self {
            Self::Approval(approval) => approval.handle_key_event(key),
            Self::UserInput(request) => request.handle_key_event(key),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BottomPane {
    queue: VecDeque<PendingInteraction>,
}

impl BottomPane {
    pub(crate) fn enqueue(&mut self, request: ServerRequest) -> Result<(), Box<ServerRequest>> {
        self.queue
            .push_back(PendingInteraction::from_server_request(request)?);
        Ok(())
    }

    pub(crate) fn is_active(&self) -> bool {
        !self.queue.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.queue.clear();
    }

    fn current(&self) -> Option<&PendingInteraction> {
        self.queue.front()
    }

    pub(crate) fn handle_event(&mut self, event: Event) -> Option<AppServerResponse> {
        match event {
            Event::Key(key) => self.handle_key_event(key),
            Event::Paste(text) => {
                let Some(PendingInteraction::UserInput(request)) = self.queue.front_mut() else {
                    return None;
                };
                request.editing = true;
                request.composer.insert(&text);
                None
            }
            _ => None,
        }
    }

    pub(crate) fn handle_key_event(&mut self, key: KeyEvent) -> Option<AppServerResponse> {
        let response = self.queue.front_mut()?.handle_key_event(key)?;
        self.queue.pop_front();
        Some(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        CommandExecutionApprovalDecision, CommandExecutionRequestApprovalParams,
        FileChangeApprovalDecision, FileChangeRequestApprovalParams,
        PermissionsRequestApprovalParams, RequestPermissionProfile, ToolRequestUserInputOption,
        ToolRequestUserInputParams, ToolRequestUserInputQuestion,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn queues_requests_and_resolves_them_in_arrival_order() {
        let mut pane = BottomPane::default();
        pane.enqueue(ServerRequest::ItemCommandExecutionRequestApproval {
            id: RequestId::Integer(1),
            params: CommandExecutionRequestApprovalParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "command-1".to_string(),
                started_at_ms: 1,
                approval_id: None,
                reason: None,
                network_approval_context: None,
                command: Some("cargo test".to_string()),
                cwd: Some("/workspace".to_string()),
                available_decisions: None,
            },
        })
        .expect("queue approval");
        pane.enqueue(ServerRequest::ItemToolRequestUserInput {
            id: RequestId::Integer(2),
            params: ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![ToolRequestUserInputQuestion {
                    id: "mode".to_string(),
                    header: "Mode".to_string(),
                    question: "Choose a mode".to_string(),
                    is_other: false,
                    is_secret: false,
                    options: Some(vec![ToolRequestUserInputOption {
                        label: "Fast".to_string(),
                        description: "Continue immediately".to_string(),
                    }]),
                }],
                auto_resolution_ms: None,
            },
        })
        .expect("queue user input");

        let first = pane.handle_event(key(KeyCode::Enter));
        assert!(matches!(
            first,
            Some(AppServerResponse::Command {
                id: RequestId::Integer(1),
                ..
            })
        ));
        assert!(pane.is_active());

        let second = pane.handle_event(key(KeyCode::Enter));
        assert!(matches!(
            second,
            Some(AppServerResponse::UserInput {
                id: RequestId::Integer(2),
                ..
            })
        ));
        assert!(!pane.is_active());
    }

    #[test]
    fn non_interactive_responses_fail_closed_for_every_supported_interaction() {
        let command =
            AppServerResponse::fail_closed(ServerRequest::ItemCommandExecutionRequestApproval {
                id: RequestId::Integer(1),
                params: CommandExecutionRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "command-1".to_string(),
                    started_at_ms: 1,
                    approval_id: None,
                    reason: None,
                    network_approval_context: None,
                    command: None,
                    cwd: None,
                    available_decisions: None,
                },
            })
            .expect("command response");
        assert!(matches!(
            command,
            AppServerResponse::Command {
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Cancel,
                },
                ..
            }
        ));

        let file_change =
            AppServerResponse::fail_closed(ServerRequest::ItemFileChangeRequestApproval {
                id: RequestId::Integer(2),
                params: FileChangeRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "patch-1".to_string(),
                    started_at_ms: 1,
                    reason: None,
                    grant_root: None,
                },
            })
            .expect("file change response");
        assert!(matches!(
            file_change,
            AppServerResponse::FileChange {
                response: FileChangeRequestApprovalResponse {
                    decision: FileChangeApprovalDecision::Cancel,
                },
                ..
            }
        ));

        let permissions =
            AppServerResponse::fail_closed(ServerRequest::ItemPermissionsRequestApproval {
                id: RequestId::Integer(3),
                params: PermissionsRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "permissions-1".to_string(),
                    environment_id: None,
                    started_at_ms: 1,
                    cwd: "/workspace".to_string(),
                    reason: None,
                    permissions: RequestPermissionProfile {
                        network: None,
                        file_system: None,
                    },
                },
            })
            .expect("permissions response");
        assert!(matches!(
            permissions,
            AppServerResponse::Permissions {
                response: PermissionsRequestApprovalResponse {
                    permissions: GrantedPermissionProfile {
                        network: None,
                        file_system: None,
                    },
                    scope: PermissionGrantScope::Turn,
                    strict_auto_review: None,
                },
                ..
            }
        ));

        let user_input = AppServerResponse::fail_closed(ServerRequest::ItemToolRequestUserInput {
            id: RequestId::Integer(4),
            params: ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: Vec::new(),
                auto_resolution_ms: None,
            },
        })
        .expect("user input response");
        assert!(matches!(
            user_input,
            AppServerResponse::UserInput {
                response: ToolRequestUserInputResponse { answers },
                ..
            } if answers.is_empty()
        ));
    }
}
