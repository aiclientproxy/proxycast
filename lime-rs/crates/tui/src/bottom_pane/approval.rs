use app_server_protocol::protocol::v2::{
    CommandExecutionApprovalDecision, CommandExecutionRequestApprovalParams,
    CommandExecutionRequestApprovalResponse, FileChangeApprovalDecision,
    FileChangeRequestApprovalParams, FileChangeRequestApprovalResponse, GrantedPermissionProfile,
    PermissionGrantScope, PermissionsRequestApprovalParams, PermissionsRequestApprovalResponse,
    ServerRequest,
};
use app_server_protocol::RequestId;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};

use super::AppServerResponse;

#[derive(Debug)]
pub(super) enum ApprovalRequest {
    Command {
        id: RequestId,
        params: CommandExecutionRequestApprovalParams,
    },
    FileChange {
        id: RequestId,
        params: FileChangeRequestApprovalParams,
    },
    Permissions {
        id: RequestId,
        params: PermissionsRequestApprovalParams,
    },
}

#[derive(Debug)]
pub(super) struct Approval {
    pub(super) request: ApprovalRequest,
    pub(super) selected: usize,
}

impl Approval {
    pub(super) fn from_server_request(request: ServerRequest) -> Self {
        let request = match request {
            ServerRequest::ItemCommandExecutionRequestApproval { id, params } => {
                ApprovalRequest::Command { id, params }
            }
            ServerRequest::ItemFileChangeRequestApproval { id, params } => {
                ApprovalRequest::FileChange { id, params }
            }
            ServerRequest::ItemPermissionsRequestApproval { id, params } => {
                ApprovalRequest::Permissions { id, params }
            }
            _ => unreachable!("approval constructor requires an approval server request"),
        };
        Self {
            request,
            selected: 0,
        }
    }

    pub(super) fn handle_event(&mut self, event: Event) -> Option<AppServerResponse> {
        let Event::Key(key) = event else {
            return None;
        };
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Some(self.cancel_response());
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = self.selected.saturating_sub(1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = (self.selected + 1).min(self.option_count().saturating_sub(1));
                None
            }
            KeyCode::Enter => Some(self.response_for_selected()),
            KeyCode::Esc => Some(self.cancel_response()),
            KeyCode::Char('y') => {
                self.selected = 0;
                Some(self.response_for_selected())
            }
            KeyCode::Char('n') => Some(self.decline_response()),
            _ => None,
        }
    }

    pub(super) fn option_labels(&self) -> Vec<String> {
        match &self.request {
            ApprovalRequest::Command { params, .. } => params
                .available_decisions
                .clone()
                .unwrap_or_else(default_command_decisions)
                .into_iter()
                .map(command_decision_label)
                .map(str::to_string)
                .collect(),
            ApprovalRequest::FileChange { .. } => default_file_decisions()
                .into_iter()
                .map(file_decision_label)
                .map(str::to_string)
                .collect(),
            ApprovalRequest::Permissions { .. } => vec![
                "Grant for this turn".to_string(),
                "Grant for this session".to_string(),
                "Decline".to_string(),
            ],
        }
    }

    fn option_count(&self) -> usize {
        self.option_labels().len()
    }

    fn response_for_selected(&self) -> AppServerResponse {
        match &self.request {
            ApprovalRequest::Command { id, params } => {
                let decisions = params
                    .available_decisions
                    .clone()
                    .unwrap_or_else(default_command_decisions);
                let decision = decisions
                    .get(self.selected)
                    .copied()
                    .unwrap_or(CommandExecutionApprovalDecision::Cancel);
                AppServerResponse::Command {
                    id: id.clone(),
                    response: CommandExecutionRequestApprovalResponse { decision },
                }
            }
            ApprovalRequest::FileChange { id, .. } => {
                let decision = default_file_decisions()
                    .get(self.selected)
                    .copied()
                    .unwrap_or(FileChangeApprovalDecision::Cancel);
                AppServerResponse::FileChange {
                    id: id.clone(),
                    response: FileChangeRequestApprovalResponse { decision },
                }
            }
            ApprovalRequest::Permissions { id, params } => {
                let (permissions, scope) = match self.selected {
                    0 => (
                        GrantedPermissionProfile {
                            network: params.permissions.network.clone(),
                            file_system: params.permissions.file_system.clone(),
                        },
                        PermissionGrantScope::Turn,
                    ),
                    1 => (
                        GrantedPermissionProfile {
                            network: params.permissions.network.clone(),
                            file_system: params.permissions.file_system.clone(),
                        },
                        PermissionGrantScope::Session,
                    ),
                    _ => (
                        GrantedPermissionProfile::default(),
                        PermissionGrantScope::Turn,
                    ),
                };
                AppServerResponse::Permissions {
                    id: id.clone(),
                    response: PermissionsRequestApprovalResponse {
                        permissions,
                        scope,
                        strict_auto_review: None,
                    },
                }
            }
        }
    }

    fn cancel_response(&self) -> AppServerResponse {
        match &self.request {
            ApprovalRequest::Command { id, .. } => AppServerResponse::Command {
                id: id.clone(),
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Cancel,
                },
            },
            ApprovalRequest::FileChange { id, .. } => AppServerResponse::FileChange {
                id: id.clone(),
                response: FileChangeRequestApprovalResponse {
                    decision: FileChangeApprovalDecision::Cancel,
                },
            },
            ApprovalRequest::Permissions { id, .. } => declined_permissions(id.clone()),
        }
    }

    fn decline_response(&self) -> AppServerResponse {
        match &self.request {
            ApprovalRequest::Command { id, params } => {
                let decision = params
                    .available_decisions
                    .as_ref()
                    .filter(|decisions| {
                        decisions.contains(&CommandExecutionApprovalDecision::Decline)
                    })
                    .map(|_| CommandExecutionApprovalDecision::Decline)
                    .unwrap_or(CommandExecutionApprovalDecision::Cancel);
                AppServerResponse::Command {
                    id: id.clone(),
                    response: CommandExecutionRequestApprovalResponse { decision },
                }
            }
            ApprovalRequest::FileChange { id, .. } => AppServerResponse::FileChange {
                id: id.clone(),
                response: FileChangeRequestApprovalResponse {
                    decision: FileChangeApprovalDecision::Decline,
                },
            },
            ApprovalRequest::Permissions { id, .. } => declined_permissions(id.clone()),
        }
    }
}

fn declined_permissions(id: RequestId) -> AppServerResponse {
    AppServerResponse::Permissions {
        id,
        response: PermissionsRequestApprovalResponse {
            permissions: GrantedPermissionProfile::default(),
            scope: PermissionGrantScope::Turn,
            strict_auto_review: None,
        },
    }
}

fn default_command_decisions() -> Vec<CommandExecutionApprovalDecision> {
    vec![
        CommandExecutionApprovalDecision::Accept,
        CommandExecutionApprovalDecision::AcceptForSession,
        CommandExecutionApprovalDecision::Decline,
        CommandExecutionApprovalDecision::Cancel,
    ]
}

fn default_file_decisions() -> Vec<FileChangeApprovalDecision> {
    vec![
        FileChangeApprovalDecision::Accept,
        FileChangeApprovalDecision::AcceptForSession,
        FileChangeApprovalDecision::Decline,
        FileChangeApprovalDecision::Cancel,
    ]
}

fn command_decision_label(decision: CommandExecutionApprovalDecision) -> &'static str {
    match decision {
        CommandExecutionApprovalDecision::Accept => "Allow once",
        CommandExecutionApprovalDecision::AcceptForSession => "Allow for this session",
        CommandExecutionApprovalDecision::Decline => "Decline",
        CommandExecutionApprovalDecision::Cancel => "Cancel turn",
    }
}

fn file_decision_label(decision: FileChangeApprovalDecision) -> &'static str {
    match decision {
        FileChangeApprovalDecision::Accept => "Allow once",
        FileChangeApprovalDecision::AcceptForSession => "Allow for this session",
        FileChangeApprovalDecision::Decline => "Decline",
        FileChangeApprovalDecision::Cancel => "Cancel turn",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyModifiers};

    fn command_approval() -> Approval {
        Approval::from_server_request(ServerRequest::ItemCommandExecutionRequestApproval {
            id: RequestId::Integer(7),
            params: CommandExecutionRequestApprovalParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "command-1".to_string(),
                started_at_ms: 1,
                approval_id: None,
                reason: Some("needs network".to_string()),
                network_approval_context: None,
                command: Some("cargo test".to_string()),
                cwd: Some("/workspace".to_string()),
                available_decisions: None,
            },
        })
    }

    #[test]
    fn enter_accepts_the_selected_command_decision() {
        let mut approval = command_approval();
        let response = approval.handle_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));

        assert_eq!(
            response,
            Some(AppServerResponse::Command {
                id: RequestId::Integer(7),
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Accept,
                },
            })
        );
    }

    #[test]
    fn escape_cancels_instead_of_approving() {
        let mut approval = command_approval();
        let response =
            approval.handle_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));

        assert!(matches!(
            response,
            Some(AppServerResponse::Command {
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Cancel,
                },
                ..
            })
        ));
    }

    #[test]
    fn ctrl_c_cancels_instead_of_approving() {
        let mut approval = command_approval();
        let response = approval.handle_event(Event::Key(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        )));

        assert!(matches!(
            response,
            Some(AppServerResponse::Command {
                response: CommandExecutionRequestApprovalResponse {
                    decision: CommandExecutionApprovalDecision::Cancel,
                },
                ..
            })
        ));
    }
}
