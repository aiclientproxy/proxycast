use super::{dispatch_result, parse_params, JsonRpcError, RequestProcessor, RpcDispatch};
use agent_protocol::ModeKind;
use app_server_protocol::{
    CollaborationModeListParams, CollaborationModeListResponse, CollaborationModeMask,
};

impl RequestProcessor {
    pub(super) async fn handle_collaboration_mode_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let CollaborationModeListParams {} = parse_params(params)?;
        dispatch_result(CollaborationModeListResponse {
            data: collaboration_mode_catalog(),
        })
    }
}

fn collaboration_mode_catalog() -> Vec<CollaborationModeMask> {
    vec![
        CollaborationModeMask {
            name: "Plan".to_string(),
            mode: Some(ModeKind::Plan),
            model: None,
            reasoning_effort: Some(Some("medium".to_string())),
        },
        CollaborationModeMask {
            name: "Default".to_string(),
            mode: Some(ModeKind::Default),
            model: None,
            reasoning_effort: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_keeps_codex_order_without_owning_models() {
        let catalog = collaboration_mode_catalog();
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0].mode, Some(ModeKind::Plan));
        assert_eq!(catalog[0].reasoning_effort, Some(Some("medium".into())));
        assert_eq!(catalog[1].mode, Some(ModeKind::Default));
        assert!(catalog.iter().all(|preset| preset.model.is_none()));
    }
}
