use super::{ProjectionReadWindow, ProjectionStore};
use crate::runtime::{article_workspace_projection, artifact_projection};
use serde_json::{Map, Value};

impl ProjectionStore {
    pub(in crate::runtime) fn read_thread_product_projection(
        &self,
        session_id: &str,
    ) -> Result<Option<Value>, String> {
        let Some(projection) =
            self.read_session_projection(session_id, ProjectionReadWindow::default())?
        else {
            return Ok(None);
        };

        let article_workspace = article_workspace_projection::article_workspace_from_events(
            &projection.session,
            &projection.item_events,
        );
        let artifacts =
            artifact_projection::artifact_summaries_for_turn(&projection.item_events, None);

        let mut product = Map::new();
        if let Some(article_workspace) = article_workspace {
            product.insert("articleWorkspace".to_string(), article_workspace);
        }
        if !artifacts.is_empty() {
            product.insert(
                "artifacts".to_string(),
                serde_json::to_value(artifacts)
                    .map_err(|error| format!("cannot serialize thread artifacts: {error}"))?,
            );
        }
        Ok((!product.is_empty()).then_some(Value::Object(product)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::AgentEvent;
    use serde_json::json;

    #[test]
    fn thread_product_projection_reads_article_workspace_from_durable_projection() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store");
        store
            .apply_event(&AgentEvent {
                event_id: "artifact-workspace".to_string(),
                sequence: 1,
                session_id: "session-1".to_string(),
                thread_id: Some("thread-1".to_string()),
                turn_id: None,
                event_type: "artifact.snapshot".to_string(),
                timestamp: "2026-07-21T00:00:00Z".to_string(),
                payload: json!({
                    "session": {
                        "createdAt": "2026-07-21T00:00:00Z",
                        "updatedAt": "2026-07-21T00:00:00Z",
                        "workspaceId": "workspace-1"
                    },
                    "artifact": {
                        "artifactId": "workspace-patch-1",
                        "kind": "workspace_patch",
                        "metadata": {
                            "workspacePatch": {
                                "schemaVersion": "article-workspace.v1",
                                "appId": "content-factory-app",
                                "sessionId": "session-1",
                                "workspaceId": "workspace-1",
                                "objects": [{
                                    "ref": {
                                        "appId": "content-factory-app",
                                        "kind": "articleDraft",
                                        "id": "article-1",
                                        "sessionId": "session-1",
                                        "artifactIds": ["article-1"]
                                    },
                                    "title": "Article",
                                    "status": "ready",
                                    "previewArtifactId": "article-1",
                                    "source": {"markdown": "# Initial"}
                                }]
                            }
                        }
                    }
                }),
            })
            .expect("workspace event");
        store
            .apply_event(&AgentEvent {
                event_id: "artifact-edited-draft".to_string(),
                sequence: 2,
                session_id: "session-1".to_string(),
                thread_id: Some("thread-1".to_string()),
                turn_id: None,
                event_type: "artifact.snapshot".to_string(),
                timestamp: "2026-07-21T00:01:00Z".to_string(),
                payload: json!({
                    "artifact": {
                        "artifactId": "article-1",
                        "kind": "article_workspace_draft",
                        "content": "# Edited",
                        "metadata": {
                            "workspacePatch": {
                                "objects": [{
                                    "ref": {
                                        "appId": "content-factory-app",
                                        "kind": "articleDraft",
                                        "id": "article-1",
                                        "sessionId": "session-1"
                                    },
                                    "source": {
                                        "documentText": "# Edited",
                                        "finalMarkdown": "# Edited",
                                        "edited": true,
                                        "updatedAt": "2026-07-21T00:01:00Z"
                                    }
                                }]
                            },
                            "editedDraft": {
                                "objectKey": "content-factory-app:session-1:articleDraft:article-1",
                                "objectRef": {
                                    "appId": "content-factory-app",
                                    "kind": "articleDraft",
                                    "id": "article-1",
                                    "sessionId": "session-1"
                                },
                                "markdown": "# Edited",
                                "updatedAt": "2026-07-21T00:01:00Z"
                            }
                        }
                    }
                }),
            })
            .expect("edited draft event");

        let projection = store
            .read_thread_product_projection("session-1")
            .expect("thread product projection")
            .expect("product projection");
        assert_eq!(
            projection["articleWorkspace"]["objects"][0]["source"]["finalMarkdown"],
            "# Edited"
        );
        assert_eq!(
            projection["articleWorkspace"]["editedDraft"]["markdown"],
            "# Edited"
        );
        assert!(projection["artifacts"]
            .as_array()
            .expect("artifacts")
            .iter()
            .any(|artifact| artifact["artifactRef"] == "article-1"));
    }
}
