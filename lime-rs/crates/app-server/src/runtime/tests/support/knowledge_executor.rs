use super::super::*;
use async_trait::async_trait;

pub(in crate::runtime::tests) struct TestKnowledgeBuilderRuntimeExecutor {
    pub(in crate::runtime::tests) calls: Mutex<Vec<lime_knowledge::KnowledgeBuilderRuntimePlan>>,
}

impl TestKnowledgeBuilderRuntimeExecutor {
    pub(in crate::runtime::tests) fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(in crate::runtime::tests) fn calls(
        &self,
    ) -> Vec<lime_knowledge::KnowledgeBuilderRuntimePlan> {
        self.calls
            .lock()
            .expect("test knowledge builder calls mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl KnowledgeBuilderRuntimeExecutor for TestKnowledgeBuilderRuntimeExecutor {
    async fn execute(
        &self,
        plan: lime_knowledge::KnowledgeBuilderRuntimePlan,
    ) -> Result<lime_knowledge::KnowledgeBuilderRuntimeExecution, RuntimeCoreError> {
        self.calls
            .lock()
            .expect("test knowledge builder calls mutex poisoned")
            .push(plan.clone());
        Ok(lime_knowledge::KnowledgeBuilderRuntimeExecution {
                skill_name: plan.skill_name,
                execution_id: plan.execution_id,
                session_id: Some(plan.session_id),
                status: "succeeded".to_string(),
                provider: plan.provider_override,
                model: plan.model_override,
                output: Some(
                    json!({
                        "primaryDocument": {
                            "path": "documents/runtime-founder.md",
                            "content": "# Runtime 创始人\n\n## 智能体应用指南\n\n- 只引用长期主义与不夸大收入。"
                        },
                        "status": "needs-review",
                        "missingFacts": ["代表案例待补充"],
                        "warnings": ["收入数据未确认"],
                        "provenance": {
                            "kind": "agent-skill",
                            "name": "personal-ip-knowledge-builder",
                            "version": "1.0.0"
                        }
                    })
                    .to_string(),
                ),
                error: None,
            })
    }
}
