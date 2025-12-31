//! 工具调用循环引擎
//!
//! 实现 Agent 工具调用循环，自动执行工具并继续对话
//! 符合 Requirements 7.1, 7.2, 7.3, 7.4, 7.5
//!
//! ## 功能
//! - 检测 Agent 响应中的工具调用
//! - 执行工具并收集结果
//! - 将工具结果发送回 Agent 继续对话
//! - 最大迭代限制防止无限循环

use crate::agent::tools::{ToolError, ToolRegistry, ToolResult as ToolsResult};
use crate::agent::types::{
    AgentMessage, MessageContent, StreamEvent, StreamResult, ToolCall, ToolExecutionResult,
};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// 工具循环错误类型
#[derive(Debug, Error)]
pub enum ToolLoopError {
    /// 超过最大迭代次数
    /// Requirements: 7.5 - THE Tool_Loop SHALL enforce a maximum iteration limit
    #[error("超过最大迭代次数限制: {0}")]
    MaxIterationsExceeded(usize),

    /// 工具执行错误
    /// Requirements: 7.4 - IF a tool execution fails, THEN THE Tool_Loop SHALL include the error
    #[error("工具执行错误: {0}")]
    ToolExecution(String),

    /// 工具未找到
    #[error("工具未找到: {0}")]
    ToolNotFound(String),

    /// JSON 解析错误
    #[error("JSON 解析错误: {0}")]
    JsonParse(String),

    /// 通道发送错误
    #[error("事件发送失败")]
    ChannelSend,
}

/// 工具执行结果（内部使用）
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// 工具调用 ID
    pub tool_call_id: String,
    /// 工具名称
    pub tool_name: String,
    /// 执行结果
    pub result: ToolsResult,
}

impl ToolCallResult {
    /// 创建新的工具调用结果
    pub fn new(tool_call_id: String, tool_name: String, result: ToolsResult) -> Self {
        Self {
            tool_call_id,
            tool_name,
            result,
        }
    }

    /// 转换为 AgentMessage（tool 角色）
    ///
    /// Requirements: 7.2 - THE Tool_Loop SHALL send tool results back to the Agent as tool role messages
    pub fn to_agent_message(&self) -> AgentMessage {
        let content = if self.result.success {
            self.result.output.clone()
        } else {
            format!(
                "Error: {}",
                self.result.error.as_deref().unwrap_or("Unknown error")
            )
        };

        AgentMessage {
            role: "tool".to_string(),
            content: MessageContent::Text(content),
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_calls: None,
            tool_call_id: Some(self.tool_call_id.clone()),
        }
    }

    /// 转换为 ToolExecutionResult（用于前端显示）
    pub fn to_execution_result(&self) -> ToolExecutionResult {
        ToolExecutionResult {
            success: self.result.success,
            output: self.result.output.clone(),
            error: self.result.error.clone(),
        }
    }
}

/// 工具循环引擎配置
#[derive(Debug, Clone)]
pub struct ToolLoopConfig {
    /// 最大迭代次数
    /// Requirements: 7.5 - THE Tool_Loop SHALL enforce a maximum iteration limit
    pub max_iterations: usize,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25, // 默认最大 25 次迭代
        }
    }
}

impl ToolLoopConfig {
    /// 创建新的配置
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

/// 工具循环引擎
///
/// 负责执行工具调用循环，直到 Agent 产生最终响应
/// Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
pub struct ToolLoopEngine {
    /// 工具注册表
    registry: Arc<ToolRegistry>,
    /// 配置
    config: ToolLoopConfig,
}

impl ToolLoopEngine {
    /// 创建新的工具循环引擎
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            config: ToolLoopConfig::default(),
        }
    }

    /// 使用自定义配置创建
    pub fn with_config(registry: Arc<ToolRegistry>, config: ToolLoopConfig) -> Self {
        Self { registry, config }
    }

    /// 获取最大迭代次数
    pub fn max_iterations(&self) -> usize {
        self.config.max_iterations
    }

    /// 检查响应是否包含工具调用
    ///
    /// Requirements: 7.1 - WHEN the Agent response contains tool_calls
    pub fn has_tool_calls(result: &StreamResult) -> bool {
        result.has_tool_calls()
    }

    /// 执行单个工具调用
    ///
    /// Requirements: 7.1 - THE Tool_Loop SHALL execute each tool and collect results
    /// Requirements: 7.4 - IF a tool execution fails, THEN THE Tool_Loop SHALL include the error
    pub async fn execute_tool_call(&self, tool_call: &ToolCall) -> ToolCallResult {
        let tool_name = &tool_call.function.name;
        let tool_id = &tool_call.id;

        debug!("[ToolLoopEngine] 执行工具: {} (id={})", tool_name, tool_id);

        // 解析参数
        let args = match serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) {
            Ok(args) => args,
            Err(e) => {
                warn!("[ToolLoopEngine] 工具参数解析失败: {} - {}", tool_name, e);
                return ToolCallResult::new(
                    tool_id.clone(),
                    tool_name.clone(),
                    ToolsResult::failure(format!("参数解析失败: {}", e)),
                );
            }
        };

        // 执行工具
        match self.registry.execute(tool_name, args).await {
            Ok(result) => {
                debug!(
                    "[ToolLoopEngine] 工具执行成功: {} success={}",
                    tool_name, result.success
                );
                ToolCallResult::new(tool_id.clone(), tool_name.clone(), result)
            }
            Err(e) => {
                warn!("[ToolLoopEngine] 工具执行失败: {} - {}", tool_name, e);
                let error_msg = match &e {
                    ToolError::NotFound(name) => format!("工具不存在: {}", name),
                    ToolError::InvalidArguments(msg) => format!("参数无效: {}", msg),
                    ToolError::ExecutionFailed(msg) => format!("执行失败: {}", msg),
                    ToolError::Security(msg) => format!("安全错误: {}", msg),
                    ToolError::Timeout => "执行超时".to_string(),
                    ToolError::Io(e) => format!("IO 错误: {}", e),
                    ToolError::Json(e) => format!("JSON 错误: {}", e),
                };
                ToolCallResult::new(
                    tool_id.clone(),
                    tool_name.clone(),
                    ToolsResult::failure(error_msg),
                )
            }
        }
    }

    /// 执行所有工具调用
    ///
    /// Requirements: 7.1 - THE Tool_Loop SHALL execute each tool and collect results
    /// Requirements: 7.6 - WHILE the Tool_Loop is executing, THE Frontend SHALL display the current tool
    pub async fn execute_all_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        event_tx: Option<&mpsc::Sender<StreamEvent>>,
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for tool_call in tool_calls {
            // 发送工具开始事件
            if let Some(tx) = event_tx {
                let _ = tx
                    .send(StreamEvent::ToolStart {
                        tool_name: tool_call.function.name.clone(),
                        tool_id: tool_call.id.clone(),
                    })
                    .await;
            }

            // 执行工具
            let result = self.execute_tool_call(tool_call).await;

            // 发送工具结束事件
            if let Some(tx) = event_tx {
                let _ = tx
                    .send(StreamEvent::ToolEnd {
                        tool_id: tool_call.id.clone(),
                        result: result.to_execution_result(),
                    })
                    .await;
            }

            results.push(result);
        }

        results
    }

    /// 将工具结果转换为 Agent 消息列表
    ///
    /// Requirements: 7.2 - THE Tool_Loop SHALL send tool results back to the Agent as tool role messages
    pub fn results_to_messages(results: &[ToolCallResult]) -> Vec<AgentMessage> {
        results.iter().map(|r| r.to_agent_message()).collect()
    }

    /// 检查是否应该继续循环
    ///
    /// Requirements: 7.3 - THE Tool_Loop SHALL continue until the Agent produces a final response without tool_calls
    /// Requirements: 7.5 - THE Tool_Loop SHALL enforce a maximum iteration limit
    pub fn should_continue(&self, result: &StreamResult, iteration: usize) -> bool {
        // 检查最大迭代次数
        if iteration >= self.config.max_iterations {
            warn!(
                "[ToolLoopEngine] 达到最大迭代次数: {}",
                self.config.max_iterations
            );
            return false;
        }

        // 检查是否有工具调用
        Self::has_tool_calls(result)
    }

    /// 创建 assistant 消息（包含工具调用）
    pub fn create_assistant_message(
        content: &str,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> AgentMessage {
        AgentMessage {
            role: "assistant".to_string(),
            content: MessageContent::Text(content.to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_calls: tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|tc| crate::agent::types::ToolCall {
                        id: tc.id,
                        call_type: tc.call_type,
                        function: tc.function,
                    })
                    .collect()
            }),
            tool_call_id: None,
        }
    }
}

/// 工具循环状态
///
/// 用于跟踪工具循环的执行状态
#[derive(Debug, Clone)]
pub struct ToolLoopState {
    /// 当前迭代次数
    pub iteration: usize,
    /// 累计执行的工具调用数
    pub total_tool_calls: usize,
    /// 是否已完成
    pub completed: bool,
    /// 最终内容
    pub final_content: Option<String>,
}

impl Default for ToolLoopState {
    fn default() -> Self {
        Self {
            iteration: 0,
            total_tool_calls: 0,
            completed: false,
            final_content: None,
        }
    }
}

impl ToolLoopState {
    /// 创建新的状态
    pub fn new() -> Self {
        Self::default()
    }

    /// 增加迭代次数
    pub fn increment_iteration(&mut self) {
        self.iteration += 1;
    }

    /// 增加工具调用计数
    pub fn add_tool_calls(&mut self, count: usize) {
        self.total_tool_calls += count;
    }

    /// 标记为完成
    pub fn mark_completed(&mut self, content: String) {
        self.completed = true;
        self.final_content = Some(content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::types::{JsonSchema, PropertySchema, ToolDefinition};
    use crate::agent::tools::{Tool, ToolRegistry};
    use crate::agent::types::FunctionCall;
    use async_trait::async_trait;

    /// 测试用的 Echo 工具
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo the input message").with_parameters(
                JsonSchema::new().add_property(
                    "message",
                    PropertySchema::string("The message to echo"),
                    true,
                ),
            )
        }

        async fn execute(&self, args: serde_json::Value) -> Result<ToolsResult, ToolError> {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("缺少 message 参数".to_string()))?;

            Ok(ToolsResult::success(message))
        }
    }

    /// 测试用的失败工具
    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("failing", "A tool that always fails")
        }

        async fn execute(&self, _args: serde_json::Value) -> Result<ToolsResult, ToolError> {
            Err(ToolError::ExecutionFailed("故意失败".to_string()))
        }
    }

    fn create_test_registry() -> Arc<ToolRegistry> {
        let registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();
        registry.register(FailingTool).unwrap();
        Arc::new(registry)
    }

    fn create_tool_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[test]
    fn test_tool_loop_engine_creation() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        assert_eq!(engine.max_iterations(), 25);
    }

    #[test]
    fn test_tool_loop_engine_with_config() {
        let registry = create_test_registry();
        let config = ToolLoopConfig::new(10);
        let engine = ToolLoopEngine::with_config(registry, config);

        assert_eq!(engine.max_iterations(), 10);
    }

    #[test]
    fn test_has_tool_calls() {
        // 无工具调用
        let result_no_tools = StreamResult::new("Hello".to_string());
        assert!(!ToolLoopEngine::has_tool_calls(&result_no_tools));

        // 有工具调用
        let result_with_tools = StreamResult::new("".to_string()).with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        assert!(ToolLoopEngine::has_tool_calls(&result_with_tools));

        // 空工具调用列表
        let result_empty_tools = StreamResult {
            content: "".to_string(),
            tool_calls: Some(vec![]),
            usage: None,
        };
        assert!(!ToolLoopEngine::has_tool_calls(&result_empty_tools));
    }

    #[tokio::test]
    async fn test_execute_tool_call_success() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let tool_call = create_tool_call("call_1", "echo", r#"{"message": "Hello, World!"}"#);
        let result = engine.execute_tool_call(&tool_call).await;

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.tool_name, "echo");
        assert!(result.result.success);
        assert_eq!(result.result.output, "Hello, World!");
    }

    #[tokio::test]
    async fn test_execute_tool_call_failure() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let tool_call = create_tool_call("call_2", "failing", "{}");
        let result = engine.execute_tool_call(&tool_call).await;

        assert_eq!(result.tool_call_id, "call_2");
        assert_eq!(result.tool_name, "failing");
        assert!(!result.result.success);
        assert!(result.result.error.is_some());
    }

    #[tokio::test]
    async fn test_execute_tool_call_not_found() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let tool_call = create_tool_call("call_3", "nonexistent", "{}");
        let result = engine.execute_tool_call(&tool_call).await;

        assert!(!result.result.success);
        assert!(result.result.error.as_ref().unwrap().contains("工具不存在"));
    }

    #[tokio::test]
    async fn test_execute_tool_call_invalid_args() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let tool_call = create_tool_call("call_4", "echo", "invalid json");
        let result = engine.execute_tool_call(&tool_call).await;

        assert!(!result.result.success);
        assert!(result
            .result
            .error
            .as_ref()
            .unwrap()
            .contains("参数解析失败"));
    }

    #[tokio::test]
    async fn test_execute_all_tool_calls() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let tool_calls = vec![
            create_tool_call("call_1", "echo", r#"{"message": "First"}"#),
            create_tool_call("call_2", "echo", r#"{"message": "Second"}"#),
        ];

        let results = engine.execute_all_tool_calls(&tool_calls, None).await;

        assert_eq!(results.len(), 2);
        assert!(results[0].result.success);
        assert_eq!(results[0].result.output, "First");
        assert!(results[1].result.success);
        assert_eq!(results[1].result.output, "Second");
    }

    #[test]
    fn test_results_to_messages() {
        let results = vec![
            ToolCallResult::new(
                "call_1".to_string(),
                "echo".to_string(),
                ToolsResult::success("Hello"),
            ),
            ToolCallResult::new(
                "call_2".to_string(),
                "failing".to_string(),
                ToolsResult::failure("Error occurred"),
            ),
        ];

        let messages = ToolLoopEngine::results_to_messages(&results);

        assert_eq!(messages.len(), 2);

        // 第一个消息（成功）
        assert_eq!(messages[0].role, "tool");
        assert_eq!(messages[0].content.as_text(), "Hello");
        assert_eq!(messages[0].tool_call_id, Some("call_1".to_string()));

        // 第二个消息（失败）
        assert_eq!(messages[1].role, "tool");
        assert!(messages[1].content.as_text().contains("Error"));
        assert_eq!(messages[1].tool_call_id, Some("call_2".to_string()));
    }

    #[test]
    fn test_should_continue() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::with_config(registry, ToolLoopConfig::new(5));

        // 有工具调用，未达到限制
        let result_with_tools = StreamResult::new("".to_string()).with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        assert!(engine.should_continue(&result_with_tools, 0));
        assert!(engine.should_continue(&result_with_tools, 4));

        // 达到最大迭代次数
        assert!(!engine.should_continue(&result_with_tools, 5));

        // 无工具调用
        let result_no_tools = StreamResult::new("Final response".to_string());
        assert!(!engine.should_continue(&result_no_tools, 0));
    }

    #[test]
    fn test_tool_loop_state() {
        let mut state = ToolLoopState::new();

        assert_eq!(state.iteration, 0);
        assert_eq!(state.total_tool_calls, 0);
        assert!(!state.completed);
        assert!(state.final_content.is_none());

        state.increment_iteration();
        assert_eq!(state.iteration, 1);

        state.add_tool_calls(3);
        assert_eq!(state.total_tool_calls, 3);

        state.mark_completed("Final content".to_string());
        assert!(state.completed);
        assert_eq!(state.final_content, Some("Final content".to_string()));
    }

    #[test]
    fn test_tool_call_result_to_agent_message() {
        // 成功结果
        let success_result = ToolCallResult::new(
            "call_1".to_string(),
            "echo".to_string(),
            ToolsResult::success("Success output"),
        );
        let success_msg = success_result.to_agent_message();
        assert_eq!(success_msg.role, "tool");
        assert_eq!(success_msg.content.as_text(), "Success output");
        assert_eq!(success_msg.tool_call_id, Some("call_1".to_string()));

        // 失败结果
        let failure_result = ToolCallResult::new(
            "call_2".to_string(),
            "failing".to_string(),
            ToolsResult::failure("Something went wrong"),
        );
        let failure_msg = failure_result.to_agent_message();
        assert_eq!(failure_msg.role, "tool");
        assert!(failure_msg.content.as_text().contains("Error"));
        assert!(failure_msg
            .content
            .as_text()
            .contains("Something went wrong"));
    }

    #[tokio::test]
    async fn test_execute_all_tool_calls_with_events() {
        let registry = create_test_registry();
        let engine = ToolLoopEngine::new(registry);

        let (tx, mut rx) = mpsc::channel::<StreamEvent>(10);

        let tool_calls = vec![create_tool_call("call_1", "echo", r#"{"message": "Test"}"#)];

        let results = engine.execute_all_tool_calls(&tool_calls, Some(&tx)).await;

        assert_eq!(results.len(), 1);
        assert!(results[0].result.success);

        // 检查事件
        let event1 = rx.recv().await.unwrap();
        assert!(matches!(event1, StreamEvent::ToolStart { .. }));

        let event2 = rx.recv().await.unwrap();
        assert!(matches!(event2, StreamEvent::ToolEnd { .. }));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::agent::tools::types::{JsonSchema, PropertySchema, ToolDefinition};
    use crate::agent::tools::{Tool, ToolError, ToolRegistry, ToolResult as ToolsResult};
    use crate::agent::types::FunctionCall;
    use async_trait::async_trait;
    use proptest::prelude::*;

    /// 测试用的 Echo 工具（用于属性测试）
    struct PropTestEchoTool;

    #[async_trait]
    impl Tool for PropTestEchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo the input message").with_parameters(
                JsonSchema::new().add_property(
                    "message",
                    PropertySchema::string("The message to echo"),
                    true,
                ),
            )
        }

        async fn execute(&self, args: serde_json::Value) -> Result<ToolsResult, ToolError> {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("缺少 message 参数".to_string()))?;

            Ok(ToolsResult::success(message))
        }
    }

    /// 测试用的计数工具
    struct PropTestCountTool;

    #[async_trait]
    impl Tool for PropTestCountTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("count", "Count characters in a string").with_parameters(
                JsonSchema::new().add_property(
                    "text",
                    PropertySchema::string("The text to count"),
                    true,
                ),
            )
        }

        async fn execute(&self, args: serde_json::Value) -> Result<ToolsResult, ToolError> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("缺少 text 参数".to_string()))?;

            Ok(ToolsResult::success(format!("{}", text.len())))
        }
    }

    fn create_proptest_registry() -> Arc<ToolRegistry> {
        let registry = ToolRegistry::new();
        registry.register(PropTestEchoTool).unwrap();
        registry.register(PropTestCountTool).unwrap();
        Arc::new(registry)
    }

    /// 生成有效的工具调用 ID
    fn arb_tool_id() -> impl Strategy<Value = String> {
        "call_[a-zA-Z0-9]{8}".prop_map(|s| s)
    }

    /// 生成有效的消息内容（用于 echo 工具）
    fn arb_message_content() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ]{1,50}".prop_map(|s| s)
    }

    /// 生成工具调用数量
    fn arb_tool_call_count() -> impl Strategy<Value = usize> {
        1..=5usize
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 12: 工具循环执行完整性**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// *For any* 包含 tool_calls 的 Agent 响应，Tool Loop 应该执行所有工具并将结果
        /// 作为 tool 角色消息发送回 Agent。
        #[test]
        fn prop_tool_loop_executes_all_tools(
            tool_ids in prop::collection::vec(arb_tool_id(), 1..=5),
            messages in prop::collection::vec(arb_message_content(), 1..=5)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let registry = create_proptest_registry();
                let engine = ToolLoopEngine::new(registry);

                // 创建工具调用列表
                let tool_calls: Vec<ToolCall> = tool_ids
                    .iter()
                    .zip(messages.iter())
                    .map(|(id, msg)| ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: "echo".to_string(),
                            arguments: serde_json::json!({"message": msg}).to_string(),
                        },
                    })
                    .collect();

                let num_calls = tool_calls.len();

                // 执行所有工具调用
                let results = engine.execute_all_tool_calls(&tool_calls, None).await;

                // 验证：结果数量等于工具调用数量
                prop_assert_eq!(
                    results.len(),
                    num_calls,
                    "结果数量应该等于工具调用数量"
                );

                // 验证：每个结果都有正确的 tool_call_id
                for (i, result) in results.iter().enumerate() {
                    prop_assert_eq!(
                        &result.tool_call_id,
                        &tool_ids[i],
                        "工具调用 ID 应该匹配"
                    );
                }

                // 验证：所有工具都成功执行
                for result in &results {
                    prop_assert!(
                        result.result.success,
                        "工具执行应该成功: {:?}",
                        result.result.error
                    );
                }

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 12: 工具循环执行完整性 - 结果转换为消息**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// *For any* 工具执行结果，转换为 AgentMessage 后应该具有正确的 role 和 tool_call_id。
        #[test]
        fn prop_tool_results_convert_to_messages(
            tool_id in arb_tool_id(),
            message in arb_message_content()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let registry = create_proptest_registry();
                let engine = ToolLoopEngine::new(registry);

                let tool_call = ToolCall {
                    id: tool_id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "echo".to_string(),
                        arguments: serde_json::json!({"message": message}).to_string(),
                    },
                };

                // 执行工具
                let result = engine.execute_tool_call(&tool_call).await;

                // 转换为 AgentMessage
                let agent_msg = result.to_agent_message();

                // 验证：role 为 "tool"
                prop_assert_eq!(
                    agent_msg.role,
                    "tool",
                    "消息角色应该为 'tool'"
                );

                // 验证：tool_call_id 正确
                prop_assert_eq!(
                    agent_msg.tool_call_id,
                    Some(tool_id.clone()),
                    "tool_call_id 应该匹配"
                );

                // 验证：成功结果的内容包含原始消息
                if result.result.success {
                    prop_assert!(
                        agent_msg.content.as_text().contains(&message),
                        "成功结果的内容应该包含原始消息"
                    );
                }

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 12: 工具循环执行完整性 - 事件发送**
        /// **Validates: Requirements 7.1, 7.6**
        ///
        /// *For any* 工具执行，应该发送 ToolStart 和 ToolEnd 事件。
        #[test]
        fn prop_tool_execution_sends_events(
            tool_id in arb_tool_id(),
            message in arb_message_content()
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let registry = create_proptest_registry();
                let engine = ToolLoopEngine::new(registry);

                let (tx, mut rx) = mpsc::channel::<StreamEvent>(10);

                let tool_calls = vec![ToolCall {
                    id: tool_id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "echo".to_string(),
                        arguments: serde_json::json!({"message": message}).to_string(),
                    },
                }];

                // 执行工具调用
                let _ = engine.execute_all_tool_calls(&tool_calls, Some(&tx)).await;

                // 验证：收到 ToolStart 事件
                let event1 = rx.recv().await;
                prop_assert!(event1.is_some(), "应该收到 ToolStart 事件");
                if let Some(StreamEvent::ToolStart { tool_name, tool_id: event_tool_id }) = event1 {
                    prop_assert_eq!(tool_name, "echo", "工具名称应该为 'echo'");
                    prop_assert_eq!(event_tool_id, tool_id.clone(), "工具 ID 应该匹配");
                } else {
                    prop_assert!(false, "第一个事件应该是 ToolStart");
                }

                // 验证：收到 ToolEnd 事件
                let event2 = rx.recv().await;
                prop_assert!(event2.is_some(), "应该收到 ToolEnd 事件");
                if let Some(StreamEvent::ToolEnd { tool_id: event_tool_id, result }) = event2 {
                    prop_assert_eq!(event_tool_id, tool_id.clone(), "工具 ID 应该匹配");
                    prop_assert!(result.success, "工具执行应该成功");
                } else {
                    prop_assert!(false, "第二个事件应该是 ToolEnd");
                }

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 12: 工具循环执行完整性 - 多工具执行顺序**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// *For any* 多个工具调用，执行顺序应该与调用顺序一致。
        #[test]
        fn prop_tool_execution_order_preserved(
            count in 2..=5usize
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let registry = create_proptest_registry();
                let engine = ToolLoopEngine::new(registry);

                // 创建多个工具调用，每个使用不同的消息
                let tool_calls: Vec<ToolCall> = (0..count)
                    .map(|i| ToolCall {
                        id: format!("call_{}", i),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: "echo".to_string(),
                            arguments: serde_json::json!({"message": format!("msg_{}", i)}).to_string(),
                        },
                    })
                    .collect();

                // 执行所有工具调用
                let results = engine.execute_all_tool_calls(&tool_calls, None).await;

                // 验证：结果顺序与调用顺序一致
                for (i, result) in results.iter().enumerate() {
                    prop_assert_eq!(
                        &result.tool_call_id,
                        &format!("call_{}", i),
                        "结果顺序应该与调用顺序一致"
                    );
                    prop_assert!(
                        result.result.output.contains(&format!("msg_{}", i)),
                        "结果内容应该对应正确的调用"
                    );
                }

                Ok(())
            })?;
        }

        /// **Feature: agent-tool-calling, Property 13: 工具循环终止**
        /// **Validates: Requirements 7.3, 7.5**
        ///
        /// *For any* 工具循环执行，当 Agent 响应不包含 tool_calls 时，循环应该终止。
        #[test]
        fn prop_tool_loop_terminates_without_tool_calls(
            content in "[a-zA-Z0-9 ]{1,100}"
        ) {
            let registry = create_proptest_registry();
            let engine = ToolLoopEngine::new(registry);

            // 创建不包含工具调用的响应
            let result = StreamResult::new(content.clone());

            // 验证：should_continue 返回 false
            prop_assert!(
                !engine.should_continue(&result, 0),
                "不包含工具调用的响应应该终止循环"
            );

            // 验证：has_tool_calls 返回 false
            prop_assert!(
                !ToolLoopEngine::has_tool_calls(&result),
                "不包含工具调用的响应 has_tool_calls 应该返回 false"
            );
        }

        /// **Feature: agent-tool-calling, Property 13: 工具循环终止 - 最大迭代次数**
        /// **Validates: Requirements 7.3, 7.5**
        ///
        /// *For any* 工具循环执行，当达到最大迭代次数时，循环应该终止。
        #[test]
        fn prop_tool_loop_terminates_at_max_iterations(
            max_iterations in 1..=20usize,
            current_iteration in 0..=25usize
        ) {
            let registry = create_proptest_registry();
            let config = ToolLoopConfig::new(max_iterations);
            let engine = ToolLoopEngine::with_config(registry, config);

            // 创建包含工具调用的响应
            let result = StreamResult::new("".to_string()).with_tool_calls(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: r#"{"message": "test"}"#.to_string(),
                },
            }]);

            let should_continue = engine.should_continue(&result, current_iteration);

            if current_iteration >= max_iterations {
                // 达到或超过最大迭代次数，应该终止
                prop_assert!(
                    !should_continue,
                    "达到最大迭代次数 {} 时应该终止循环（当前迭代: {}）",
                    max_iterations,
                    current_iteration
                );
            } else {
                // 未达到最大迭代次数，应该继续
                prop_assert!(
                    should_continue,
                    "未达到最大迭代次数 {} 时应该继续循环（当前迭代: {}）",
                    max_iterations,
                    current_iteration
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 13: 工具循环终止 - 空工具调用列表**
        /// **Validates: Requirements 7.3, 7.5**
        ///
        /// *For any* 包含空工具调用列表的响应，循环应该终止。
        #[test]
        fn prop_tool_loop_terminates_with_empty_tool_calls(
            content in "[a-zA-Z0-9 ]{1,100}"
        ) {
            let registry = create_proptest_registry();
            let engine = ToolLoopEngine::new(registry);

            // 创建包含空工具调用列表的响应
            let result = StreamResult {
                content: content.clone(),
                tool_calls: Some(vec![]),
                usage: None,
            };

            // 验证：should_continue 返回 false
            prop_assert!(
                !engine.should_continue(&result, 0),
                "空工具调用列表应该终止循环"
            );

            // 验证：has_tool_calls 返回 false
            prop_assert!(
                !ToolLoopEngine::has_tool_calls(&result),
                "空工具调用列表 has_tool_calls 应该返回 false"
            );
        }

        /// **Feature: agent-tool-calling, Property 13: 工具循环终止 - 配置一致性**
        /// **Validates: Requirements 7.5**
        ///
        /// *For any* 配置的最大迭代次数，engine.max_iterations() 应该返回相同的值。
        #[test]
        fn prop_tool_loop_config_consistency(
            max_iterations in 1..=100usize
        ) {
            let registry = create_proptest_registry();
            let config = ToolLoopConfig::new(max_iterations);
            let engine = ToolLoopEngine::with_config(registry, config);

            prop_assert_eq!(
                engine.max_iterations(),
                max_iterations,
                "max_iterations() 应该返回配置的值"
            );
        }

        /// **Feature: agent-tool-calling, Property 13: 工具循环终止 - 边界条件**
        /// **Validates: Requirements 7.3, 7.5**
        ///
        /// *For any* 最大迭代次数，在边界处的行为应该正确。
        #[test]
        fn prop_tool_loop_boundary_conditions(
            max_iterations in 1..=20usize
        ) {
            let registry = create_proptest_registry();
            let config = ToolLoopConfig::new(max_iterations);
            let engine = ToolLoopEngine::with_config(registry, config);

            // 创建包含工具调用的响应
            let result_with_tools = StreamResult::new("".to_string()).with_tool_calls(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: r#"{"message": "test"}"#.to_string(),
                },
            }]);

            // 在 max_iterations - 1 处应该继续
            if max_iterations > 0 {
                prop_assert!(
                    engine.should_continue(&result_with_tools, max_iterations - 1),
                    "在 max_iterations - 1 处应该继续"
                );
            }

            // 在 max_iterations 处应该终止
            prop_assert!(
                !engine.should_continue(&result_with_tools, max_iterations),
                "在 max_iterations 处应该终止"
            );

            // 在 max_iterations + 1 处应该终止
            prop_assert!(
                !engine.should_continue(&result_with_tools, max_iterations + 1),
                "在 max_iterations + 1 处应该终止"
            );
        }
    }
}
