//! 工具注册表和 Tool trait
//!
//! 提供工具的注册、查找和验证功能
//! 符合 Requirements 2.2, 2.4

use super::types::{ToolDefinition, ToolError, ToolResult, ToolValidationError};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// 工具 trait
///
/// 所有工具必须实现此 trait
/// Requirements: 2.2 - WHEN tools are registered, THE Tool_Registry SHALL validate the tool definitions
#[async_trait]
pub trait Tool: Send + Sync {
    /// 获取工具定义
    fn definition(&self) -> ToolDefinition;

    /// 执行工具
    ///
    /// # Arguments
    /// * `args` - 工具调用参数（JSON 对象）
    ///
    /// # Returns
    /// * `Ok(ToolResult)` - 执行结果
    /// * `Err(ToolError)` - 执行错误
    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError>;

    /// 获取工具名称（便捷方法）
    fn name(&self) -> String {
        self.definition().name
    }

    /// 验证参数
    ///
    /// 默认实现检查必需参数是否存在
    fn validate_args(&self, args: &serde_json::Value) -> Result<(), ToolError> {
        let def = self.definition();
        let obj = args
            .as_object()
            .ok_or_else(|| ToolError::InvalidArguments("参数必须是 JSON 对象".to_string()))?;

        // 检查必需参数
        for required in &def.parameters.required {
            if !obj.contains_key(required) {
                return Err(ToolError::InvalidArguments(format!(
                    "缺少必需参数: {}",
                    required
                )));
            }
        }

        Ok(())
    }
}

/// 工具注册表
///
/// 管理所有已注册的工具
/// Requirements: 2.4 - WHEN a new tool is added, THE Tool_Registry SHALL make it available to the Agent without restart
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// 创建新的工具注册表
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// 注册工具
    ///
    /// Requirements: 2.2 - WHEN tools are registered, THE Tool_Registry SHALL validate the tool definitions
    pub fn register<T: Tool + 'static>(&self, tool: T) -> Result<(), ToolValidationError> {
        let definition = tool.definition();

        // 验证工具定义
        definition.validate()?;

        let name = definition.name.clone();

        // 检查是否已存在
        {
            let tools = self.tools.read();
            if tools.contains_key(&name) {
                return Err(ToolValidationError::DuplicateName(name));
            }
        }

        // 注册工具
        {
            let mut tools = self.tools.write();
            tools.insert(name.clone(), Arc::new(tool));
        }

        info!("[ToolRegistry] 注册工具: {}", name);
        Ok(())
    }

    /// 注册工具（Arc 版本）
    pub fn register_arc(&self, tool: Arc<dyn Tool>) -> Result<(), ToolValidationError> {
        let definition = tool.definition();

        // 验证工具定义
        definition.validate()?;

        let name = definition.name.clone();

        // 检查是否已存在
        {
            let tools = self.tools.read();
            if tools.contains_key(&name) {
                return Err(ToolValidationError::DuplicateName(name));
            }
        }

        // 注册工具
        {
            let mut tools = self.tools.write();
            tools.insert(name.clone(), tool);
        }

        info!("[ToolRegistry] 注册工具: {}", name);
        Ok(())
    }

    /// 注销工具
    pub fn unregister(&self, name: &str) -> bool {
        let mut tools = self.tools.write();
        let removed = tools.remove(name).is_some();
        if removed {
            info!("[ToolRegistry] 注销工具: {}", name);
        }
        removed
    }

    /// 获取工具
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().get(name).cloned()
    }

    /// 检查工具是否存在
    pub fn contains(&self, name: &str) -> bool {
        self.tools.read().contains_key(name)
    }

    /// 获取所有工具定义
    pub fn list_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.read().values().map(|t| t.definition()).collect()
    }

    /// 获取所有工具名称
    pub fn list_names(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }

    /// 获取工具数量
    pub fn len(&self) -> usize {
        self.tools.read().len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.tools.read().is_empty()
    }

    /// 执行工具
    ///
    /// 查找并执行指定的工具
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let tool = self
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        debug!("[ToolRegistry] 执行工具: {} args={:?}", name, args);

        // 验证参数
        tool.validate_args(&args)?;

        // 执行工具
        let result = tool.execute(args).await?;

        debug!(
            "[ToolRegistry] 工具执行完成: {} success={}",
            name, result.success
        );

        Ok(result)
    }

    /// 验证工具定义
    ///
    /// 用于在注册前验证工具定义
    pub fn validate_definition(
        &self,
        definition: &ToolDefinition,
    ) -> Result<(), ToolValidationError> {
        definition.validate()?;

        // 检查名称是否已存在
        if self.contains(&definition.name) {
            return Err(ToolValidationError::DuplicateName(definition.name.clone()));
        }

        Ok(())
    }

    /// 清空所有工具
    pub fn clear(&self) {
        let mut tools = self.tools.write();
        let count = tools.len();
        tools.clear();
        warn!("[ToolRegistry] 清空所有工具: {} 个", count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用的简单工具
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo the input message").with_parameters(
                super::super::types::JsonSchema::new().add_property(
                    "message",
                    super::super::types::PropertySchema::string("The message to echo"),
                    true,
                ),
            )
        }

        async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("缺少 message 参数".to_string()))?;

            Ok(ToolResult::success(message))
        }
    }

    /// 无效工具（空名称）
    struct InvalidTool;

    #[async_trait]
    impl Tool for InvalidTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("", "Invalid tool with empty name")
        }

        async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::success(""))
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ToolRegistry::new();

        // 注册工具
        assert!(registry.register(EchoTool).is_ok());
        assert_eq!(registry.len(), 1);

        // 获取工具
        let tool = registry.get("echo");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "echo");

        // 检查存在性
        assert!(registry.contains("echo"));
        assert!(!registry.contains("nonexistent"));
    }

    #[test]
    fn test_registry_reject_invalid_tool() {
        let registry = ToolRegistry::new();

        // 注册无效工具应该失败
        let result = registry.register(InvalidTool);
        assert!(result.is_err());
        assert!(matches!(result, Err(ToolValidationError::EmptyName)));
    }

    #[test]
    fn test_registry_reject_duplicate() {
        let registry = ToolRegistry::new();

        // 第一次注册成功
        assert!(registry.register(EchoTool).is_ok());

        // 第二次注册应该失败
        let result = registry.register(EchoTool);
        assert!(result.is_err());
        assert!(matches!(result, Err(ToolValidationError::DuplicateName(_))));
    }

    #[test]
    fn test_registry_unregister() {
        let registry = ToolRegistry::new();

        registry.register(EchoTool).unwrap();
        assert_eq!(registry.len(), 1);

        // 注销工具
        assert!(registry.unregister("echo"));
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("echo"));

        // 再次注销应该返回 false
        assert!(!registry.unregister("echo"));
    }

    #[test]
    fn test_registry_list_definitions() {
        let registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();

        let definitions = registry.list_definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "echo");
    }

    #[tokio::test]
    async fn test_registry_execute() {
        let registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();

        // 执行工具
        let result = registry
            .execute("echo", serde_json::json!({"message": "Hello, World!"}))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "Hello, World!");
    }

    #[tokio::test]
    async fn test_registry_execute_not_found() {
        let registry = ToolRegistry::new();

        let result = registry.execute("nonexistent", serde_json::json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_registry_execute_missing_required_arg() {
        let registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();

        // 缺少必需参数
        let result = registry.execute("echo", serde_json::json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[test]
    fn test_registry_clear() {
        let registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }
}
