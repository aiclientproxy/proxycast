//! 工具类型定义
//!
//! 定义工具系统的核心类型，包括工具定义、调用、结果和错误
//! 符合 Requirements 2.1, 2.5

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// 工具定义结构
///
/// 包含工具的名称、描述和参数 JSON Schema
/// Requirements: 2.1 - THE Tool_Definition SHALL include name, description, and JSON Schema parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// 工具名称（唯一标识）
    pub name: String,
    /// 工具描述（供 LLM 理解）
    pub description: String,
    /// 参数 JSON Schema
    pub parameters: JsonSchema,
}

impl ToolDefinition {
    /// 创建新的工具定义
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: JsonSchema::default(),
        }
    }

    /// 设置参数 schema
    pub fn with_parameters(mut self, parameters: JsonSchema) -> Self {
        self.parameters = parameters;
        self
    }

    /// 验证工具定义是否有效
    pub fn validate(&self) -> Result<(), ToolValidationError> {
        if self.name.is_empty() {
            return Err(ToolValidationError::EmptyName);
        }
        if self.description.is_empty() {
            return Err(ToolValidationError::EmptyDescription);
        }
        self.parameters.validate()?;
        Ok(())
    }
}

/// JSON Schema 参数定义
///
/// 定义工具参数的类型和结构
/// Requirements: 2.5 - THE Tool_Definition SHALL support required and optional parameters with type validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema 类型（通常为 "object"）
    #[serde(rename = "type")]
    pub schema_type: String,
    /// 属性定义
    #[serde(default)]
    pub properties: HashMap<String, PropertySchema>,
    /// 必需参数列表
    #[serde(default)]
    pub required: Vec<String>,
}

impl Default for JsonSchema {
    fn default() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }
}

impl JsonSchema {
    /// 创建新的 JSON Schema
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加属性
    pub fn add_property(
        mut self,
        name: impl Into<String>,
        prop: PropertySchema,
        required: bool,
    ) -> Self {
        let name = name.into();
        if required {
            self.required.push(name.clone());
        }
        self.properties.insert(name, prop);
        self
    }

    /// 验证 schema 是否有效
    pub fn validate(&self) -> Result<(), ToolValidationError> {
        // 检查 required 中的字段是否都在 properties 中定义
        for req in &self.required {
            if !self.properties.contains_key(req) {
                return Err(ToolValidationError::RequiredPropertyNotDefined(req.clone()));
            }
        }
        Ok(())
    }
}

/// 属性 Schema
///
/// 定义单个参数的类型、描述和默认值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// 属性类型（string, number, boolean, array, object）
    #[serde(rename = "type")]
    pub prop_type: String,
    /// 属性描述
    pub description: String,
    /// 默认值（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// 枚举值（可选，用于限制取值范围）
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<serde_json::Value>>,
}

impl PropertySchema {
    /// 创建字符串类型属性
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            prop_type: "string".to_string(),
            description: description.into(),
            default: None,
            enum_values: None,
        }
    }

    /// 创建数字类型属性
    pub fn number(description: impl Into<String>) -> Self {
        Self {
            prop_type: "number".to_string(),
            description: description.into(),
            default: None,
            enum_values: None,
        }
    }

    /// 创建整数类型属性
    pub fn integer(description: impl Into<String>) -> Self {
        Self {
            prop_type: "integer".to_string(),
            description: description.into(),
            default: None,
            enum_values: None,
        }
    }

    /// 创建布尔类型属性
    pub fn boolean(description: impl Into<String>) -> Self {
        Self {
            prop_type: "boolean".to_string(),
            description: description.into(),
            default: None,
            enum_values: None,
        }
    }

    /// 创建数组类型属性
    pub fn array(description: impl Into<String>) -> Self {
        Self {
            prop_type: "array".to_string(),
            description: description.into(),
            default: None,
            enum_values: None,
        }
    }

    /// 设置默认值
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// 设置枚举值
    pub fn with_enum(mut self, values: Vec<serde_json::Value>) -> Self {
        self.enum_values = Some(values);
        self
    }
}

/// 工具调用请求
///
/// 表示 LLM 发起的工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// 工具调用 ID（用于关联结果）
    pub id: String,
    /// 工具名称
    pub name: String,
    /// 调用参数（JSON 对象）
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// 创建新的工具调用
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// 从 JSON 字符串参数创建
    pub fn from_json_str(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments_str: &str,
    ) -> Result<Self, serde_json::Error> {
        let arguments: serde_json::Value = serde_json::from_str(arguments_str)?;
        Ok(Self {
            id: id.into(),
            name: name.into(),
            arguments,
        })
    }
}

/// 工具执行结果
///
/// 表示工具执行后的返回值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// 是否成功
    pub success: bool,
    /// 输出内容
    pub output: String,
    /// 错误信息（如果失败）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    /// 创建成功结果
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    /// 创建失败结果
    pub fn failure(error: impl Into<String>) -> Self {
        let error_msg = error.into();
        Self {
            success: false,
            output: String::new(),
            error: Some(error_msg),
        }
    }

    /// 创建带输出的失败结果
    pub fn failure_with_output(output: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: output.into(),
            error: Some(error.into()),
        }
    }
}

/// 工具错误类型
#[derive(Debug, Error)]
pub enum ToolError {
    /// 工具不存在
    #[error("工具不存在: {0}")]
    NotFound(String),

    /// 参数验证失败
    #[error("参数验证失败: {0}")]
    InvalidArguments(String),

    /// 执行失败
    #[error("执行失败: {0}")]
    ExecutionFailed(String),

    /// 安全错误
    #[error("安全错误: {0}")]
    Security(String),

    /// 超时
    #[error("执行超时")]
    Timeout,

    /// IO 错误
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    /// JSON 解析错误
    #[error("JSON 解析错误: {0}")]
    Json(#[from] serde_json::Error),
}

/// 工具定义验证错误
#[derive(Debug, Error)]
pub enum ToolValidationError {
    /// 工具名称为空
    #[error("工具名称不能为空")]
    EmptyName,

    /// 工具描述为空
    #[error("工具描述不能为空")]
    EmptyDescription,

    /// 必需属性未定义
    #[error("必需属性 '{0}' 未在 properties 中定义")]
    RequiredPropertyNotDefined(String),

    /// 重复的工具名称
    #[error("工具名称 '{0}' 已存在")]
    DuplicateName(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_validation() {
        // 有效的工具定义
        let valid = ToolDefinition::new("bash", "Execute bash commands");
        assert!(valid.validate().is_ok());

        // 空名称
        let empty_name = ToolDefinition::new("", "Some description");
        assert!(matches!(
            empty_name.validate(),
            Err(ToolValidationError::EmptyName)
        ));

        // 空描述
        let empty_desc = ToolDefinition::new("bash", "");
        assert!(matches!(
            empty_desc.validate(),
            Err(ToolValidationError::EmptyDescription)
        ));
    }

    #[test]
    fn test_json_schema_validation() {
        // 有效的 schema
        let valid = JsonSchema::new().add_property(
            "command",
            PropertySchema::string("The command to run"),
            true,
        );
        assert!(valid.validate().is_ok());

        // required 中有未定义的属性
        let invalid = JsonSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec!["undefined_prop".to_string()],
        };
        assert!(matches!(
            invalid.validate(),
            Err(ToolValidationError::RequiredPropertyNotDefined(_))
        ));
    }

    #[test]
    fn test_tool_result_creation() {
        let success = ToolResult::success("Hello, World!");
        assert!(success.success);
        assert_eq!(success.output, "Hello, World!");
        assert!(success.error.is_none());

        let failure = ToolResult::failure("Something went wrong");
        assert!(!failure.success);
        assert!(failure.output.is_empty());
        assert_eq!(failure.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_tool_call_from_json() {
        let call = ToolCall::from_json_str("call_1", "bash", r#"{"command": "ls -la"}"#).unwrap();
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "bash");
        assert_eq!(call.arguments["command"], "ls -la");
    }

    #[test]
    fn test_property_schema_builders() {
        let string_prop = PropertySchema::string("A string value");
        assert_eq!(string_prop.prop_type, "string");

        let number_prop =
            PropertySchema::number("A number value").with_default(serde_json::json!(0));
        assert_eq!(number_prop.prop_type, "number");
        assert_eq!(number_prop.default, Some(serde_json::json!(0)));

        let enum_prop = PropertySchema::string("A choice")
            .with_enum(vec![serde_json::json!("a"), serde_json::json!("b")]);
        assert!(enum_prop.enum_values.is_some());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// 生成有效的工具名称
    fn arb_valid_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,30}".prop_map(|s| s)
    }

    /// 生成有效的工具描述
    fn arb_valid_description() -> impl Strategy<Value = String> {
        ".{1,200}".prop_map(|s| s)
    }

    /// 生成有效的属性名称
    fn arb_property_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,20}".prop_map(|s| s)
    }

    /// 生成有效的 PropertySchema
    fn arb_property_schema() -> impl Strategy<Value = PropertySchema> {
        prop_oneof![
            ".{1,50}".prop_map(|desc| PropertySchema::string(desc)),
            ".{1,50}".prop_map(|desc| PropertySchema::number(desc)),
            ".{1,50}".prop_map(|desc| PropertySchema::integer(desc)),
            ".{1,50}".prop_map(|desc| PropertySchema::boolean(desc)),
        ]
    }

    /// 生成有效的 JsonSchema
    fn arb_valid_json_schema() -> impl Strategy<Value = JsonSchema> {
        prop::collection::vec(
            (arb_property_name(), arb_property_schema(), any::<bool>()),
            0..5,
        )
        .prop_map(|props| {
            let mut schema = JsonSchema::new();
            for (name, prop, required) in props {
                schema = schema.add_property(name, prop, required);
            }
            schema
        })
    }

    /// 生成有效的 ToolDefinition
    fn arb_valid_tool_definition() -> impl Strategy<Value = ToolDefinition> {
        (
            arb_valid_name(),
            arb_valid_description(),
            arb_valid_json_schema(),
        )
            .prop_map(|(name, description, parameters)| ToolDefinition {
                name,
                description,
                parameters,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 2: 工具定义验证**
        /// **Validates: Requirements 2.1, 2.2**
        ///
        /// *For any* 工具定义，如果缺少 name、description 或 parameters 中的任何一个字段，
        /// 注册时应该返回验证错误。
        #[test]
        fn prop_valid_tool_definition_passes_validation(def in arb_valid_tool_definition()) {
            // 有效的工具定义应该通过验证
            prop_assert!(def.validate().is_ok(), "有效的工具定义应该通过验证: {:?}", def);
        }

        /// **Feature: agent-tool-calling, Property 2: 工具定义验证 - 空名称**
        /// **Validates: Requirements 2.1, 2.2**
        #[test]
        fn prop_empty_name_fails_validation(description in arb_valid_description()) {
            let def = ToolDefinition::new("", description);
            prop_assert!(
                matches!(def.validate(), Err(ToolValidationError::EmptyName)),
                "空名称应该返回 EmptyName 错误"
            );
        }

        /// **Feature: agent-tool-calling, Property 2: 工具定义验证 - 空描述**
        /// **Validates: Requirements 2.1, 2.2**
        #[test]
        fn prop_empty_description_fails_validation(name in arb_valid_name()) {
            let def = ToolDefinition::new(name, "");
            prop_assert!(
                matches!(def.validate(), Err(ToolValidationError::EmptyDescription)),
                "空描述应该返回 EmptyDescription 错误"
            );
        }

        /// **Feature: agent-tool-calling, Property 2: 工具定义验证 - required 属性未定义**
        /// **Validates: Requirements 2.1, 2.2**
        #[test]
        fn prop_undefined_required_property_fails_validation(
            name in arb_valid_name(),
            description in arb_valid_description(),
            undefined_prop in arb_property_name()
        ) {
            let schema = JsonSchema {
                schema_type: "object".to_string(),
                properties: HashMap::new(),
                required: vec![undefined_prop.clone()],
            };
            let def = ToolDefinition {
                name,
                description,
                parameters: schema,
            };
            prop_assert!(
                matches!(def.validate(), Err(ToolValidationError::RequiredPropertyNotDefined(_))),
                "required 中未定义的属性应该返回 RequiredPropertyNotDefined 错误"
            );
        }

        /// **Feature: agent-tool-calling, Property 2: 工具定义验证 - required 属性已定义**
        /// **Validates: Requirements 2.1, 2.2, 2.5**
        #[test]
        fn prop_defined_required_property_passes_validation(
            name in arb_valid_name(),
            description in arb_valid_description(),
            prop_name in arb_property_name(),
            prop_schema in arb_property_schema()
        ) {
            let schema = JsonSchema::new().add_property(prop_name, prop_schema, true);
            let def = ToolDefinition {
                name,
                description,
                parameters: schema,
            };
            prop_assert!(def.validate().is_ok(), "已定义的 required 属性应该通过验证");
        }
    }
}
