//! 工具 Prompt 生成器模块
//!
//! 提供工具定义到 System Prompt 的转换功能
//! 符合 Requirements 2.3 - THE System_Prompt SHALL include all available tool definitions
//!
//! ## 功能
//! - 工具定义到 XML 格式转换
//! - 工具定义到 JSON 格式转换
//! - System Prompt 模板生成

use super::types::{JsonSchema, ToolDefinition};

/// Prompt 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptFormat {
    /// XML 格式（Claude 风格）
    #[default]
    Xml,
    /// JSON 格式（OpenAI 风格）
    Json,
}

/// 工具 Prompt 生成器
///
/// 将工具定义转换为 LLM 可理解的 System Prompt 格式
/// Requirements: 2.3 - THE System_Prompt SHALL include all available tool definitions
pub struct ToolPromptGenerator {
    /// 输出格式
    format: PromptFormat,
}

impl Default for ToolPromptGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolPromptGenerator {
    /// 创建新的 Prompt 生成器
    pub fn new() -> Self {
        Self {
            format: PromptFormat::Xml,
        }
    }

    /// 设置输出格式
    pub fn with_format(mut self, format: PromptFormat) -> Self {
        self.format = format;
        self
    }

    /// 生成包含工具定义的 System Prompt
    ///
    /// Requirements: 2.3 - THE System_Prompt SHALL include all available tool definitions
    /// in a format the LLM can understand
    pub fn generate_system_prompt(&self, tools: &[ToolDefinition]) -> String {
        match self.format {
            PromptFormat::Xml => self.generate_xml_prompt(tools),
            PromptFormat::Json => self.generate_json_prompt(tools),
        }
    }

    /// 生成 XML 格式的 System Prompt（Claude 风格）
    fn generate_xml_prompt(&self, tools: &[ToolDefinition]) -> String {
        let mut prompt = String::new();

        // 添加工具使用说明
        prompt.push_str(TOOL_USAGE_INSTRUCTIONS);
        prompt.push_str("\n\n");

        // 添加工具定义
        prompt.push_str("<tools>\n");
        for tool in tools {
            prompt.push_str(&self.tool_to_xml(tool));
            prompt.push('\n');
        }
        prompt.push_str("</tools>\n");

        prompt
    }

    /// 生成 JSON 格式的 System Prompt（OpenAI 风格）
    fn generate_json_prompt(&self, tools: &[ToolDefinition]) -> String {
        let mut prompt = String::new();

        // 添加工具使用说明
        prompt.push_str(TOOL_USAGE_INSTRUCTIONS);
        prompt.push_str("\n\n");

        // 添加工具定义
        prompt.push_str("Available tools:\n```json\n");
        let tools_json = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());
        prompt.push_str(&tools_json);
        prompt.push_str("\n```\n");

        prompt
    }

    /// 将单个工具定义转换为 XML 格式
    pub fn tool_to_xml(&self, tool: &ToolDefinition) -> String {
        let mut xml = String::new();

        xml.push_str(&format!("<tool name=\"{}\">\n", escape_xml(&tool.name)));
        xml.push_str(&format!(
            "  <description>{}</description>\n",
            escape_xml(&tool.description)
        ));
        xml.push_str("  <parameters>\n");
        xml.push_str(&self.json_schema_to_xml(&tool.parameters, 4));
        xml.push_str("  </parameters>\n");
        xml.push_str("</tool>");

        xml
    }

    /// 将 JsonSchema 转换为 XML 格式
    fn json_schema_to_xml(&self, schema: &JsonSchema, indent: usize) -> String {
        let mut xml = String::new();
        let indent_str = " ".repeat(indent);

        for (name, prop) in &schema.properties {
            let required = if schema.required.contains(name) {
                " required=\"true\""
            } else {
                ""
            };

            xml.push_str(&format!(
                "{}<parameter name=\"{}\" type=\"{}\"{}>\n",
                indent_str,
                escape_xml(name),
                escape_xml(&prop.prop_type),
                required
            ));
            xml.push_str(&format!(
                "{}  <description>{}</description>\n",
                indent_str,
                escape_xml(&prop.description)
            ));

            // 添加默认值（如果有）
            if let Some(default) = &prop.default {
                xml.push_str(&format!(
                    "{}  <default>{}</default>\n",
                    indent_str,
                    escape_xml(&default.to_string())
                ));
            }

            // 添加枚举值（如果有）
            if let Some(enum_values) = &prop.enum_values {
                xml.push_str(&format!("{}  <enum>\n", indent_str));
                for value in enum_values {
                    xml.push_str(&format!(
                        "{}    <value>{}</value>\n",
                        indent_str,
                        escape_xml(&value.to_string())
                    ));
                }
                xml.push_str(&format!("{}  </enum>\n", indent_str));
            }

            xml.push_str(&format!("{}</parameter>\n", indent_str));
        }

        xml
    }

    /// 将单个工具定义转换为 JSON 格式
    pub fn tool_to_json(&self, tool: &ToolDefinition) -> String {
        serde_json::to_string_pretty(tool).unwrap_or_else(|_| "{}".to_string())
    }

    /// 获取当前格式
    pub fn format(&self) -> PromptFormat {
        self.format
    }
}

/// 工具使用说明模板
const TOOL_USAGE_INSTRUCTIONS: &str = r#"You have access to a set of tools that you can use to help accomplish tasks. When you need to use a tool, respond with a tool call in the following format:

<tool_call>
<name>tool_name</name>
<arguments>
{
  "param1": "value1",
  "param2": "value2"
}
</arguments>
</tool_call>

Important guidelines for tool usage:
1. Only use tools when necessary to accomplish the task
2. Provide all required parameters for each tool call
3. Wait for tool results before making additional tool calls that depend on them
4. If a tool call fails, analyze the error and try an alternative approach
5. Always explain your reasoning before and after using tools"#;

/// XML 特殊字符转义
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// 从 ToolRegistry 生成 System Prompt 的便捷函数
pub fn generate_tools_prompt(tools: &[ToolDefinition], format: PromptFormat) -> String {
    ToolPromptGenerator::new()
        .with_format(format)
        .generate_system_prompt(tools)
}

#[cfg(test)]
mod tests {
    use super::super::types::PropertySchema;
    use super::*;

    fn create_test_tool() -> ToolDefinition {
        ToolDefinition::new("bash", "Execute a bash command in the shell").with_parameters(
            JsonSchema::new()
                .add_property(
                    "command",
                    PropertySchema::string("The bash command to execute"),
                    true,
                )
                .add_property(
                    "timeout",
                    PropertySchema::integer("Optional timeout in seconds")
                        .with_default(serde_json::json!(120)),
                    false,
                ),
        )
    }

    fn create_test_tools() -> Vec<ToolDefinition> {
        vec![
            create_test_tool(),
            ToolDefinition::new("read_file", "Read the contents of a file").with_parameters(
                JsonSchema::new()
                    .add_property(
                        "path",
                        PropertySchema::string("The file path to read"),
                        true,
                    )
                    .add_property(
                        "start_line",
                        PropertySchema::integer("Starting line number (1-based)"),
                        false,
                    )
                    .add_property(
                        "end_line",
                        PropertySchema::integer("Ending line number (inclusive)"),
                        false,
                    ),
            ),
        ]
    }

    #[test]
    fn test_generator_default_format() {
        let generator = ToolPromptGenerator::new();
        assert_eq!(generator.format(), PromptFormat::Xml);
    }

    #[test]
    fn test_generator_with_format() {
        let generator = ToolPromptGenerator::new().with_format(PromptFormat::Json);
        assert_eq!(generator.format(), PromptFormat::Json);
    }

    #[test]
    fn test_tool_to_xml() {
        let generator = ToolPromptGenerator::new();
        let tool = create_test_tool();
        let xml = generator.tool_to_xml(&tool);

        // 验证 XML 包含工具名称
        assert!(xml.contains("name=\"bash\""));
        // 验证 XML 包含描述
        assert!(xml.contains("Execute a bash command"));
        // 验证 XML 包含必需参数
        assert!(xml.contains("name=\"command\""));
        assert!(xml.contains("required=\"true\""));
        // 验证 XML 包含可选参数
        assert!(xml.contains("name=\"timeout\""));
        // 验证 XML 包含默认值
        assert!(xml.contains("<default>120</default>"));
    }

    #[test]
    fn test_tool_to_json() {
        let generator = ToolPromptGenerator::new();
        let tool = create_test_tool();
        let json = generator.tool_to_json(&tool);

        // 验证 JSON 包含工具名称
        assert!(json.contains("\"name\": \"bash\""));
        // 验证 JSON 包含描述
        assert!(json.contains("Execute a bash command"));
        // 验证 JSON 包含参数
        assert!(json.contains("\"command\""));
    }

    #[test]
    fn test_generate_xml_prompt() {
        let generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
        let tools = create_test_tools();
        let prompt = generator.generate_system_prompt(&tools);

        // 验证包含工具使用说明
        assert!(prompt.contains("You have access to a set of tools"));
        // 验证包含 tools 标签
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("</tools>"));
        // 验证包含所有工具
        assert!(prompt.contains("name=\"bash\""));
        assert!(prompt.contains("name=\"read_file\""));
    }

    #[test]
    fn test_generate_json_prompt() {
        let generator = ToolPromptGenerator::new().with_format(PromptFormat::Json);
        let tools = create_test_tools();
        let prompt = generator.generate_system_prompt(&tools);

        // 验证包含工具使用说明
        assert!(prompt.contains("You have access to a set of tools"));
        // 验证包含 JSON 代码块
        assert!(prompt.contains("```json"));
        // 验证包含所有工具
        assert!(prompt.contains("\"bash\""));
        assert!(prompt.contains("\"read_file\""));
    }

    #[test]
    fn test_generate_tools_prompt_convenience_function() {
        let tools = create_test_tools();

        let xml_prompt = generate_tools_prompt(&tools, PromptFormat::Xml);
        assert!(xml_prompt.contains("<tools>"));

        let json_prompt = generate_tools_prompt(&tools, PromptFormat::Json);
        assert!(json_prompt.contains("```json"));
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("hello"), "hello");
        assert_eq!(escape_xml("<script>"), "&lt;script&gt;");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
    }

    #[test]
    fn test_empty_tools() {
        let generator = ToolPromptGenerator::new();
        let prompt = generator.generate_system_prompt(&[]);

        // 即使没有工具，也应该包含使用说明
        assert!(prompt.contains("You have access to a set of tools"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("</tools>"));
    }

    #[test]
    fn test_tool_with_enum_values() {
        let tool = ToolDefinition::new("select", "Select an option").with_parameters(
            JsonSchema::new().add_property(
                "choice",
                PropertySchema::string("The choice to make").with_enum(vec![
                    serde_json::json!("option_a"),
                    serde_json::json!("option_b"),
                ]),
                true,
            ),
        );

        let generator = ToolPromptGenerator::new();
        let xml = generator.tool_to_xml(&tool);

        // 验证包含枚举值
        assert!(xml.contains("<enum>"));
        // JSON 序列化会包含引号，所以检查转义后的值
        assert!(
            xml.contains("option_a"),
            "XML should contain option_a: {}",
            xml
        );
        assert!(
            xml.contains("option_b"),
            "XML should contain option_b: {}",
            xml
        );
        assert!(xml.contains("</enum>"));
    }

    #[test]
    fn test_prompt_contains_all_tool_names_and_descriptions() {
        let tools = vec![
            ToolDefinition::new("tool_a", "Description for tool A"),
            ToolDefinition::new("tool_b", "Description for tool B"),
            ToolDefinition::new("tool_c", "Description for tool C"),
        ];

        let generator = ToolPromptGenerator::new();
        let prompt = generator.generate_system_prompt(&tools);

        // 验证所有工具名称都在 prompt 中
        for tool in &tools {
            assert!(
                prompt.contains(&tool.name),
                "Prompt should contain tool name: {}",
                tool.name
            );
            assert!(
                prompt.contains(&tool.description),
                "Prompt should contain tool description: {}",
                tool.description
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::super::types::PropertySchema;
    use super::*;
    use proptest::prelude::*;

    /// 生成有效的工具名称
    fn arb_valid_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,30}".prop_map(|s| s)
    }

    /// 生成有效的工具描述
    fn arb_valid_description() -> impl Strategy<Value = String> {
        // 生成不包含 XML 特殊字符的描述，避免转义问题
        "[a-zA-Z0-9 ,.!?]{1,100}".prop_map(|s| s)
    }

    /// 生成有效的属性名称
    fn arb_property_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,20}".prop_map(|s| s)
    }

    /// 生成有效的 PropertySchema
    fn arb_property_schema() -> impl Strategy<Value = PropertySchema> {
        prop_oneof![
            "[a-zA-Z0-9 ]{1,50}".prop_map(|desc| PropertySchema::string(desc)),
            "[a-zA-Z0-9 ]{1,50}".prop_map(|desc| PropertySchema::number(desc)),
            "[a-zA-Z0-9 ]{1,50}".prop_map(|desc| PropertySchema::integer(desc)),
            "[a-zA-Z0-9 ]{1,50}".prop_map(|desc| PropertySchema::boolean(desc)),
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

    /// 生成有效的工具定义列表
    fn arb_tool_definitions() -> impl Strategy<Value = Vec<ToolDefinition>> {
        prop::collection::vec(arb_valid_tool_definition(), 0..10)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 3: System Prompt 工具包含**
        /// **Validates: Requirements 2.3**
        ///
        /// *For any* 已注册的工具集合，生成的 System Prompt 应该包含所有工具的 name 和 description。
        #[test]
        fn prop_system_prompt_contains_all_tool_names(tools in arb_tool_definitions()) {
            let generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
            let prompt = generator.generate_system_prompt(&tools);

            // 验证所有工具名称都在 prompt 中
            for tool in &tools {
                prop_assert!(
                    prompt.contains(&tool.name),
                    "System Prompt 应该包含工具名称 '{}'\nPrompt:\n{}",
                    tool.name,
                    prompt
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 3: System Prompt 工具包含 - 描述**
        /// **Validates: Requirements 2.3**
        ///
        /// *For any* 已注册的工具集合，生成的 System Prompt 应该包含所有工具的 description。
        #[test]
        fn prop_system_prompt_contains_all_tool_descriptions(tools in arb_tool_definitions()) {
            let generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
            let prompt = generator.generate_system_prompt(&tools);

            // 验证所有工具描述都在 prompt 中
            for tool in &tools {
                prop_assert!(
                    prompt.contains(&tool.description),
                    "System Prompt 应该包含工具描述 '{}'\nPrompt:\n{}",
                    tool.description,
                    prompt
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 3: System Prompt 工具包含 - JSON 格式**
        /// **Validates: Requirements 2.3**
        ///
        /// *For any* 已注册的工具集合，JSON 格式的 System Prompt 也应该包含所有工具的 name 和 description。
        #[test]
        fn prop_system_prompt_json_contains_all_tools(tools in arb_tool_definitions()) {
            let generator = ToolPromptGenerator::new().with_format(PromptFormat::Json);
            let prompt = generator.generate_system_prompt(&tools);

            // 验证所有工具名称和描述都在 prompt 中
            for tool in &tools {
                prop_assert!(
                    prompt.contains(&tool.name),
                    "JSON System Prompt 应该包含工具名称 '{}'\nPrompt:\n{}",
                    tool.name,
                    prompt
                );
                prop_assert!(
                    prompt.contains(&tool.description),
                    "JSON System Prompt 应该包含工具描述 '{}'\nPrompt:\n{}",
                    tool.description,
                    prompt
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 3: System Prompt 工具包含 - 格式一致性**
        /// **Validates: Requirements 2.3**
        ///
        /// *For any* 已注册的工具集合，无论使用 XML 还是 JSON 格式，
        /// 生成的 System Prompt 都应该包含相同的工具信息。
        #[test]
        fn prop_system_prompt_format_consistency(tools in arb_tool_definitions()) {
            let xml_generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
            let json_generator = ToolPromptGenerator::new().with_format(PromptFormat::Json);

            let xml_prompt = xml_generator.generate_system_prompt(&tools);
            let json_prompt = json_generator.generate_system_prompt(&tools);

            // 两种格式都应该包含所有工具名称和描述
            for tool in &tools {
                prop_assert!(
                    xml_prompt.contains(&tool.name) && json_prompt.contains(&tool.name),
                    "两种格式都应该包含工具名称 '{}'",
                    tool.name
                );
                prop_assert!(
                    xml_prompt.contains(&tool.description) && json_prompt.contains(&tool.description),
                    "两种格式都应该包含工具描述 '{}'",
                    tool.description
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 3: System Prompt 工具包含 - 工具数量**
        /// **Validates: Requirements 2.3**
        ///
        /// *For any* 已注册的工具集合，生成的 System Prompt 中工具名称出现的次数
        /// 应该至少等于工具数量（每个工具至少出现一次）。
        #[test]
        fn prop_system_prompt_tool_count(tools in arb_tool_definitions()) {
            let generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
            let prompt = generator.generate_system_prompt(&tools);

            // 统计每个工具名称在 prompt 中出现的次数
            for tool in &tools {
                let count = prompt.matches(&tool.name).count();
                prop_assert!(
                    count >= 1,
                    "工具 '{}' 应该在 System Prompt 中至少出现一次，实际出现 {} 次",
                    tool.name,
                    count
                );
            }
        }
    }
}
