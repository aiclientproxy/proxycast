//! 语音文本润色
//!
//! 提供与桌面宿主无关的文本模板处理和本地 LLM 调用能力。

/// 语音润色的 System Prompt
pub const VOICE_POLISH_SYSTEM_PROMPT: &str = r#"你是一个语音转文字的后处理助手。

## 背景说明
用户通过麦克风录音，然后使用语音识别 API（如讯飞、百度等）将语音转换为文字。由于录音环境、口音、语速等因素，识别结果可能存在以下问题：

1. **重复字词**：语音识别的流式返回机制可能导致字词重复，如"你你好好"实际是"你好"
2. **语气词和填充词**：如"嗯"、"啊"、"那个"、"就是"、"然后"等口语习惯
3. **同音字错误**：识别引擎可能选错同音字，如"准备"识别成"准被"
4. **环境噪音干扰**：背景噪音可能被误识别为无意义的字词
5. **断句错误**：缺少标点或标点位置不当
6. **多余内容**：录音开始或结束时的杂音可能被识别为无关文字

## 你的任务
根据上下文语义，智能还原用户真正想表达的内容：
- 去除明显的重复字词
- 去除无意义的语气词和填充词
- 根据语义修正可能的同音字错误
- 过滤掉噪音产生的无意义内容
- 添加合适的标点符号
- 保持用户的原意，不要添加或臆测内容

## 输出要求
只输出处理后的文本，不要添加任何解释、说明或前缀。"#;

/// 应用 Prompt 模板（将 `{{text}}` 替换为实际文本）
pub fn apply_prompt_template(text: &str, template: &str) -> String {
    template.replace("{{text}}", text)
}

/// 调用本地 API 服务器进行 LLM 润色
pub async fn polish_with_local_api(
    base_url: &str,
    api_key: &str,
    prompt: &str,
    provider: Option<&str>,
    model: Option<&str>,
    instruction_id: &str,
) -> Result<String, String> {
    // 使用配置的模型，如果没有配置则使用 deepseek-chat
    let model_name = model.filter(|m| !m.is_empty()).unwrap_or("deepseek-chat");

    tracing::info!(
        "[语音润色] 使用模型: {}, provider: {:?}, 指令: {}",
        model_name,
        provider,
        instruction_id
    );

    #[derive(serde::Serialize)]
    struct Message {
        role: String,
        content: String,
    }

    #[derive(serde::Serialize)]
    struct ChatRequest {
        model: String,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
    }

    let messages = if instruction_id == "default" {
        vec![
            Message {
                role: "system".to_string(),
                content: VOICE_POLISH_SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ]
    } else {
        vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }]
    };

    let request = ChatRequest {
        model: model_name.to_string(),
        messages,
        max_tokens: 2048,
        temperature: 0.3,
    };

    let endpoint = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut request_builder = reqwest::Client::new()
        .post(endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&request);

    if let Some(provider_id) = provider.filter(|value| !value.trim().is_empty()) {
        request_builder = request_builder.header("X-Provider-Id", provider_id);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|e| format!("请求失败: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("LLM API 错误: {status} - {body}"));
    }

    #[derive(serde::Deserialize)]
    struct Choice {
        message: ResponseMessage,
    }

    #[derive(serde::Deserialize)]
    struct ResponseMessage {
        content: Option<String>,
    }

    #[derive(serde::Deserialize)]
    struct ChatResponse {
        choices: Vec<Choice>,
    }

    let result: ChatResponse = response
        .json()
        .await
        .map_err(|e| format!("解析响应失败: {e}"))?;

    result
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .ok_or_else(|| "LLM 返回空内容".to_string())
}
