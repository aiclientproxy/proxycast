//! Kiro/CodeWhisperer Provider
use crate::converter::openai_to_cw::convert_openai_to_codewhisperer;
use crate::models::openai::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KiroCredentials {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub profile_arn: Option<String>,
    pub expires_at: Option<String>,
    pub region: Option<String>,
    pub auth_method: Option<String>,
}

impl Default for KiroCredentials {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            client_id: None,
            client_secret: None,
            profile_arn: None,
            expires_at: None,
            region: Some("us-east-1".to_string()),
            auth_method: Some("social".to_string()),
        }
    }
}

pub struct KiroProvider {
    pub credentials: KiroCredentials,
    pub client: Client,
}

impl Default for KiroProvider {
    fn default() -> Self {
        Self {
            credentials: KiroCredentials::default(),
            client: Client::new(),
        }
    }
}

impl KiroProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_creds_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".aws")
            .join("sso")
            .join("cache")
            .join("kiro-auth-token.json")
    }

    pub fn load_credentials(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();
        let dir = path.parent().unwrap();

        let mut merged = KiroCredentials::default();

        // 读取主凭证文件
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let creds: KiroCredentials = serde_json::from_str(&content)?;
            merge_credentials(&mut merged, &creds);
        }

        // 读取目录中其他 JSON 文件
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().map(|e| e == "json").unwrap_or(false) && file_path != path
                {
                    if let Ok(content) = std::fs::read_to_string(&file_path) {
                        if let Ok(creds) = serde_json::from_str::<KiroCredentials>(&content) {
                            merge_credentials(&mut merged, &creds);
                        }
                    }
                }
            }
        }

        self.credentials = merged;
        Ok(())
    }

    pub fn get_base_url(&self) -> String {
        let region = self.credentials.region.as_deref().unwrap_or("us-east-1");
        format!("https://codewhisperer.{region}.amazonaws.com/generateAssistantResponse")
    }

    pub fn get_refresh_url(&self) -> String {
        let region = self.credentials.region.as_deref().unwrap_or("us-east-1");
        let auth_method = self
            .credentials
            .auth_method
            .as_deref()
            .unwrap_or("social")
            .to_lowercase();

        if auth_method == "idc" {
            format!("https://oidc.{region}.amazonaws.com/token")
        } else {
            format!("https://prod.{region}.auth.desktop.kiro.dev/refreshToken")
        }
    }

    pub async fn refresh_token(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or("No refresh token")?;

        let auth_method = self
            .credentials
            .auth_method
            .as_deref()
            .unwrap_or("social")
            .to_lowercase();
        let refresh_url = self.get_refresh_url();

        let body = if auth_method == "idc" {
            serde_json::json!({
                "refreshToken": refresh_token,
                "clientId": self.credentials.client_id,
                "clientSecret": self.credentials.client_secret,
                "grantType": "refresh_token"
            })
        } else {
            serde_json::json!({ "refreshToken": refresh_token })
        };

        let resp = self
            .client
            .post(&refresh_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(format!("Refresh failed: {status} {body_text}").into());
        }

        let data: serde_json::Value = resp.json().await?;
        let new_token = data["accessToken"]
            .as_str()
            .ok_or("No access token in response")?;

        self.credentials.access_token = Some(new_token.to_string());

        if let Some(rt) = data["refreshToken"].as_str() {
            self.credentials.refresh_token = Some(rt.to_string());
        }
        if let Some(arn) = data["profileArn"].as_str() {
            self.credentials.profile_arn = Some(arn.to_string());
        }

        // 保存更新后的凭证到文件
        self.save_credentials()?;

        Ok(new_token.to_string())
    }

    pub fn save_credentials(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();

        // 读取现有文件内容
        let mut existing: serde_json::Value = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
        } else {
            serde_json::json!({})
        };

        // 更新字段
        if let Some(token) = &self.credentials.access_token {
            existing["accessToken"] = serde_json::json!(token);
        }
        if let Some(token) = &self.credentials.refresh_token {
            existing["refreshToken"] = serde_json::json!(token);
        }
        if let Some(arn) = &self.credentials.profile_arn {
            existing["profileArn"] = serde_json::json!(arn);
        }

        // 写回文件
        let content = serde_json::to_string_pretty(&existing)?;
        std::fs::write(&path, content)?;

        Ok(())
    }

    pub async fn call_api(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or("No access token")?;

        let profile_arn = if self.credentials.auth_method.as_deref() == Some("social") {
            self.credentials.profile_arn.clone()
        } else {
            None
        };

        let cw_request = convert_openai_to_codewhisperer(request, profile_arn);
        let url = self.get_base_url();

        // Debug: 记录转换后的请求
        if let Ok(json_str) = serde_json::to_string_pretty(&cw_request) {
            // 保存到文件用于调试
            let debug_path = dirs::home_dir()
                .unwrap_or_default()
                .join(".proxycast")
                .join("logs")
                .join(format!("cw_request_{}.json", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("unknown")));
            let _ = std::fs::write(&debug_path, &json_str);
            tracing::debug!("[CW_REQ] Request saved to {:?}", debug_path);
            
            // 记录历史消息数量和 tool_results 情况
            let history_len = cw_request.conversation_state.history.as_ref().map(|h| h.len()).unwrap_or(0);
            let current_has_tools = cw_request.conversation_state.current_message.user_input_message
                .user_input_message_context.as_ref()
                .map(|ctx| ctx.tool_results.as_ref().map(|tr| tr.len()).unwrap_or(0))
                .unwrap_or(0);
            tracing::info!("[CW_REQ] history={} current_tool_results={}", history_len, current_has_tools);
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("amz-sdk-invocation-id", uuid::Uuid::new_v4().to_string())
            .json(&cw_request)
            .send()
            .await?;

        Ok(resp)
    }
}

fn merge_credentials(target: &mut KiroCredentials, source: &KiroCredentials) {
    if source.access_token.is_some() {
        target.access_token = source.access_token.clone();
    }
    if source.refresh_token.is_some() {
        target.refresh_token = source.refresh_token.clone();
    }
    if source.client_id.is_some() {
        target.client_id = source.client_id.clone();
    }
    if source.client_secret.is_some() {
        target.client_secret = source.client_secret.clone();
    }
    if source.profile_arn.is_some() {
        target.profile_arn = source.profile_arn.clone();
    }
    if source.expires_at.is_some() {
        target.expires_at = source.expires_at.clone();
    }
    if source.region.is_some() {
        target.region = source.region.clone();
    }
    if source.auth_method.is_some() {
        target.auth_method = source.auth_method.clone();
    }
}
