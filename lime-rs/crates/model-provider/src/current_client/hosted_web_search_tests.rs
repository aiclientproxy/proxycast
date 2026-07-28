use super::{
    lowering::responses_request, CurrentProviderRequest, CurrentProviderTool,
    RuntimeProviderConfig, RuntimeReplyProviderRequestWireShape,
};
use crate::runtime_provider::{RuntimeProviderAuth, RuntimeProviderProtocol as RouteProtocol};
use serde_json::json;
use std::collections::BTreeMap;

fn config(provider_name: &str, base_url: Option<&str>) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: provider_name.to_string(),
        provider_selector: None,
        model_name: "gpt-5".to_string(),
        api_key: Some("sk-test".to_string()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url: base_url.map(str::to_string),
        api_version: None,
        credential_uuid: "credential-1".to_string(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(RouteProtocol::Responses),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    }
}

fn request(tool_name: &str) -> CurrentProviderRequest {
    CurrentProviderRequest::new(Vec::new()).with_tools(vec![CurrentProviderTool {
        name: tool_name.to_string(),
        description: "Search the web".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
        }),
    }])
}

fn lowered_tools(config: &RuntimeProviderConfig, tool_name: &str) -> serde_json::Value {
    let canonical = request(tool_name)
        .into_canonical("gpt-5")
        .expect("canonical request");
    responses_request(
        config,
        &canonical,
        &RuntimeReplyProviderRequestWireShape::default(),
        &BTreeMap::new(),
    )["tools"]
        .clone()
}

#[test]
fn official_responses_route_lowers_web_search_as_hosted_tool() {
    assert_eq!(
        lowered_tools(
            &config("openai", Some("https://api.openai.com/v1")),
            "WebSearch",
        ),
        json!([{
            "type": "web_search",
            "external_web_access": true,
        }])
    );
}

#[test]
fn non_official_responses_routes_keep_web_search_as_local_function() {
    for config in [
        config("openai", Some("https://gateway.example.com/v1")),
        config("ollama", Some("http://127.0.0.1:11434")),
    ] {
        assert_eq!(lowered_tools(&config, "WebSearch")[0]["type"], "function");
        assert_eq!(lowered_tools(&config, "WebSearch")[0]["name"], "WebSearch");
    }
}

#[test]
fn hosted_web_search_rejects_non_canonical_tool_aliases() {
    let config = config("openai", Some("https://api.openai.com/v1"));
    for alias in ["web_search", "WebSearchTool", "mcp.system.WebSearch"] {
        assert_eq!(lowered_tools(&config, alias)[0]["type"], "function");
        assert_eq!(lowered_tools(&config, alias)[0]["name"], alias);
    }
}

#[test]
fn official_responses_route_lowers_image_generation_as_hosted_tool() {
    assert_eq!(
        lowered_tools(
            &config("openai", Some("https://api.openai.com/v1")),
            "ImageGeneration",
        ),
        json!([{ "type": "image_generation" }])
    );
}

#[test]
fn non_official_responses_routes_keep_image_generation_as_local_function() {
    for config in [
        config("openai", Some("https://gateway.example.com/v1")),
        config("ollama", Some("http://127.0.0.1:11434")),
    ] {
        assert_eq!(
            lowered_tools(&config, "ImageGeneration")[0]["type"],
            "function"
        );
        assert_eq!(
            lowered_tools(&config, "ImageGeneration")[0]["name"],
            "ImageGeneration"
        );
    }
}

#[test]
fn hosted_image_generation_rejects_aliases_and_local_media_task_tool() {
    let config = config("openai", Some("https://api.openai.com/v1"));
    for alias in [
        "image_generation",
        "ImageGenerationTool",
        "lime_create_image_generation_task",
    ] {
        assert_eq!(lowered_tools(&config, alias)[0]["type"], "function");
        assert_eq!(lowered_tools(&config, alias)[0]["name"], alias);
    }
}
