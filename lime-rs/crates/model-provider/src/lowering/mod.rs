//! Provider wire lowering owner.
//!
//! Provider-neutral requests use `runtime-core::llm_protocol::canonical`.
//! Media-specific body lowering remains here beside the provider network
//! boundary. OpenCode reference:
//! `packages/llm/src/route/protocol.ts` and `protocols/*.ts` at commit
//! `9976269ab1accfc9f9dc98a4a688c516934de422` (MIT).

mod common;
mod fal_video_generation;
mod openai_images;
mod openai_responses_image_generation;
mod xai_video_generation;

pub use common::ProtocolMappingError;

use runtime_core::CanonicalRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesImageGenerationInputShape {
    #[default]
    PromptString,
    InputList,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ResponsesImageGenerationOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer_model: Option<String>,
    #[serde(default)]
    pub input_shape: ResponsesImageGenerationInputShape,
}

pub fn build_openai_images_generation_body(
    model_id: &str,
    request: &CanonicalRequest,
) -> Result<serde_json::Value, ProtocolMappingError> {
    openai_images::body_for_model(model_id, request)
}

pub fn build_fal_video_generation_body(
    model_id: &str,
    request: &CanonicalRequest,
) -> Result<serde_json::Value, ProtocolMappingError> {
    fal_video_generation::body_for_model(model_id, request)
}

pub fn build_xai_video_generation_body(
    model_id: &str,
    request: &CanonicalRequest,
) -> Result<serde_json::Value, ProtocolMappingError> {
    xai_video_generation::body_for_model(model_id, request)
}

pub fn build_responses_image_generation_body(
    model_id: &str,
    request: &CanonicalRequest,
    options: &ResponsesImageGenerationOptions,
) -> Result<serde_json::Value, ProtocolMappingError> {
    openai_responses_image_generation::body_for_model(model_id, request, options)
}
