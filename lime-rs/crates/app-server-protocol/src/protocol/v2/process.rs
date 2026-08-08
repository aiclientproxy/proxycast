use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessTerminalSize {
    pub rows: u16,
    pub cols: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSpawnParams {
    pub command: Vec<String>,
    pub process_handle: String,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tty: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream_stdin: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream_stdout_stderr: bool,
    #[serde(
        default,
        deserialize_with = "super::serde_helpers::deserialize_double_option",
        serialize_with = "super::serde_helpers::serialize_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_bytes_cap: Option<Option<usize>>,
    #[serde(
        default,
        deserialize_with = "super::serde_helpers::deserialize_double_option",
        serialize_with = "super::serde_helpers::serialize_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub timeout_ms: Option<Option<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<std::collections::HashMap<String, Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ProcessTerminalSize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSpawnResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessWriteStdinParams {
    pub process_handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_base64: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub close_stdin: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessWriteStdinResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessResizePtyParams {
    pub process_handle: String,
    pub size: ProcessTerminalSize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessResizePtyResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessKillParams {
    pub process_handle: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessKillResponse {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ProcessOutputStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessOutputDeltaNotification {
    pub process_handle: String,
    pub stream: ProcessOutputStream,
    pub delta_base64: String,
    pub cap_reached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProcessExitedNotification {
    pub process_handle: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stdout_cap_reached: bool,
    pub stderr: String,
    pub stderr_cap_reached: bool,
}
