//! Request validation at the gRPC host boundary.

use code_mode_protocol::grpc as proto;
use tonic::Status;
use uuid::Uuid;

pub(super) use code_mode_protocol::grpc::MAX_IDENTIFIER_BYTES;
pub(super) const MAX_TOOL_FILTERS: usize = 64;

pub(super) fn require_identifier(value: &str, field: &str) -> Result<(), Status> {
    if value.trim().is_empty() {
        return Err(Status::invalid_argument(format!(
            "{field} must not be empty"
        )));
    }
    bounded(value, MAX_IDENTIFIER_BYTES, field)
}

pub(super) fn require_uuid(value: &str, field: &str) -> Result<Uuid, Status> {
    require_identifier(value, field)?;
    Uuid::parse_str(value).map_err(|_| Status::invalid_argument(format!("{field} must be a UUID")))
}

pub(super) fn bounded(value: &str, maximum: usize, field: &str) -> Result<(), Status> {
    if value.len() > maximum {
        return Err(Status::invalid_argument(format!(
            "{field} exceeds {maximum} bytes"
        )));
    }
    Ok(())
}

pub(super) fn tool_name(name: &proto::ToolName) -> Result<(), Status> {
    require_identifier(&name.name, "tool name")?;
    if let Some(namespace) = name.namespace.as_deref() {
        bounded(namespace, MAX_IDENTIFIER_BYTES, "tool namespace")?;
    }
    Ok(())
}

pub(super) fn tool_filters(filters: &[proto::ToolName]) -> Result<(), Status> {
    if filters.len() > MAX_TOOL_FILTERS {
        return Err(Status::invalid_argument(format!(
            "code-mode tool subscription exceeds {MAX_TOOL_FILTERS} filters"
        )));
    }
    filters.iter().try_for_each(tool_name)
}
