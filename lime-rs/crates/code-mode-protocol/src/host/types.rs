//! Stable connection constants and negotiated host limits.

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_IN_FLIGHT_REQUESTS: usize = 1_024;
pub const MAX_PENDING_DELEGATE_CALLS: usize = 1_024;
pub const SESSION_LIMITS_CAPABILITY: &str = "session-cell-execution-resource-limits";
