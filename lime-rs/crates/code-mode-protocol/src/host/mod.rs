//! Code Mode host connection protocol.
//!
//! This module is intentionally only a boundary index. Wire envelopes live in
//! [`message`], operation payloads in [`payload`], and connection constants in
//! [`types`]. Keeping those owners separate mirrors the Codex host protocol
//! and prevents transport implementations from defining their own copies.

mod codec;
mod error;
mod message;
mod payload;
mod types;

pub use codec::{EncodedFrame, FramedReader, FramedWriter, MAX_FRAME_BYTES};
pub use error::HandshakeRejectReason;
pub use message::{
    ClientHello, ClientToHost, DelegateRequest, DelegateResponse, HostHello, HostRequest,
    HostResponse, HostToClient, WireResult,
};
pub use payload::{WireExecuteRequest, WireSessionCellExecutionLimits, WireWaitRequest};
pub use types::{
    MAX_IN_FLIGHT_REQUESTS, MAX_PENDING_DELEGATE_CALLS, PROTOCOL_VERSION, SESSION_LIMITS_CAPABILITY,
};
