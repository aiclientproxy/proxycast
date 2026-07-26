use super::{CircuitState, RouteHealthKey};
use sha2::{Digest, Sha256};
use std::time::Duration;

#[derive(Clone, Debug)]
pub(crate) struct RouteHealthTelemetry {
    pub(super) provider: String,
    pub(super) model: String,
    pub(super) protocol: &'static str,
    pub(super) route_id: String,
    pub(super) credential_kind: &'static str,
}

impl RouteHealthTelemetry {
    pub(super) fn from_key(key: &RouteHealthKey) -> Self {
        let mut digest = Sha256::new();
        for value in [
            key.provider.as_str(),
            key.model.as_str(),
            key.base_url.as_str(),
            key.protocol,
            key.credential_scope.as_str(),
        ] {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        Self {
            provider: key.provider.clone(),
            model: key.model.clone(),
            protocol: key.protocol,
            route_id: format!("{:x}", digest.finalize()),
            credential_kind: if key.credential_scope.starts_with("stored:") {
                "stored"
            } else {
                "direct"
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TransportRetryEvent {
    transport: &'static str,
    reason: &'static str,
    failed_attempt: u8,
    next_attempt: u8,
    max_attempts: u8,
    delay: Duration,
    delay_source: &'static str,
    status_code: Option<u16>,
}

impl TransportRetryEvent {
    pub(crate) fn new(
        transport: &'static str,
        reason: &'static str,
        failed_attempt: u8,
        next_attempt: u8,
        max_attempts: u8,
        delay: Duration,
        delay_source: &'static str,
        status_code: Option<u16>,
    ) -> Self {
        Self {
            transport,
            reason,
            failed_attempt,
            next_attempt,
            max_attempts,
            delay,
            delay_source,
            status_code,
        }
    }
}

pub(super) trait CircuitObserver: Send + Sync {
    fn on_state_change(&self, _old: CircuitState, _new: CircuitState, _reason: &'static str) {}

    fn on_probe_admission(&self, _allowed: bool) {}

    fn on_rejected(&self, _state: CircuitState, _retry_after: Duration) {}

    fn on_failure(&self, _state: CircuitState) {}

    fn on_transport_retry(&self, _event: TransportRetryEvent) {}
}

pub(super) struct TracingCircuitObserver {
    route: RouteHealthTelemetry,
}

impl TracingCircuitObserver {
    pub(super) fn new(route: RouteHealthTelemetry) -> Self {
        Self { route }
    }
}

impl CircuitObserver for TracingCircuitObserver {
    fn on_state_change(&self, old: CircuitState, new: CircuitState, reason: &'static str) {
        let old = old.as_str();
        let new = new.as_str();
        match new {
            "open" => tracing::warn!(
                target: "provider_health",
                provider = %self.route.provider,
                model = %self.route.model,
                protocol = self.route.protocol,
                route_id = %self.route.route_id,
                credential_kind = self.route.credential_kind,
                old,
                new,
                reason,
                "provider circuit opened"
            ),
            "half_open" => tracing::debug!(
                target: "provider_health",
                provider = %self.route.provider,
                model = %self.route.model,
                protocol = self.route.protocol,
                route_id = %self.route.route_id,
                credential_kind = self.route.credential_kind,
                old,
                new,
                reason,
                "provider circuit half-open"
            ),
            _ => tracing::info!(
                target: "provider_health",
                provider = %self.route.provider,
                model = %self.route.model,
                protocol = self.route.protocol,
                route_id = %self.route.route_id,
                credential_kind = self.route.credential_kind,
                old,
                new,
                reason,
                "provider circuit closed"
            ),
        }
    }

    fn on_probe_admission(&self, allowed: bool) {
        tracing::debug!(
            target: "provider_health",
            provider = %self.route.provider,
            model = %self.route.model,
            protocol = self.route.protocol,
            route_id = %self.route.route_id,
            credential_kind = self.route.credential_kind,
            allowed,
            "provider circuit probe admission"
        );
    }

    fn on_rejected(&self, state: CircuitState, retry_after: Duration) {
        let retry_after_ms = u64::try_from(retry_after.as_millis()).unwrap_or(u64::MAX);
        tracing::debug!(
            target: "provider_health",
            provider = %self.route.provider,
            model = %self.route.model,
            protocol = self.route.protocol,
            route_id = %self.route.route_id,
            credential_kind = self.route.credential_kind,
            state = state.as_str(),
            retry_after_ms,
            "provider circuit rejected request"
        );
    }

    fn on_failure(&self, state: CircuitState) {
        tracing::trace!(
            target: "provider_health",
            provider = %self.route.provider,
            model = %self.route.model,
            protocol = self.route.protocol,
            route_id = %self.route.route_id,
            credential_kind = self.route.credential_kind,
            state = state.as_str(),
            "provider circuit recorded failure"
        );
    }

    fn on_transport_retry(&self, event: TransportRetryEvent) {
        let delay_ms = u64::try_from(event.delay.as_millis()).unwrap_or(u64::MAX);
        tracing::info!(
            target: "provider_retry",
            provider = %self.route.provider,
            model = %self.route.model,
            protocol = self.route.protocol,
            route_id = %self.route.route_id,
            credential_kind = self.route.credential_kind,
            transport = event.transport,
            reason = event.reason,
            failed_attempt = event.failed_attempt,
            next_attempt = event.next_attempt,
            max_attempts = event.max_attempts,
            delay_ms,
            delay_source = event.delay_source,
            status_code = event.status_code,
            "provider transport retry scheduled"
        );
    }
}
