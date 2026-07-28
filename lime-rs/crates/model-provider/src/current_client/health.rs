use crate::runtime_provider::{
    RuntimeProviderAuth, RuntimeProviderConfig, RuntimeProviderProtocol,
};
use sha2::{Digest, Sha256};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod telemetry;

pub(super) use telemetry::TransportRetryEvent;
use telemetry::{CircuitObserver, RouteHealthTelemetry, TracingCircuitObserver};

#[derive(Clone)]
pub struct CurrentProviderHealthRegistry {
    config: HealthConfig,
    breakers: Arc<Mutex<HashMap<RouteHealthKey, Arc<CircuitBreaker>>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RouteHealthKey {
    provider: String,
    model: String,
    base_url: String,
    api_version: String,
    protocol: &'static str,
    credential_scope: String,
}

impl RouteHealthKey {
    fn from_config(config: &RuntimeProviderConfig) -> Self {
        let default_api_version =
            if config.protocol == Some(RuntimeProviderProtocol::AzureResponses) {
                "v1"
            } else {
                "default"
            };
        Self {
            provider: config.provider_name.trim().to_ascii_lowercase(),
            model: config.model_name.trim().to_string(),
            base_url: normalized_base_url(config),
            api_version: config
                .api_version
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(default_api_version)
                .to_string(),
            protocol: match config.protocol {
                Some(RuntimeProviderProtocol::ChatCompletions) => "chat_completions",
                Some(RuntimeProviderProtocol::Responses) => "responses",
                Some(RuntimeProviderProtocol::AzureResponses) => "azure_responses",
                Some(RuntimeProviderProtocol::AnthropicMessages) => "anthropic_messages",
                Some(RuntimeProviderProtocol::GeminiGenerateContent) => "gemini_generate_content",
                Some(RuntimeProviderProtocol::VertexGemini) => "vertex_gemini",
                None => "missing",
            },
            credential_scope: credential_scope(config),
        }
    }
}

fn credential_scope(config: &RuntimeProviderConfig) -> String {
    match config.auth {
        RuntimeProviderAuth::NoAuth => return "no-auth".to_string(),
        RuntimeProviderAuth::OemManaged => return "oem-managed".to_string(),
        RuntimeProviderAuth::ApiKey => {}
    }
    let credential_uuid = config.credential_uuid.trim();
    if !credential_uuid.is_empty() {
        return format!("stored:{credential_uuid}");
    }

    // Direct runtime credentials have no durable UUID. The registry only keeps
    // a collision-resistant fingerprint, never the credential itself.
    format!(
        "direct:{:x}",
        Sha256::digest(config.api_key.as_deref().unwrap_or_default().as_bytes())
    )
}

impl CurrentProviderHealthRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn circuit_for(&self, config: &RuntimeProviderConfig) -> Arc<CircuitBreaker> {
        let key = RouteHealthKey::from_config(config);
        let mut breakers = self
            .breakers
            .lock()
            .expect("provider health registry mutex poisoned");
        match breakers.entry(key) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let breaker = Arc::new(CircuitBreaker::for_route(self.config, entry.key()));
                Arc::clone(entry.insert(breaker))
            }
        }
    }

    /// Reads the health state for an exact resolved route without creating a
    /// synthetic closed entry when the route has never executed.
    pub fn snapshot_for(
        &self,
        config: &RuntimeProviderConfig,
    ) -> Option<CurrentProviderHealthSnapshot> {
        let key = RouteHealthKey::from_config(config);
        let breaker = self
            .breakers
            .lock()
            .expect("provider health registry mutex poisoned")
            .get(&key)
            .cloned()?;
        Some(breaker.snapshot())
    }

    #[cfg(test)]
    fn with_config(config: HealthConfig) -> Self {
        Self {
            config,
            breakers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for CurrentProviderHealthRegistry {
    fn default() -> Self {
        Self {
            config: HealthConfig::default(),
            breakers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

fn normalized_base_url(config: &RuntimeProviderConfig) -> String {
    let fallback = match config.protocol {
        Some(RuntimeProviderProtocol::AnthropicMessages) => "https://api.anthropic.com",
        Some(RuntimeProviderProtocol::GeminiGenerateContent) => {
            "https://generativelanguage.googleapis.com/v1beta"
        }
        Some(RuntimeProviderProtocol::VertexGemini) => "vertex-project-endpoint-required",
        Some(RuntimeProviderProtocol::AzureResponses) => "azure-resource-url-required",
        _ => "https://api.openai.com",
    };
    let value = config
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback);
    let Ok(mut url) = url::Url::parse(value) else {
        return value.trim_end_matches('/').to_string();
    };
    url.set_fragment(None);
    let path = url.path().trim_end_matches('/').to_string();
    url.set_path(if path.is_empty() { "/" } else { &path });
    url.to_string().trim_end_matches('/').to_string()
}

/// Provider 传输健康熔断策略。
///
/// 窗口有明确上限；只有积累足够观测后才统计失败率，避免单次瞬态错误
/// 压制原本健康的 route。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HealthConfig {
    pub(crate) window_duration: Duration,
    pub(crate) min_samples: usize,
    pub(crate) error_rate_threshold: f64,
    pub(crate) open_duration: Duration,
}

const MAX_OUTCOMES: usize = 10_000;

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_secs(60),
            min_samples: 10,
            error_rate_threshold: 0.5,
            open_duration: Duration::from_secs(10),
        }
    }
}

impl HealthConfig {
    fn normalized(self) -> Self {
        let window_duration = self.window_duration.max(Duration::from_millis(1));
        Self {
            window_duration,
            min_samples: self.min_samples.max(1),
            error_rate_threshold: if self.error_rate_threshold.is_finite() {
                self.error_rate_threshold.clamp(0.01, 1.0)
            } else {
                0.5
            },
            open_duration: self.open_duration,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    at: Instant,
    failed: bool,
}

#[derive(Debug)]
enum State {
    Closed { outcomes: VecDeque<Outcome> },
    Open { opened_at: Instant },
    HalfOpen { probe_in_flight: bool },
}

impl State {
    fn kind(&self) -> CircuitState {
        match self {
            Self::Closed { .. } => CircuitState::Closed,
            Self::Open { .. } => CircuitState::Open,
            Self::HalfOpen { .. } => CircuitState::HalfOpen,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Runtime circuit state for one exact provider route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrentProviderHealthState {
    Closed,
    Open,
    HalfOpen,
}

impl From<CircuitState> for CurrentProviderHealthState {
    fn from(state: CircuitState) -> Self {
        match state {
            CircuitState::Closed => Self::Closed,
            CircuitState::Open => Self::Open,
            CircuitState::HalfOpen => Self::HalfOpen,
        }
    }
}

/// Sanitized runtime health facts for one exact provider route.
///
/// Window counts are available while the circuit is closed. The current
/// breaker intentionally discards that window when it opens, so open and
/// half-open snapshots report them as unknown instead of fabricating data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CurrentProviderHealthSnapshot {
    pub state: CurrentProviderHealthState,
    pub window_sample_count: Option<usize>,
    pub window_failure_count: Option<usize>,
    pub probe_in_flight: bool,
    pub retry_after: Option<Duration>,
}

impl CircuitState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

#[derive(Debug)]
struct Inner {
    config: HealthConfig,
    state: State,
}

/// 共享 provider 健康熔断器。
pub(crate) struct CircuitBreaker {
    inner: Mutex<Inner>,
    observer: Option<Arc<dyn CircuitObserver>>,
}

impl CircuitBreaker {
    #[cfg(test)]
    pub(crate) fn new(config: HealthConfig) -> Self {
        Self::build(config, None)
    }

    fn for_route(config: HealthConfig, key: &RouteHealthKey) -> Self {
        Self::build(
            config,
            Some(Arc::new(TracingCircuitObserver::new(
                RouteHealthTelemetry::from_key(key),
            ))),
        )
    }

    fn build(config: HealthConfig, observer: Option<Arc<dyn CircuitObserver>>) -> Self {
        let config = config.normalized();
        Self {
            inner: Mutex::new(Inner {
                config,
                state: State::Closed {
                    outcomes: VecDeque::new(),
                },
            }),
            observer,
        }
    }

    #[cfg(test)]
    fn with_observer(config: HealthConfig, observer: Arc<dyn CircuitObserver>) -> Self {
        Self::build(config, Some(observer))
    }

    pub(crate) fn acquire(self: &Arc<Self>) -> Result<CircuitPermit, CircuitOpen> {
        let (result, transition, probe_admission, rejected_state) = {
            let mut inner = self
                .inner
                .lock()
                .expect("provider health circuit mutex poisoned");
            let window_duration = inner.config.window_duration;
            if let State::Closed { outcomes } = &mut inner.state {
                prune_outcomes(outcomes, window_duration, Instant::now());
            }
            match &mut inner.state {
                State::Closed { .. } => (
                    Ok(CircuitPermit {
                        breaker: Arc::clone(self),
                        mode: PermitMode::Closed,
                        settled: false,
                    }),
                    None,
                    None,
                    None,
                ),
                State::Open { opened_at } => {
                    let elapsed = opened_at.elapsed();
                    if elapsed < inner.config.open_duration {
                        (
                            Err(CircuitOpen {
                                retry_after: inner.config.open_duration.saturating_sub(elapsed),
                            }),
                            None,
                            None,
                            Some(CircuitState::Open),
                        )
                    } else {
                        inner.state = State::HalfOpen {
                            probe_in_flight: true,
                        };
                        (
                            Ok(CircuitPermit {
                                breaker: Arc::clone(self),
                                mode: PermitMode::Probe,
                                settled: false,
                            }),
                            Some((CircuitState::Open, CircuitState::HalfOpen, "open_elapsed")),
                            Some(true),
                            None,
                        )
                    }
                }
                State::HalfOpen { probe_in_flight } => {
                    if *probe_in_flight {
                        (
                            Err(CircuitOpen {
                                retry_after: HALF_OPEN_PROBE_BACKOFF
                                    .min(inner.config.open_duration),
                            }),
                            None,
                            Some(false),
                            Some(CircuitState::HalfOpen),
                        )
                    } else {
                        *probe_in_flight = true;
                        (
                            Ok(CircuitPermit {
                                breaker: Arc::clone(self),
                                mode: PermitMode::Probe,
                                settled: false,
                            }),
                            None,
                            Some(true),
                            None,
                        )
                    }
                }
            }
        };
        if let Some(observer) = self.observer.as_deref() {
            if let Some((old, new, reason)) = transition {
                observer.on_state_change(old, new, reason);
            }
            if let Some(allowed) = probe_admission {
                observer.on_probe_admission(allowed);
            }
            if let (Some(state), Err(error)) = (rejected_state, &result) {
                observer.on_rejected(state, error.retry_after);
            }
        }
        result
    }

    fn snapshot(&self) -> CurrentProviderHealthSnapshot {
        let mut inner = self
            .inner
            .lock()
            .expect("provider health circuit mutex poisoned");
        let now = Instant::now();
        let window_duration = inner.config.window_duration;
        if let State::Closed { outcomes } = &mut inner.state {
            prune_outcomes(outcomes, window_duration, now);
        }

        match &inner.state {
            State::Closed { outcomes } => CurrentProviderHealthSnapshot {
                state: CircuitState::Closed.into(),
                window_sample_count: Some(outcomes.len()),
                window_failure_count: Some(
                    outcomes.iter().filter(|outcome| outcome.failed).count(),
                ),
                probe_in_flight: false,
                retry_after: None,
            },
            State::Open { opened_at } => CurrentProviderHealthSnapshot {
                state: CircuitState::Open.into(),
                window_sample_count: None,
                window_failure_count: None,
                probe_in_flight: false,
                retry_after: Some(
                    inner
                        .config
                        .open_duration
                        .saturating_sub(now.saturating_duration_since(*opened_at)),
                ),
            },
            State::HalfOpen { probe_in_flight } => CurrentProviderHealthSnapshot {
                state: CircuitState::HalfOpen.into(),
                window_sample_count: None,
                window_failure_count: None,
                probe_in_flight: *probe_in_flight,
                retry_after: probe_in_flight
                    .then_some(HALF_OPEN_PROBE_BACKOFF.min(inner.config.open_duration)),
            },
        }
    }

    fn record(&self, mode: PermitMode, success: bool) {
        let (transition, state) = {
            let mut inner = self
                .inner
                .lock()
                .expect("provider health circuit mutex poisoned");
            let config = inner.config;
            let old = inner.state.kind();
            let mut reason = None;
            match (&mut inner.state, mode) {
                (State::Closed { outcomes }, PermitMode::Closed) => {
                    let now = Instant::now();
                    outcomes.push_back(Outcome {
                        at: now,
                        failed: !success,
                    });
                    while outcomes.len() > MAX_OUTCOMES {
                        outcomes.pop_front();
                    }
                    prune_outcomes(outcomes, config.window_duration, now);
                    let failures = outcomes.iter().filter(|outcome| outcome.failed).count();
                    let error_rate = failures as f64 / outcomes.len().max(1) as f64;
                    if outcomes.len() >= config.min_samples
                        && error_rate >= config.error_rate_threshold
                    {
                        inner.state = State::Open {
                            opened_at: Instant::now(),
                        };
                        reason = Some("trip");
                    }
                }
                (State::HalfOpen { .. }, PermitMode::Probe) if success => {
                    inner.state = State::Closed {
                        outcomes: VecDeque::new(),
                    };
                    reason = Some("probe_success");
                }
                (State::HalfOpen { .. }, PermitMode::Probe) => {
                    inner.state = State::Open {
                        opened_at: Instant::now(),
                    };
                    reason = Some("probe_failure");
                }
                // 较早开始的 closed 请求可能晚于新请求完成；不能因此关闭或覆盖
                // half-open probe 状态。
                _ => {}
            }
            let new = inner.state.kind();
            (reason.map(|reason| (old, new, reason)), inner.state.kind())
        };
        if let Some(observer) = self.observer.as_deref() {
            if let Some((old, new, reason)) = transition {
                observer.on_state_change(old, new, reason);
            }
            if !success {
                observer.on_failure(state);
            }
        }
    }

    pub(super) fn observe_transport_retry(&self, event: TransportRetryEvent) {
        if let Some(observer) = self.observer.as_deref() {
            observer.on_transport_retry(event);
        }
    }

    fn release_probe(&self, mode: PermitMode) {
        if mode != PermitMode::Probe {
            return;
        }
        let mut inner = self
            .inner
            .lock()
            .expect("provider health circuit mutex poisoned");
        if let State::HalfOpen { probe_in_flight } = &mut inner.state {
            *probe_in_flight = false;
        }
    }
}

const HALF_OPEN_PROBE_BACKOFF: Duration = Duration::from_millis(50);

fn prune_outcomes(outcomes: &mut VecDeque<Outcome>, window: Duration, now: Instant) {
    while outcomes
        .front()
        .is_some_and(|outcome| now.duration_since(outcome.at) > window)
    {
        outcomes.pop_front();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PermitMode {
    Closed,
    Probe,
}

pub(crate) struct CircuitPermit {
    breaker: Arc<CircuitBreaker>,
    mode: PermitMode,
    settled: bool,
}

impl CircuitPermit {
    pub(crate) fn success(&mut self) {
        self.settle(true);
    }

    pub(crate) fn failure(&mut self) {
        self.settle(false);
    }

    pub(crate) fn ignore(&mut self) {
        if !self.settled {
            self.breaker.release_probe(self.mode);
        }
        self.settled = true;
    }

    fn settle(&mut self, success: bool) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.breaker.record(self.mode, success);
    }
}

impl Drop for CircuitPermit {
    fn drop(&mut self) {
        if !self.settled {
            self.breaker.release_probe(self.mode);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CircuitOpen {
    retry_after: Duration,
}

impl CircuitOpen {
    #[cfg(test)]
    pub(crate) fn retry_after(self) -> Duration {
        self.retry_after
    }
}

impl fmt::Display for CircuitOpen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "provider health circuit is open; retry after {} ms",
            self.retry_after.as_millis()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn route_config(
        provider_name: &str,
        model_name: &str,
        base_url: Option<&str>,
        protocol: RuntimeProviderProtocol,
    ) -> RuntimeProviderConfig {
        RuntimeProviderConfig {
            provider_name: provider_name.to_string(),
            provider_selector: Some(provider_name.to_string()),
            model_name: model_name.to_string(),
            api_key: Some("test-key".to_string()),
            auth: RuntimeProviderAuth::ApiKey,
            base_url: base_url.map(str::to_string),
            api_version: None,
            credential_uuid: "credential-1".to_string(),
            reasoning_effort: None,
            service_tier: None,
            protocol: Some(protocol),
            supports_websockets: false,
            toolshim: false,
            toolshim_model: None,
        }
    }

    fn breaker(config: HealthConfig) -> Arc<CircuitBreaker> {
        Arc::new(CircuitBreaker::new(config))
    }

    fn fail(breaker: &Arc<CircuitBreaker>) {
        let mut permit = breaker.acquire().expect("circuit permit");
        permit.failure();
    }

    fn succeed(breaker: &Arc<CircuitBreaker>) {
        let mut permit = breaker.acquire().expect("circuit permit");
        permit.success();
    }

    #[derive(Default)]
    struct RecordingObserver {
        breaker: Mutex<Option<std::sync::Weak<CircuitBreaker>>>,
        transitions: Mutex<Vec<(CircuitState, CircuitState, &'static str)>>,
        probe_admissions: Mutex<Vec<bool>>,
        rejections: Mutex<Vec<CircuitState>>,
        failure_states: Mutex<Vec<CircuitState>>,
        transport_retries: Mutex<Vec<TransportRetryEvent>>,
    }

    impl RecordingObserver {
        fn assert_breaker_unlocked(&self) {
            let breaker = self
                .breaker
                .lock()
                .expect("read observed breaker")
                .as_ref()
                .and_then(std::sync::Weak::upgrade);
            if let Some(breaker) = breaker {
                assert!(
                    breaker.inner.try_lock().is_ok(),
                    "observer callbacks must run outside the breaker mutex"
                );
            }
        }
    }

    impl CircuitObserver for RecordingObserver {
        fn on_state_change(&self, old: CircuitState, new: CircuitState, reason: &'static str) {
            self.assert_breaker_unlocked();
            self.transitions
                .lock()
                .expect("record transitions")
                .push((old, new, reason));
        }

        fn on_probe_admission(&self, allowed: bool) {
            self.assert_breaker_unlocked();
            self.probe_admissions
                .lock()
                .expect("record probe admission")
                .push(allowed);
        }

        fn on_rejected(&self, state: CircuitState, _retry_after: Duration) {
            self.assert_breaker_unlocked();
            self.rejections
                .lock()
                .expect("record rejection")
                .push(state);
        }

        fn on_failure(&self, state: CircuitState) {
            self.assert_breaker_unlocked();
            self.failure_states
                .lock()
                .expect("record failure")
                .push(state);
        }

        fn on_transport_retry(&self, event: TransportRetryEvent) {
            self.assert_breaker_unlocked();
            self.transport_retries
                .lock()
                .expect("record transport retries")
                .push(event);
        }
    }

    #[test]
    fn bounded_window_opens_after_threshold() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 3,
            error_rate_threshold: 0.5,
            open_duration: Duration::from_secs(60),
        });
        fail(&breaker);
        succeed(&breaker);
        fail(&breaker);

        let error = match breaker.acquire() {
            Err(error) => error,
            Ok(_) => panic!("threshold should open circuit"),
        };
        assert!(error.retry_after() <= Duration::from_secs(60));
    }

    #[test]
    fn half_open_allows_one_probe_and_success_closes() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::ZERO,
        });
        fail(&breaker);

        let mut probe = breaker.acquire().expect("half-open probe");
        assert!(
            breaker.acquire().is_err(),
            "only one probe may be in flight"
        );
        probe.success();
        assert!(breaker.acquire().is_ok(), "successful probe closes circuit");
    }

    #[test]
    fn observer_records_transitions_probe_admission_and_rejection() {
        let observer = Arc::new(RecordingObserver::default());
        let breaker = Arc::new(CircuitBreaker::with_observer(
            HealthConfig {
                window_duration: Duration::from_secs(60),
                min_samples: 1,
                error_rate_threshold: 1.0,
                open_duration: Duration::ZERO,
            },
            observer.clone(),
        ));
        *observer.breaker.lock().expect("set observed breaker") = Some(Arc::downgrade(&breaker));
        fail(&breaker);

        let mut probe = breaker.acquire().expect("half-open probe");
        assert!(breaker.acquire().is_err(), "second probe must be rejected");
        probe.success();

        assert_eq!(
            *observer.transitions.lock().expect("read transitions"),
            vec![
                (CircuitState::Closed, CircuitState::Open, "trip"),
                (CircuitState::Open, CircuitState::HalfOpen, "open_elapsed"),
                (
                    CircuitState::HalfOpen,
                    CircuitState::Closed,
                    "probe_success"
                ),
            ]
        );
        assert_eq!(
            *observer.probe_admissions.lock().expect("read probes"),
            vec![true, false]
        );
        assert_eq!(
            *observer.rejections.lock().expect("read rejections"),
            vec![CircuitState::HalfOpen]
        );
        assert_eq!(
            *observer.failure_states.lock().expect("read failures"),
            vec![CircuitState::Open]
        );
    }

    #[test]
    fn half_open_probe_rejection_uses_short_backoff() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::from_millis(60),
        });
        fail(&breaker);
        thread::sleep(Duration::from_millis(70));

        let _probe = breaker.acquire().expect("half-open probe");
        let rejected = match breaker.acquire() {
            Err(error) => error,
            Ok(_) => panic!("second probe must wait"),
        };

        assert_eq!(rejected.retry_after(), Duration::from_millis(50));
    }

    #[test]
    fn observer_records_structured_transport_retry_outside_breaker_mutex() {
        let observer = Arc::new(RecordingObserver::default());
        let breaker = Arc::new(CircuitBreaker::with_observer(
            HealthConfig::default(),
            observer.clone(),
        ));
        *observer.breaker.lock().expect("set observed breaker") = Some(Arc::downgrade(&breaker));
        let event = TransportRetryEvent::new(
            "http",
            "server_error",
            1,
            2,
            5,
            Duration::from_secs(2),
            "retry_after",
            Some(503),
        );

        breaker.observe_transport_retry(event);

        assert_eq!(
            *observer
                .transport_retries
                .lock()
                .expect("read transport retries"),
            vec![event]
        );
    }

    #[test]
    fn route_telemetry_excludes_endpoint_and_credential_values() {
        let mut config = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://gateway.example.com/v1?token=endpoint-secret"),
            RuntimeProviderProtocol::Responses,
        );
        config.credential_uuid.clear();
        config.api_key = Some("direct-key-secret".to_string());

        let telemetry = RouteHealthTelemetry::from_key(&RouteHealthKey::from_config(&config));
        let rendered = format!("{telemetry:?}");

        assert_eq!(telemetry.provider, "openai");
        assert_eq!(telemetry.model, "gpt-5-codex");
        assert_eq!(telemetry.protocol, "responses");
        assert_eq!(telemetry.credential_kind, "direct");
        assert_eq!(telemetry.route_id.len(), 64);
        assert!(!rendered.contains("endpoint-secret"));
        assert!(!rendered.contains("direct-key-secret"));
    }

    #[test]
    fn dropped_probe_releases_slot_without_recording_outcome() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::ZERO,
        });
        fail(&breaker);
        let probe = breaker.acquire().expect("half-open probe");
        drop(probe);
        let _probe = breaker
            .acquire()
            .expect("dropped probe should not leave half-open circuit wedged");
    }

    #[test]
    fn ignored_probe_releases_half_open_slot_without_closing_circuit() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::ZERO,
        });
        fail(&breaker);
        let mut probe = breaker.acquire().expect("half-open probe");
        probe.ignore();
        assert!(
            breaker.acquire().is_ok(),
            "ignored probe should release slot"
        );
    }

    #[test]
    fn old_closed_request_cannot_close_new_half_open_probe() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::from_secs(60),
        });
        let mut old_request = breaker.acquire().expect("closed request");
        let mut trigger = breaker.acquire().expect("second closed request");
        trigger.failure();
        old_request.success();
        assert!(breaker.acquire().is_err());
    }

    #[test]
    fn registry_snapshot_is_exact_route_read_only_and_sanitized() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 3,
            error_rate_threshold: 1.0,
            ..HealthConfig::default()
        });
        let mut route = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://gateway.example.com/v1?token=endpoint-secret"),
            RuntimeProviderProtocol::Responses,
        );
        route.credential_uuid.clear();
        route.api_key = Some("direct-key-secret".to_string());
        let mut different_model = route.clone();
        different_model.model_name = "gpt-5.5".to_string();

        assert!(registry.snapshot_for(&route).is_none());
        assert!(registry.snapshot_for(&different_model).is_none());
        assert!(
            registry
                .breakers
                .lock()
                .expect("read breaker registry")
                .is_empty(),
            "snapshot must not create a synthetic closed route"
        );

        let breaker = registry.circuit_for(&route);
        succeed(&breaker);
        fail(&breaker);
        let snapshot = registry.snapshot_for(&route).expect("known route snapshot");

        assert_eq!(snapshot.state, CurrentProviderHealthState::Closed);
        assert_eq!(snapshot.window_sample_count, Some(2));
        assert_eq!(snapshot.window_failure_count, Some(1));
        assert!(!snapshot.probe_in_flight);
        assert_eq!(snapshot.retry_after, None);
        assert!(registry.snapshot_for(&different_model).is_none());

        let rendered = format!("{snapshot:?}");
        assert!(!rendered.contains("gateway.example.com"));
        assert!(!rendered.contains("endpoint-secret"));
        assert!(!rendered.contains("direct-key-secret"));
    }

    #[test]
    fn registry_snapshot_reports_open_and_half_open_retry_state() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::from_millis(80),
            ..HealthConfig::default()
        });
        let route = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let breaker = registry.circuit_for(&route);
        fail(&breaker);

        let open = registry.snapshot_for(&route).expect("open route snapshot");
        assert_eq!(open.state, CurrentProviderHealthState::Open);
        assert_eq!(open.window_sample_count, None);
        assert_eq!(open.window_failure_count, None);
        assert!(!open.probe_in_flight);
        assert!(open
            .retry_after
            .is_some_and(|retry_after| retry_after <= Duration::from_millis(80)));

        thread::sleep(Duration::from_millis(90));
        let probe = breaker.acquire().expect("half-open probe");
        let half_open = registry
            .snapshot_for(&route)
            .expect("half-open route snapshot");
        assert_eq!(half_open.state, CurrentProviderHealthState::HalfOpen);
        assert!(half_open.probe_in_flight);
        assert_eq!(half_open.retry_after, Some(Duration::from_millis(50)));
        drop(probe);
    }

    #[test]
    fn registry_reuses_breaker_for_normalized_route() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 1,
            error_rate_threshold: 1.0,
            ..HealthConfig::default()
        });
        let first = route_config(
            " OpenAI ",
            " gpt-5-codex ",
            Some("HTTPS://API.OPENAI.COM/v1/"),
            RuntimeProviderProtocol::Responses,
        );
        let second = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let first_breaker = registry.circuit_for(&first);
        let second_breaker = registry.circuit_for(&second);

        assert!(Arc::ptr_eq(&first_breaker, &second_breaker));
    }

    #[test]
    fn azure_default_api_version_shares_health_with_explicit_v1() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig::default());
        let implicit = route_config(
            "azure",
            "gpt-5.4",
            Some("https://resource.openai.azure.com"),
            RuntimeProviderProtocol::AzureResponses,
        );
        let mut explicit = implicit.clone();
        explicit.api_version = Some("v1".to_string());
        let mut preview = implicit.clone();
        preview.api_version = Some("2025-04-01-preview".to_string());

        assert!(Arc::ptr_eq(
            &registry.circuit_for(&implicit),
            &registry.circuit_for(&explicit)
        ));
        assert!(!Arc::ptr_eq(
            &registry.circuit_for(&implicit),
            &registry.circuit_for(&preview)
        ));
    }

    #[test]
    fn registry_isolates_model_base_url_and_protocol() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 1,
            error_rate_threshold: 1.0,
            ..HealthConfig::default()
        });
        let route = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let different_model = route_config(
            "openai",
            "gpt-5.5",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let different_base = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://gateway.example.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let different_protocol = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::ChatCompletions,
        );
        let route_breaker = registry.circuit_for(&route);
        let mut permit = route_breaker.acquire().expect("route starts closed");
        permit.failure();

        assert!(registry.circuit_for(&route).acquire().is_err());
        assert!(registry.circuit_for(&different_model).acquire().is_ok());
        assert!(registry.circuit_for(&different_base).acquire().is_ok());
        assert!(registry.circuit_for(&different_protocol).acquire().is_ok());
    }

    #[test]
    fn registry_does_not_share_health_between_gemini_api_key_and_vertex_protocols() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 1,
            error_rate_threshold: 1.0,
            ..HealthConfig::default()
        });
        let gemini = route_config(
            "google",
            "gemini-2.5-pro",
            Some("https://gateway.example.com"),
            RuntimeProviderProtocol::GeminiGenerateContent,
        );
        let vertex = route_config(
            "google",
            "gemini-2.5-pro",
            Some("https://gateway.example.com"),
            RuntimeProviderProtocol::VertexGemini,
        );
        let gemini_breaker = registry.circuit_for(&gemini);
        let mut permit = gemini_breaker
            .acquire()
            .expect("Gemini API key route starts closed");
        permit.failure();

        let vertex_breaker = registry.circuit_for(&vertex);
        assert!(gemini_breaker.acquire().is_err());
        assert!(vertex_breaker.acquire().is_ok());
        assert!(!Arc::ptr_eq(&gemini_breaker, &vertex_breaker));
    }

    #[test]
    fn registry_isolates_stored_and_direct_credential_scopes() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig {
            min_samples: 1,
            error_rate_threshold: 1.0,
            ..HealthConfig::default()
        });
        let stored = route_config(
            "openai",
            "gpt-5-codex",
            Some("https://api.openai.com/v1"),
            RuntimeProviderProtocol::Responses,
        );
        let mut different_stored = stored.clone();
        different_stored.credential_uuid = "credential-2".to_string();
        let mut direct_first = stored.clone();
        direct_first.credential_uuid.clear();
        direct_first.api_key = Some("direct-key-1".to_string());
        let mut direct_second = direct_first.clone();
        direct_second.api_key = Some("direct-key-2".to_string());

        let mut permit = registry
            .circuit_for(&stored)
            .acquire()
            .expect("stored route starts closed");
        permit.failure();

        assert!(registry.circuit_for(&stored).acquire().is_err());
        assert!(registry.circuit_for(&different_stored).acquire().is_ok());
        let direct_first_breaker = registry.circuit_for(&direct_first);
        assert!(direct_first_breaker.acquire().is_ok());
        assert!(registry.circuit_for(&direct_second).acquire().is_ok());
        assert!(
            !Arc::ptr_eq(&direct_first_breaker, &registry.circuit_for(&direct_second),),
            "direct credentials must not share health state"
        );
        assert!(
            Arc::ptr_eq(&direct_first_breaker, &registry.circuit_for(&direct_first)),
            "the same direct credential must retain its route health state"
        );
    }

    #[test]
    fn no_auth_routes_share_health_without_credential_identity() {
        let registry = CurrentProviderHealthRegistry::with_config(HealthConfig::default());
        let mut first = route_config(
            "openai",
            "local-model",
            Some("http://127.0.0.1:11434/v1"),
            RuntimeProviderProtocol::ChatCompletions,
        );
        first.auth = RuntimeProviderAuth::NoAuth;
        first.api_key = None;
        first.credential_uuid.clear();
        let mut second = first.clone();
        second.api_key = Some("must-not-affect-no-auth-scope".to_string());
        second.credential_uuid = "must-not-affect-no-auth-scope".to_string();

        assert!(Arc::ptr_eq(
            &registry.circuit_for(&first),
            &registry.circuit_for(&second)
        ));
    }

    #[test]
    fn normalized_config_is_finite_and_usable() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::ZERO,
            min_samples: 0,
            error_rate_threshold: f64::NAN,
            open_duration: Duration::ZERO,
        });
        fail(&breaker);
        let _ = breaker.acquire().expect("normalized one-entry circuit");
        let _ = thread::yield_now();
    }

    #[test]
    fn outcomes_expire_from_time_window() {
        let breaker = breaker(HealthConfig {
            window_duration: Duration::from_millis(5),
            min_samples: 2,
            error_rate_threshold: 1.0,
            open_duration: Duration::from_secs(60),
        });
        fail(&breaker);
        thread::sleep(Duration::from_millis(10));
        succeed(&breaker);
        assert!(
            breaker.acquire().is_ok(),
            "expired failure must leave window"
        );
    }
}
