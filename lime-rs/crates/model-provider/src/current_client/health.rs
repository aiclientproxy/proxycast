use crate::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    protocol: &'static str,
}

impl RouteHealthKey {
    fn from_config(config: &RuntimeProviderConfig) -> Self {
        Self {
            provider: config.provider_name.trim().to_ascii_lowercase(),
            model: config.model_name.trim().to_string(),
            base_url: normalized_base_url(config),
            protocol: match config.protocol {
                Some(RuntimeProviderProtocol::ChatCompletions) => "chat_completions",
                Some(RuntimeProviderProtocol::Responses) => "responses",
                Some(RuntimeProviderProtocol::AnthropicMessages) => "anthropic_messages",
                None => "missing",
            },
        }
    }
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
        Arc::clone(
            breakers
                .entry(key)
                .or_insert_with(|| Arc::new(CircuitBreaker::new(self.config))),
        )
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

#[derive(Debug)]
struct Inner {
    config: HealthConfig,
    state: State,
}

/// 共享 provider 健康熔断器。
pub(crate) struct CircuitBreaker {
    inner: Mutex<Inner>,
}

impl CircuitBreaker {
    pub(crate) fn new(config: HealthConfig) -> Self {
        let config = config.normalized();
        Self {
            inner: Mutex::new(Inner {
                config,
                state: State::Closed {
                    outcomes: VecDeque::new(),
                },
            }),
        }
    }

    pub(crate) fn acquire(self: &Arc<Self>) -> Result<CircuitPermit, CircuitOpen> {
        let mut inner = self
            .inner
            .lock()
            .expect("provider health circuit mutex poisoned");
        let window_duration = inner.config.window_duration;
        if let State::Closed { outcomes } = &mut inner.state {
            prune_outcomes(outcomes, window_duration, Instant::now());
        }
        match &mut inner.state {
            State::Closed { .. } => Ok(CircuitPermit {
                breaker: Arc::clone(self),
                mode: PermitMode::Closed,
                settled: false,
            }),
            State::Open { opened_at } => {
                let elapsed = opened_at.elapsed();
                if elapsed < inner.config.open_duration {
                    return Err(CircuitOpen {
                        retry_after: inner.config.open_duration.saturating_sub(elapsed),
                    });
                }
                inner.state = State::HalfOpen {
                    probe_in_flight: true,
                };
                Ok(CircuitPermit {
                    breaker: Arc::clone(self),
                    mode: PermitMode::Probe,
                    settled: false,
                })
            }
            State::HalfOpen { probe_in_flight } => {
                if *probe_in_flight {
                    return Err(CircuitOpen {
                        retry_after: inner.config.open_duration,
                    });
                }
                *probe_in_flight = true;
                Ok(CircuitPermit {
                    breaker: Arc::clone(self),
                    mode: PermitMode::Probe,
                    settled: false,
                })
            }
        }
    }

    fn record(&self, mode: PermitMode, success: bool) {
        let mut inner = self
            .inner
            .lock()
            .expect("provider health circuit mutex poisoned");
        let config = inner.config;
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
                if outcomes.len() >= config.min_samples && error_rate >= config.error_rate_threshold
                {
                    inner.state = State::Open {
                        opened_at: Instant::now(),
                    };
                }
            }
            (State::HalfOpen { .. }, PermitMode::Probe) if success => {
                inner.state = State::Closed {
                    outcomes: VecDeque::new(),
                };
            }
            (State::HalfOpen { .. }, PermitMode::Probe) => {
                inner.state = State::Open {
                    opened_at: Instant::now(),
                };
            }
            // 较早开始的 closed 请求可能晚于新请求完成；不能因此关闭或覆盖
            // half-open probe 状态。
            _ => {}
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
            base_url: base_url.map(str::to_string),
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
