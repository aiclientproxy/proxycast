use super::health::{CircuitBreaker, TransportRetryEvent};
use super::CurrentProviderError;
use rand::Rng;
use reqwest::{header::HeaderMap, StatusCode};
use std::error::Error;
use std::time::Duration;

// Matches the Codex provider default: four retries after the initial request.
pub(super) const MAX_STREAM_REQUEST_ATTEMPTS: u8 = 5;
const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(200);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RetryDelay {
    pub(super) duration: Duration,
    pub(super) source: &'static str,
}

pub(super) fn should_retry_stream_request_status(status: StatusCode) -> bool {
    // Codex defaults to retry_5xx=true and retry_429=false at the request layer.
    status.is_server_error()
}

pub(super) fn retry_delay(headers: &HeaderMap, completed_attempts: u8) -> RetryDelay {
    match retry_after(headers) {
        Some(duration) => RetryDelay {
            duration,
            source: "retry_after",
        },
        None => RetryDelay {
            duration: exponential_backoff(completed_attempts),
            source: "exponential_backoff",
        },
    }
}

pub(super) fn request_retry_reason(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect_error"
    } else {
        "transport_error"
    }
}

pub(super) fn observed_retry_delay(
    breaker: &CircuitBreaker,
    headers: &HeaderMap,
    completed_attempts: u8,
    transport: &'static str,
    reason: &'static str,
    status_code: Option<u16>,
) -> Duration {
    let delay = retry_delay(headers, completed_attempts);
    breaker.observe_transport_retry(TransportRetryEvent::new(
        transport,
        reason,
        completed_attempts,
        completed_attempts.saturating_add(1),
        MAX_STREAM_REQUEST_ATTEMPTS,
        delay.duration,
        delay.source,
        status_code,
    ));
    delay.duration
}

pub(super) fn request_failure(error: reqwest::Error) -> CurrentProviderError {
    CurrentProviderError::transport(format!(
        "Provider 请求失败 ({})",
        request_retry_reason(&error)
    ))
}

pub(super) fn error_chain(error: &(dyn Error + 'static)) -> String {
    let mut messages = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        let message = error.to_string();
        if messages.last() != Some(&message) {
            messages.push(message);
        }
        source = error.source();
    }
    messages.join(": ")
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let seconds = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    (seconds > 0).then(|| Duration::from_secs(seconds).min(MAX_RETRY_AFTER))
}

fn exponential_backoff(completed_attempts: u8) -> Duration {
    let delay = INITIAL_RETRY_DELAY.saturating_mul(1_u32 << completed_attempts.saturating_sub(1));
    let jitter = rand::thread_rng().gen_range(0.9_f64..1.1_f64);
    Duration::from_millis((delay.as_millis() as f64 * jitter) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderValue, RETRY_AFTER};
    use std::fmt;

    #[derive(Debug)]
    struct OuterError {
        source: InnerError,
    }

    impl fmt::Display for OuterError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("outer")
        }
    }

    impl Error for OuterError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.source)
        }
    }

    #[derive(Debug)]
    struct InnerError;

    impl fmt::Display for InnerError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("inner")
        }
    }

    impl Error for InnerError {}

    #[test]
    fn error_chain_keeps_nested_transport_cause() {
        assert_eq!(
            error_chain(&OuterError { source: InnerError }),
            "outer: inner"
        );
    }

    #[tokio::test]
    async fn request_failure_redacts_url_and_error_chain() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind stalled server");
        let url = format!(
            "http://{}/private-provider-path?credential=secret-ref",
            listener.local_addr().expect("stalled server address")
        );
        let error = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(50))
            .build()
            .expect("build request client")
            .get(&url)
            .send()
            .await
            .expect_err("server that never responds must time out");
        assert!(
            error.to_string().contains(&url),
            "fixture must prove the source error contains the sensitive URL"
        );

        let failure = request_failure(error);

        assert_eq!(failure.message, "Provider 请求失败 (timeout)");
        assert!(!failure.message.contains("private-provider-path"));
        assert!(!failure.message.contains("secret-ref"));
    }

    #[test]
    fn retries_codex_default_request_statuses() {
        for code in 500..=599 {
            let status = StatusCode::from_u16(code).expect("server status");
            assert!(should_retry_stream_request_status(status), "{status}");
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::CONFLICT,
            StatusCode::TOO_EARLY,
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            assert!(!should_retry_stream_request_status(status), "{status}");
        }
    }

    #[test]
    fn retry_after_overrides_and_is_capped() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("2"));
        assert_eq!(
            retry_delay(&headers, 1),
            RetryDelay {
                duration: Duration::from_secs(2),
                source: "retry_after",
            }
        );

        headers.insert(RETRY_AFTER, HeaderValue::from_static("120"));
        assert_eq!(retry_delay(&headers, 1).duration, MAX_RETRY_AFTER);
    }

    #[test]
    fn zero_retry_after_falls_back_to_exponential_backoff() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("0"));

        let delay = retry_delay(&headers, 1);

        assert_eq!(delay.source, "exponential_backoff");
        assert!((Duration::from_millis(180)..=Duration::from_millis(220)).contains(&delay.duration));
    }

    #[test]
    fn retry_delay_uses_jittered_exponential_backoff() {
        let first = retry_delay(&HeaderMap::new(), 1);
        let second = retry_delay(&HeaderMap::new(), 2);

        assert_eq!(first.source, "exponential_backoff");
        assert_eq!(second.source, "exponential_backoff");
        assert!((Duration::from_millis(180)..=Duration::from_millis(220)).contains(&first.duration));
        assert!(
            (Duration::from_millis(360)..=Duration::from_millis(440)).contains(&second.duration)
        );
    }
}
