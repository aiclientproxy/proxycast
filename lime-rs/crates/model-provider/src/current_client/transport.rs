use super::health::{CircuitBreaker, TransportRetryEvent};
use super::CurrentProviderError;
use chrono::{DateTime, Utc};
use rand::Rng;
use reqwest::{header::HeaderMap, StatusCode};
use std::error::Error;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

pub(super) fn server_disallows_retry(headers: &HeaderMap) -> bool {
    headers
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .is_some_and(|value| value.eq_ignore_ascii_case("false"))
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

pub(super) fn provider_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let now = SystemTime::now();
    retry_after_at(headers, now).or_else(|| quota_reset_after(headers, now))
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    provider_retry_after(headers).map(|duration| duration.min(MAX_RETRY_AFTER))
}

fn retry_after_at(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return (seconds > 0).then(|| Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(now)
        .ok()
        .filter(|duration| !duration.is_zero())
}

fn quota_reset_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    const RESET_HEADERS: [(&str, Option<&str>); 5] = [
        (
            "x-ratelimit-reset-requests",
            Some("x-ratelimit-remaining-requests"),
        ),
        (
            "x-ratelimit-reset-tokens",
            Some("x-ratelimit-remaining-tokens"),
        ),
        (
            "anthropic-ratelimit-requests-reset",
            Some("anthropic-ratelimit-requests-remaining"),
        ),
        (
            "anthropic-ratelimit-tokens-reset",
            Some("anthropic-ratelimit-tokens-remaining"),
        ),
        ("x-ratelimit-reset", None),
    ];

    RESET_HEADERS
        .iter()
        .filter(|(_, remaining_header)| {
            remaining_header.is_none_or(|remaining_header| {
                headers
                    .get(remaining_header)
                    .and_then(|value| value.to_str().ok())
                    .is_none_or(|value| value.trim() == "0")
            })
        })
        .filter_map(|(reset_header, _)| {
            headers
                .get(*reset_header)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| parse_reset_value(value.trim(), now))
        })
        .max()
}

fn parse_reset_value(value: &str, now: SystemTime) -> Option<Duration> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return SystemTime::from(timestamp.with_timezone(&Utc))
            .duration_since(now)
            .ok()
            .filter(|duration| !duration.is_zero());
    }
    if let Ok(raw) = value.parse::<u64>() {
        let now_epoch = now.duration_since(UNIX_EPOCH).ok()?.as_secs();
        return if raw > now_epoch / 2 {
            raw.checked_sub(now_epoch)
                .filter(|seconds| *seconds > 0)
                .map(Duration::from_secs)
        } else {
            (raw > 0).then(|| Duration::from_secs(raw))
        };
    }
    parse_compound_duration(value)
}

fn parse_compound_duration(value: &str) -> Option<Duration> {
    let mut rest = value;
    let mut seconds = 0.0_f64;
    while !rest.is_empty() {
        let number_len = rest
            .find(|character: char| !character.is_ascii_digit() && character != '.')
            .unwrap_or(rest.len());
        if number_len == 0 || number_len == rest.len() {
            return None;
        }
        let number = rest[..number_len].parse::<f64>().ok()?;
        rest = &rest[number_len..];
        let (multiplier, suffix_len) = if rest.starts_with("ms") {
            (0.001, 2)
        } else if rest.starts_with('s') {
            (1.0, 1)
        } else if rest.starts_with('m') {
            (60.0, 1)
        } else if rest.starts_with('h') {
            (3600.0, 1)
        } else {
            return None;
        };
        seconds += number * multiplier;
        rest = &rest[suffix_len..];
    }
    (seconds.is_finite() && seconds > 0.0).then(|| Duration::from_secs_f64(seconds))
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
    fn explicit_server_retry_false_overrides_status_policy() {
        let mut headers = HeaderMap::new();
        headers.insert("x-should-retry", HeaderValue::from_static(" FALSE "));
        assert!(server_disallows_retry(&headers));

        headers.insert("x-should-retry", HeaderValue::from_static("true"));
        assert!(!server_disallows_retry(&headers));
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
    fn provider_retry_after_preserves_server_window_and_http_date() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("120"));
        assert_eq!(
            provider_retry_after(&headers),
            Some(Duration::from_secs(120))
        );

        let now = SystemTime::now();
        let retry_at = now + Duration::from_secs(90);
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str(&httpdate::fmt_http_date(retry_at)).expect("HTTP date"),
        );
        let parsed = retry_after_at(&headers, now).expect("HTTP-date retry window");
        assert!((Duration::from_secs(89)..=Duration::from_secs(90)).contains(&parsed));
    }

    #[test]
    fn provider_retry_after_consumes_exhausted_quota_reset() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("0"),
        );
        headers.insert(
            "x-ratelimit-reset-requests",
            HeaderValue::from_static("1m250ms"),
        );
        assert_eq!(
            provider_retry_after(&headers),
            Some(Duration::from_millis(60_250))
        );

        headers.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("0"),
        );
        headers.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("2m"));
        assert_eq!(
            provider_retry_after(&headers),
            Some(Duration::from_secs(120)),
            "all exhausted dimensions must recover before the key is selected again"
        );

        headers.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("1"),
        );
        assert_eq!(
            provider_retry_after(&headers),
            Some(Duration::from_secs(120))
        );
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
