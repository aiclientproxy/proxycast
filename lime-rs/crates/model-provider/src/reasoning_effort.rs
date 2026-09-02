pub fn reasoning_effort_for_request(
    requested: Option<&str>,
    multi_agent_override: Option<&str>,
    supported: &[String],
) -> Option<String> {
    let requested = requested
        .map(str::trim)
        .filter(|effort| !effort.is_empty())?;
    if requested.eq_ignore_ascii_case("persistent") {
        return Some("disabled".to_string());
    }
    if !requested.eq_ignore_ascii_case("ultra") {
        return Some(requested.to_string());
    }

    multi_agent_override
        .map(str::trim)
        .filter(|effort| !effort.eq_ignore_ascii_case("ultra"))
        .and_then(|effort| supported_effort(supported, effort))
        .or_else(|| supported_effort(supported, "max"))
        .or_else(|| {
            supported
                .iter()
                .rev()
                .find(|effort| !effort.eq_ignore_ascii_case("ultra"))
                .cloned()
        })
        .or_else(|| Some("medium".to_string()))
}

fn supported_effort(supported: &[String], requested: &str) -> Option<String> {
    supported
        .iter()
        .find(|effort| effort.eq_ignore_ascii_case(requested))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::reasoning_effort_for_request;

    fn efforts(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn ultra_uses_valid_catalog_override() {
        assert_eq!(
            reasoning_effort_for_request(
                Some("ultra"),
                Some("high"),
                &efforts(&["low", "high", "max", "ultra"]),
            )
            .as_deref(),
            Some("high")
        );
    }

    #[test]
    fn ultra_uses_codex_fallback_order() {
        let invalid_override = Some("unsupported");
        assert_eq!(
            reasoning_effort_for_request(
                Some("ultra"),
                invalid_override,
                &efforts(&["low", "max", "xhigh", "ultra"]),
            )
            .as_deref(),
            Some("max")
        );
        assert_eq!(
            reasoning_effort_for_request(
                Some("ultra"),
                invalid_override,
                &efforts(&["low", "xhigh", "ultra"]),
            )
            .as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            reasoning_effort_for_request(Some("ultra"), None, &[]).as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn persistent_uses_responses_wire_name() {
        assert_eq!(
            reasoning_effort_for_request(Some("persistent"), None, &[]).as_deref(),
            Some("disabled")
        );
    }
}
