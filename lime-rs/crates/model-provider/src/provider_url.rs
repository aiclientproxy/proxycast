//! Provider URL 中的路由元数据。

pub const LIME_TENANT_HEADER: &str = "X-Lime-Tenant-ID";
const LIME_TENANT_PARAM: &str = "lime_tenant_id";

pub fn parse_provider_base_url(base_url: &str) -> Option<url::Url> {
    let base_url = base_url.trim();
    if base_url.is_empty() {
        return None;
    }
    if let Ok(url) = url::Url::parse(base_url) {
        if url.has_host() {
            return Some(url);
        }
    }
    url::Url::parse(&format!("https://{base_url}")).ok()
}

pub fn lime_tenant_id_from_base_url(base_url: &str) -> Option<String> {
    let url = parse_provider_base_url(base_url)?;
    tenant_id_from_pairs(url.query()).or_else(|| tenant_id_from_pairs(url.fragment()))
}

/// 清除不属于 HTTP request target 的 Provider 路由元数据。
///
/// Fragment 始终只在本地解释；query 中仅移除 Lime tenant 参数，其他网关参数保留。
pub fn strip_provider_routing_metadata(url: &mut url::Url) {
    url.set_fragment(None);
    let query = url
        .query_pairs()
        .filter(|(name, _)| name != LIME_TENANT_PARAM)
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    url.set_query(None);
    if !query.is_empty() {
        let mut pairs = url.query_pairs_mut();
        for (name, value) in query {
            pairs.append_pair(&name, &value);
        }
    }
}

fn tenant_id_from_pairs(value: Option<&str>) -> Option<String> {
    url::form_urlencoded::parse(value?.as_bytes()).find_map(|(name, value)| {
        (name == LIME_TENANT_PARAM)
            .then(|| normalize_lime_tenant_id(&value))
            .flatten()
    })
}

fn normalize_lime_tenant_id(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')))
    .then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_id_accepts_query_or_fragment_and_rejects_unsafe_values() {
        assert_eq!(
            lime_tenant_id_from_base_url("https://llm.limeai.run#lime_tenant_id=tenant-0001")
                .as_deref(),
            Some("tenant-0001")
        );
        assert_eq!(
            lime_tenant_id_from_base_url("https://llm.limeai.run?lime_tenant_id=tenant_0002")
                .as_deref(),
            Some("tenant_0002")
        );
        assert_eq!(
            lime_tenant_id_from_base_url(
                "https://llm.limeai.run#lime_tenant_id=tenant%0D%0Aunsafe"
            ),
            None
        );
        assert_eq!(
            lime_tenant_id_from_base_url("llm.limeai.run#lime_tenant_id=tenant-0003").as_deref(),
            Some("tenant-0003")
        );
        assert_eq!(
            parse_provider_base_url("localhost:11434/v1")
                .expect("scheme-less localhost provider URL")
                .as_str(),
            "https://localhost:11434/v1"
        );
    }

    #[test]
    fn request_url_keeps_gateway_query_but_removes_local_routing_metadata() {
        let mut url =
            url::Url::parse("https://gateway.example/v1?region=cn&lime_tenant_id=tenant-1#local")
                .expect("provider URL");

        strip_provider_routing_metadata(&mut url);

        assert_eq!(url.as_str(), "https://gateway.example/v1?region=cn");
    }
}
