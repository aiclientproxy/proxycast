//! 应用通用工具函数
//!
//! 包含 API Key 生成、绑定地址校验、Token 掩码等与桌面宿主无关的逻辑。

use crate::config;

/// 生成安全的 API Key
pub fn generate_api_key() -> String {
    config::generate_secure_api_key()
}

/// 检查是否为回环地址
pub fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(address) => address.is_loopback(),
        Err(_) => false,
    }
}

/// 检查是否为有效的绑定地址
/// 允许回环地址、0.0.0.0 和私有网络地址。
pub fn is_valid_bind_host(host: &str) -> bool {
    if is_loopback_host(host) {
        return true;
    }

    if host == "0.0.0.0" || host == "::" {
        return true;
    }

    if let Ok(std::net::IpAddr::V4(ipv4)) = host.parse::<std::net::IpAddr>() {
        let octets = ipv4.octets();
        return octets[0] == 10
            || (octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31))
            || (octets[0] == 192 && octets[1] == 168);
    }

    false
}

/// 检查是否为非本地绑定地址（需要强 API Key）
pub fn is_non_local_bind(host: &str) -> bool {
    if host == "0.0.0.0" || host == "::" {
        return true;
    }

    if let Ok(std::net::IpAddr::V4(ipv4)) = host.parse::<std::net::IpAddr>() {
        let octets = ipv4.octets();
        return octets[0] == 10
            || (octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31))
            || (octets[0] == 192 && octets[1] == 168);
    }

    false
}

/// 掩码敏感 Token
pub fn mask_token(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 12 {
        "****".to_string()
    } else {
        let prefix: String = chars[..6].iter().collect();
        let suffix: String = chars[chars.len() - 4..].iter().collect();
        format!("{prefix}****{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_loopback_host() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.1"));
    }

    #[test]
    fn test_is_valid_bind_host() {
        assert!(is_valid_bind_host("localhost"));
        assert!(is_valid_bind_host("127.0.0.1"));
        assert!(is_valid_bind_host("::1"));
        assert!(is_valid_bind_host("0.0.0.0"));
        assert!(is_valid_bind_host("::"));
        assert!(is_valid_bind_host("192.168.1.1"));
        assert!(is_valid_bind_host("10.0.0.1"));
        assert!(is_valid_bind_host("172.16.0.1"));
        assert!(is_valid_bind_host("172.31.255.255"));
        assert!(!is_valid_bind_host("8.8.8.8"));
        assert!(!is_valid_bind_host("1.1.1.1"));
    }

    #[test]
    fn test_is_non_local_bind() {
        assert!(is_non_local_bind("0.0.0.0"));
        assert!(is_non_local_bind("::"));
        assert!(!is_non_local_bind("127.0.0.1"));
        assert!(!is_non_local_bind("localhost"));
        assert!(is_non_local_bind("192.168.1.1"));
        assert!(is_non_local_bind("10.0.0.1"));
        assert!(is_non_local_bind("172.16.0.1"));
    }

    #[test]
    fn test_mask_token() {
        assert_eq!(mask_token("short"), "****");
        assert_eq!(mask_token("abcdefghijklmnop"), "abcdef****mnop");
    }
}
