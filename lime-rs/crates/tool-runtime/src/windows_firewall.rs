use std::io;

#[cfg(not(windows))]
pub fn install_offline_rules(_account: &str) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows Firewall is only available on Windows",
    ))
}

#[cfg(not(windows))]
pub fn verify_offline_rules(_account: &str) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows Firewall is only available on Windows",
    ))
}

#[cfg(windows)]
mod platform {
    use super::io;
    use std::ffi::c_void;
    use windows::core::{Interface, BSTR};
    use windows::Win32::Foundation::{RPC_E_CHANGED_MODE, S_OK, VARIANT_TRUE};
    use windows::Win32::NetworkManagement::WindowsFirewall::{
        INetFwPolicy2, INetFwRule3, INetFwRules, NetFwPolicy2, NetFwRule, NET_FW_ACTION_BLOCK,
        NET_FW_IP_PROTOCOL_ANY, NET_FW_MODIFY_STATE_OK, NET_FW_PROFILE2_ALL,
        NET_FW_PROFILE2_DOMAIN, NET_FW_PROFILE2_PRIVATE, NET_FW_PROFILE2_PUBLIC,
        NET_FW_RULE_DIR_OUT,
    };
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER,
        COINIT_APARTMENTTHREADED,
    };
    use windows_sys::Win32::Foundation::{LocalFree, HLOCAL};
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;

    const RULE_GROUP: &str = "Lime Windows Sandbox";
    const LOOPBACK_REMOTE_ADDRESSES: &str = "127.0.0.0/8,::/127";
    const NON_LOOPBACK_REMOTE_ADDRESSES: &str = "0.0.0.0-126.255.255.255,128.0.0.0-255.255.255.255,::,::2-ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff";

    const RULES: &[FirewallRuleSpec] = &[
        FirewallRuleSpec {
            name: "lime_sandbox_offline_block_non_loopback_outbound",
            description: "Block non-loopback outbound traffic for the Lime offline sandbox account",
            remote_addresses: NON_LOOPBACK_REMOTE_ADDRESSES,
        },
        FirewallRuleSpec {
            name: "lime_sandbox_offline_block_loopback_outbound",
            description: "Block loopback outbound traffic for the Lime offline sandbox account",
            remote_addresses: LOOPBACK_REMOTE_ADDRESSES,
        },
    ];

    struct FirewallRuleSpec {
        name: &'static str,
        description: &'static str,
        remote_addresses: &'static str,
    }

    struct ComApartment {
        should_uninitialize: bool,
    }

    impl ComApartment {
        fn initialize() -> io::Result<Self> {
            let result = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
            if result == RPC_E_CHANGED_MODE {
                return Ok(Self {
                    should_uninitialize: false,
                });
            }
            if result.is_err() {
                return Err(other(format!("CoInitializeEx failed: {result:?}")));
            }
            Ok(Self {
                should_uninitialize: true,
            })
        }
    }

    impl Drop for ComApartment {
        fn drop(&mut self) {
            if self.should_uninitialize {
                unsafe { CoUninitialize() };
            }
        }
    }

    pub fn install_offline_rules(account: &str) -> io::Result<usize> {
        let _apartment = ComApartment::initialize()?;
        let policy = open_policy()?;
        verify_policy_enforcement(&policy)?;
        let rules = unsafe { policy.Rules() }.map_err(|error| windows_error("Rules", error))?;
        let account_sid = account_sid_string(account)?;
        for spec in RULES {
            ensure_rule(&rules, spec, &account_sid)?;
        }
        Ok(RULES.len())
    }

    pub fn verify_offline_rules(account: &str) -> io::Result<()> {
        let _apartment = ComApartment::initialize()?;
        let policy = open_policy()?;
        verify_policy_enforcement(&policy)?;
        let rules = unsafe { policy.Rules() }.map_err(|error| windows_error("Rules", error))?;
        let account_sid = account_sid_string(account)?;
        for spec in RULES {
            let name = BSTR::from(spec.name);
            let rule = unsafe { rules.Item(&name) }
                .map_err(|error| windows_error(&format!("Rules::Item({})", spec.name), error))?
                .cast::<INetFwRule3>()
                .map_err(|error| windows_error(&format!("cast rule {}", spec.name), error))?;
            validate_rule(&rule, spec, &account_sid)?;
        }
        Ok(())
    }

    fn open_policy() -> io::Result<INetFwPolicy2> {
        unsafe { CoCreateInstance(&NetFwPolicy2, None, CLSCTX_INPROC_SERVER) }
            .map_err(|error| windows_error("CoCreateInstance(NetFwPolicy2)", error))
    }

    fn verify_policy_enforcement(policy: &INetFwPolicy2) -> io::Result<()> {
        let mut modify_state = Default::default();
        let result = unsafe {
            (Interface::vtable(policy).LocalPolicyModifyState)(
                Interface::as_raw(policy),
                &mut modify_state,
            )
        };
        if result != S_OK {
            return Err(other(format!(
                "Windows Firewall local policy does not cover every active profile: {result:?}"
            )));
        }
        if modify_state != NET_FW_MODIFY_STATE_OK {
            return Err(other(format!(
                "Windows Firewall local rules are ineffective: LocalPolicyModifyState={modify_state:?}"
            )));
        }

        let active_profiles = unsafe { policy.CurrentProfileTypes() }
            .map_err(|error| windows_error("CurrentProfileTypes", error))?;
        if active_profiles
            & (NET_FW_PROFILE2_DOMAIN.0 | NET_FW_PROFILE2_PRIVATE.0 | NET_FW_PROFILE2_PUBLIC.0)
            == 0
        {
            return Err(other("Windows Firewall has no recognized active profile"));
        }
        for profile in [
            NET_FW_PROFILE2_DOMAIN,
            NET_FW_PROFILE2_PRIVATE,
            NET_FW_PROFILE2_PUBLIC,
        ] {
            if active_profiles & profile.0 == 0 {
                continue;
            }
            let enabled = unsafe { policy.get_FirewallEnabled(profile) }
                .map_err(|error| windows_error("get_FirewallEnabled", error))?;
            if enabled != VARIANT_TRUE {
                return Err(other(format!(
                    "Windows Firewall is disabled for active profile bit {}",
                    profile.0
                )));
            }
        }
        Ok(())
    }

    fn ensure_rule(
        rules: &INetFwRules,
        spec: &FirewallRuleSpec,
        account_sid: &str,
    ) -> io::Result<()> {
        let name = BSTR::from(spec.name);
        let rule = match unsafe { rules.Item(&name) } {
            Ok(existing) => existing
                .cast::<INetFwRule3>()
                .map_err(|error| windows_error(&format!("cast rule {}", spec.name), error))?,
            Err(_) => {
                let created: INetFwRule3 =
                    unsafe { CoCreateInstance(&NetFwRule, None, CLSCTX_INPROC_SERVER) }
                        .map_err(|error| windows_error("CoCreateInstance(NetFwRule)", error))?;
                unsafe { created.SetName(&name) }
                    .map_err(|error| windows_error("SetName", error))?;
                configure_rule(&created, spec, account_sid)?;
                unsafe { rules.Add(&created) }
                    .map_err(|error| windows_error("Rules::Add", error))?;
                created
            }
        };
        configure_rule(&rule, spec, account_sid)?;
        validate_rule(&rule, spec, account_sid)
    }

    fn configure_rule(
        rule: &INetFwRule3,
        spec: &FirewallRuleSpec,
        account_sid: &str,
    ) -> io::Result<()> {
        let local_user = local_user_sddl(account_sid);
        unsafe {
            rule.SetDescription(&BSTR::from(spec.description))
                .map_err(|error| windows_error("SetDescription", error))?;
            rule.SetDirection(NET_FW_RULE_DIR_OUT)
                .map_err(|error| windows_error("SetDirection", error))?;
            rule.SetAction(NET_FW_ACTION_BLOCK)
                .map_err(|error| windows_error("SetAction", error))?;
            rule.SetEnabled(VARIANT_TRUE)
                .map_err(|error| windows_error("SetEnabled", error))?;
            rule.SetProfiles(NET_FW_PROFILE2_ALL.0)
                .map_err(|error| windows_error("SetProfiles", error))?;
            rule.SetGrouping(&BSTR::from(RULE_GROUP))
                .map_err(|error| windows_error("SetGrouping", error))?;
            rule.SetProtocol(NET_FW_IP_PROTOCOL_ANY.0)
                .map_err(|error| windows_error("SetProtocol", error))?;
            rule.SetRemoteAddresses(&BSTR::from(spec.remote_addresses))
                .map_err(|error| windows_error("SetRemoteAddresses", error))?;
            rule.SetLocalUserAuthorizedList(&BSTR::from(local_user))
                .map_err(|error| windows_error("SetLocalUserAuthorizedList", error))?;
        }
        Ok(())
    }

    fn validate_rule(
        rule: &INetFwRule3,
        spec: &FirewallRuleSpec,
        account_sid: &str,
    ) -> io::Result<()> {
        let direction =
            unsafe { rule.Direction() }.map_err(|error| windows_error("Direction", error))?;
        let action = unsafe { rule.Action() }.map_err(|error| windows_error("Action", error))?;
        let enabled = unsafe { rule.Enabled() }.map_err(|error| windows_error("Enabled", error))?;
        let profiles =
            unsafe { rule.Profiles() }.map_err(|error| windows_error("Profiles", error))?;
        let protocol =
            unsafe { rule.Protocol() }.map_err(|error| windows_error("Protocol", error))?;
        let remote_addresses = unsafe { rule.RemoteAddresses() }
            .map_err(|error| windows_error("RemoteAddresses", error))?
            .to_string();
        let local_user = unsafe { rule.LocalUserAuthorizedList() }
            .map_err(|error| windows_error("LocalUserAuthorizedList", error))?
            .to_string();
        if direction != NET_FW_RULE_DIR_OUT
            || action != NET_FW_ACTION_BLOCK
            || enabled != VARIANT_TRUE
            || profiles != NET_FW_PROFILE2_ALL.0
            || protocol != NET_FW_IP_PROTOCOL_ANY.0
            || !same_csv_values(&remote_addresses, spec.remote_addresses)
            || !local_user
                .to_ascii_lowercase()
                .contains(&account_sid.to_ascii_lowercase())
        {
            return Err(other(format!(
                "Windows Firewall rule {} failed read-back validation: direction={direction:?}, action={action:?}, enabled={enabled:?}, profiles={profiles}, protocol={protocol}, remoteAddresses={remote_addresses:?}, localUserAuthorizedList={local_user:?}",
                spec.name,
            )));
        }
        Ok(())
    }

    fn account_sid_string(account: &str) -> io::Result<String> {
        let sid = crate::windows_setup::resolve_windows_account_sid(account)
            .map_err(|error| other(format!("resolve account SID failed: {error}")))?;
        let mut value = std::ptr::null_mut();
        if unsafe { ConvertSidToStringSidW(sid.as_ptr() as *mut c_void, &mut value) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut length = 0usize;
        unsafe {
            while *value.add(length) != 0 {
                length += 1;
            }
        }
        let result = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(value, length) });
        unsafe { LocalFree(value as HLOCAL) };
        if result.is_empty() {
            Err(other("resolved Windows account SID is empty"))
        } else {
            Ok(result)
        }
    }

    fn local_user_sddl(account_sid: &str) -> String {
        format!("O:LSD:(A;;CC;;;{account_sid})")
    }

    fn same_csv_values(actual: &str, expected: &str) -> bool {
        let normalize = |value: &str| {
            let mut values = value
                .split(',')
                .map(|part| canonical_remote_address(part.trim()))
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>();
            values.sort();
            values
        };
        normalize(actual) == normalize(expected)
    }

    fn canonical_remote_address(value: &str) -> String {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            // Windows Firewall COM read-back expands the IPv6 any-address
            // into an explicit range while preserving the same address set.
            "::-::" => "::".to_owned(),
            _ => value,
        }
    }

    fn windows_error(operation: &str, error: windows::core::Error) -> io::Error {
        other(format!("{operation} failed: {error}"))
    }

    fn other(message: impl Into<String>) -> io::Error {
        io::Error::other(message.into())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn csv_comparison_ignores_order_case_and_spacing() {
            assert!(same_csv_values(
                " ::/127, 127.0.0.0/8 ",
                "127.0.0.0/8,::/127"
            ));
        }

        #[test]
        fn csv_comparison_accepts_windows_ipv6_any_address_readback() {
            assert!(same_csv_values(
                "::-::, 0.0.0.0-126.255.255.255",
                "::,0.0.0.0-126.255.255.255"
            ));
        }

        #[test]
        fn offline_rules_cover_loopback_and_non_loopback_ranges() {
            assert_eq!(RULES.len(), 2);
            assert!(RULES
                .iter()
                .any(|rule| rule.remote_addresses == LOOPBACK_REMOTE_ADDRESSES));
            assert!(RULES
                .iter()
                .any(|rule| rule.remote_addresses == NON_LOOPBACK_REMOTE_ADDRESSES));
        }
    }
}

#[cfg(windows)]
pub use platform::{install_offline_rules, verify_offline_rules};

#[cfg(test)]
mod tests {
    #[cfg(not(windows))]
    #[test]
    fn non_windows_firewall_is_fail_closed() {
        let error = super::verify_offline_rules("unused")
            .expect_err("Windows Firewall must be unavailable off Windows");
        assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    }
}
