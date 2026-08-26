use std::io;

#[cfg(not(windows))]
pub fn install_filters(_account: &str) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows Filtering Platform is only available on Windows",
    ))
}

#[cfg(not(windows))]
pub fn verify_filters(_account: &str) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows Filtering Platform is only available on Windows",
    ))
}

#[cfg(windows)]
mod platform {
    use super::io;
    use sha2::{Digest, Sha256};
    use std::ffi::{c_void, OsStr};
    use std::mem::zeroed;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::{null, null_mut};
    use windows_sys::core::GUID;
    use windows_sys::Win32::Foundation::{
        LocalFree, FWP_E_ALREADY_EXISTS, FWP_E_FILTER_NOT_FOUND, FWP_E_NOT_FOUND, HANDLE, HLOCAL,
    };
    use windows_sys::Win32::NetworkManagement::WindowsFilteringPlatform::{
        FwpmEngineClose0, FwpmEngineOpen0, FwpmFilterAdd0, FwpmFilterDeleteByKey0,
        FwpmFilterGetByKey0, FwpmFreeMemory0, FwpmProviderAdd0, FwpmSubLayerAdd0,
        FwpmTransactionAbort0, FwpmTransactionBegin0, FwpmTransactionCommit0, FWPM_ACTION0,
        FWPM_ACTION0_0, FWPM_CONDITION_ALE_USER_ID, FWPM_CONDITION_IP_PROTOCOL,
        FWPM_CONDITION_IP_REMOTE_PORT, FWPM_DISPLAY_DATA0, FWPM_FILTER0, FWPM_FILTER0_0,
        FWPM_FILTER_CONDITION0, FWPM_FILTER_FLAG_PERSISTENT, FWPM_LAYER_ALE_AUTH_CONNECT_V4,
        FWPM_LAYER_ALE_AUTH_CONNECT_V6, FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V4,
        FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V6, FWPM_PROVIDER0, FWPM_PROVIDER_FLAG_PERSISTENT,
        FWPM_SESSION0, FWPM_SUBLAYER0, FWPM_SUBLAYER_FLAG_PERSISTENT, FWP_ACTION_BLOCK,
        FWP_ACTRL_MATCH_FILTER, FWP_BYTE_BLOB, FWP_CONDITION_VALUE0, FWP_CONDITION_VALUE0_0,
        FWP_EMPTY, FWP_MATCH_EQUAL, FWP_SECURITY_DESCRIPTOR_TYPE, FWP_UINT16, FWP_UINT8,
        FWP_VALUE0,
    };
    use windows_sys::Win32::Security::Authorization::{
        BuildExplicitAccessWithNameW, BuildSecurityDescriptorW, EXPLICIT_ACCESS_W, GRANT_ACCESS,
    };
    use windows_sys::Win32::Security::PSECURITY_DESCRIPTOR;
    use windows_sys::Win32::System::Rpc::RPC_C_AUTHN_DEFAULT;
    use windows_sys::Win32::System::Threading::INFINITE;

    const SESSION_NAME: &str = "Lime Windows Sandbox WFP";
    const PROVIDER_NAME: &str = "Lime Windows Sandbox WFP";
    const PROVIDER_DESCRIPTION: &str = "Persistent WFP provider for Lime restricted execution";
    const SUBLAYER_NAME: &str = "Lime Windows Sandbox WFP";
    const SUBLAYER_DESCRIPTION: &str = "Persistent per-account network deny filters";

    // These identities are Lime-owned and stable across upgrades. They are not
    // reused from the upstream Codex installation to avoid cross-product drift.
    pub(super) const PROVIDER_KEY: GUID = GUID::from_u128(0x3e0a7d6c_8c49_4f98_b8be_3e4a7a5b0a21);
    const SUBLAYER_KEY: GUID = GUID::from_u128(0x7b694c24_5e23_49f2_b7f9_8e8fdc8fb7e2);

    #[derive(Clone, Copy)]
    enum ConditionSpec {
        User,
        Protocol(u8),
        RemotePort(u16),
    }

    #[derive(Clone, Copy)]
    struct FilterSpec {
        key: GUID,
        name: &'static str,
        description: &'static str,
        layer_key: GUID,
        conditions: &'static [ConditionSpec],
    }

    const USER_ICMP_V4: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::Protocol(1)];
    const USER_ICMP_V6: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::Protocol(58)];
    const USER_DNS: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::RemotePort(53)];
    const USER_DNS_TLS: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::RemotePort(853)];
    const USER_SMB: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::RemotePort(445)];
    const USER_NETBIOS: &[ConditionSpec] = &[ConditionSpec::User, ConditionSpec::RemotePort(139)];

    const FILTER_SPECS: &[FilterSpec] = &[
        FilterSpec {
            key: GUID::from_u128(0xd3792b9a_2f8d_4d54_a401_8d7b92589e02),
            name: "lime_wfp_icmp_connect_v4",
            description: "Block sandbox-account ICMP connect v4",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V4,
            conditions: USER_ICMP_V4,
        },
        FilterSpec {
            key: GUID::from_u128(0x9578bc1f_3c6d_4a92_aa89_43d70287ad30),
            name: "lime_wfp_icmp_connect_v6",
            description: "Block sandbox-account ICMP connect v6",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V6,
            conditions: USER_ICMP_V6,
        },
        FilterSpec {
            key: GUID::from_u128(0xf4edc0f0_8d46_47ef_a12e_0f55fa1a3e6f),
            name: "lime_wfp_icmp_assign_v4",
            description: "Block sandbox-account ICMP resource assignment v4",
            layer_key: FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V4,
            conditions: USER_ICMP_V4,
        },
        FilterSpec {
            key: GUID::from_u128(0xcb7d0d10_b8b1_43d0_8df4_9a9224b4d4c8),
            name: "lime_wfp_icmp_assign_v6",
            description: "Block sandbox-account ICMP resource assignment v6",
            layer_key: FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V6,
            conditions: USER_ICMP_V6,
        },
        FilterSpec {
            key: GUID::from_u128(0x5b3ca0d7_2f5d_4e1c_9a68_79f43a24ae75),
            name: "lime_wfp_dns_53_v4",
            description: "Block sandbox-account DNS port 53 v4",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V4,
            conditions: USER_DNS,
        },
        FilterSpec {
            key: GUID::from_u128(0x6f30fb88_7db5_4d6c_97f8_a20567c31bc8),
            name: "lime_wfp_dns_53_v6",
            description: "Block sandbox-account DNS port 53 v6",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V6,
            conditions: USER_DNS,
        },
        FilterSpec {
            key: GUID::from_u128(0x1f978f67_0cd3_4471_89d9_24a465b1d6f4),
            name: "lime_wfp_dns_853_v4",
            description: "Block sandbox-account DNS-over-TLS port 853 v4",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V4,
            conditions: USER_DNS_TLS,
        },
        FilterSpec {
            key: GUID::from_u128(0x9db3d7fc_90ed_44c4_972c_91cb2acb1d5e),
            name: "lime_wfp_dns_853_v6",
            description: "Block sandbox-account DNS-over-TLS port 853 v6",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V6,
            conditions: USER_DNS_TLS,
        },
        FilterSpec {
            key: GUID::from_u128(0x50a63dd2_6c52_4130_8dd7_0c4efb3c21c3),
            name: "lime_wfp_smb_445_v4",
            description: "Block sandbox-account SMB port 445 v4",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V4,
            conditions: USER_SMB,
        },
        FilterSpec {
            key: GUID::from_u128(0x66a51abf_07d7_427c_b3f7_8cf9d8ef4c01),
            name: "lime_wfp_smb_445_v6",
            description: "Block sandbox-account SMB port 445 v6",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V6,
            conditions: USER_SMB,
        },
        FilterSpec {
            key: GUID::from_u128(0x2b991ad5_6e29_45f8_b76f_4f7a3695d802),
            name: "lime_wfp_netbios_139_v4",
            description: "Block sandbox-account NetBIOS port 139 v4",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V4,
            conditions: USER_NETBIOS,
        },
        FilterSpec {
            key: GUID::from_u128(0xa1ab0f6d_0d66_45fb_8471_1dfbaf7e0529),
            name: "lime_wfp_netbios_139_v6",
            description: "Block sandbox-account NetBIOS port 139 v6",
            layer_key: FWPM_LAYER_ALE_AUTH_CONNECT_V6,
            conditions: USER_NETBIOS,
        },
    ];

    pub fn install_filters(account: &str) -> io::Result<usize> {
        let engine = Engine::open()?;
        let mut transaction = engine.begin_transaction()?;
        ensure_provider(engine.handle)?;
        ensure_sublayer(engine.handle)?;
        let user_condition = UserMatchCondition::for_account(account)?;
        for spec in FILTER_SPECS {
            let filter_key = filter_key_for_account(spec, account);
            delete_filter_if_present(engine.handle, &filter_key)?;
            add_filter(engine.handle, spec, &filter_key, &user_condition)?;
        }
        transaction.commit()?;
        Ok(FILTER_SPECS.len())
    }

    pub fn verify_filters(account: &str) -> io::Result<()> {
        let engine = Engine::open()?;
        let expected_user = UserMatchCondition::for_account(account)?;
        for spec in FILTER_SPECS {
            let filter_key = filter_key_for_account(spec, account);
            let mut filter = null_mut();
            let result = unsafe { FwpmFilterGetByKey0(engine.handle, &filter_key, &mut filter) };
            ensure_success(result, &format!("FwpmFilterGetByKey0({})", spec.name))?;
            let validation = unsafe { validate_filter(filter, spec, &expected_user) };
            unsafe { free_filter(filter) };
            validation?;
        }
        Ok(())
    }

    struct Engine {
        handle: HANDLE,
    }

    impl Engine {
        fn open() -> io::Result<Self> {
            let name = to_wide(SESSION_NAME);
            let mut session: FWPM_SESSION0 = unsafe { zeroed() };
            session.displayData = FWPM_DISPLAY_DATA0 {
                name: name.as_ptr() as *mut _,
                description: null_mut(),
            };
            session.txnWaitTimeoutInMSec = INFINITE;
            let mut handle = 0;
            let result = unsafe {
                FwpmEngineOpen0(
                    null(),
                    RPC_C_AUTHN_DEFAULT as u32,
                    null(),
                    &session,
                    &mut handle,
                )
            };
            ensure_success(result, "FwpmEngineOpen0")?;
            Ok(Self { handle })
        }

        fn begin_transaction(&self) -> io::Result<Transaction<'_>> {
            let result = unsafe { FwpmTransactionBegin0(self.handle, 0) };
            ensure_success(result, "FwpmTransactionBegin0")?;
            Ok(Transaction {
                engine: self,
                committed: false,
            })
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe { FwpmEngineClose0(self.handle) };
        }
    }

    struct Transaction<'a> {
        engine: &'a Engine,
        committed: bool,
    }

    impl Transaction<'_> {
        fn commit(&mut self) -> io::Result<()> {
            let result = unsafe { FwpmTransactionCommit0(self.engine.handle) };
            ensure_success(result, "FwpmTransactionCommit0")?;
            self.committed = true;
            Ok(())
        }
    }

    impl Drop for Transaction<'_> {
        fn drop(&mut self) {
            if !self.committed {
                unsafe { FwpmTransactionAbort0(self.engine.handle) };
            }
        }
    }

    struct UserMatchCondition {
        security_descriptor: PSECURITY_DESCRIPTOR,
        blob: FWP_BYTE_BLOB,
    }

    impl UserMatchCondition {
        fn for_account(account: &str) -> io::Result<Self> {
            let account_w = to_wide(account);
            let mut access: EXPLICIT_ACCESS_W = unsafe { zeroed() };
            unsafe {
                BuildExplicitAccessWithNameW(
                    &mut access,
                    account_w.as_ptr(),
                    FWP_ACTRL_MATCH_FILTER,
                    GRANT_ACCESS,
                    0,
                );
            }
            let mut descriptor = null_mut();
            let mut descriptor_len = 0;
            let result = unsafe {
                BuildSecurityDescriptorW(
                    null(),
                    null(),
                    1,
                    &access,
                    0,
                    null(),
                    null_mut(),
                    &mut descriptor_len,
                    &mut descriptor,
                )
            };
            ensure_success(result, "BuildSecurityDescriptorW")?;
            Ok(Self {
                security_descriptor: descriptor,
                blob: FWP_BYTE_BLOB {
                    size: descriptor_len,
                    data: descriptor as *mut u8,
                },
            })
        }
    }

    impl Drop for UserMatchCondition {
        fn drop(&mut self) {
            if !self.security_descriptor.is_null() {
                unsafe { LocalFree(self.security_descriptor as HLOCAL) };
            }
        }
    }

    fn ensure_provider(engine: HANDLE) -> io::Result<()> {
        let name = to_wide(PROVIDER_NAME);
        let description = to_wide(PROVIDER_DESCRIPTION);
        let provider = FWPM_PROVIDER0 {
            providerKey: PROVIDER_KEY,
            displayData: FWPM_DISPLAY_DATA0 {
                name: name.as_ptr() as *mut _,
                description: description.as_ptr() as *mut _,
            },
            flags: FWPM_PROVIDER_FLAG_PERSISTENT,
            providerData: empty_blob(),
            serviceName: null_mut(),
        };
        let result = unsafe { FwpmProviderAdd0(engine, &provider, null_mut()) };
        ensure_success_or(result, "FwpmProviderAdd0", &[FWP_E_ALREADY_EXISTS as u32])
    }

    fn ensure_sublayer(engine: HANDLE) -> io::Result<()> {
        let name = to_wide(SUBLAYER_NAME);
        let description = to_wide(SUBLAYER_DESCRIPTION);
        let provider_key = PROVIDER_KEY;
        let sublayer = FWPM_SUBLAYER0 {
            subLayerKey: SUBLAYER_KEY,
            displayData: FWPM_DISPLAY_DATA0 {
                name: name.as_ptr() as *mut _,
                description: description.as_ptr() as *mut _,
            },
            flags: FWPM_SUBLAYER_FLAG_PERSISTENT,
            providerKey: &provider_key as *const _ as *mut _,
            providerData: empty_blob(),
            weight: 0x8000,
        };
        let result = unsafe { FwpmSubLayerAdd0(engine, &sublayer, null_mut()) };
        ensure_success_or(result, "FwpmSubLayerAdd0", &[FWP_E_ALREADY_EXISTS as u32])
    }

    fn add_filter(
        engine: HANDLE,
        spec: &FilterSpec,
        filter_key: &GUID,
        user_condition: &UserMatchCondition,
    ) -> io::Result<()> {
        let name = to_wide(spec.name);
        let description = to_wide(spec.description);
        let mut conditions = build_conditions(spec.conditions, user_condition);
        let provider_key = PROVIDER_KEY;
        let filter = FWPM_FILTER0 {
            filterKey: *filter_key,
            displayData: FWPM_DISPLAY_DATA0 {
                name: name.as_ptr() as *mut _,
                description: description.as_ptr() as *mut _,
            },
            flags: FWPM_FILTER_FLAG_PERSISTENT,
            providerKey: &provider_key as *const _ as *mut _,
            providerData: empty_blob(),
            layerKey: spec.layer_key,
            subLayerKey: SUBLAYER_KEY,
            weight: empty_value(),
            numFilterConditions: conditions.len() as u32,
            filterCondition: conditions.as_mut_ptr(),
            action: FWPM_ACTION0 {
                r#type: FWP_ACTION_BLOCK,
                Anonymous: FWPM_ACTION0_0 {
                    filterType: zero_guid(),
                },
            },
            Anonymous: FWPM_FILTER0_0 { rawContext: 0 },
            reserved: null_mut(),
            filterId: 0,
            effectiveWeight: empty_value(),
        };
        let mut filter_id = 0u64;
        let result = unsafe { FwpmFilterAdd0(engine, &filter, null_mut(), &mut filter_id) };
        ensure_success(result, &format!("FwpmFilterAdd0({})", spec.name))
    }

    fn filter_key_for_account(spec: &FilterSpec, account: &str) -> GUID {
        let mut hasher = Sha256::new();
        hasher.update(PROVIDER_KEY.data1.to_le_bytes());
        hasher.update(PROVIDER_KEY.data2.to_le_bytes());
        hasher.update(PROVIDER_KEY.data3.to_le_bytes());
        hasher.update(PROVIDER_KEY.data4);
        hasher.update(spec.key.data1.to_le_bytes());
        hasher.update(spec.key.data2.to_le_bytes());
        hasher.update(spec.key.data3.to_le_bytes());
        hasher.update(spec.key.data4);
        hasher.update(account.as_bytes());
        let digest = hasher.finalize();
        let mut data4 = [0u8; 8];
        data4.copy_from_slice(&digest[8..16]);
        GUID {
            data1: u32::from_le_bytes(digest[0..4].try_into().expect("digest segment")),
            data2: u16::from_le_bytes(digest[4..6].try_into().expect("digest segment")),
            data3: u16::from_le_bytes(digest[6..8].try_into().expect("digest segment")),
            data4,
        }
    }

    fn build_conditions(
        specs: &[ConditionSpec],
        user_condition: &UserMatchCondition,
    ) -> Vec<FWPM_FILTER_CONDITION0> {
        specs
            .iter()
            .map(|spec| match spec {
                ConditionSpec::User => FWPM_FILTER_CONDITION0 {
                    fieldKey: FWPM_CONDITION_ALE_USER_ID,
                    matchType: FWP_MATCH_EQUAL,
                    conditionValue: FWP_CONDITION_VALUE0 {
                        r#type: FWP_SECURITY_DESCRIPTOR_TYPE,
                        Anonymous: FWP_CONDITION_VALUE0_0 {
                            sd: &user_condition.blob as *const _ as *mut _,
                        },
                    },
                },
                ConditionSpec::Protocol(protocol) => FWPM_FILTER_CONDITION0 {
                    fieldKey: FWPM_CONDITION_IP_PROTOCOL,
                    matchType: FWP_MATCH_EQUAL,
                    conditionValue: FWP_CONDITION_VALUE0 {
                        r#type: FWP_UINT8,
                        Anonymous: FWP_CONDITION_VALUE0_0 { uint8: *protocol },
                    },
                },
                ConditionSpec::RemotePort(port) => FWPM_FILTER_CONDITION0 {
                    fieldKey: FWPM_CONDITION_IP_REMOTE_PORT,
                    matchType: FWP_MATCH_EQUAL,
                    conditionValue: FWP_CONDITION_VALUE0 {
                        r#type: FWP_UINT16,
                        Anonymous: FWP_CONDITION_VALUE0_0 { uint16: *port },
                    },
                },
            })
            .collect()
    }

    unsafe fn validate_filter(
        filter: *mut FWPM_FILTER0,
        spec: &FilterSpec,
        expected_user: &UserMatchCondition,
    ) -> io::Result<()> {
        if filter.is_null() {
            return Err(other("WFP returned a null filter"));
        }
        let filter = &*filter;
        if filter.providerKey.is_null()
            || !guid_equal(&*filter.providerKey, &PROVIDER_KEY)
            || !guid_equal(&filter.subLayerKey, &SUBLAYER_KEY)
            || !guid_equal(&filter.layerKey, &spec.layer_key)
            || filter.action.r#type != FWP_ACTION_BLOCK
            || filter.numFilterConditions != spec.conditions.len() as u32
        {
            return Err(other(&format!(
                "WFP filter {} has unexpected metadata",
                spec.name
            )));
        }
        if filter.numFilterConditions > 0 && filter.filterCondition.is_null() {
            return Err(other(&format!(
                "WFP filter {} returned null conditions",
                spec.name
            )));
        }
        let conditions =
            std::slice::from_raw_parts(filter.filterCondition, filter.numFilterConditions as usize);
        let expected = build_conditions(spec.conditions, expected_user);
        for (actual, expected) in conditions.iter().zip(expected.iter()) {
            if !guid_equal(&actual.fieldKey, &expected.fieldKey)
                || actual.matchType != expected.matchType
                || !condition_equal(actual, expected)
            {
                return Err(other(&format!(
                    "WFP filter {} has unexpected condition",
                    spec.name
                )));
            }
        }
        Ok(())
    }

    unsafe fn condition_equal(
        actual: &FWPM_FILTER_CONDITION0,
        expected: &FWPM_FILTER_CONDITION0,
    ) -> bool {
        if actual.conditionValue.r#type != expected.conditionValue.r#type {
            return false;
        }
        match actual.conditionValue.r#type {
            FWP_SECURITY_DESCRIPTOR_TYPE => {
                let actual_blob = actual.conditionValue.Anonymous.sd;
                let expected_blob = expected.conditionValue.Anonymous.sd;
                if actual_blob.is_null() || expected_blob.is_null() {
                    return false;
                }
                let actual_blob = &*actual_blob;
                let expected_blob = &*expected_blob;
                actual_blob.size == expected_blob.size
                    && !actual_blob.data.is_null()
                    && !expected_blob.data.is_null()
                    && std::slice::from_raw_parts(actual_blob.data, actual_blob.size as usize)
                        == std::slice::from_raw_parts(
                            expected_blob.data,
                            expected_blob.size as usize,
                        )
            }
            FWP_UINT8 => {
                actual.conditionValue.Anonymous.uint8 == expected.conditionValue.Anonymous.uint8
            }
            FWP_UINT16 => {
                actual.conditionValue.Anonymous.uint16 == expected.conditionValue.Anonymous.uint16
            }
            _ => false,
        }
    }

    unsafe fn free_filter(filter: *mut FWPM_FILTER0) {
        if !filter.is_null() {
            let mut memory = filter as *mut c_void;
            FwpmFreeMemory0(&mut memory);
        }
    }

    fn delete_filter_if_present(engine: HANDLE, key: &GUID) -> io::Result<()> {
        let result = unsafe { FwpmFilterDeleteByKey0(engine, key) };
        ensure_success_or(
            result,
            "FwpmFilterDeleteByKey0",
            &[FWP_E_FILTER_NOT_FOUND as u32, FWP_E_NOT_FOUND as u32],
        )
    }

    fn ensure_success(result: u32, operation: &str) -> io::Result<()> {
        ensure_success_or(result, operation, &[])
    }

    fn ensure_success_or(result: u32, operation: &str, allowed: &[u32]) -> io::Result<()> {
        if result == 0 || allowed.contains(&result) {
            Ok(())
        } else {
            Err(other(&format!("{operation} failed: 0x{result:08X}")))
        }
    }

    fn to_wide(value: &str) -> Vec<u16> {
        OsStr::new(value)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    fn empty_blob() -> FWP_BYTE_BLOB {
        FWP_BYTE_BLOB {
            size: 0,
            data: null_mut(),
        }
    }

    fn empty_value() -> FWP_VALUE0 {
        FWP_VALUE0 {
            r#type: FWP_EMPTY,
            Anonymous: unsafe { zeroed() },
        }
    }

    fn zero_guid() -> GUID {
        GUID::from_u128(0)
    }

    fn guid_equal(left: &GUID, right: &GUID) -> bool {
        left.data1 == right.data1
            && left.data2 == right.data2
            && left.data3 == right.data3
            && left.data4 == right.data4
    }

    fn other(message: &str) -> io::Error {
        io::Error::new(io::ErrorKind::Other, message.to_string())
    }
}

#[cfg(windows)]
pub use platform::{install_filters, verify_filters};

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    #[test]
    fn provider_identity_is_stable() {
        assert_ne!(super::platform::PROVIDER_KEY.data1, 0);
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_wfp_is_fail_closed() {
        let error = super::verify_filters("unused").expect_err("WFP must be unavailable");
        assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    }
}
