pub mod v0;
pub mod v2;

pub fn app_server_method_catalog() -> Vec<v0::AppServerMethodSpec> {
    let mut catalog = v0::APP_SERVER_METHODS.to_vec();
    catalog.extend(v2::METHODS.iter().map(|method| v0::AppServerMethodSpec {
        method: *method,
        kind: v0::AppServerMethodKind::Request,
    }));
    catalog.extend(
        v2::NOTIFICATION_METHODS
            .iter()
            .map(|method| v0::AppServerMethodSpec {
                method: *method,
                kind: v0::AppServerMethodKind::Notification,
            }),
    );
    catalog.extend(
        v2::SERVER_REQUEST_METHODS
            .iter()
            .map(|method| v0::AppServerMethodSpec {
                method: *method,
                kind: v0::AppServerMethodKind::ServerRequest,
            }),
    );

    let mut methods = std::collections::BTreeSet::new();
    for spec in &catalog {
        assert!(
            methods.insert(spec.method),
            "duplicate App Server method `{}` in the central catalog",
            spec.method
        );
    }
    catalog
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_method_catalog_has_one_owner_per_wire_method() {
        let catalog = app_server_method_catalog();

        assert_eq!(
            catalog
                .iter()
                .filter(|spec| spec.kind == v0::AppServerMethodKind::Request)
                .count(),
            v0::APP_SERVER_METHODS
                .iter()
                .filter(|spec| spec.kind == v0::AppServerMethodKind::Request)
                .count()
                + v2::METHODS.len()
        );
        assert_eq!(
            catalog
                .iter()
                .find(|spec| spec.method == v2::METHOD_MCP_SERVER_ELICITATION_REQUEST)
                .map(|spec| spec.kind),
            Some(v0::AppServerMethodKind::ServerRequest)
        );
    }

    #[test]
    fn central_method_catalog_includes_all_mcp_lifecycle_notifications() {
        let catalog = app_server_method_catalog();
        for method in [
            v2::METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED,
            v2::METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED,
        ] {
            assert_eq!(
                catalog
                    .iter()
                    .find(|spec| spec.method == method)
                    .map(|spec| spec.kind),
                Some(v0::AppServerMethodKind::Notification),
                "missing MCP lifecycle notification {method}"
            );
        }
    }
}
