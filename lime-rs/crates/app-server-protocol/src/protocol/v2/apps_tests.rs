use super::*;
use crate::{JsonRpcNotification, RequestId};
use serde_json::json;

#[test]
fn app_requests_round_trip_exact_methods_and_camel_case_fields() {
    let requests = [
        ClientRequest::AppList {
            id: RequestId::Integer(1),
            params: AppsListParams {
                cursor: Some("1".to_string()),
                limit: Some(25),
                thread_id: Some("thread-1".to_string()),
                force_refetch: true,
            },
        },
        ClientRequest::AppRead {
            id: RequestId::Integer(2),
            params: AppsReadParams {
                app_ids: vec!["writer".to_string(), "reader".to_string()],
                include_tools: true,
            },
        },
        ClientRequest::AppInstalled {
            id: RequestId::Integer(3),
            params: AppsInstalledParams {
                thread_id: Some("thread-1".to_string()),
                force_refresh: true,
            },
        },
    ];

    for (request, method, expected_params) in [
        (
            &requests[0],
            METHOD_APP_LIST,
            json!({
                "cursor": "1",
                "limit": 25,
                "threadId": "thread-1",
                "forceRefetch": true
            }),
        ),
        (
            &requests[1],
            METHOD_APP_READ,
            json!({"appIds": ["writer", "reader"], "includeTools": true}),
        ),
        (
            &requests[2],
            METHOD_APP_INSTALLED,
            json!({"threadId": "thread-1", "forceRefresh": true}),
        ),
    ] {
        let value = serde_json::to_value(request).expect("serialize app request");
        assert_eq!(value["method"], method);
        assert_eq!(value["params"], expected_params);
        let decoded: ClientRequest = serde_json::from_value(value).expect("decode app request");
        assert_eq!(decoded.method().as_str(), method);
    }
}

#[test]
fn app_list_updated_notification_round_trips_typed_payload() {
    let notification = ServerNotification::AppListUpdated(AppListUpdatedNotification {
        data: vec![AppInfo {
            id: "writer".to_string(),
            name: "Writer".to_string(),
            description: Some("Write documents".to_string()),
            logo_url: None,
            logo_url_dark: None,
            icon_assets: None,
            icon_dark_assets: None,
            distribution_channel: Some("local".to_string()),
            branding: None,
            app_metadata: None,
            labels: None,
            install_url: None,
            is_accessible: true,
            is_enabled: true,
            plugin_display_names: vec!["writer-plugin".to_string()],
        }],
    });
    let jsonrpc: JsonRpcNotification = notification.clone().into();
    assert_eq!(jsonrpc.method, METHOD_APP_LIST_UPDATED);
    assert_eq!(
        jsonrpc.params,
        Some(json!({
            "data": [{
                "id": "writer",
                "name": "Writer",
                "description": "Write documents",
                "logoUrl": null,
                "logoUrlDark": null,
                "iconAssets": null,
                "iconDarkAssets": null,
                "distributionChannel": "local",
                "branding": null,
                "appMetadata": null,
                "labels": null,
                "installUrl": null,
                "isAccessible": true,
                "isEnabled": true,
                "pluginDisplayNames": ["writer-plugin"]
            }]
        }))
    );
    assert_eq!(
        ServerNotification::try_from(jsonrpc).expect("decode app update"),
        notification
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_APP_LIST_UPDATED));
}

#[test]
fn app_response_payloads_lower_to_jsonrpc_results() {
    let response = ClientResponsePayload::AppList(AppsListResponse {
        data: Vec::new(),
        next_cursor: Some("2".to_string()),
    })
    .into_response(RequestId::Integer(4))
    .expect("lower app/list response");
    let value = serde_json::to_value(response).expect("serialize app/list response");
    assert_eq!(
        value,
        json!({"id": 4, "result": {"data": [], "nextCursor": "2"}})
    );
}
