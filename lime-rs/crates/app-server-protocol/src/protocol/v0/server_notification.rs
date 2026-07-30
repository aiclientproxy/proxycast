use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::JsonRpcNotification;

use super::{
    AgentSessionEventParams, AppServerNotificationMethod, WorkspaceRightSurfacePendingChangedParams,
};

macro_rules! app_server_server_notification_definitions {
    ($($variant:ident => $wire:literal ($payload:ty)),* $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
        #[serde(tag = "method", content = "params")]
        pub enum ServerNotification {
            $(
                #[serde(rename = $wire)]
                $variant($payload),
            )*
        }

        impl ServerNotification {
            pub fn method(&self) -> AppServerNotificationMethod {
                match self {
                    $(Self::$variant(_) => AppServerNotificationMethod::$variant,)*
                }
            }
        }

        impl TryFrom<JsonRpcNotification> for ServerNotification {
            type Error = String;

            fn try_from(notification: JsonRpcNotification) -> Result<Self, Self::Error> {
                match AppServerNotificationMethod::parse(&notification.method) {
                    $(
                        Some(AppServerNotificationMethod::$variant) => {
                            serde_json::from_value(
                                notification.params.unwrap_or_else(|| serde_json::json!({})),
                            )
                            .map(Self::$variant)
                            .map_err(|error| error.to_string())
                        }
                    )*
                    Some(method) => Err(format!(
                        "notification is not a server notification: {}",
                        method.as_str()
                    )),
                    None => Err(format!("unknown notification method: {}", notification.method)),
                }
            }
        }

        impl From<ServerNotification> for JsonRpcNotification {
            fn from(notification: ServerNotification) -> Self {
                match notification {
                    $(
                        ServerNotification::$variant(params) => JsonRpcNotification::new(
                            $wire,
                            Some(serde_json::to_value(params).expect("serialize app-server notification")),
                        ),
                    )*
                }
            }
        }
    };
}

app_server_server_notification_definitions! {
    AgentSessionEvent => "agentSession/event" (AgentSessionEventParams),
    WorkspaceRightSurfacePendingChanged => "workspaceRightSurface/pendingChanged" (
        WorkspaceRightSurfacePendingChangedParams
    ),
}
