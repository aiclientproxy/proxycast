use code_mode_protocol::RuntimeToolDefinition;
use tool_runtime::gateway_dispatch_execution::RuntimeGatewayToolExecutionRegistration;
use tool_runtime::native_overlay::{
    runtime_native_tool_definition, runtime_native_tool_install_plan,
    runtime_native_tool_registration_is_allowed,
};

pub(crate) struct NativeRegistration {
    definition: RuntimeToolDefinition,
    gateway_execution: RuntimeGatewayToolExecutionRegistration,
}

impl NativeRegistration {
    pub(crate) fn gateway(registration: RuntimeGatewayToolExecutionRegistration) -> Self {
        Self {
            definition: registration.definition(),
            gateway_execution: registration,
        }
    }

    pub(crate) fn definition(&self) -> RuntimeToolDefinition {
        self.definition.clone()
    }

    pub(crate) fn into_gateway_execution(self) -> RuntimeGatewayToolExecutionRegistration {
        self.gateway_execution
    }
}

pub(crate) fn current_native_tool_definitions() -> Vec<RuntimeToolDefinition> {
    let mut definitions = runtime_native_tool_install_plan()
        .iter()
        .map(|step| runtime_native_tool_definition(step.tool()))
        .filter(|definition| runtime_native_tool_registration_is_allowed(&definition.name))
        .collect::<Vec<_>>();
    definitions.extend(tool_runtime::unified_exec::unified_exec_tool_definitions());
    definitions
}
