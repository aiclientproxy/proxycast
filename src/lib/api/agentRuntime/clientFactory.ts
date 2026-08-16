import { createAgentClient } from "./agentClient";
import {
  createExportClient,
  type AgentRuntimeExportAppServerClient,
} from "./exportClient";
import {
  createInventoryClient,
  type AgentRuntimeWorkspaceSkillBindingsAppServerClient,
} from "./inventoryClient";
import type { AppServerSessionRpcClient } from "./appServerSessionClient";
import { createSessionClient } from "./sessionClient";
import {
  createThreadClient,
  type AgentRuntimeThreadClientDeps,
} from "./threadClient";
import {
  createAgentRuntimeBridgeInvoke,
  createAgentRuntimeCommandInvoke,
  type AgentRuntimeBridgeInvoke,
  type AgentRuntimeCommandInvoke,
  type AgentRuntimeTransportDeps,
} from "./transport";

export type AgentRuntimeAppServerClient =
  AgentRuntimeThreadClientDeps["appServerClient"] &
    AppServerSessionRpcClient &
    AgentRuntimeExportAppServerClient &
    AgentRuntimeWorkspaceSkillBindingsAppServerClient;

export interface AgentRuntimeClientDeps extends AgentRuntimeTransportDeps {
  bridgeInvoke?: AgentRuntimeBridgeInvoke;
  invokeCommand?: AgentRuntimeCommandInvoke;
  appServerClient?: AgentRuntimeAppServerClient;
  standardRuntimeClient?: AgentRuntimeThreadClientDeps["standardRuntimeClient"];
  isAppServerTurnLifecycleAvailable?: AgentRuntimeThreadClientDeps["isAppServerTurnLifecycleAvailable"];
}

export function createAgentRuntimeClient({
  appServerClient,
  bridgeInvoke,
  invoke,
  invokeCommand,
  standardRuntimeClient,
  isAppServerTurnLifecycleAvailable,
}: AgentRuntimeClientDeps = {}) {
  const resolvedBridgeInvoke =
    bridgeInvoke ?? createAgentRuntimeBridgeInvoke({ invoke });
  const resolvedInvokeCommand =
    invokeCommand ??
    createAgentRuntimeCommandInvoke({
      bridgeInvoke: resolvedBridgeInvoke,
    });

  return {
    ...createAgentClient({ bridgeInvoke: resolvedBridgeInvoke }),
    ...createExportClient({
      appServerClient,
    }),
    ...createInventoryClient({
      appServerClient,
    }),
    ...createSessionClient({
      appServerClient,
    }),
    ...createThreadClient({
      appServerClient,
      invokeCommand: resolvedInvokeCommand,
      standardRuntimeClient,
      isAppServerTurnLifecycleAvailable,
    }),
  };
}

export type AgentRuntimeClient = ReturnType<typeof createAgentRuntimeClient>;
