#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import {
  checkMcpRuntimeCurrentContracts,
  checkWorkspaceRightSurfaceCurrentContracts,
} from "./mcp/lib/contract-guards.mjs";

const repoRoot = process.cwd();
function normalizeContractSnippet(value) {
  return value
    .replace(/\b(?:protocol|appServer|constants)\./gu, "")
    .replace(/(\w+)\s*:\s*([A-Za-z0-9_<>,\[\]\s|&]+)\s*=\s*\{\}/gu, "$1?: $2")
    .replace(/\basync\s+(?=[A-Za-z_$][\w$]*\()/gu, "")
    .replace(/\.await\b/gu, "")
    .replace(/\.boxed\(\)/gu, "")
    .replace(/\bready\(/gu, "")
    .replace(/,\s*\)/gu, ")")
    .replace(/\s+/gu, "");
}

function contractContentIncludes(content, snippet) {
  if (content.includes(snippet)) {
    return true;
  }
  const normalizedContent = normalizeContractSnippet(content);
  const normalizedSnippet = normalizeContractSnippet(snippet);
  if (normalizedContent.includes(normalizedSnippet)) {
    return true;
  }

  const importedTypeSnippet = snippet.match(/^type\s+([A-Za-z0-9_]+),$/u);
  if (importedTypeSnippet) {
    return normalizedContent.includes(
      normalizeContractSnippet(`protocol.${importedTypeSnippet[1]}`),
    );
  }

  const dynamicClientCall = snippet.match(
    /^this\.client\.([A-Za-z0-9_]+)\(params\)$/u,
  );
  if (dynamicClientCall) {
    const clientMethodName = dynamicClientCall[1];
    return (
      (content.includes(`clientMethod: "${clientMethodName}"`) ||
        content.includes(`name: "${clientMethodName}"`)) &&
      content.includes("APP_SERVER_REQUEST_CLIENT_METHODS") &&
      content.includes("client[spec.clientMethod](...clientArgs)")
    );
  }

  return false;
}
const appServerClientIndexFile = "packages/app-server-client/src/index.ts";
const appServerGeneratedProtocolFile =
  "packages/app-server-client/src/generated/protocol-types.ts";
const appServerClientSplitSourceFiles = [
  appServerClientIndexFile,
  "packages/app-server-client/src/request-client.ts",
  "packages/app-server-client/src/request-client-methods.ts",
  "packages/app-server-client/src/connection.ts",
  "packages/app-server-client/src/connection-methods.ts",
  "packages/app-server-client/src/sidecar.ts",
  "packages/app-server-client/src/sidecar-types.ts",
  "packages/app-server-client/src/sidecar-manifest.ts",
  "packages/app-server-client/src/sidecar-process.ts",
  "packages/app-server-client/src/sidecar-lifecycle.ts",
  "packages/app-server-client/src/agent-runtime.ts",
];
const rendererAppServerIndexFile = "src/lib/api/appServer.ts";
const rendererAppServerSplitSourceFiles = [
  rendererAppServerIndexFile,
  "src/lib/api/appServerConstants.ts",
  "src/lib/api/appServerTypes.ts",
  "src/lib/api/appServerTransport.ts",
  "src/lib/api/appServerResponse.ts",
  "src/lib/api/appServerClient.ts",
  "src/lib/api/appServerClientMethods.ts",
  "src/lib/api/appServerClientMethodSpecs.ts",
];
const rendererQueuedTurnWriteSurfaceRoots = [
  "src/components/agent/chat",
  "src/lib/api/agentRuntime",
];
const rendererQueuedTurnWriteSurfaceForbiddenSnippets = [
  '"agentSession/queuedTurn/promote"',
  '"agentSession/queuedTurn/remove"',
  "promoteAgentRuntimeQueuedTurn",
  "removeAgentRuntimeQueuedTurn",
  "promoteAgentSessionQueuedTurn",
  "removeAgentSessionQueuedTurn",
  "onPromoteQueuedTurn",
  "onRemoveQueuedTurn",
  "QueuedTurnSnapshot",
  "normalizeQueuedTurnSnapshots",
  "queued_turns",
];
const retiredAgentStreamToolMessageSynthesisFiles = [
  "src/components/agent/chat/hooks/agentStreamToolItemMessageSync.ts",
  "src/components/agent/chat/hooks/agentStreamToolItemMessageSync.unit.test.ts",
];
const retiredAgentStreamToolMessageSynthesisSnippets = [
  "syncMessageToolCallFromThreadItem",
  "toolCallStateFromThreadItem",
  "mergeToolCallStateFromItem",
  "agentStreamToolItemMessageSync",
];
const retiredPublicQueuedTurnSurfaceSpecs = [
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/agent_session.rs",
    snippets: [
      "AgentSessionQueuedTurnRemoveParams",
      "AgentSessionQueuedTurnRemoveResponse",
      "AgentSessionQueuedTurnPromoteParams",
      "AgentSessionQueuedTurnPromoteResponse",
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
    snippets: [
      "AGENT_SESSION_QUEUED_TURN_REMOVE",
      "AGENT_SESSION_QUEUED_TURN_PROMOTE",
      '"agentSession/queuedTurn/remove"',
      '"agentSession/queuedTurn/promote"',
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/client_request.rs",
    snippets: ["AgentSessionQueuedTurnRemove", "AgentSessionQueuedTurnPromote"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
    snippets: [
      "AgentSessionQueuedTurnRemove",
      "AgentSessionQueuedTurnPromote",
      '"agentSession/queuedTurn/remove"',
      '"agentSession/queuedTurn/promote"',
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/schema_types.rs",
    snippets: [
      "AgentSessionQueuedTurnRemoveParams",
      "AgentSessionQueuedTurnPromoteParams",
    ],
  },
  {
    file: "lime-rs/crates/app-server/src/processor/agent_session.rs",
    snippets: [
      "handle_agent_session_queued_turn_remove",
      "handle_agent_session_queued_turn_promote",
    ],
  },
  {
    file: "lime-rs/crates/app-server/src/runtime/session_control.rs",
    snippets: [
      "remove_agent_session_queued_turn",
      "promote_agent_session_queued_turn",
      "promote_queued_turn_in_session",
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v2/turn.rs",
    snippets: ["queue_if_busy", "TurnQueuePromote"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v2/common.rs",
    snippets: ["TurnQueueState", "pub queue:"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v2/methods.rs",
    snippets: ['"turn/queue/promote"', "METHOD_TURN_QUEUE_PROMOTE"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs",
    snippets: ["TurnQueuePromote"],
  },
  {
    file: "lime-rs/crates/app-server/src/processor/dispatch.rs",
    snippets: ["METHOD_TURN_QUEUE_PROMOTE"],
  },
  {
    file: "lime-rs/crates/app-server/src/processor/dispatch/v2_ingress.rs",
    snippets: ["TurnQueuePromote"],
  },
  {
    file: "lime-rs/crates/app-server/src/processor/turn.rs",
    snippets: ["handle_turn_queue_promote_v2_impl", "params.queue_if_busy"],
  },
  {
    file: "packages/app-server-client/src/generated/protocol-types.ts",
    snippets: [
      '"turn/queue/promote"',
      "TurnQueuePromote",
      '"agentSession/queuedTurn/remove"',
      '"agentSession/queuedTurn/promote"',
      "AgentSessionQueuedTurnRemove",
      "AgentSessionQueuedTurnPromote",
    ],
  },
  {
    file: "packages/app-server-client/src/request-client.ts",
    snippets: [
      "promoteQueuedTurn",
      "TurnQueuePromote",
      "promoteAgentSessionQueuedTurn",
      "removeAgentSessionQueuedTurn",
    ],
  },
  {
    file: "packages/app-server-client/src/request-client-methods.ts",
    snippets: [
      "promoteQueuedTurn",
      "METHOD_TURN_QUEUE_PROMOTE",
      "promoteAgentSessionQueuedTurn",
      "removeAgentSessionQueuedTurn",
    ],
  },
  {
    file: "packages/app-server-client/src/connection-methods.ts",
    snippets: [
      "promoteQueuedTurn",
      "TurnQueuePromote",
      "promoteAgentSessionQueuedTurn",
      "removeAgentSessionQueuedTurn",
    ],
  },
  {
    file: "src/lib/api/appServerConstants.ts",
    snippets: [
      "APP_SERVER_METHOD_TURN_QUEUE_PROMOTE",
      "APP_SERVER_METHOD_AGENT_SESSION_QUEUED_TURN_REMOVE",
      "APP_SERVER_METHOD_AGENT_SESSION_QUEUED_TURN_PROMOTE",
    ],
  },
  {
    file: "src/lib/api/appServerTypes.ts",
    snippets: [
      "AppServerTurnQueuePromote",
      "AppServerAgentSessionQueuedTurnRemove",
      "AppServerAgentSessionQueuedTurnPromote",
    ],
  },
  {
    file: "src/lib/api/appServerClientMethods.ts",
    snippets: [
      "promoteQueuedTurn",
      "promoteAgentSessionQueuedTurn",
      "removeAgentSessionQueuedTurn",
    ],
  },
  {
    file: "src/lib/api/appServerClientMethodSpecs.ts",
    snippets: [
      "promoteQueuedTurn",
      "APP_SERVER_METHOD_TURN_QUEUE_PROMOTE",
      "promoteAgentSessionQueuedTurn",
      "removeAgentSessionQueuedTurn",
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/schema/json/app_server_protocol.schemas.json",
    snippets: ['"turn/queue/promote"', "TurnQueuePromote"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/schema/json/manifest.json",
    snippets: ['"turn/queue/promote"', "TurnQueuePromote"],
  },
];
const retiredPublicQueuedTurnSchemaFiles = [
  "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionQueuedTurnRemoveParams.json",
  "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionQueuedTurnRemoveResponse.json",
  "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionQueuedTurnPromoteParams.json",
  "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionQueuedTurnPromoteResponse.json",
  "lime-rs/crates/app-server-protocol/schema/json/v2/TurnQueuePromoteParams.json",
  "lime-rs/crates/app-server-protocol/schema/json/v2/TurnQueuePromoteResponse.json",
];
const retiredRendererQueuedTurnFiles = [
  "src/lib/api/queuedTurn.ts",
  "src/lib/api/queuedTurn.test.ts",
  "src/lib/api/queuedTurn.d.ts",
];
const retiredPendingSteerFixtureFiles = [
  "scripts/agent-runtime/claw-chat-current-fixture-inputbar-pending-steer.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-gui-actions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-read-model.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-assertions.mjs",
];
const retiredPendingSteerScenarios = [
  "inputbar-pending-steer-rich-restore",
  "inputbar-pending-steer-multi-queue",
  "inputbar-pending-steer-pop-front-resume",
];
const retiredExecutionProcessFiles = [
  "lime-rs/crates/app-server-protocol/src/protocol/v0/execution_process.rs",
  "src/lib/api/executionProcess.ts",
  "src/lib/api/executionProcess.test.ts",
  ...[
    "DrainOutputParams",
    "DrainOutputResponse",
    "EmptyResponse",
    "IdParams",
    "OutputDelta",
    "OutputKind",
    "Snapshot",
    "StartParams",
    "StartResponse",
    "Status",
    "StatusResponse",
    "WriteStdinParams",
  ].map(
    (name) =>
      `lime-rs/crates/app-server-protocol/schema/json/v0/ExecutionProcess${name}.json`,
  ),
];
const retiredFileSystemFiles = [
  "lime-rs/crates/app-server-protocol/src/protocol/v0/file_system.rs",
  "lime-rs/crates/app-server/src/processor/file.rs",
  "lime-rs/crates/app-server/src/processor/tests/file.rs",
  "lime-rs/crates/app-server/src/runtime/file_system.rs",
  "lime-rs/crates/app-server/src/services/file_browser_service.rs",
  "lime-rs/crates/services/src/file_browser_service.rs",
  ...[
    "CreateDirectoryParams",
    "CreateFileParams",
    "DeleteFileParams",
    "DirectoryListing",
    "FileEntry",
    "FilePreview",
    "ListDirectoryParams",
    "MutationResponse",
    "ReadFilePreviewParams",
    "RenameFileParams",
  ].map(
    (name) =>
      `lime-rs/crates/app-server-protocol/schema/json/v0/FileSystem${name}.json`,
  ),
];
const retiredFileSystemSurfaceSpecs = [
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0.rs",
    snippets: ["mod file_system;", "pub use file_system::*;"],
  },
  {
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/client_request.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/schema_types.rs",
      "lime-rs/crates/app-server-protocol/src/schema_export/registry.rs",
    ],
    snippets: [
      "METHOD_FILE_SYSTEM_",
      '"fileSystem/',
      "FileSystemListDirectoryParams",
      "FileSystemReadFilePreviewParams",
      "FileSystemMutationResponse",
    ],
  },
  {
    files: [
      "lime-rs/crates/app-server-client/src/lib.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/connection-methods.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
      ...rendererAppServerSplitSourceFiles,
    ],
    snippets: [
      "METHOD_FILE_SYSTEM_",
      '"fileSystem/',
      "FileSystemListDirectoryParams",
      "FileSystemReadFilePreviewParams",
      "FileSystemMutationResponse",
      "readFilePreview",
      "renameFile",
      "deleteFile",
    ],
  },
  {
    files: [
      "lime-rs/crates/app-server-protocol/schema/json/app_server_protocol.schemas.json",
      "lime-rs/crates/app-server-protocol/schema/json/manifest.json",
    ],
    snippets: [
      '"fileSystem/',
      "FileSystemListDirectoryParams",
      "FileSystemReadFilePreviewParams",
      "FileSystemMutationResponse",
    ],
  },
];
const retiredExecutionProcessSurfaceSpecs = [
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0.rs",
    snippets: ["mod execution_process;", "pub use execution_process::*;"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
    snippets: ["METHOD_EXECUTION_PROCESS_", '"executionProcess/'],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/client_request.rs",
    snippets: ["ExecutionProcessStart", "ExecutionProcessDrainOutput"],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
    snippets: ["ExecutionProcessStart", '"executionProcess/'],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/protocol/v0/schema_types.rs",
    snippets: [
      "ExecutionProcessStartParams",
      "ExecutionProcessDrainOutputResponse",
    ],
  },
  {
    file: "lime-rs/crates/app-server-protocol/src/schema_export/registry.rs",
    snippets: [
      "ExecutionProcessStartParams",
      "ExecutionProcessDrainOutputResponse",
    ],
  },
  {
    file: "packages/app-server-client/src/generated/protocol-types.ts",
    snippets: [
      "METHOD_EXECUTION_PROCESS_",
      '"executionProcess/',
      "ExecutionProcessStartParams",
      "ExecutionProcessDrainOutputResponse",
    ],
  },
  {
    files: [
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/connection-methods.ts",
    ],
    snippets: [
      "startExecutionProcess",
      "writeExecutionProcessStdin",
      "interruptExecutionProcess",
      "terminateExecutionProcess",
      "readExecutionProcessStatus",
      "drainExecutionProcessOutput",
    ],
  },
  {
    files: rendererAppServerSplitSourceFiles,
    snippets: ["APP_SERVER_METHOD_EXECUTION_PROCESS_", '"executionProcess/'],
  },
  {
    files: [
      "lime-rs/crates/app-server-protocol/schema/json/app_server_protocol.schemas.json",
      "lime-rs/crates/app-server-protocol/schema/json/manifest.json",
    ],
    snippets: [
      '"executionProcess/',
      "ExecutionProcessStartParams",
      "ExecutionProcessDrainOutputResponse",
    ],
  },
];

function expandContractFiles(files) {
  return [
    ...new Set(
      files.flatMap((file) => {
        if (file === appServerClientIndexFile) {
          return appServerClientSplitSourceFiles;
        }
        if (file === rendererAppServerIndexFile) {
          return rendererAppServerSplitSourceFiles;
        }
        return [file];
      }),
    ),
  ];
}

function requiredContractContent(files, content) {
  if (!files.includes("packages/app-server-client/src/protocol.ts")) {
    return content;
  }
  const generatedPath = path.join(repoRoot, appServerGeneratedProtocolFile);
  if (!fs.existsSync(generatedPath)) {
    return content;
  }
  return `${content}\n${fs.readFileSync(generatedPath, "utf8")}`;
}

function collectRustFiles(relativeDir) {
  const absoluteDir = path.join(repoRoot, relativeDir);
  return fs
    .readdirSync(absoluteDir, { withFileTypes: true })
    .flatMap((entry) => {
      const relativePath = `${relativeDir}/${entry.name}`;
      if (entry.isDirectory()) {
        return collectRustFiles(relativePath);
      }
      return entry.isFile() && entry.name.endsWith(".rs") ? [relativePath] : [];
    })
    .sort();
}

const protocolV0ModuleFiles = collectRustFiles(
  "lime-rs/crates/app-server-protocol/src/protocol/v0",
);
const protocolV2ModuleFiles = collectRustFiles(
  "lime-rs/crates/app-server-protocol/src/protocol/v2",
);
const schemaExportModuleFiles = collectRustFiles(
  "lime-rs/crates/app-server-protocol/src/schema_export",
);
const appServerRuntimeFiles = [
  "lime-rs/crates/app-server/src/runtime.rs",
  ...collectRustFiles("lime-rs/crates/app-server/src/runtime"),
];
const appServerRuntimeThreadReadProjectionFiles = [
  "lime-rs/crates/app-server/src/runtime/load_context.rs",
  "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
  "lime-rs/crates/app-server/src/runtime/read_model.rs",
  "lime-rs/crates/app-server/src/runtime/tool_item_projection.rs",
  "lime-rs/crates/app-server/src/runtime/tool_item_projection/extract.rs",
  "lime-rs/crates/app-server/src/runtime/artifact_projection.rs",
  "lime-rs/crates/app-server/src/runtime/output_refs.rs",
  "lime-rs/crates/app-server/src/runtime/tests/read_model/tool_calls.rs",
  "lime-rs/crates/app-server/src/runtime/tests/read_model/imports_items.rs",
  "lime-rs/crates/app-server/src/runtime/tests/read_model/artifacts.rs",
];
const appServerRuntimeBackendFiles = [
  "lime-rs/crates/app-server/src/runtime_backend.rs",
  ...collectRustFiles("lime-rs/crates/app-server/src/runtime_backend"),
];
const appServerRuntimeBackendExecutionChainFiles = [
  "lime-rs/crates/app-server/src/runtime_backend.rs",
  "lime-rs/crates/app-server/src/runtime/turn_execution.rs",
  "lime-rs/crates/app-server/src/runtime_backend/action_response.rs",
  "lime-rs/crates/app-server/src/runtime_backend/event_mapper.rs",
  "lime-rs/crates/app-server/src/runtime_backend/execution_backend.rs",
  "lime-rs/crates/app-server/src/runtime_backend/provider_config.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context/turn_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context/workspace_scope.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tool_events.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tool_process_runtime_metadata.rs",
  "lime-rs/crates/app-server/src/runtime_backend/initialization_tests.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/coding_event_projection.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/model_selection.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/tool_policy_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/tool_surface.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/turn_flows.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/workspace_scope_context.rs",
];
const appServerProcessorFiles = [
  "lime-rs/crates/app-server/src/processor/mod.rs",
  ...collectRustFiles("lime-rs/crates/app-server/src/processor"),
];
const agentRuntimeBoundaryFiles = [
  "lime-rs/crates/agent/src/lib.rs",
  "lime-rs/crates/agent/src/runtime_state.rs",
  "lime-rs/crates/agent/src/runtime_state_support.rs",
  "lime-rs/crates/agent/src/provider_configuration.rs",
  "lime-rs/crates/agent/src/session_configuration.rs",
  "lime-rs/crates/agent/src/turn_context_configuration.rs",
  "lime-rs/crates/agent/src/current_provider_turn.rs",
  "lime-rs/crates/agent/src/current_provider_turn/tool_executor.rs",
  "lime-rs/crates/agent/src/direct_text_generation.rs",
  "lime-rs/crates/agent/src/turn_execution.rs",
];
const agentRequestToolPolicyFiles = [
  "lime-rs/crates/agent/src/request_tool_policy.rs",
  ...collectRustFiles("lime-rs/crates/agent/src/request_tool_policy"),
];
const rustProtocolFiles = [
  "lime-rs/crates/app-server-protocol/src/lib.rs",
  "lime-rs/crates/app-server-protocol/src/jsonrpc_lite.rs",
  "lime-rs/crates/app-server-protocol/src/protocol/v0.rs",
  ...protocolV0ModuleFiles,
  ...protocolV2ModuleFiles,
  "lime-rs/crates/app-server-protocol/src/schema_export.rs",
  ...schemaExportModuleFiles,
  "lime-rs/crates/app-server-protocol/src/schema_fixtures.rs",
];
const retiredTreeProjectionNames = [
  ["View", "Tree"].join(""),
  ["View", "tree"].join(""),
  ["view", "tree"].join(""),
  ["Process", "Tree"].join(""),
  ["Process", "tree"].join(""),
  ["process", "tree"].join(""),
];
const canonicalAgentUiPackages = [
  "@limecloud/agent-runtime-client",
  "@limecloud/agent-ui-contracts",
  "@limecloud/agent-runtime-projection",
  "@limecloud/agent-runtime-ui",
];
const retiredAgentUiPackageNames = [
  "@limecloud/agent-ui-projection",
  "@limecloud/agent-ui-react",
];
const agentUiPackageNamingGuardFiles = [
  "package.json",
  "package-lock.json",
  "pnpm-lock.yaml",
  "tsconfig.json",
  "vite.config.ts",
];
const retiredAgentRuntimeSessionFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.createSession",
  "AGENT_RUNTIME_COMMANDS.listSessions",
  "AGENT_RUNTIME_COMMANDS.getSession",
  "AGENT_RUNTIME_COMMANDS.updateSession",
  'createSession: "agent_runtime_create_session"',
  'listSessions: "agent_runtime_list_sessions"',
  'getSession: "agent_runtime_get_session"',
  'updateSession: "agent_runtime_update_session"',
  'readonly createSession: "agent_runtime_create_session"',
  'readonly listSessions: "agent_runtime_list_sessions"',
  'readonly getSession: "agent_runtime_get_session"',
  'readonly updateSession: "agent_runtime_update_session"',
  '"agent_runtime_create_session"',
  '"agent_runtime_list_sessions"',
  '"agent_runtime_get_session"',
  '"agent_runtime_update_session"',
];
const retiredAgentRuntimeSessionFacadeProductionFiles = [
  "electron/hostCommands.ts",
  "electron/ipcChannels.ts",
  "src/lib/dev-bridge/commandPolicy.ts",
  "src/lib/governance/agentCommandCatalog.json",
];
const retiredAgentRuntimeEvidenceExportFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.exportEvidencePack",
  'exportEvidencePack: "agent_runtime_export_evidence_pack"',
  'readonly exportEvidencePack: "agent_runtime_export_evidence_pack"',
  '"agent_runtime_export_evidence_pack"',
];
const retiredAgentRuntimeEvidenceExportFacadeProductionFiles = [
  "electron/hostCommands.ts",
  "electron/ipcChannels.ts",
  "src/lib/dev-bridge/commandPolicy.ts",
  "src/lib/governance/agentCommandCatalog.json",
];
const retiredAgentRuntimeThreadReadFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.getThreadRead",
  'getThreadRead: "agent_runtime_get_thread_read"',
  'readonly getThreadRead: "agent_runtime_get_thread_read"',
  '"agent_runtime_get_thread_read"',
];
const retiredAgentRuntimeThreadReadFacadeProductionFiles =
  retiredAgentRuntimeEvidenceExportFacadeProductionFiles;
const retiredAgentRuntimeSubmitTurnFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.submitTurn",
  'submitTurn: "agent_runtime_submit_turn"',
  'readonly submitTurn: "agent_runtime_submit_turn"',
  '"agent_runtime_submit_turn"',
];
const retiredAgentRuntimeSubmitTurnFacadeProductionFiles =
  retiredAgentRuntimeEvidenceExportFacadeProductionFiles;
const retiredAgentRuntimeInterruptTurnFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.interruptTurn",
  'interruptTurn: "agent_runtime_interrupt_turn"',
  'readonly interruptTurn: "agent_runtime_interrupt_turn"',
  '"agent_runtime_interrupt_turn"',
];
const retiredAgentRuntimeInterruptTurnFacadeProductionFiles =
  retiredAgentRuntimeEvidenceExportFacadeProductionFiles;
const retiredAgentRuntimeRespondActionFacadeSnippets = [
  "AGENT_RUNTIME_COMMANDS.respondAction",
  'respondAction: "agent_runtime_respond_action"',
  'readonly respondAction: "agent_runtime_respond_action"',
  '"agent_runtime_respond_action"',
];
const retiredAgentRuntimeRespondActionFacadeProductionFiles =
  retiredAgentRuntimeEvidenceExportFacadeProductionFiles;
const activeAgentRuntimeAipromptFiles = [
  "AGENTS.md",
  ...fs
    .readdirSync(path.join(repoRoot, "internal/aiprompts"), {
      withFileTypes: true,
    })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
    .map((entry) => `internal/aiprompts/${entry.name}`)
    .sort(),
];
const activeAgentRuntimeMarkdownFiles = [
  ...activeAgentRuntimeAipromptFiles,
  ...collectMarkdownFiles("src"),
  ...collectMarkdownFiles("packages"),
  ...collectMarkdownFiles("electron"),
];
const allowedRetiredAgentRuntimeDocContextPattern =
  /(旧|已删除|retired|legacy|历史|history|迁移|migration|compat|deprecated|dead|guard|test-only|fixture|只允许|不得|禁止|不再|不能|不要|not|forbid|forbidden|retire|退场|退役|残留|residual|reference|参考|cleanup|删除|现有迁移锚点)/i;
const forbiddenAgentRuntimeCurrentDocContextPattern =
  /(current|主链|事实源|唯一|继续收敛|新增|新增能力|当前.*入口|当前.*主路径|必须回到|继续走|default tool surface|统一提交|统一会话管理|统一中断|统一响应)/i;
const retiredAgentRuntimeScriptCallPatterns = [
  /\bcmd\s*:\s*["'`]agent_runtime_[A-Za-z0-9_]+["'`]/u,
  /\bcommand\s*:\s*["'`]agent_runtime_[A-Za-z0-9_]+["'`]/u,
  /\b(?:safeInvoke|invoke|invokeCommand|invokeAgentRuntimeBridge|bridgeInvoke|invokeViaHttp|postInvoke|postJson)\s*\([^)]*["'`]agent_runtime_[A-Za-z0-9_]+["'`]/su,
];
const agentRuntimeThinGatewayForbiddenSnippets = [
  {
    snippet: "mockPriorityCommands",
    reason:
      "frontend agentRuntime gateway cannot import renderer mock priority commands",
  },
  {
    snippet: "defaultMocks",
    reason:
      "frontend agentRuntime gateway cannot use default mocks as fallback",
  },
  {
    snippet: "invokeMockOnly",
    reason: "frontend agentRuntime gateway cannot call test-only desktop mocks",
  },
  {
    snippet: "explicitMockFallback",
    reason:
      "frontend agentRuntime gateway cannot use explicit renderer mock fallback",
  },
  {
    snippet: "invokeExplicitMock",
    reason:
      "frontend agentRuntime gateway cannot invoke renderer mock fallback",
  },
  {
    snippet: "listenExplicitMock",
    reason:
      "frontend agentRuntime gateway cannot listen to renderer mock fallback",
  },
  {
    snippet: "mockCommand",
    reason:
      "frontend agentRuntime gateway cannot register renderer mock commands",
  },
  {
    snippet: "clearMocks",
    reason: "frontend agentRuntime gateway cannot clear renderer mock commands",
  },
  {
    snippet: "fetch(",
    reason:
      "frontend agentRuntime gateway cannot call Provider or backend HTTP directly",
  },
  {
    snippet: "XMLHttpRequest",
    reason:
      "frontend agentRuntime gateway cannot call Provider or backend HTTP directly",
  },
  {
    snippet: "EventSource",
    reason:
      "frontend agentRuntime gateway cannot open a parallel stream transport",
  },
  {
    snippet: "new WebSocket",
    reason:
      "frontend agentRuntime gateway cannot open a parallel stream transport",
  },
  {
    snippet: "/v1/chat/completions",
    reason: "frontend agentRuntime gateway cannot call model APIs directly",
  },
  {
    snippet: "/v1/messages",
    reason: "frontend agentRuntime gateway cannot call model APIs directly",
  },
  {
    snippet: "chat/completions",
    reason: "frontend agentRuntime gateway cannot call model APIs directly",
  },
  {
    snippet: "LIME_GATEWAY",
    reason: "frontend agentRuntime gateway cannot receive Gateway credentials",
  },
  {
    snippet: "APP_SERVER_BACKEND_MODE=mock",
    reason:
      "frontend agentRuntime gateway cannot depend on App Server mock backend",
  },
  {
    snippet: 'APP_SERVER_BACKEND_MODE: "mock"',
    reason:
      "frontend agentRuntime gateway cannot depend on App Server mock backend",
  },
  {
    snippet: "APP_SERVER_BACKEND_MODE: 'mock'",
    reason:
      "frontend agentRuntime gateway cannot depend on App Server mock backend",
  },
];
const retiredSkillExecutionSurfaceFiles = [
  "src/hooks/useSkillExecution.ts",
  "src/components/skills/SkillExecutionDialog.tsx",
  "src/components/skills/SkillExecutionDialog.test.tsx",
];
const retiredAgentRuntimeMockFiles = [
  "src/lib/desktop-host/agentRuntimeMocks.ts",
  "src/lib/desktop-host/agentRuntimeMocks.d.ts",
  "src/lib/desktop-host/agentRuntimeObjectiveMocks.ts",
  "src/lib/desktop-host/agentRuntimeMocks.test.ts",
  "src/lib/desktop-host/agentRuntimeObjectiveMocks.test.ts",
];
const retiredAgentRuntimeCommandManifestFiles = [
  "src/lib/governance/agentRuntimeCommandSchema.json",
  "src/lib/api/agentRuntime/commandManifest.generated.ts",
  "src/lib/api/agentRuntime/commandManifest.generated.d.ts",
  "scripts/generate-agent-runtime-clients.mjs",
];
const retiredAgentRuntimeAdapterFiles = [
  "lime-rs/crates/app-server/src/runtime_backend_adapter.rs",
  "lime-rs/crates/agent/src/message_content_adapter.rs",
  "lime-rs/crates/agent/src/event_converter.rs",
];
const retiredRendererProjectionFiles = [
  "src/lib/api/agentRuntime/canonicalApprovalItemProjection.ts",
  "src/lib/api/agentRuntime/canonicalApprovalItemProjection.test.ts",
  "src/lib/api/agentRuntime/appServerEventPayloadProjection.ts",
  "src/components/agent/chat/projection/queueProjection.ts",
  "src/components/agent/chat/projection/queueProjection.test.ts",
  "packages/agent-runtime-projection/src/queueEvents.ts",
  "src/components/agent/chat/hooks/agentStreamReadModelParsing.ts",
  "src/components/agent/chat/hooks/agentQueuedTurnProjection.ts",
  "src/components/agent/chat/hooks/agentQueuedTurnProjection.unit.test.ts",
];
const retiredRendererQueuedTurnProjectionProductionFiles = [
  "src/lib/api/agentProtocolEventTypes.ts",
  "src/lib/api/agentProtocol.d.ts",
  "src/lib/api/agentProtocolRuntimeParsers.ts",
  "src/components/agent/chat/projection/agentUiEventProjection.ts",
  "src/components/agent/chat/hooks/agentStreamRuntimeLifecycleEvents.ts",
  "src/components/agent/chat/hooks/agentStreamRuntimeHandler.ts",
];
const retiredRendererQueuedTurnProjectionSnippets = [
  "AgentEventQueueAdded",
  "AgentEventQueueRemoved",
  "AgentEventQueueStarted",
  "AgentEventQueueCleared",
  'case "queue_added"',
  'case "queue_removed"',
  'case "queue_started"',
  'case "queue_cleared"',
  "buildQueueProjectionEvents",
  "handleAgentStreamQueueEvent",
];
const retiredRendererQueuedTurnSecondaryProjectionSpecs = [
  {
    file: "src/components/agent/chat/projection/threadReadActivity.ts",
    snippets: [
      "includeQueuedActivity",
      '"queued_turns"',
      '"queuedTurns"',
      "hasPendingOrQueuedActivity",
    ],
  },
  {
    file: "src/components/agent/chat/hooks/agentSessionTopicViewModel.ts",
    snippets: [
      "shouldAutoResumeHydratedRuntimeThread",
      "resolveRuntimePreviewFromSessionDetail",
      ".queued_turns",
    ],
  },
  {
    file: "src/components/agent/chat/hooks/agentChatHistoryNormalize.ts",
    snippets: [".queued_turns"],
  },
  {
    file: "src/lib/api/agentRuntime/sessionClient.ts",
    snippets: ["queuedTurnsCount:"],
  },
  {
    file: "src/components/agent/chat/hooks/agentSessionState.ts",
    snippets: [
      "QueuedTurnSnapshot",
      "normalizeQueuedTurnSnapshots",
      "queuedTurns:",
    ],
  },
  {
    file: "src/components/agent/chat/hooks/agentSessionRefresh.ts",
    snippets: [
      "QueuedTurnSnapshot",
      "normalizeQueuedTurnSnapshots",
      "queuedTurns:",
    ],
  },
  {
    file: "src/components/agent/chat/hooks/useAgentSession.ts",
    snippets: ["QueuedTurnSnapshot", "setQueuedTurns", "queuedTurns:"],
  },
  {
    file: "src/components/agent/chat/hooks/useAgentChat.ts",
    snippets: ["session.queuedTurns", "queuedTurns: session.queuedTurns"],
  },
  {
    file: "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
    snippets: ["queuedTurns = []", "queuedTurns,"],
  },
  {
    file: "src/components/agent/chat/workspace/useAgentChatWorkspaceCommandRuntime.ts",
    snippets: ["queuedTurns.length"],
  },
];
const retiredAgentRuntimeLegacyQueueFiles = [
  "lime-rs/crates/core/src/database/agent_runtime_queue_repository.rs",
  "lime-rs/crates/agent/src/agent_runtime_support.rs",
];
const retiredAgentRuntimeLegacyQueueSurfaceFiles = [
  "lime-rs/crates/core/src/database/mod.rs",
  "src/lib/governance/legacySurfaceCatalog.json",
];
const retiredAgentUiResumeContractFiles = [
  "packages/agent-ui-contracts/schemas/agent-runtime-resume-contract.v0.1.schema.json",
];
const retiredAgentRuntimeLegacyQueueSnippets = [
  "agent_runtime_queued_turns",
  "agent_runtime_queue_repository",
  "migrate_legacy_runtime_queue_to_agent_store",
  "LegacyRuntimeQueueMigrationReport",
  "LegacyRuntimeQueuedTurn",
];
const retiredAgentRuntimeToolInventoryMockFiles = [
  "src/lib/desktop-host/runtimeToolInventoryMocks.ts",
  "src/lib/desktop-host/runtimeToolInventoryMocks.d.ts",
];

const checks = [
  {
    name: "Rust protocol exposes capability discovery request and response DTOs",
    files: rustProtocolFiles,
    snippets: [
      "schemars::JsonSchema",
      'pub const METHOD_CAPABILITY_LIST: &str = "capability/list"',
      "pub struct CapabilityListParams",
      "pub app_id: Option<String>",
      "pub workspace_id: Option<String>",
      "pub session_id: Option<String>",
      "pub cursor: Option<String>",
      "pub limit: Option<u32>",
      "pub struct CapabilityListResponse",
      "pub next_cursor: Option<String>",
      "pub struct CapabilityDescriptor",
      "pub enum AppServerMethodKind",
      "pub struct AppServerMethodSpec",
      "pub const APP_SERVER_METHODS: &[AppServerMethodSpec]",
      "pub enum AppServerRequestSerializationScope",
      "pub struct AppServerRequestSerializationScopeSpec",
      "pub const APP_SERVER_REQUEST_SERIALIZATION_SCOPES",
      "pub fn app_server_request_serialization_scope",
      "pub fn is_app_server_request_method(method: &str) -> bool",
      "pub fn is_app_server_notification_method(method: &str) -> bool",
      "pub const CAPABILITY_DENIED: i64 = -32020",
      '#[serde(rename_all = "camelCase")]',
    ],
  },
  {
    name: "Rust protocol exports JSON schema fixtures for JSON-RPC, v0, and v2 DTOs",
    files: rustProtocolFiles,
    snippets: [
      "pub const JSONRPC_SCHEMA_TYPE_NAMES: &[&str]",
      "pub const V0_SCHEMA_TYPE_NAMES: &[&str]",
      "pub const V2_SCHEMA_TYPE_NAMES: &[&str]",
      "JsonRpcRequest",
      "AgentSessionTurnStartParams",
      "RuntimeOptions",
      "pub include_protocol_types: bool",
      "fn jsonrpc_schemas() -> Vec<GeneratedJsonSchema>",
      "fn v0_schemas() -> Vec<GeneratedJsonSchema>",
      "fn v2_schemas() -> Vec<GeneratedJsonSchema>",
      'typed_schema::<AppServerRequestSerializationScope>("AppServerRequestSerializationScope")',
      'typed_schema::<AppServerRequestSerializationScopeSpec>("AppServerRequestSerializationScopeSpec")',
      'typed_schema::<AgentSessionTurnStartParams>("AgentSessionTurnStartParams")',
      'PathBuf::from("json")',
      '.join("v0")',
      '.join("v2")',
      '.join("jsonrpc")',
      '"schemas": {',
      "schema_registry_matches_declared_type_names",
    ],
  },
  {
    name: "configWarning stays current from v2 producer through Electron and Renderer toast",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v2/config.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs",
      "lime-rs/crates/app-server-protocol/schema/json/v2/ConfigWarningNotification.json",
      appServerGeneratedProtocolFile,
      "lime-rs/crates/app-server/src/processor/config_warning.rs",
      "packages/app-server-client/tests/client.test.mjs",
      "electron/appServerHost.test.ts",
      "src/lib/api/appServerResponse.ts",
      "src/lib/api/appServerClient.ts",
      "src/lib/api/appServerConfigWarnings.ts",
      "src/components/AppServerConfigWarningToastBridge.tsx",
      "src/App.tsx",
      "src/i18n/__tests__/loadNamespace.test.ts",
    ],
    snippets: [
      "pub struct ConfigWarningNotification",
      "ConfigWarning(ConfigWarningNotification)",
      '"title": "ConfigWarningNotification"',
      'export const METHOD_CONFIG_WARNING = "configWarning"',
      'method: "configWarning"',
      "isAppServerNotificationMethod(METHOD_CONFIG_WARNING), true",
      "ConfigWarningScope::Initialize",
      "connected.connection.nextNotification(",
      "warmup 后应保留 initialize 阶段已缓冲的 configWarning",
      "readAppServerConfigWarnings",
      "publishAppServerConfigWarnings",
      "subscribeAppServerConfigWarnings",
      "AppServerConfigWarningToastBridge",
      "<AppServerConfigWarningToastBridge />",
      "common.app.configWarning.descriptionWithPathAndDetails",
    ],
  },
  {
    name: "runtime.warning stays typed from App Server projection through localized Renderer warning",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v2/notification.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs",
      "lime-rs/crates/app-server-protocol/schema/json/v2/WarningNotification.json",
      appServerGeneratedProtocolFile,
      "lime-rs/crates/app-server/src/processor/v2_notifications.rs",
      "lime-rs/crates/app-server/src/processor/v2_notifications/warning.rs",
      "lime-rs/crates/app-server/src/runtime/read_model/runtime_items.rs",
      "src/lib/api/agentRuntime/appServerV2Notification.ts",
      "src/lib/api/agentRuntime/eventSequenceGate.ts",
      "src/lib/api/agentRuntime/appServerEventStream.test.ts",
      "packages/app-server-client/tests/client.test.mjs",
    ],
    snippets: [
      "pub struct WarningNotification",
      "Warning(WarningNotification)",
      'export const METHOD_WARNING = "warning"',
      '"runtime.warning" => return warning::project(event)',
      "ServerNotification::Warning",
      "runtime_warning_code_from_event",
      'case "warning"',
      'notification.method === "warning"',
      '"runtime.warning",',
      "isAppServerNotificationMethod(METHOD_WARNING), true",
    ],
  },
  {
    name: "runtime errors stay typed from App Server retry policy through Renderer state",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v2/common.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v2/notification.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs",
      "lime-rs/crates/app-server-protocol/schema/json/v2/ErrorNotification.json",
      appServerGeneratedProtocolFile,
      "packages/app-server-client/src/server-notifications.ts",
      "packages/app-server-client/src/agent-runtime.ts",
      "lime-rs/crates/app-server/src/processor/v2_notifications.rs",
      "lime-rs/crates/app-server/src/processor/v2_notifications/error.rs",
      "src/lib/api/agentRuntime/appServerV2Notification.ts",
      "src/components/agent/chat/hooks/agentStreamTypedErrorController.ts",
      "src/components/agent/chat/hooks/agentStreamRuntimeHandler.ts",
    ],
    snippets: [
      "pub enum CodexErrorInfo",
      '#[serde(rename = "httpStatusCode")]',
      '#[serde(rename = "turnKind")]',
      "pub struct ErrorNotification",
      "Error(ErrorNotification)",
      'export const METHOD_ERROR = "error"',
      "export interface ErrorNotification",
      "httpStatusCode?: number | null;",
      "turnKind: NonSteerableTurnKind;",
      '"runtime.error" => return self.project_error(event, None)',
      "error::project(event, Some(false))",
      "ServerNotification::Error",
      "isErrorNotification(notification)",
      "notification && !isAgentRuntimeSignalNotification(notification)",
      'params.event.protocol_method !== "error"',
      'typeof params.event.will_retry !== "boolean"',
      "buildAgentStreamTypedErrorPlan",
    ],
  },
  {
    name: "Renderer turn binding cannot restore raw runtime error fallback",
    file: "src/components/agent/chat/hooks/agentStreamTurnEventBinding.ts",
    snippets: ["handleTurnStreamEvent"],
    absentSnippets: [
      '"runtime_error"',
      '"runtime.error"',
      "readRuntimeErrorMessage",
    ],
  },
  {
    name: "Rust App Server gates AgentUI runtime events before storage",
    files: [
      "lime-rs/crates/app-server/src/lib.rs",
      "lime-rs/crates/app-server/src/agent_ui_event_schema.rs",
      "lime-rs/crates/app-server/src/agent_ui_sequence_verifier.rs",
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/runtime/event_store.rs",
      "lime-rs/crates/app-server/src/runtime/event_store/validation.rs",
      "lime-rs/crates/app-server/src/runtime/tool_lifecycle.rs",
      "lime-rs/crates/app-server/src/runtime/tool_lifecycle_tests.rs",
      "lime-rs/crates/app-server/src/runtime/tests.rs",
      "lime-rs/crates/app-server/src/runtime/tests/external_events.rs",
      ...collectRustFiles(
        "lime-rs/crates/app-server/src/runtime/tests/external_events",
      ),
    ],
    snippets: [
      "mod agent_ui_event_schema;",
      "mod agent_ui_sequence_verifier;",
      "mod tool_lifecycle;",
      "agent-runtime-event.v0.1.schema.json",
      "agent-runtime-state-delta.v0.1.schema.json",
      "jsonschema::validator_for",
      "mod validation;",
      "use self::validation::EventValidationContext;",
      "EventValidationContext::from_events(&stored.events, session_id, turn_id)",
      "agent_ui_event_schema::validate_agent_event(&event).map_err(RuntimeCoreError::Backend)?",
      "validation.validate_and_observe(",
      "events.push(event)",
      "stored.events.push(event)",
      "struct EventValidationContext",
      "AgentEventSequenceValidator::from_events",
      "ToolLifecycleValidator::from_events",
      "self.sequence.validate_and_observe(event)?",
      "self.tool_lifecycle.validate_and_observe(event)?",
      "agent runtime event sequence validation failed",
      "agent runtime tool lifecycle validation failed",
      "rejects_policy_event_for_inactive_tool",
      "rejects_tool_output_before_action_resolution",
      "rejects_tool_result_after_action_denial",
      "rejects_tool_result_owner_mismatch",
      "append_external_runtime_events_rejects_invalid_state_delta_before_storage",
      "invalid state.delta must fail closed",
      "append_external_runtime_events_rejects_canonical_tool_completed_without_start",
      "append_external_runtime_events_rejects_duplicate_canonical_tool_start",
      "append_external_runtime_events_rejects_retired_raw_tool_wire",
      "append_external_runtime_events_rejects_retired_raw_tool_wire_with_import_markers",
    ],
  },
  {
    name: "Rust protocol exposes agentSession/action/respond DTOs and method catalog",
    files: rustProtocolFiles,
    snippets: [
      'pub const METHOD_AGENT_SESSION_ACTION_RESPOND: &str = "agentSession/action/respond"',
      "pub enum AgentSessionActionType",
      "ToolConfirmation",
      "AskUser",
      "Elicitation",
      "pub struct AgentSessionActionScope",
      "pub struct AgentSessionActionRespondParams",
      "pub request_id: String",
      "pub action_type: AgentSessionActionType",
      "pub confirmed: bool",
      "pub response: Option<String>",
      "pub user_data: Option<serde_json::Value>",
      "pub metadata: Option<serde_json::Value>",
      "pub event_name: Option<String>",
      "pub action_scope: Option<AgentSessionActionScope>",
      "pub struct AgentSessionActionRespondResponse",
      "method: METHOD_AGENT_SESSION_ACTION_RESPOND,",
      "agent_session_action_respond_request_matches_protocol_fixture_shape",
    ],
  },
  {
    name: "Rust protocol exposes artifact/read DTOs and method catalog",
    files: rustProtocolFiles,
    snippets: [
      'pub const METHOD_ARTIFACT_READ: &str = "artifact/read"',
      "pub struct ArtifactReadParams",
      "pub session_id: String",
      "pub turn_id: Option<String>",
      "pub artifact_ref: Option<String>",
      "pub include_content: Option<bool>",
      "pub struct ArtifactSummary",
      "pub artifact_ref: String",
      "pub event_id: String",
      "pub content: Option<String>",
      "pub enum ArtifactContentStatus",
      "pub content_status: ArtifactContentStatus",
      "pub metadata: Option<serde_json::Value>",
      "pub struct ArtifactReadResponse",
      "pub artifacts: Vec<ArtifactSummary>",
      "method: METHOD_ARTIFACT_READ,",
      "artifact_read_request_matches_protocol_fixture_shape",
      "artifact_summary_content_status_matches_protocol_fixture_shape",
    ],
  },
  {
    name: "Rust protocol exposes exact fs DTOs and method catalog",
    files: rustProtocolFiles,
    snippets: [
      'pub const METHOD_FS_READ_FILE: &str = "fs/readFile"',
      'pub const METHOD_FS_WRITE_FILE: &str = "fs/writeFile"',
      'pub const METHOD_FS_CREATE_DIRECTORY: &str = "fs/createDirectory"',
      'pub const METHOD_FS_GET_METADATA: &str = "fs/getMetadata"',
      'pub const METHOD_FS_READ_DIRECTORY: &str = "fs/readDirectory"',
      'pub const METHOD_FS_REMOVE: &str = "fs/remove"',
      'pub const METHOD_FS_COPY: &str = "fs/copy"',
      'pub const METHOD_FS_WATCH: &str = "fs/watch"',
      'pub const METHOD_FS_UNWATCH: &str = "fs/unwatch"',
      'pub const METHOD_FS_CHANGED: &str = "fs/changed"',
      "pub struct FsReadFileParams",
      "pub struct FsReadFileResponse",
      "pub data_base64: String",
      "pub struct FsWriteFileParams",
      "pub struct FsCreateDirectoryParams",
      "pub struct FsGetMetadataResponse",
      "pub struct FsReadDirectoryResponse",
      "pub struct FsRemoveParams",
      "pub recursive: Option<bool>",
      "pub force: Option<bool>",
      "pub struct FsCopyParams",
      "pub source_path: String",
      "pub destination_path: String",
      "pub struct FsWatchParams",
      "pub watch_id: String",
      "pub struct FsChangedNotification",
      "pub changed_paths: Vec<String>",
    ],
  },
  {
    name: "Rust server initializes capability discovery and dispatches capability/list",
    files: appServerProcessorFiles,
    snippets: [
      "METHOD_CAPABILITY_LIST => self.handle_capability_list(params)",
      "let params: CapabilityListParams = parse_params(params)?",
      ".list_capabilities(params)",
      ".map_err(to_jsonrpc_error)?",
      "capability_discovery: true",
    ],
  },
  {
    name: "Rust JSON-RPC router dispatches agentSession/action/respond into RuntimeCore",
    files: appServerProcessorFiles,
    snippets: [
      "METHOD_AGENT_SESSION_ACTION_RESPOND => self.handle_action_respond(params).await",
      "let params: AgentSessionActionRespondParams = parse_params(params)?",
      ".respond_action(params, host)",
      "dispatch_result_with_events(output.response, output.events)",
    ],
  },
  {
    name: "Rust JSON-RPC router dispatches artifact/read into RuntimeCore",
    files: appServerProcessorFiles,
    snippets: [
      "METHOD_ARTIFACT_READ => self.handle_artifact_read(params)",
      "fn handle_artifact_read(",
      "let params: ArtifactReadParams = parse_params(params)?",
      ".read_artifacts(params)",
      "artifact: true",
      "artifact_read_requires_initialized_and_returns_artifact_summaries",
    ],
  },
  {
    name: "Rust JSON-RPC router dispatches exact fs methods into FsServer",
    files: appServerProcessorFiles,
    snippets: [
      "v2::METHOD_FS_READ_FILE => self.handle_fs_read_file_impl(params).boxed()",
      "v2::METHOD_FS_WRITE_FILE => self.handle_fs_write_file_impl(params).boxed()",
      "v2::METHOD_FS_CREATE_DIRECTORY => self.handle_fs_create_directory_impl(params).boxed()",
      "v2::METHOD_FS_GET_METADATA => self.handle_fs_get_metadata_impl(params).boxed()",
      "v2::METHOD_FS_READ_DIRECTORY => self.handle_fs_read_directory_impl(params).boxed()",
      "v2::METHOD_FS_REMOVE => self.handle_fs_remove_impl(params).boxed()",
      "v2::METHOD_FS_COPY => self.handle_fs_copy_impl(params).boxed()",
      ".handle_fs_watch_impl(params, connection_request_id.clone())",
      ".handle_fs_unwatch_impl(params, connection_request_id.clone())",
      "let params: FsReadFileParams = parse_params(params)?",
      "let params: FsWriteFileParams = parse_params(params)?",
      "let params: FsCreateDirectoryParams = parse_params(params)?",
      "let params: FsGetMetadataParams = parse_params(params)?",
      "let params: FsReadDirectoryParams = parse_params(params)?",
      "let params: FsRemoveParams = parse_params(params)?",
      "let params: FsCopyParams = parse_params(params)?",
      "let params: FsWatchParams = parse_params(params)?",
      "let params: FsUnwatchParams = parse_params(params)?",
      "dispatch_result(self.fs.read_file(params).await?)",
      "dispatch_result(self.fs.unwatch(connection_id(request)?, params).await?)",
    ],
  },
  {
    name: "Rust capability module exposes host-independent inventory source",
    file: "lime-rs/crates/app-server/src/capability.rs",
    snippets: [
      "pub struct CapabilityInventoryRecord",
      "pub struct CapabilityInventorySource",
      "pub struct CapabilityListContext",
      "pub app_id: Option<String>",
      "pub workspace_id: Option<String>",
      "pub session_id: Option<String>",
      "pub fn executable_agent_turn(",
      "pub fn capability_descriptor_allows_agent_turn_start(",
      "pub fn for_sessions(",
      "pub trait CapabilitySource",
      "fn list_capabilities(&self, context: &CapabilityListContext) -> Vec<CapabilityDescriptor>",
      "fn scope_matches",
      "inventory_source_filters_by_session_scope",
      'id: "agent.session".to_string()',
      "METHOD_TURN_START.to_string()",
      "METHOD_TURN_INTERRUPT.to_string()",
    ],
  },
  {
    name: "Rust runtime uses injected capability source",
    files: appServerRuntimeFiles,
    snippets: [
      "capability_source: Arc<dyn CapabilitySource>",
      "pub fn with_backend_and_capability_source(",
      "capability_source: Arc<dyn CapabilitySource>",
      "CapabilityInventorySource::default()",
      "pub fn list_capabilities(",
      "Result<CapabilityListResponse, RuntimeCoreError>",
      "fn capability_list_context(",
      "SessionNotFound(session_id.clone())",
      "session_id: Some(session_id)",
      "self.capability_source.list_capabilities(&context)",
      "capability_descriptor_allows_agent_turn_start(capability)",
      "fn paginate_capabilities(",
      "next_cursor",
      "fn ensure_capability_allowed_with_context(",
      "METHOD_TURN_START",
      "capability_list_with_session_id_uses_stored_session_scope",
      "start_turn_allows_session_scoped_capability_id",
      "RuntimeCoreError::CapabilityDenied",
    ],
  },
  {
    name: "Rust runtime routes action responses through injected backend",
    files: appServerRuntimeFiles,
    snippets: [
      "pub struct ActionRespondRequest",
      "async fn respond_action(",
      "pub async fn respond_action(",
      "params: AgentSessionActionRespondParams",
      "AgentSessionActionRespondResponse",
      "RuntimeCoreError::TurnNotActive",
      "ActionRespondRequest {",
      "action_scope: params.action_scope",
    ],
  },
  {
    name: "Rust FsServer owns exact file IO and connection-scoped watches",
    files: [
      "lime-rs/crates/app-server/src/fs.rs",
      "lime-rs/crates/app-server/src/fs/tests.rs",
    ],
    snippets: [
      "pub(crate) struct FsServer",
      "pub(crate) async fn read_file(",
      'absolute_path(&params.path, "fs/readFile.path")',
      "STANDARD.encode(bytes)",
      "pub(crate) async fn write_file(",
      "STANDARD.decode(params.data_base64)",
      "pub(crate) async fn create_directory(",
      "pub(crate) async fn get_metadata(",
      "pub(crate) async fn read_directory(",
      "pub(crate) async fn remove(",
      "pub(crate) async fn copy(",
      "pub(crate) async fn watch(",
      "pub(crate) async fn unwatch(",
      "ServerNotification::FsChanged(FsChangedNotification {",
      "watch_ids_are_connection_scoped_and_disconnect_cleans_only_the_owner",
    ],
  },
  {
    name: "Rust runtime indexes artifact summaries from stored events",
    files: appServerRuntimeFiles,
    snippets: [
      "pub fn read_artifacts(",
      "params: ArtifactReadParams",
      "Result<ArtifactReadResponse, RuntimeCoreError>",
      "ArtifactContentProvider",
      "ArtifactContentRequest",
      "InlineArtifactContentProvider",
      "FilesystemArtifactContentProvider",
      "DEFAULT_ARTIFACT_CONTENT_MAX_BYTES",
      "ArtifactContentStatus::Available",
      "ArtifactContentStatus::Unavailable",
      "ArtifactContentStatus::NotRequested",
      "with_backend_capability_source_and_artifact_content_provider",
      "fn read_limited_relative_utf8_file(",
      "fn is_safe_relative_path(",
      "fn paginate_artifact_summaries(",
      "fn artifact_summary_from_event(",
      "fn string_field(",
      "read_artifacts_indexes_latest_artifact_events_for_session",
      "read_artifacts_uses_injected_content_provider_for_current_page",
      "filesystem_artifact_content_provider_reads_allowed_relative_path",
      "filesystem_artifact_content_provider_rejects_escape_and_oversized_files",
    ],
  },
  {
    name: "Rust App Server keeps retired backend adapter out of RuntimeBackend",
    files: [
      "lime-rs/crates/app-server/src/main.rs",
      "lime-rs/crates/app-server/src/runtime_backend.rs",
      "lime-rs/crates/app-server/src/runtime_factory.rs",
    ],
    snippets: [
      "parse_args_rejects_runtime_backend_for_standalone_binary",
      "unsupported app-server backend mode: agent",
      "RuntimeBackend::with_execution_process_server",
      "RuntimeBackend::with_db_and_execution_process_server",
    ],
    absentSnippets: [
      "RuntimeBackendAdapter",
      "RuntimeBackendHost",
      "runtime_adapter_core",
    ],
  },
  {
    name: "Standalone App Server local data source implements workspace and skill surfaces",
    files: [
      "lime-rs/crates/app-server/src/runtime/app_data.rs",
      "lime-rs/crates/app-server/src/runtime/app_data/skills.rs",
      "lime-rs/crates/app-server/src/runtime/app_data/workspaces.rs",
      "lime-rs/crates/app-server/src/local_data_source.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/skills.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/workspace_skill_bindings.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/workspaces.rs",
      "lime-rs/crates/app-server/src/local_data_source/workspaces.rs",
      "lime-rs/crates/app-server/src/local_data_source/skills/workspace.rs",
    ],
    snippets: [
      "pub struct LocalAppDataSource",
      "pub trait AppDataSource:",
      "impl<T> AppDataSource for T where",
      "pub trait WorkspaceAppDataSource: Send + Sync",
      "pub trait SkillAppDataSource: Send + Sync",
      "pub trait WorkspaceSkillBindingAppDataSource: Send + Sync",
      "impl WorkspaceAppDataSource for LocalAppDataSource",
      "impl SkillAppDataSource for LocalAppDataSource",
      "impl WorkspaceSkillBindingAppDataSource for LocalAppDataSource",
      "async fn list_workspaces(&self) -> Result<WorkspaceListResponse, RuntimeCoreError>",
      "async fn read_workspace(",
      "async fn read_workspace_by_path(",
      "async fn ensure_workspace_ready(",
      "async fn read_workspace_projects_root(",
      "async fn resolve_workspace_project_path(",
      "async fn read_skill(",
      "async fn list_workspace_skill_bindings(",
      "fn row_to_workspace_value(row: &Row<'_>)",
      "fn ensure_current_default_workspace(",
      "fn list_workspace_skill_bindings_value(",
      '"当前只返回 workspace 本地注册 Skill 的只读 readiness；不会 reload Skill，也不会注入默认工具面。"',
    ],
  },
  {
    name: "Runtime projection store preserves current session list cwd filters",
    files: [
      "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
      "lime-rs/crates/app-server/src/runtime/session_list_scope.rs",
      "lime-rs/crates/app-server/src/runtime/projection_store.rs",
      "lime-rs/crates/app-server/src/runtime/tests/session_list_projection.rs",
      "lime-rs/crates/app-server/src/runtime/tests/sessions.rs",
    ],
    snippets: [
      "pub struct SessionListScope",
      "pub fn from_params(params: &AgentSessionListParams) -> Self",
      "normalize_cwd_filter(params.cwd.as_ref())",
      "workspace_id_filters: normalize_id_filter(params.workspace_id.as_deref())",
      "pub fn matches_session(&self, workspace_id: Option<&str>, cwd: Option<&str>) -> bool",
      "let scope = SessionListScope::from_params(params);",
      "let cwd_filters = scope.cwd_filters();",
      'Some(format!("working_dir IN ({placeholders})"))',
      'format!(" AND (({cwd}) OR ({workspace_id}))")',
      "params.include_archived.unwrap_or(false)",
      "params.archived_only.unwrap_or(false)",
      "(?1 = 1 AND archived_at IS NOT NULL)",
      "(?1 = 0 AND (?2 = 1 OR archived_at IS NULL))",
      "pub fn list_session_overviews(",
      ".list_session_overviews(&params)",
      "list_agent_sessions_filters_projection_by_cwd",
    ],
    absentSnippets: [
      "pub(crate) fn list_current_timeline_sessions(",
      "pub(crate) fn resolve_session_list_scope(",
      "fn query_current_timeline_session_overviews(",
      "pub fn archive_many_sessions(",
      ".archive_many_sessions(",
    ],
  },
  {
    name: "App Server protocol exposes current app data surface methods and DTOs",
    files: rustProtocolFiles,
    snippets: [
      'pub const METHOD_KNOWLEDGE_PACK_LIST: &str = "knowledgePack/list"',
      'pub const METHOD_KNOWLEDGE_PACK_READ: &str = "knowledgePack/read"',
      'pub const METHOD_KNOWLEDGE_SOURCE_IMPORT: &str = "knowledgePack/source/import"',
      'pub const METHOD_KNOWLEDGE_PACK_COMPILE: &str = "knowledgePack/compile"',
      'pub const METHOD_KNOWLEDGE_PACK_DEFAULT_SET: &str = "knowledgePack/default/set"',
      'pub const METHOD_KNOWLEDGE_PACK_STATUS_UPDATE: &str = "knowledgePack/status/update"',
      'pub const METHOD_KNOWLEDGE_CONTEXT_RESOLVE: &str = "knowledgeContext/resolve"',
      'pub const METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE: &str = "knowledgeContextRun/validate"',
      'pub const METHOD_SCHEDULED_TASK_LIST: &str = "scheduledTask/list"',
      'pub const METHOD_PROJECT_MEMORY_READ: &str = "projectMemory/read"',
      'pub const METHOD_MEMORY_STORE_LIST: &str = "memoryStore/list"',
      'pub const METHOD_MEMORY_STORE_READ: &str = "memoryStore/read"',
      'pub const METHOD_MEMORY_STORE_SEARCH: &str = "memoryStore/search"',
      'pub const METHOD_MEMORY_STORE_ADD_NOTE: &str = "memoryStore/addNote"',
      'pub const METHOD_MEMORY_STORE_CONSOLIDATE: &str = "memoryStore/consolidate"',
      'pub const METHOD_MEMORY_STORE_REVIEW_LIST: &str = "memoryStore/review/list"',
      'pub const METHOD_MEMORY_STORE_REVIEW_RESOLVE: &str = "memoryStore/review/resolve"',
      'pub const METHOD_MEMORY_STORE_HEALTH: &str = "memoryStore/health"',
      'pub const METHOD_MEMORY_RESET: &str = "memory/reset"',
      'pub const METHOD_MEMORY_STORE_INDEX_REBUILD: &str = "memoryStore/index/rebuild"',
      "pub struct KnowledgeListPacksParams",
      "pub working_dir: String",
      "pub include_archived: bool",
      "pub struct KnowledgeListPacksResponse",
      "pub root_path: String",
      "pub packs: Vec<serde_json::Value>",
      "pub struct KnowledgeReadPackParams",
      "pub name: String",
      "pub struct KnowledgeReadPackResponse",
      "pub pack: serde_json::Value",
      "pub struct KnowledgeImportSourceParams",
      "pub pack_name: String",
      "pub source_text: Option<String>",
      "pub struct KnowledgeImportSourceResponse",
      "pub source: serde_json::Value",
      "pub struct KnowledgeCompilePackParams",
      "pub builder_runtime: Option<serde_json::Value>",
      "pub struct KnowledgeCompilePackResponse",
      "pub selected_source_count: u32",
      "pub compiled_view: serde_json::Value",
      "pub struct KnowledgeSetDefaultPackParams",
      "pub struct KnowledgeSetDefaultPackResponse",
      "pub default_pack_name: String",
      "pub struct KnowledgeUpdatePackStatusParams",
      "pub status: String",
      "pub struct KnowledgeUpdatePackStatusResponse",
      "pub previous_status: String",
      "pub struct KnowledgeResolveContextParams",
      "pub struct KnowledgeContextResolutionResponse",
      "pub fenced_context: String",
      "pub struct KnowledgeValidateContextRunParams",
      "pub struct KnowledgeValidateContextRunResponse",
      "pub valid: bool",
      "pub struct ScheduledTaskListResponse",
      "pub items: Vec<ScheduledTaskSummary>",
      "pub struct ProjectMemoryReadParams",
      "pub project_id: String",
      "pub struct ProjectMemoryReadResponse",
      "pub memory: serde_json::Value",
      "pub struct MemoryStoreListParams",
      "pub struct MemoryStoreReadParams",
      "pub struct MemoryStoreSearchParams",
      "pub struct MemoryStoreAddNoteParams",
      "pub struct MemoryStoreConsolidateParams",
      "pub struct MemoryStoreReviewListParams",
      "pub enum MemoryStoreReviewResolveAction",
      "pub struct MemoryStoreReviewResolveParams",
      "pub struct MemoryResetResponse",
      "pub struct MemoryStoreListResponse",
      "pub struct MemoryStoreReadResponse",
      "pub struct MemoryStoreSearchResponse",
      "pub struct MemoryStoreAddNoteResponse",
      "pub struct MemoryStoreConsolidateResponse",
      "pub struct MemoryStoreReviewNote",
      "pub struct MemoryStoreReviewListResponse",
      "pub struct MemoryStoreReviewResolveResponse",
      "pub struct MemoryStoreHealthResponse",
      "method: METHOD_KNOWLEDGE_PACK_LIST,",
      "method: METHOD_KNOWLEDGE_PACK_READ,",
      "method: METHOD_SCHEDULED_TASK_LIST,",
      "method: METHOD_PROJECT_MEMORY_READ,",
      "method: METHOD_MEMORY_STORE_LIST,",
      "method: METHOD_MEMORY_STORE_READ,",
      "method: METHOD_MEMORY_STORE_SEARCH,",
      "method: METHOD_MEMORY_STORE_ADD_NOTE,",
      "method: METHOD_MEMORY_STORE_CONSOLIDATE,",
      "method: METHOD_MEMORY_STORE_REVIEW_LIST,",
      "method: METHOD_MEMORY_STORE_REVIEW_RESOLVE,",
      "method: METHOD_MEMORY_STORE_HEALTH,",
      "method: METHOD_MEMORY_STORE_INDEX_REBUILD,",
      "KnowledgeListPacksParams",
      "KnowledgeListPacksResponse",
      "KnowledgeReadPackParams",
      "KnowledgeReadPackResponse",
      "ScheduledTaskListResponse",
      "ProjectMemoryReadParams",
      "ProjectMemoryReadResponse",
      "MemoryStoreListParams",
      "MemoryStoreReadParams",
      "MemoryStoreSearchParams",
      "MemoryStoreAddNoteParams",
      "MemoryStoreConsolidateParams",
      "MemoryStoreReviewListParams",
      "MemoryStoreReviewResolveAction",
      "MemoryStoreReviewResolveParams",
      "MemoryResetResponse",
      "MemoryStoreListResponse",
      "MemoryStoreReadResponse",
      "MemoryStoreSearchResponse",
      "MemoryStoreAddNoteResponse",
      "MemoryStoreConsolidateResponse",
      "MemoryStoreReviewNote",
      "MemoryStoreReviewListResponse",
      "MemoryStoreReviewResolveResponse",
      "MemoryStoreHealthResponse",
    ],
    absentSnippets: [
      "UnifiedMemory",
      "unifiedMemory/list",
      "unifiedMemory/create",
      "unifiedMemory/update",
      "unifiedMemory/delete",
      "unifiedMemory/search",
      "unifiedMemory/stats",
      "unifiedMemory/analyze",
      "unifiedMemory/semanticSearch",
      "METHOD_AUTOMATION_",
      "AutomationJobListResponse",
      "automationJob/",
      "automationSchedule/",
      "automationScheduler/",
      "unifiedMemory/hybridSearch",
      "METHOD_PLUGIN_INSTALLED_LIST",
      "METHOD_PLUGIN_HOST_LIFECYCLE_LIST",
      "METHOD_PLUGIN_SHELL_PREPARE",
      "METHOD_PLUGIN_UI_RUNTIME_START",
      "METHOD_PLUGIN_UI_RUNTIME_STATUS",
      "METHOD_PLUGIN_UI_RUNTIME_STOP",
      '"pluginInstalled/list"',
      '"pluginHostLifecycle/list"',
      '"pluginShell/prepare"',
      '"pluginUiRuntime/start"',
      '"pluginUiRuntime/status"',
      '"pluginUiRuntime/stop"',
      "PluginInstalledListResponse",
      "PluginShellPrepareParams",
      "PluginUiRuntimeStartParams",
      "PluginUiRuntimeStatusParams",
      "PluginUiRuntimeStopParams",
      "PluginUiRuntimeStatusResponse",
    ],
  },
  {
    name: "Rust App Server runtime and data source expose current app data surface",
    files: [
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/runtime/plugins.rs",
      "lime-rs/crates/app-server/src/runtime/app_data.rs",
      "lime-rs/crates/app-server/src/runtime/automation.rs",
      "lime-rs/crates/app-server/src/runtime/knowledge.rs",
      "lime-rs/crates/app-server/src/runtime/memory.rs",
      "lime-rs/crates/app-server/src/local_data_source.rs",
      "lime-rs/crates/app-server/src/local_data_source/automation.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/plugins.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/automation_overview.rs",
      "lime-rs/crates/app-server/src/local_data_source/impls/memory.rs",
      "lime-rs/crates/app-server/src/local_data_source/knowledge.rs",
    ],
    snippets: [
      "async fn list_knowledge_packs(",
      "params: KnowledgeListPacksParams",
      "Result<KnowledgeListPacksResponse, RuntimeCoreError>",
      "async fn read_knowledge_pack(",
      "params: KnowledgeReadPackParams",
      "Result<KnowledgeReadPackResponse, RuntimeCoreError>",
      "async fn import_knowledge_source(",
      "params: KnowledgeImportSourceParams",
      "Result<KnowledgeImportSourceResponse, RuntimeCoreError>",
      "async fn compile_knowledge_pack(",
      "params: KnowledgeCompilePackParams",
      "Result<KnowledgeCompilePackResponse, RuntimeCoreError>",
      "async fn set_default_knowledge_pack(",
      "params: KnowledgeSetDefaultPackParams",
      "async fn update_knowledge_pack_status(",
      "params: KnowledgeUpdatePackStatusParams",
      "async fn resolve_knowledge_context(",
      "params: KnowledgeResolveContextParams",
      "async fn validate_knowledge_context_run(",
      "params: KnowledgeValidateContextRunParams",
      "async fn list_scheduled_tasks(",
      "Result<ScheduledTaskListResponse, RuntimeCoreError>",
      "async fn read_project_memory(",
      "params: ProjectMemoryReadParams",
      "Result<ProjectMemoryReadResponse, RuntimeCoreError>",
      "async fn list_memory_store(",
      "params: MemoryStoreListParams",
      "Result<MemoryStoreListResponse, RuntimeCoreError>",
      "async fn read_memory_store(",
      "params: MemoryStoreReadParams",
      "Result<MemoryStoreReadResponse, RuntimeCoreError>",
      "async fn search_memory_store(",
      "params: MemoryStoreSearchParams",
      "Result<MemoryStoreSearchResponse, RuntimeCoreError>",
      "async fn add_memory_store_note(",
      "params: MemoryStoreAddNoteParams",
      "Result<MemoryStoreAddNoteResponse, RuntimeCoreError>",
      "async fn consolidate_memory_store(",
      "params: MemoryStoreConsolidateParams",
      "Result<MemoryStoreConsolidateResponse, RuntimeCoreError>",
      "async fn list_memory_store_review_notes(",
      "params: MemoryStoreReviewListParams",
      "Result<MemoryStoreReviewListResponse, RuntimeCoreError>",
      "async fn resolve_memory_store_review_note(",
      "params: MemoryStoreReviewResolveParams",
      "Result<MemoryStoreReviewResolveResponse, RuntimeCoreError>",
      "async fn health_memory_store(",
      "params: MemoryStoreRootParams",
      "Result<MemoryStoreHealthResponse, RuntimeCoreError>",
      "async fn reset_memory(&self) -> Result<(), RuntimeCoreError>",
      "self.app_data_source.list_knowledge_packs(params).await",
      "self.app_data_source.read_knowledge_pack(params).await",
      "self.app_data_source.import_knowledge_source(params).await",
      "lime_knowledge::plan_knowledge_builder_runtime(&request)",
      "self.knowledge_builder_runtime_executor",
      "self.app_data_source.compile_knowledge_pack(request).await",
      ".set_default_knowledge_pack(params)",
      ".update_knowledge_pack_status(params)",
      "self.app_data_source.resolve_knowledge_context(params).await",
      ".validate_knowledge_context_run(params)",
      "self.app_data_source.list_scheduled_tasks(params).await",
      "self.app_data_source.read_project_memory(params).await",
      "self.app_data_source.list_memory_store(params).await",
      "self.app_data_source.read_memory_store(params).await",
      "self.app_data_source.search_memory_store(params).await",
      "self.app_data_source.add_memory_store_note(params).await",
      "self.app_data_source.consolidate_memory_store(params).await",
      "self.app_data_source.health_memory_store(params).await",
      "self.app_data_source.reset_memory().await",
      "lime_knowledge::list_knowledge_packs(lime_knowledge::KnowledgeListPacksRequest",
      "lime_knowledge::get_knowledge_pack(lime_knowledge::KnowledgeGetPackRequest",
      "lime_knowledge::import_knowledge_source(lime_knowledge::KnowledgeImportSourceRequest",
      "lime_knowledge::compile_knowledge_pack(request)",
      "lime_knowledge::set_default_knowledge_pack(",
      "lime_knowledge::update_knowledge_pack_status(",
      "lime_knowledge::resolve_knowledge_context(",
      "lime_knowledge::validate_knowledge_context_run(",
      "automation::list_scheduled_tasks(&self.db, params)",
      "lime_core::memory::read_project_memory(self.db.clone(), &params.project_id)",
      "LocalMemoryBackend::new(data_root)",
      "memory_backend: Arc<dyn MemoryBackend>",
    ],
    absentSnippets: [
      "list_plugin_installed",
      "prepare_plugin_shell",
      "start_plugin_ui_runtime",
      "plugin_ui_runtime_status",
      "stop_plugin_ui_runtime",
      "PluginInstalledListResponse",
      "PluginShellPrepareParams",
      "PluginUiRuntimeStartParams",
      "PluginUiRuntimeStatusParams",
      "PluginUiRuntimeStopParams",
      "PluginUiRuntimeStatusResponse",
    ],
  },
  {
    name: "Rust JSON-RPC router dispatches current app data surface into RuntimeCore",
    files: appServerProcessorFiles,
    snippets: [
      "METHOD_KNOWLEDGE_PACK_LIST => self.handle_knowledge_pack_list_impl(params).await",
      "METHOD_KNOWLEDGE_PACK_READ => self.handle_knowledge_pack_read_impl(params).await",
      "METHOD_KNOWLEDGE_SOURCE_IMPORT =>",
      "METHOD_KNOWLEDGE_PACK_COMPILE =>",
      "METHOD_KNOWLEDGE_PACK_DEFAULT_SET =>",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE =>",
      "METHOD_KNOWLEDGE_CONTEXT_RESOLVE =>",
      "self.handle_knowledge_context_resolve_impl(params).await",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE =>",
      "METHOD_SCHEDULED_TASK_LIST => self.handle_scheduled_task_list_impl(params).boxed()",
      "METHOD_PROJECT_MEMORY_READ => self.handle_project_memory_read_impl(params).await",
      "METHOD_MEMORY_STORE_LIST => self.handle_memory_store_list_impl(params).await",
      "METHOD_MEMORY_STORE_READ => self.handle_memory_store_read_impl(params).await",
      "METHOD_MEMORY_STORE_SEARCH => self.handle_memory_store_search_impl(params).await",
      "METHOD_MEMORY_STORE_ADD_NOTE =>",
      "METHOD_MEMORY_STORE_CONSOLIDATE =>",
      "METHOD_MEMORY_STORE_HEALTH => self.handle_memory_store_health_impl(params).await",
      "METHOD_MEMORY_RESET => {",
      "self.handle_memory_reset_impl().boxed()",
      "METHOD_MEMORY_STORE_INDEX_REBUILD =>",
      "let params: KnowledgeListPacksParams = parse_params(params)?",
      ".list_knowledge_packs(params)",
      "let params: KnowledgeReadPackParams = parse_params(params)?",
      ".read_knowledge_pack(params)",
      "let params: KnowledgeImportSourceParams = parse_params(params)?",
      ".import_knowledge_source(params)",
      "let params: KnowledgeCompilePackParams = parse_params(params)?",
      ".compile_knowledge_pack(params)",
      "let params: KnowledgeSetDefaultPackParams = parse_params(params)?",
      ".set_default_knowledge_pack(params)",
      "let params: KnowledgeUpdatePackStatusParams = parse_params(params)?",
      ".update_knowledge_pack_status(params)",
      "let params: KnowledgeResolveContextParams = parse_params(params)?",
      ".resolve_knowledge_context(params)",
      "let params: KnowledgeValidateContextRunParams = parse_params(params)?",
      ".validate_knowledge_context_run(params)",
      "async fn handle_scheduled_task_list_impl(",
      ".list_scheduled_tasks(params)",
      "let params: ProjectMemoryReadParams = parse_params(params)?",
      ".read_project_memory(params)",
      "let params: MemoryStoreListParams = parse_params(params)?",
      ".list_memory_store(params)",
      "let params: MemoryStoreReadParams = parse_params(params)?",
      ".read_memory_store(params)",
      "let params: MemoryStoreSearchParams = parse_params(params)?",
      ".search_memory_store(params)",
      "let params: MemoryStoreAddNoteParams = parse_params(params)?",
      ".add_memory_store_note(params)",
      "let params: MemoryStoreConsolidateParams = parse_params(params)?",
      ".consolidate_memory_store(params)",
      "let params: MemoryStoreReviewListParams = parse_params(params)?",
      ".list_memory_store_review_notes(params)",
      "let params: MemoryStoreReviewResolveParams = parse_params(params)?",
      ".resolve_memory_store_review_note(params)",
      "let params: MemoryStoreRootParams = parse_params(params)?",
      ".health_memory_store(params)",
      "async fn handle_memory_reset_impl(&self)",
      "self.runtime.reset_memory().await",
    ],
    absentSnippets: [
      "mod unified;",
      "processor/unified.rs",
      'METHOD_MEMORY_STORE_RESET: &str = "memoryStore/reset"',
      "MemoryStoreResetParams",
      "MemoryStoreResetResponse",
      "handle_memory_store_reset_impl",
      "reset_memory_store",
      "METHOD_PLUGIN_INSTALLED_LIST",
      "METHOD_PLUGIN_SHELL_PREPARE",
      "METHOD_PLUGIN_UI_RUNTIME_START",
      "METHOD_PLUGIN_UI_RUNTIME_STATUS",
      "METHOD_PLUGIN_UI_RUNTIME_STOP",
      "handle_plugin_installed_list_impl",
      "handle_plugin_shell_prepare_impl",
      "handle_plugin_ui_runtime_start_impl",
      "handle_plugin_ui_runtime_status_impl",
      "handle_plugin_ui_runtime_stop_impl",
    ],
  },
  {
    name: "Rust app-server-client exposes typed helpers for current app data surface",
    file: "lime-rs/crates/app-server-client/src/lib.rs",
    snippets: [
      "pub use app_server_protocol::KnowledgeListPacksParams",
      "pub use app_server_protocol::KnowledgeListPacksResponse",
      "pub use app_server_protocol::KnowledgeReadPackParams",
      "pub use app_server_protocol::KnowledgeReadPackResponse",
      "pub use app_server_protocol::KnowledgeImportSourceParams",
      "pub use app_server_protocol::KnowledgeImportSourceResponse",
      "pub use app_server_protocol::KnowledgeCompilePackParams",
      "pub use app_server_protocol::KnowledgeCompilePackResponse",
      "pub use app_server_protocol::KnowledgeSetDefaultPackParams",
      "pub use app_server_protocol::KnowledgeSetDefaultPackResponse",
      "pub use app_server_protocol::KnowledgeUpdatePackStatusParams",
      "pub use app_server_protocol::KnowledgeUpdatePackStatusResponse",
      "pub use app_server_protocol::KnowledgeResolveContextParams",
      "pub use app_server_protocol::KnowledgeContextResolutionResponse",
      "pub use app_server_protocol::KnowledgeValidateContextRunParams",
      "pub use app_server_protocol::KnowledgeValidateContextRunResponse",
      "pub use app_server_protocol::ProjectMemoryReadParams",
      "pub use app_server_protocol::ProjectMemoryReadResponse",
      "pub use app_server_protocol::MemoryStoreListParams",
      "pub use app_server_protocol::MemoryStoreReadParams",
      "pub use app_server_protocol::MemoryStoreSearchParams",
      "pub use app_server_protocol::MemoryStoreAddNoteParams",
      "pub use app_server_protocol::MemoryStoreConsolidateParams",
      "pub use app_server_protocol::MemoryStoreReviewListParams",
      "pub use app_server_protocol::MemoryStoreReviewResolveAction",
      "pub use app_server_protocol::MemoryStoreReviewResolveParams",
      "pub use app_server_protocol::MemoryResetResponse",
      "pub use app_server_protocol::MemoryStoreListResponse",
      "pub use app_server_protocol::MemoryStoreReadResponse",
      "pub use app_server_protocol::MemoryStoreSearchResponse",
      "pub use app_server_protocol::MemoryStoreAddNoteResponse",
      "pub use app_server_protocol::MemoryStoreConsolidateResponse",
      "pub use app_server_protocol::MemoryStoreReviewNote",
      "pub use app_server_protocol::MemoryStoreReviewListResponse",
      "pub use app_server_protocol::MemoryStoreReviewResolveResponse",
      "pub use app_server_protocol::MemoryStoreHealthResponse",
      "pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_LIST",
      "pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_READ",
      "pub use app_server_protocol::METHOD_PROJECT_MEMORY_READ",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_LIST",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_READ",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_SEARCH",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_ADD_NOTE",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_CONSOLIDATE",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_HEALTH",
      "pub use app_server_protocol::METHOD_MEMORY_RESET",
      "pub use app_server_protocol::METHOD_MEMORY_STORE_INDEX_REBUILD",
      "pub fn list_knowledge_packs(",
      "pub fn read_knowledge_pack(",
      "pub fn read_project_memory(",
      "pub fn list_memory_store(",
      "pub fn read_memory_store(",
      "pub fn search_memory_store(",
      "pub fn add_memory_store_note(",
      "pub fn consolidate_memory_store(",
      "pub fn health_memory_store(",
      "pub fn reset_memory(&mut self)",
      "TypedRequest::new(METHOD_KNOWLEDGE_PACK_LIST, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_PACK_READ, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_SOURCE_IMPORT, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_PACK_COMPILE, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_PACK_DEFAULT_SET, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_PACK_STATUS_UPDATE, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_CONTEXT_RESOLVE, params)",
      "TypedRequest::new(METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE, params)",
      "TypedRequest::new(METHOD_PROJECT_MEMORY_READ, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_LIST, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_READ, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_SEARCH, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_ADD_NOTE, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_CONSOLIDATE, params)",
      "TypedRequest::new(METHOD_MEMORY_STORE_HEALTH, params)",
      "TypedRequest::new(METHOD_MEMORY_RESET, serde_json::json!({}))",
      "TypedRequest::new(METHOD_MEMORY_STORE_INDEX_REBUILD, params)",
      "fn app_data_surface_helpers_use_current_methods()",
    ],
    absentSnippets: [
      "PluginInstalledListResponse",
      "PluginLocalPackageExportParams",
      "PluginLocalPackageExportResponse",
      "PluginShellPrepareParams",
      "PluginUiRuntimeStartParams",
      "PluginUiRuntimeStatusParams",
      "PluginUiRuntimeStopParams",
      "METHOD_PLUGIN_INSTALLED_LIST",
      "METHOD_PLUGIN_LOCAL_PACKAGE_EXPORT",
      "METHOD_PLUGIN_SHELL_PREPARE",
      "METHOD_PLUGIN_UI_RUNTIME_START",
      "METHOD_PLUGIN_UI_RUNTIME_STATUS",
      "METHOD_PLUGIN_UI_RUNTIME_STOP",
      "list_plugin_installed",
      "export_plugin_local_package",
      "prepare_plugin_shell",
      "plugin_ui_runtime",
      "AutomationJob",
      "AutomationSchedule",
      "AutomationScheduler",
      "METHOD_AUTOMATION_",
      "automation_job",
    ],
  },
  {
    name: "TypeScript app-server-client mirrors current app data surface",
    files: [
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/index.ts",
      "packages/app-server-client/tests/client.test.mjs",
    ],
    snippets: [
      'export const METHOD_KNOWLEDGE_PACK_LIST = "knowledgePack/list"',
      'export const METHOD_KNOWLEDGE_PACK_READ = "knowledgePack/read"',
      'export const METHOD_KNOWLEDGE_SOURCE_IMPORT = "knowledgePack/source/import"',
      'export const METHOD_KNOWLEDGE_PACK_COMPILE = "knowledgePack/compile"',
      'export const METHOD_KNOWLEDGE_PACK_DEFAULT_SET = "knowledgePack/default/set"',
      "export const METHOD_KNOWLEDGE_PACK_STATUS_UPDATE =",
      'export const METHOD_KNOWLEDGE_CONTEXT_RESOLVE = "knowledgeContext/resolve"',
      "export const METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE =",
      'export const METHOD_SCHEDULED_TASK_LIST = "scheduledTask/list"',
      'export const METHOD_PROJECT_MEMORY_READ = "projectMemory/read"',
      'export const METHOD_MEMORY_STORE_LIST = "memoryStore/list"',
      'export const METHOD_MEMORY_STORE_READ = "memoryStore/read"',
      'export const METHOD_MEMORY_STORE_SEARCH = "memoryStore/search"',
      'export const METHOD_MEMORY_STORE_ADD_NOTE = "memoryStore/addNote"',
      'export const METHOD_MEMORY_STORE_CONSOLIDATE = "memoryStore/consolidate"',
      'export const METHOD_MEMORY_STORE_REVIEW_LIST = "memoryStore/review/list"',
      "export const METHOD_MEMORY_STORE_REVIEW_RESOLVE =",
      'export const METHOD_MEMORY_STORE_HEALTH = "memoryStore/health"',
      'export const METHOD_MEMORY_RESET = "memory/reset"',
      'export const METHOD_MEMORY_STORE_INDEX_REBUILD = "memoryStore/index/rebuild"',
      "export type KnowledgeListPacksParams",
      "export type KnowledgeListPacksResponse",
      "export type KnowledgeReadPackParams",
      "export type KnowledgeReadPackResponse",
      "export type KnowledgeImportSourceParams",
      "export type KnowledgeImportSourceResponse",
      "export type KnowledgeCompilePackParams",
      "export type KnowledgeCompilePackResponse",
      "export type KnowledgeSetDefaultPackParams",
      "export type KnowledgeSetDefaultPackResponse",
      "export type KnowledgeUpdatePackStatusParams",
      "export type KnowledgeUpdatePackStatusResponse",
      "export type KnowledgeResolveContextParams",
      "export type KnowledgeContextResolutionResponse",
      "export type KnowledgeValidateContextRunParams",
      "export type KnowledgeValidateContextRunResponse",
      "export interface ScheduledTaskListResponse",
      "export type ProjectMemoryReadParams",
      "export type ProjectMemoryReadResponse",
      "export interface MemoryStoreListParams",
      "export interface MemoryStoreReadParams",
      "export interface MemoryStoreSearchParams",
      "export interface MemoryStoreAddNoteParams",
      "export interface MemoryStoreConsolidateParams",
      "export interface MemoryStoreReviewListParams",
      "export type MemoryStoreReviewResolveAction",
      "export interface MemoryStoreReviewResolveParams",
      "export type MemoryResetResponse",
      "export interface MemoryStoreListResponse",
      "export interface MemoryStoreReadResponse",
      "export interface MemoryStoreSearchResponse",
      "export interface MemoryStoreAddNoteResponse",
      "export interface MemoryStoreConsolidateResponse",
      "export interface MemoryStoreReviewNote",
      "export interface MemoryStoreReviewListResponse",
      "export interface MemoryStoreReviewResolveResponse",
      "export interface MemoryStoreHealthResponse",
      "listKnowledgePacks(params: protocol.KnowledgeListPacksParams): protocol.JsonRpcRequest",
      "readKnowledgePack(params: protocol.KnowledgeReadPackParams): protocol.JsonRpcRequest",
      "importKnowledgeSource(params: protocol.KnowledgeImportSourceParams): protocol.JsonRpcRequest",
      "compileKnowledgePack(params: protocol.KnowledgeCompilePackParams): protocol.JsonRpcRequest",
      "setDefaultKnowledgePack(",
      "updateKnowledgePackStatus(",
      "resolveKnowledgeContext(",
      "validateKnowledgeContextRun(",
      "readProjectMemory(params: protocol.ProjectMemoryReadParams): protocol.JsonRpcRequest",
      "listMemoryStore(params: protocol.MemoryStoreListParams): protocol.JsonRpcRequest",
      "readMemoryStore(params: protocol.MemoryStoreReadParams): protocol.JsonRpcRequest",
      "searchMemoryStore(params: protocol.MemoryStoreSearchParams): protocol.JsonRpcRequest",
      "addMemoryStoreNote(params: protocol.MemoryStoreAddNoteParams): protocol.JsonRpcRequest",
      "consolidateMemoryStore(params: protocol.MemoryStoreConsolidateParams): protocol.JsonRpcRequest",
      "listMemoryStoreReviewNotes(",
      "resolveMemoryStoreReviewNote(",
      "healthMemoryStore(params: protocol.MemoryStoreRootParams): protocol.JsonRpcRequest",
      "resetMemory(): protocol.JsonRpcRequest",
      "builds app data surface requests with current methods",
      "assert.equal(knowledge.method, METHOD_KNOWLEDGE_PACK_LIST)",
      "assert.equal(knowledgeDetail.method, METHOD_KNOWLEDGE_PACK_READ)",
      "assert.equal(importedKnowledgeSource.method, METHOD_KNOWLEDGE_SOURCE_IMPORT)",
      "assert.equal(compiledKnowledgePack.method, METHOD_KNOWLEDGE_PACK_COMPILE)",
      "assert.equal(defaultKnowledgePack.method, METHOD_KNOWLEDGE_PACK_DEFAULT_SET)",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE",
      "assert.equal(knowledgeContext.method, METHOD_KNOWLEDGE_CONTEXT_RESOLVE)",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE",
      "assert.equal(memory.method, METHOD_PROJECT_MEMORY_READ)",
      "assert.equal(memoryStoreList.method, METHOD_MEMORY_STORE_LIST)",
      "assert.equal(memoryStoreRead.method, METHOD_MEMORY_STORE_READ)",
      "assert.equal(memoryStoreSearch.method, METHOD_MEMORY_STORE_SEARCH)",
      "assert.equal(memoryStoreAddNote.method, METHOD_MEMORY_STORE_ADD_NOTE)",
      "assert.equal(memoryStoreConsolidate.method, METHOD_MEMORY_STORE_CONSOLIDATE)",
      "assert.equal(memoryStoreReviewList.method, METHOD_MEMORY_STORE_REVIEW_LIST)",
      "memoryStoreReviewResolve.method",
      "METHOD_MEMORY_STORE_REVIEW_RESOLVE",
      "assert.equal(memoryStoreHealth.method, METHOD_MEMORY_STORE_HEALTH)",
      "assert.equal(memoryReset.method, METHOD_MEMORY_RESET)",
      "METHOD_MEMORY_STORE_INDEX_REBUILD",
    ],
    absentSnippets: [
      "METHOD_UNIFIED_MEMORY",
      "UnifiedMemory",
      "unifiedMemory/",
      "listUnifiedMemories(",
      "createUnifiedMemory(",
      "updateUnifiedMemory(",
      "METHOD_AUTOMATION_",
      "AutomationJobListResponse",
      "listAutomationJobs",
      "automationJob/",
      "automationSchedule/",
      "automationScheduler/",
      "deleteUnifiedMemory(",
      'METHOD_MEMORY_STORE_RESET = "memoryStore/reset"',
      "MemoryStoreResetParams",
      "MemoryStoreResetResponse",
      "resetMemoryStore(",
      "METHOD_PLUGIN_INSTALLED_LIST",
      "METHOD_PLUGIN_HOST_LIFECYCLE_LIST",
      "METHOD_PLUGIN_SHELL_PREPARE",
      "METHOD_PLUGIN_UI_RUNTIME_START",
      "METHOD_PLUGIN_UI_RUNTIME_STATUS",
      "METHOD_PLUGIN_UI_RUNTIME_STOP",
      "PluginInstalledListResponse",
      "PluginShellPrepareParams",
      "PluginUiRuntimeStartParams",
      "PluginUiRuntimeStatusParams",
      "PluginUiRuntimeStopParams",
      "listPluginInstalled(",
      "startPluginUiRuntime(",
      "getPluginUiRuntimeStatus(",
      "stopPluginUiRuntime(",
    ],
  },
  {
    name: "Renderer project memory aggregate read uses App Server projectMemory/read",
    file: "src/lib/api/projectMemory.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_PROJECT_MEMORY_READ",
      'from "../../../packages/app-server-client/src/protocol"',
      'export type ProjectMemoryAppServerClient = Pick<AppServerClient, "request">',
      "type ProjectMemoryReadResponse = Omit<",
      "AppServerProjectMemoryReadResponse",
      "memory?: ProjectMemory | null",
      "projectId is required to read App Server project memory",
      "const appServerClient = options.appServerClient ?? new AppServerClient()",
      "appServerClient.request<ProjectMemoryReadResponse>",
      "METHOD_PROJECT_MEMORY_READ",
      "{ projectId: normalizedProjectId }",
      "App Server projectMemory/read did not return project memory",
      "projectMemoryCache.set(normalizedProjectId",
      "projectMemoryInflight.set(normalizedProjectId",
    ],
    absentSnippets: [
      '"project_memory_get"',
      'const APP_SERVER_METHOD_PROJECT_MEMORY_READ = "projectMemory/read"',
      "safeInvoke<ProjectMemory>",
      "invokeMockOnly",
    ],
  },
  {
    name: "Renderer project memory tests lock App Server current path and no legacy fallback",
    file: "src/lib/api/projectMemory.test.ts",
    snippets: [
      "createProjectMemoryClient",
      "ProjectMemoryAppServerClient",
      "projectMemory/read",
      "并发读取同一项目上下文时应复用同一个 projectMemory/read",
      "项目上下文读取缺少 projectId 时应 fail closed",
      "App Server 未返回项目上下文时不应返回空对象",
      "projectId is required to read App Server project memory",
      "App Server projectMemory/read did not return project memory",
    ],
  },
  {
    name: "Workspace Right Surface uses App Server current JSON-RPC contract",
    files: [
      ...rustProtocolFiles,
      ...appServerRuntimeFiles,
      ...appServerProcessorFiles,
      "lime-rs/crates/app-server-client/src/lib.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/index.ts",
      "packages/app-server-client/tests/client.test.mjs",
      "src/lib/api/appServer.ts",
      "src/lib/api/appServer.test.ts",
      "src/lib/api/workspaceRightSurface.ts",
      "src/lib/api/workspaceRightSurface.test.ts",
      "src/lib/governance/agentCommandCatalog.json",
    ],
    snippets: [
      'pub const METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST: &str = "workspaceRightSurface/request"',
      '"workspaceRightSurface/pending/list"',
      '"workspaceRightSurface/pending/consume"',
      '"workspaceRightSurface/pending/dismiss"',
      '"workspaceRightSurface/pendingChanged"',
      "pub struct WorkspaceRightSurfaceRequestParams",
      "pub struct WorkspaceRightSurfacePendingListParams",
      "pub struct WorkspaceRightSurfacePendingConsumeParams",
      "pub struct WorkspaceRightSurfacePendingDismissParams",
      "pub struct WorkspaceRightSurfacePendingRequest",
      "pub struct WorkspaceRightSurfaceRequestResponse",
      "pub struct WorkspaceRightSurfacePendingListResponse",
      "pub struct WorkspaceRightSurfacePendingConsumeResponse",
      "pub struct WorkspaceRightSurfacePendingDismissResponse",
      "pub struct WorkspaceRightSurfacePendingChangedParams",
      "method: METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,",
      "method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,",
      "method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,",
      "method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,",
      "method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED,",
      "request_workspace_right_surface(",
      "list_workspace_right_surface_pending(",
      "consume_workspace_right_surface_pending(",
      "dismiss_workspace_right_surface_pending(",
      "METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST =>",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST =>",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME =>",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS =>",
      "handle_workspace_right_surface_request_impl",
      "handle_workspace_right_surface_pending_list_impl",
      "handle_workspace_right_surface_pending_consume_impl",
      "handle_workspace_right_surface_pending_dismiss_impl",
      "pub use app_server_protocol::WorkspaceRightSurfaceRequestParams",
      "pub use app_server_protocol::WorkspaceRightSurfacePendingConsumeParams",
      "pub use app_server_protocol::WorkspaceRightSurfacePendingDismissParams",
      "pub use app_server_protocol::WorkspaceRightSurfacePendingChangedParams",
      "pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST",
      "pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME",
      "pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS",
      "pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED",
      "export const METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST =",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS",
      "METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED",
      "export interface WorkspaceRightSurfaceRequestParams",
      "export interface WorkspaceRightSurfacePendingListResponse",
      "export interface WorkspaceRightSurfacePendingConsumeResponse",
      "export interface WorkspaceRightSurfacePendingDismissResponse",
      "export interface WorkspaceRightSurfacePendingChangedParams",
      "workspaceRightSurfacePendingChangedNotification(",
      "requestWorkspaceRightSurface(",
      "listWorkspaceRightSurfacePending(",
      "consumeWorkspaceRightSurfacePending(",
      "dismissWorkspaceRightSurfacePending(",
      "APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST",
      "APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME",
      "APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS",
      "APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED",
      'import { AppServerClient } from "@/lib/api/appServer"',
      "export type WorkspaceRightSurfaceAppServerClient = Pick<",
      "App Server workspaceRightSurface/request did not return a valid pending request",
      "App Server workspaceRightSurface/pending/list did not return valid pending requests",
      "App Server workspaceRightSurface/pending/consume did not return consumed request ids",
      "App Server workspaceRightSurface/pending/dismiss did not return dismissed request ids",
      "workspaceRightSurface methods 应通过 App Server JSON-RPC 调度右侧 surface",
      "Right Surface 请求应走 App Server current method",
      '"appServerWorkspaceRightSurfaceMethods"',
      '"workspaceRightSurface/request"',
      '"workspaceRightSurface/pending/list"',
      '"workspaceRightSurface/pending/consume"',
      '"workspaceRightSurface/pending/dismiss"',
      '"workspaceRightSurface/pendingChanged"',
    ],
    absentSnippets: [
      '"agent_runtime_request_right_surface"',
      '"right_surface_request"',
      "invokeMockOnly",
      "defaultMocks",
      "mockPriorityCommands",
    ],
  },
  {
    name: "Retired Browser Session protocol stays out of current contracts",
    files: [
      ...rustProtocolFiles,
      ...appServerRuntimeFiles,
      ...appServerProcessorFiles,
      "lime-rs/Cargo.toml",
      "lime-rs/crates/app-server/Cargo.toml",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/connection-methods.ts",
      "packages/app-server-client/tests/client.test.mjs",
      "src/lib/governance/agentCommandCatalog.json",
    ],
    snippets: [],
    absentSnippets: [
      "browserSession/",
      "METHOD_BROWSER_SESSION",
      "BrowserSessionTargetListParams",
      "BrowserSessionOpenParams",
      "BrowserSessionActionExecuteParams",
      "listBrowserSessionTargets",
      "openBrowserSession",
      "executeBrowserSessionAction",
      "lime-browser-runtime",
      "lime_browser_runtime",
      '"appServerBrowserSessionMethods"',
    ],
  },
  {
    name: "Renderer project workspace read and ensure use App Server current methods",
    file: "src/lib/api/project.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_WORKSPACE_LIST",
      "METHOD_WORKSPACE_READ",
      "METHOD_WORKSPACE_BY_PATH_READ",
      "METHOD_WORKSPACE_ENSURE",
      "METHOD_WORKSPACE_DEFAULT_READ",
      "METHOD_WORKSPACE_DEFAULT_ENSURE",
      "METHOD_WORKSPACE_PROJECTS_ROOT_READ",
      "METHOD_WORKSPACE_PROJECT_PATH_RESOLVE",
      "METHOD_WORKSPACE_ENSURE_READY",
      'from "../../../packages/app-server-client/src/protocol"',
      'type ProjectAppServerClient = Pick<AppServerClient, "request">',
      "async function requestProjectAppServer<T>",
      "appServerClient.request<T>(method, params)",
      "return response.result",
      "METHOD_WORKSPACE_PROJECTS_ROOT_READ",
      "METHOD_WORKSPACE_PROJECT_PATH_RESOLVE",
      "METHOD_WORKSPACE_LIST",
      "METHOD_WORKSPACE_DEFAULT_READ",
      "METHOD_WORKSPACE_ENSURE",
      "METHOD_WORKSPACE_ENSURE_READY",
      "METHOD_WORKSPACE_DEFAULT_ENSURE",
      "METHOD_WORKSPACE_BY_PATH_READ",
      "METHOD_WORKSPACE_READ",
      "workspace id is required to ensure App Server workspace",
      "workspace rootPath is required to ensure project",
      "App Server workspace/projectsRoot/read did not return rootPath",
      "App Server workspace/projectPath/resolve did not return rootPath",
      "App Server workspace/ensure did not return workspace",
      "App Server workspace/ensureReady did not return result",
      "App Server workspace/default/ensure did not return workspace",
    ],
    absentSnippets: [
      'const APP_SERVER_METHOD_WORKSPACE_LIST = "workspace/list"',
      '"workspace_get_projects_root"',
      '"workspace_resolve_project_path"',
      '"workspace_get_by_path"',
      '"workspace_get_default"',
      '"workspace_ensure_ready"',
      '"workspace_ensure_default_ready"',
      '"get_or_create_default_project"',
      '"workspace_list"',
    ],
  },
  {
    name: "Renderer project workspace tests lock App Server current path and no legacy fallback",
    file: "src/lib/api/project.test.ts",
    snippets: [
      "appServerRequestMock",
      'vi.mock("@/lib/api/appServer"',
      "应该通过 App Server 获取 workspace 根目录",
      "应该通过 App Server 解析项目目录",
      "应该通过 App Server 按路径获取项目",
      "应该通过 App Server ensure 本地项目工作区",
      "应该通过 App Server 获取并标准化默认项目",
      "应该通过 App Server 确保工作区目录就绪",
      "确保默认工作区目录就绪时应先 ensure 默认 workspace 再 ensureReady",
      "workspace read/ensure 缺少必需 App Server result 时不应回退 legacy",
      "ensureWorkspaceReady 缺少 workspace id 时应 fail closed",
      '"workspace/projectsRoot/read"',
      '"workspace/projectPath/resolve"',
      '"workspace/byPath/read"',
      '"workspace/ensure"',
      '"workspace/default/read"',
      '"workspace/default/ensure"',
      '"workspace/ensureReady"',
      '"workspace/list"',
      '"workspace/read"',
      '"workspace_get_projects_root"',
      '"workspace_resolve_project_path"',
      '"workspace_ensure_ready"',
      '"get_or_create_default_project"',
      "expect(safeInvoke).not.toHaveBeenCalledWith(",
    ],
  },
  {
    name: "Renderer model registry reads use App Server current methods",
    file: "src/lib/api/modelRegistry.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_MODEL_LIST",
      "METHOD_MODEL_PREFERENCES_LIST",
      "METHOD_MODEL_PROVIDER_ALIAS_LIST",
      "METHOD_MODEL_PROVIDER_ALIAS_READ",
      "METHOD_MODEL_PROVIDER_FETCH_MODELS",
      "METHOD_MODEL_SYNC_STATE_READ",
      'from "../../../packages/app-server-client/src/protocol"',
      'type ModelRegistryAppServerClient = Pick<AppServerClient, "request">',
      "async function requestModelRegistryAppServer<T>",
      "appServerClient.request<T>(method, params)",
      "return response.result",
      "async function readModelsFromAppServer(",
      "METHOD_MODEL_LIST",
      "METHOD_MODEL_PREFERENCES_LIST",
      "METHOD_MODEL_SYNC_STATE_READ",
      "METHOD_MODEL_PROVIDER_ALIAS_READ",
      "METHOD_MODEL_PROVIDER_ALIAS_LIST",
      "METHOD_MODEL_PROVIDER_FETCH_MODELS",
      "App Server model/list did not return data",
      "response.nextCursor",
      "includeHidden",
      "App Server modelPreferences/list did not return preferences",
      "App Server modelSyncState/read did not return syncState",
      "App Server modelProviderAlias/list did not return configs",
    ],
    absentSnippets: [
      '"get_model_registry"',
      '"get_model_preferences"',
      '"get_model_sync_state"',
      '"get_models_for_provider"',
      '"get_models_by_tier"',
      '"get_provider_alias_config"',
      '"get_all_alias_configs"',
      '"fetch_provider_models_auto"',
    ],
  },
  {
    name: "Renderer model registry tests lock App Server current path and no legacy fallback",
    file: "src/lib/api/modelRegistry.test.ts",
    snippets: [
      "appServerRequestMock",
      'vi.mock("@/lib/api/appServer"',
      "getModelRegistry 应缓存并复用同一轮读取结果",
      "getModelRegistry 应聚合 model/list 的全部分页",
      "getModelRegistry 遇到重复 cursor 时应 fail closed",
      "getModelRegistry 应隔离默认目录与 includeHidden 缓存",
      "模型偏好与同步状态读取应走 App Server current",
      "单个 provider alias 应通过 App Server 读取并缓存",
      "App Server 模型读链缺少必需 result 时不应回退 legacy",
      '"model/list"',
      '"modelPreferences/list"',
      '"modelSyncState/read"',
      '"modelProviderAlias/read"',
      '"modelProviderAlias/list"',
      '"modelProvider/fetchModels"',
      "App Server model/list did not return data",
      "App Server model/list repeated cursor: 1",
      "includeHidden: true",
      "App Server modelPreferences/list did not return preferences",
      "App Server modelSyncState/read did not return syncState",
      "App Server modelProviderAlias/list did not return configs",
      "expect(safeInvoke).not.toHaveBeenCalledWith(",
      '"get_model_registry"',
      '"get_model_preferences"',
      '"get_model_sync_state"',
      '"get_all_alias_configs"',
      '"fetch_provider_models_auto"',
    ],
  },
  {
    name: "Renderer API key provider uses App Server modelProvider methods",
    file: "src/lib/api/apiKeyProvider.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_MODEL_PROVIDER_CATALOG_LIST",
      "METHOD_MODEL_PROVIDER_CONFIG_EXPORT",
      "METHOD_MODEL_PROVIDER_CONFIG_IMPORT",
      "METHOD_MODEL_PROVIDER_CREATE",
      "METHOD_MODEL_PROVIDER_DELETE",
      "METHOD_MODEL_PROVIDER_KEY_CREATE",
      "METHOD_MODEL_PROVIDER_KEY_DELETE",
      "METHOD_MODEL_PROVIDER_KEY_UPDATE",
      "METHOD_MODEL_PROVIDER_LIST",
      "METHOD_MODEL_PROVIDER_READ",
      "METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE",
      "METHOD_MODEL_PROVIDER_TEST_CHAT",
      "METHOD_MODEL_PROVIDER_TEST_CONNECTION",
      "METHOD_MODEL_PROVIDER_UI_STATE_READ",
      "METHOD_MODEL_PROVIDER_UI_STATE_WRITE",
      "METHOD_MODEL_PROVIDER_UPDATE",
      'from "../../../packages/app-server-client/src/protocol"',
      'type ApiKeyProviderAppServerClient = Pick<AppServerClient, "request">',
      "async function requestApiKeyProviderAppServer<T>",
      "appServerClient.request<T>(method, params)",
      "return response.result",
      "function normalizeModelProviderListResponse(",
      "function normalizeModelProviderCatalogListResponse(",
      "METHOD_MODEL_PROVIDER_LIST",
      "METHOD_MODEL_PROVIDER_CATALOG_LIST",
      "App Server modelProvider/list did not return providers",
      "App Server modelProvider/catalog/list did not return providers",
    ],
    absentSnippets: [
      '"get_api_key_providers"',
      '"get_system_provider_catalog"',
      '"get_api_key_provider"',
      '"add_custom_api_key_provider"',
      '"update_api_key_provider"',
      '"delete_custom_api_key_provider"',
      '"add_api_key"',
      '"delete_api_key"',
      '"toggle_api_key"',
      '"update_api_key_alias"',
      '"get_next_api_key"',
      '"record_api_key_usage"',
      '"record_api_key_error"',
      '"modelProviderKey/next"',
      '"modelProviderKey/usage/record"',
      '"modelProviderKey/error/record"',
      '"get_provider_ui_state"',
      '"set_provider_ui_state"',
      '"update_provider_sort_orders"',
      '"export_api_key_providers"',
      '"import_api_key_providers"',
      '"test_api_key_provider_connection"',
      '"test_api_key_provider_chat"',
      'safeInvoke<ProviderWithKeysDisplay[]>("get_api_key_providers"',
      'safeInvoke("get_system_provider_catalog"',
      "safeInvoke(",
      "invokeMockOnly",
    ],
  },
  {
    name: "Renderer API key provider tests lock App Server current read/write path",
    file: "src/lib/api/apiKeyProvider.test.ts",
    snippets: [
      "appServerRequestMock",
      'vi.mock("@/lib/api/appServer"',
      "Provider 列表应通过 App Server modelProvider/list 读取",
      "系统 Provider Catalog 应通过 App Server modelProvider/catalog/list 读取",
      "Provider 读链缺少必需 result 时不应回退 legacy",
      "Provider 读取失败时不应注入本地 mock 或写入缓存",
      "写操作成功后应失效缓存",
      '"modelProvider/list"',
      '"modelProvider/catalog/list"',
      '"modelProviderKey/create"',
      '"modelProvider/testConnection"',
      "App Server modelProvider/list did not return providers",
      "App Server modelProvider/catalog/list did not return providers",
      "expect(safeInvoke).not.toHaveBeenCalledWith(",
      '"get_api_key_providers"',
      '"get_system_provider_catalog"',
    ],
  },
  {
    name: "Renderer knowledge gateway uses App Server Knowledge current methods",
    file: "src/lib/api/knowledge.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_KNOWLEDGE_PACK_LIST",
      "METHOD_KNOWLEDGE_PACK_READ",
      "METHOD_KNOWLEDGE_SOURCE_IMPORT",
      "METHOD_KNOWLEDGE_PACK_COMPILE",
      "METHOD_KNOWLEDGE_PACK_DEFAULT_SET",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE",
      "METHOD_KNOWLEDGE_CONTEXT_RESOLVE",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE",
      'from "../../../packages/app-server-client/src/protocol"',
      'export type KnowledgeAppServerClient = Pick<AppServerClient, "request">',
      "async function requestKnowledgeAppServer<T>",
      "appServerClient.request<T>(method, params)",
      "return response.result",
      "function normalizeKnowledgeListPacksResponse(",
      "function normalizeKnowledgeReadPackResponse(",
      "function normalizeKnowledgeImportSourceResponse(",
      "function normalizeKnowledgeCompilePackResponse(",
      "function normalizeKnowledgeSetDefaultPackResponse(",
      "function normalizeKnowledgeUpdatePackStatusResponse(",
      "function normalizeKnowledgeContextResolutionResponse(",
      "function normalizeKnowledgeValidateContextRunResponse(",
      "workingDir is required to list App Server knowledge packs",
      "workingDir is required to read App Server knowledge pack",
      "name is required to read App Server knowledge pack",
      "workingDir is required to import App Server knowledge source",
      "workingDir is required to compile App Server knowledge pack",
      "METHOD_KNOWLEDGE_PACK_LIST",
      "METHOD_KNOWLEDGE_PACK_READ",
      "METHOD_KNOWLEDGE_SOURCE_IMPORT",
      "METHOD_KNOWLEDGE_PACK_COMPILE",
      "METHOD_KNOWLEDGE_PACK_DEFAULT_SET",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE",
      "METHOD_KNOWLEDGE_CONTEXT_RESOLVE",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE",
      "workingDir: normalizedWorkingDir",
      "name: normalizedName",
      "includeArchived: request.includeArchived ?? false",
      "App Server knowledgePack/list did not return packs",
      "App Server knowledgePack/list did not return rootPath",
      "App Server knowledgePack/list did not return workingDir",
      "App Server knowledgePack/read did not return pack",
    ],
    absentSnippets: [
      '"knowledge_list_packs"',
      '"knowledge_get_pack"',
      '"knowledge_import_source"',
      '"knowledge_compile_pack"',
      '"knowledge_set_default_pack"',
      '"knowledge_update_pack_status"',
      '"knowledge_resolve_context"',
      '"knowledge_validate_context_run"',
      'safeInvoke("knowledge_list_packs"',
      'safeInvoke("knowledge_get_pack"',
      "safeInvoke",
      "invokeKnowledgeCommand",
      "invokeMockOnly",
    ],
  },
  {
    name: "Renderer knowledge tests lock App Server current path and no legacy fallback",
    file: "src/lib/api/knowledge.test.ts",
    snippets: [
      "appServerRequestMock",
      'vi.mock("@/lib/api/appServer"',
      "知识包列表应通过 App Server knowledgePack/list 读取",
      "应通过 App Server current 网关代理全部知识包命令",
      '"knowledgePack/read"',
      '"knowledgePack/source/import"',
      '"knowledgePack/compile"',
      '"knowledgePack/default/set"',
      '"knowledgePack/status/update"',
      '"knowledgeContext/resolve"',
      '"knowledgeContextRun/validate"',
      "知识包列表应透传 includeArchived 到 App Server",
      "知识包列表缺少 workingDir 时应 fail closed",
      "App Server 知识包列表缺少 packs 时应 fail closed",
      "App Server Knowledge 详情缺少 pack 时应 fail closed",
      '"knowledgePack/list"',
      "workingDir is required to list App Server knowledge packs",
      "App Server knowledgePack/list did not return packs",
      "knowledgePack/read did not return a knowledge pack detail",
    ],
    absentSnippets: [
      "safeInvoke",
      '"knowledge_get_pack"',
      '"knowledge_import_source"',
      '"knowledge_compile_pack"',
      '"knowledge_set_default_pack"',
      '"knowledge_update_pack_status"',
      '"knowledge_resolve_context"',
      '"knowledge_validate_context_run"',
    ],
  },
  {
    name: "Renderer skill execution reads use App Server skills/list and skill/read",
    file: "src/lib/api/skill-execution.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_SKILLS_LIST",
      "METHOD_SKILL_READ",
      'from "../../../packages/app-server-client/src/protocol"',
      'type SkillExecutionAppServerClient = Pick<AppServerClient, "request">',
      "async function requestSkillExecutionAppServer<T>",
      "appServerClient.request<T>(method, params)",
      "return response.result",
      "function normalizeSkillListResponse(",
      "function normalizeSkillReadResponse(",
      "METHOD_SKILLS_LIST",
      "METHOD_SKILL_READ",
      "App Server skills/list did not return data",
      "App Server skill/read did not return skill",
    ],
    absentSnippets: [
      "executeSkill",
      "EXECUTE_SKILL_RETIRED_ERROR",
      "ExecuteSkillRequest",
      "SkillExecutionResult",
      "skill:step_start",
      "skill:step_complete",
      "skill:step_error",
      "skill:complete",
      "safeInvoke",
      "invokeSkillExecutionCommand",
      'safeInvoke("list_executable_skills"',
      'safeInvoke("get_skill_detail"',
      'safeInvoke("execute_skill"',
      "invokeMockOnly",
      "METHOD_SKILL_LIST",
      '"skill/list"',
    ],
  },
  {
    name: "Renderer skill execution tests lock App Server current read path and retired execute boundary",
    file: "src/lib/api/skill-execution.test.ts",
    snippets: [
      "appServerRequestMock",
      'vi.mock("@/lib/api/appServer"',
      "可执行 Skill 列表应通过 App Server skills/list 读取并过滤禁用项",
      "Skill 详情应通过 App Server skill/read 读取",
      "App Server Skill 读链缺少必需 result 时不应回退 legacy",
      "Skill 独立执行 API 不再暴露 executeSkill",
      '"skills/list"',
      '"skill/read"',
      "App Server skills/list did not return data",
      "App Server skill/read did not return skill",
      "expect(safeInvoke).not.toHaveBeenCalledWith(",
      '"list_executable_skills"',
      '"get_skill_detail"',
    ],
    absentSnippets: [
      "Skill 执行 side-effect 仍保持 Desktop compat 命令",
      'toHaveBeenCalledWith("execute_skill"',
      "EXECUTE_SKILL_RETIRED_ERROR",
    ],
  },
  {
    name: "Renderer slash skill command parser stays execution-free",
    file: "src/components/agent/chat/hooks/skillCommand.ts",
    snippets: [
      "export interface ParsedSkillCommand",
      "export function parseSkillSlashCommand(",
      "Skill 执行必须走 Agent Runtime turn / SkillTool current 主链",
    ],
    absentSnippets: [
      "tryExecuteSlashSkillCommand",
      "skillExecutionApi",
      "safeListen",
      "execute_skill",
      "skill:step_start",
      "skill-exec-",
      "SKILL_INLINE_PROCESS_RETENTION",
    ],
  },
  {
    name: "Renderer slash skill command tests stay parser-only",
    file: "src/components/agent/chat/hooks/skillCommand.test.ts",
    snippets: [
      'describe("parseSkillSlashCommand"',
      "解析英文 slash skill 与参数",
      "中文 scene slash 不应被旧 slash skill 解析器误判",
    ],
    absentSnippets: [
      "tryExecuteSlashSkillCommand",
      "executeSkill",
      "safeListen",
      "execute_skill",
      "skill-exec-",
    ],
  },
  {
    name: "Renderer Plugin catalog uses current App Server JSON-RPC methods",
    files: [
      "src/lib/api/pluginCatalog.ts",
      "src/lib/api/appServerClientMethods.ts",
      "src/lib/api/appServerClientMethodSpecs.ts",
      "src/lib/api/appServerConstants.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
    ],
    snippets: [
      "listPluginCatalog",
      "searchPlugins",
      "readPluginCatalog",
      "installPluginCatalog",
      "uninstallPluginCatalog",
      "listInstalledPluginCatalog",
      "setPluginCatalogEnabled",
      "METHOD_PLUGIN_LIST",
      "METHOD_PLUGIN_SEARCH",
      "METHOD_PLUGIN_READ",
      "METHOD_PLUGIN_INSTALL",
      "METHOD_PLUGIN_UNINSTALL",
      "METHOD_PLUGIN_INSTALLED",
      "METHOD_PLUGIN_ENABLED_SET",
      '"plugin/list"',
      '"plugin/search"',
      '"plugin/read"',
      '"plugin/install"',
      '"plugin/uninstall"',
      '"plugin/installed"',
      '"plugin/enabled/set"',
    ],
    absentSnippets: [
      "pluginInstalled/list",
      "pluginHostLifecycle/list",
      "pluginShell/prepare",
      "pluginUiRuntime/start",
      "pluginUiRuntime/status",
      "pluginUiRuntime/stop",
      "pluginLocalPackage/inspect",
      "pluginLocalPackage/export",
      "pluginPackage/fetchCloud",
      "pluginInstalled/save",
      "pluginInstalled/disabled/set",
      "pluginInstalled/uninstall/rehearsal",
      "pluginInstalled/uninstall",
      "plugin_runtime_",
      "plugin_worker",
    ],
  },
  {
    name: "Renderer Scheduled Tasks gateway uses current App Server methods",
    file: "src/lib/api/scheduledTasks.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_SCHEDULED_TASK_LIST",
      "METHOD_SCHEDULED_TASK_READ",
      "METHOD_SCHEDULED_TASK_CREATE",
      "METHOD_SCHEDULED_TASK_UPDATE",
      "METHOD_SCHEDULED_TASK_DELETE",
      "METHOD_SCHEDULED_TASK_ENABLED_SET",
      "METHOD_SCHEDULED_TASK_RUN_START",
      "METHOD_SCHEDULED_TASK_RUN_LIST",
      "METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW",
      'from "../../../packages/app-server-client/src/protocol"',
      'type ScheduledTaskAppServerClient = Pick<AppServerClient, "request">',
      "async function requestScheduledTask<T>",
      "client.request<T>(method, params)",
      "return response.result",
      "createScheduledTasksApi",
      "did not return items",
      "did not return runs",
    ],
    absentSnippets: [
      "safeInvoke",
      "invokeMockOnly",
      "METHOD_AUTOMATION_",
      "automationJob/",
      "automationSchedule/",
      "automationScheduler/",
      "getAutomationJobs",
      "createAutomationJob",
      "runAutomationJobNow",
    ],
  },
  {
    name: "Renderer Scheduled Tasks tests lock current methods and fail-closed validation",
    file: "src/lib/api/scheduledTasks.test.ts",
    snippets: [
      'describe("scheduledTasks gateway"',
      "通过 exact App Server method 读取任务目录",
      "创建时保留 typed task envelope",
      "运行历史形状无效时 fail closed",
      "只投影严格的 typed Scheduled Task notifications",
      "METHOD_SCHEDULED_TASK_LIST",
      "METHOD_SCHEDULED_TASK_CREATE",
      "METHOD_SCHEDULED_TASK_RUN_LIST",
    ],
    absentSnippets: [
      "safeInvoke",
      "METHOD_AUTOMATION_",
      "automationJob/",
      "automationSchedule/",
      "automationScheduler/",
      "get_automation_jobs",
      "create_automation_job",
      "run_automation_job_now",
    ],
  },
  {
    name: "Electron Desktop Host keeps retired Plugin and Knowledge facades out of the current bridge",
    files: ["electron/hostCommands.ts", "electron/ipcChannels.ts"],
    snippets: [],
    absentSnippets: [
      "plugin_runtime_start_task",
      "plugin_runtime_get_task",
      "plugin_runtime_cancel_task",
      "plugin_runtime_submit_host_response",
      "plugin_start_ui_runtime",
      "plugin_get_ui_runtime_status",
      "plugin_stop_ui_runtime",
      "plugin_list_installed",
      "plugin_save_installed_state",
      "plugin_set_disabled",
      "plugin_uninstall_rehearsal",
      "plugin_uninstall",
      "METHOD_PROJECT_MEMORY_READ",
      'case "project_memory_get":',
      "return await this.#readProjectMemory(args)",
      "async #readProjectMemory(args: HostArgs)",
      '"project_memory_get"',
      "METHOD_KNOWLEDGE_PACK_LIST",
      "METHOD_KNOWLEDGE_PACK_READ",
      "METHOD_KNOWLEDGE_SOURCE_IMPORT",
      "METHOD_KNOWLEDGE_PACK_COMPILE",
      "METHOD_KNOWLEDGE_PACK_DEFAULT_SET",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE",
      "METHOD_KNOWLEDGE_CONTEXT_RESOLVE",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE",
      'case "plugin_list_installed":',
      "return await this.#listPluginInstalled()",
      "async #listPluginInstalled()",
      '"plugin_list_installed"',
      'case "knowledge_list_packs":',
      'case "knowledge_get_pack":',
      'case "knowledge_import_source":',
      'case "knowledge_compile_pack":',
      'case "knowledge_set_default_pack":',
      'case "knowledge_update_pack_status":',
      'case "knowledge_resolve_context":',
      'case "knowledge_validate_context_run":',
      "return await this.#listKnowledgePacks(args)",
      "return await this.#readKnowledgePack(args)",
      "return await this.#importKnowledgeSource(args)",
      "return await this.#compileKnowledgePack(args)",
      "return await this.#setDefaultKnowledgePack(args)",
      "return await this.#updateKnowledgePackStatus(args)",
      "return await this.#resolveKnowledgeContext(args)",
      "return await this.#validateKnowledgeContextRun(args)",
      "async #listKnowledgePacks(args: HostArgs)",
      "async #readKnowledgePack(args: HostArgs)",
      "async #importKnowledgeSource(args: HostArgs)",
      "async #compileKnowledgePack(args: HostArgs)",
      "async #setDefaultKnowledgePack(args: HostArgs)",
      "async #updateKnowledgePackStatus(args: HostArgs)",
      "async #resolveKnowledgeContext(args: HostArgs)",
      "async #validateKnowledgeContextRun(args: HostArgs)",
    ],
  },
  {
    name: "Electron Desktop Host tests lock retired Knowledge legacy facade",
    files: ["electron/hostCommands.test.ts", "electron/ipcChannels.test.ts"],
    snippets: [
      "ElectronHostCommands retired Knowledge legacy facade",
      "已从 Electron Host 退场，生产只能走 App Server JSONL current",
      "expect(isElectronHostCommand(command)).toBe(false)",
      '"knowledge_list_packs"',
      '"knowledge_get_pack"',
      '"knowledge_import_source"',
      '"knowledge_compile_pack"',
      '"knowledge_set_default_pack"',
      '"knowledge_update_pack_status"',
      '"knowledge_resolve_context"',
      '"knowledge_validate_context_run"',
    ],
    absentSnippets: [
      "METHOD_KNOWLEDGE_PACK_LIST",
      "METHOD_KNOWLEDGE_PACK_READ",
      "METHOD_KNOWLEDGE_SOURCE_IMPORT",
      "METHOD_KNOWLEDGE_PACK_COMPILE",
      "METHOD_KNOWLEDGE_PACK_DEFAULT_SET",
      "METHOD_KNOWLEDGE_PACK_STATUS_UPDATE",
      "METHOD_KNOWLEDGE_CONTEXT_RESOLVE",
      "METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE",
      'case "knowledge_list_packs":',
      'case "knowledge_get_pack":',
      'case "knowledge_import_source":',
      'case "knowledge_compile_pack":',
      'case "knowledge_set_default_pack":',
      'case "knowledge_update_pack_status":',
      'case "knowledge_resolve_context":',
      'case "knowledge_validate_context_run":',
      "return await this.#listKnowledgePacks(args)",
      "return await this.#readKnowledgePack(args)",
      "return await this.#importKnowledgeSource(args)",
      "return await this.#compileKnowledgePack(args)",
      "return await this.#setDefaultKnowledgePack(args)",
      "return await this.#updateKnowledgePackStatus(args)",
      "return await this.#resolveKnowledgeContext(args)",
      "return await this.#validateKnowledgeContextRun(args)",
      "async #listKnowledgePacks(args: HostArgs)",
      "async #readKnowledgePack(args: HostArgs)",
      "async #importKnowledgeSource(args: HostArgs)",
      "async #compileKnowledgePack(args: HostArgs)",
      "async #setDefaultKnowledgePack(args: HostArgs)",
      "async #updateKnowledgePackStatus(args: HostArgs)",
      "async #resolveKnowledgeContext(args: HostArgs)",
      "async #validateKnowledgeContextRun(args: HostArgs)",
    ],
  },
  {
    name: "Electron Desktop Host automation legacy facade is retired",
    files: [
      "electron/hostCommands.ts",
      "electron/hostCommands.test.ts",
      "electron/ipcChannels.ts",
      "electron/ipcChannels.test.ts",
    ],
    snippets: [
      'describe("ElectronHostCommands retired automation facade"',
      "get_automation_scheduler_config",
      "get_automation_status",
      "get_automation_health",
      "get_automation_jobs",
      'isElectronHostCommand("get_automation_jobs")',
      "toBe(false)",
    ],
    absentSnippets: [
      'case "get_automation_scheduler_config":',
      'case "get_automation_status":',
      'case "get_automation_job":',
      'case "create_automation_job":',
      'case "update_automation_job":',
      'case "delete_automation_job":',
      'case "run_automation_job_now":',
      'case "get_automation_health":',
      'case "get_automation_run_history":',
      'case "preview_automation_schedule":',
      'case "validate_automation_schedule":',
      "return await this.#listAutomationJobs()",
      "async #listAutomationJobs()",
    ],
  },
  {
    name: "App Server exposes full Agent runtime tool inventory current method",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/agent_session.rs",
      "lime-rs/crates/app-server/src/processor/mod.rs",
      "lime-rs/crates/app-server/src/processor/agent_session.rs",
      "lime-rs/crates/app-server/src/processor/dispatch.rs",
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/runtime_backend.rs",
      "lime-rs/crates/app-server/src/runtime_backend/execution_backend.rs",
      "lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs",
      "lime-rs/crates/agent/src/agent_tools/inventory.rs",
      "lime-rs/crates/agent/src/agent_tools/tool_inventory_runtime_snapshot.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/index.ts",
      "src/lib/api/appServer.ts",
    ],
    snippets: [
      'METHOD_AGENT_SESSION_TOOL_INVENTORY_READ: &str = "agentSession/toolInventory/read"',
      "method: METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,",
      "pub struct AgentSessionToolInventoryReadParams",
      "pub struct AgentSessionToolInventoryReadResponse",
      "METHOD_AGENT_SESSION_TOOL_INVENTORY_READ =>",
      "handle_tool_inventory_read_impl",
      "read_agent_session_tool_inventory",
      "ToolInventoryReadRequest",
      "async fn read_tool_inventory(",
      "read_agent_tool_inventory_runtime_snapshot",
      "build_tool_inventory(AgentToolInventoryBuildInput",
      'METHOD_AGENT_SESSION_TOOL_INVENTORY_READ =\n  "agentSession/toolInventory/read"',
      "readAgentSessionToolInventory(",
      "APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ",
    ],
  },
  {
    name: "App Server session read projects runtime events into thread_read",
    files: appServerRuntimeThreadReadProjectionFiles,
    snippets: [
      "let detail = read_model::runtime_session_read_detail_with_options(",
      "detail: Some(detail)",
      "pub(super) fn runtime_session_read_detail_with_options(",
      "pub(super) fn from_params(params: &AgentSessionReadParams) -> Self",
      "fn runtime_thread_read_from_stored_session_with_usage_events(",
      "fn runtime_events_with_workflow_audit<'a>(",
      "article_workspace: Option<serde_json::Value>",
      '"thread_read": thread_read',
      '"tool_calls": tool_item_projection::tool_calls_from_events(&stored.events)',
      "items.extend(tool_item_projection::tool_items_from_events(stored))",
      '"artifacts": artifact_projection::stored_artifact_summaries_for_turn(stored, None)',
      '"outputs": output_refs::read_model_outputs(stored.output_blobs.values(), None)',
      "pub(super) fn tool_calls_from_events(events: &[AgentEvent]) -> Vec<Value>",
      "pub(super) fn tool_items_from_events(stored: &StoredSession) -> Vec<Value>",
      "pub(super) fn current_tool_item_from_event(event: &AgentEvent) -> Option<CurrentToolItem>",
      "fn apply_current_item(",
      "ToolState::from_current_item(",
      "self.tools[index].merge_current_item(event, item)",
      '"item.started" | "item.updated" | "item.completed"',
      "read_session_projects_runtime_events_into_thread_read_tool_calls",
      "read_session_merges_tool_started_arguments_into_completed_tool_calls",
      "canonical_tool_projection_does_not_downgrade_completed_item_with_late_update",
      "read_session_keeps_workspace_patch_in_thread_read_artifacts_only",
      '"artifact.snapshot"',
      '"workspacePatch"',
      'assert_eq!(web_fetch["status"], "completed")',
      'assert_eq!(web_search["status"], "completed")',
      'assert_eq!(artifacts[0]["artifactRef"], "artifact-content-batch")',
      'assert_eq!(artifacts[0]["contentStatus"], "notRequested")',
    ],
    absentSnippets: [
      'tool_name: "WebFetch"',
      'tool_name: "WebSearch"',
      "success: true",
    ],
  },
  {
    name: "App Server default capability source exposes current runtime tool catalog",
    files: [
      "lime-rs/crates/app-server/src/capability.rs",
      "lime-rs/crates/app-server/src/runtime_factory.rs",
    ],
    snippets: [
      "lime_agent::agent_tools::catalog",
      "tool_catalog_entries_for_surface",
      "WorkspaceToolSurface::workbench()",
      "ToolLifecycle::Current",
      "pub fn default_current_surface() -> Self",
      "current_tool_capability_record(tool.name)",
      "fn current_tool_capability_record(tool_name: &str) -> CapabilityInventoryRecord",
      'format!("tool.{tool_name}")',
      "default_inventory_source_exposes_current_tool_capabilities",
      "factory_default_mock_runtime_uses_inventory_capability_source",
      '"tool.WebFetch"',
      '"tool.WebSearch"',
    ],
    absentSnippets: [
      "workbench_with_browser_assist",
      "APP_SERVER_BACKEND_MODE=mock",
      '"backend": "mock"',
    ],
  },
  {
    name: "Conversation import preserves tools after canonical Item lowering",
    files: [
      "lime-rs/crates/app-server/src/runtime/conversation_import/commit.rs",
      "lime-rs/crates/app-server/src/runtime/conversation_import/codex/history_builder.rs",
      "lime-rs/crates/app-server/src/runtime/conversation_import/codex/canonical_items.rs",
      "lime-rs/crates/app-server/src/runtime/conversation_import/tests/runtime_events.rs",
    ],
    snippets: [
      "build_canonical_history_events",
      "project_rollout_events_to_canonical",
      '"item.started"',
      '"item.completed"',
      "commit_preserves_codex_tool_command_and_patch_timeline",
      "commit_preserves_high_volume_codex_tool_events_in_canonical_projection",
      "commit_preserves_imported_commands_across_turns_without_projection_budget",
      "commit_preserves_imported_assistant_message_order_between_runtime_events",
      "commit_delays_imported_turn_terminal_until_after_late_runtime_events",
      "commit_preserves_imported_update_plan_timeline_item",
      "commit_imports_user_and_agent_items_with_canonical_lifecycle",
      "commit_projects_codex_runtime_specialized_items_into_existing_timeline_types",
      "commit_closes_incomplete_imported_lifecycles_as_failed_timeline_items",
    ],
    absentSnippets: ["lower_imported_runtime_events_for_commit"],
  },
  {
    name: "Request tool policy validates required WebSearch from current item lifecycle",
    files: agentRequestToolPolicyFiles,
    snippets: [
      "pub struct RequestToolPolicy",
      "pub fn resolve_request_tool_policy_with_mode(",
      "pub struct WebSearchExecutionTracker",
      "record_tool_item",
      "observed_item_lifecycle",
      "validate_web_search_requirement",
    ],
    absentSnippets: [
      "PreflightSearchPlan",
      "PlannedWebSearchQuery",
      "build_preflight_queries",
      "should_run_web_search_preflight",
      ".execute(&preflight_tool_name",
    ],
  },
  {
    name: "Electron session history fixture proves v2 archive lifecycle and current history read path",
    file: "scripts/electron/session-history-fixture-smoke.mjs",
    snippets: [
      "[smoke:agent-session-history-electron-fixture]",
      "import { _electron as electron }",
      "electron.launch({",
      '"--use-mock-keychain"',
      'APP_SERVER_BACKEND_MODE: "unavailable"',
      'LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"',
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI.supportsCommand",
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      '"initialize"',
      '"thread/start"',
      '"thread/archive"',
      '"thread/unarchive"',
      '"thread/read"',
      '"thread/list"',
      '"thread/turns/list"',
      '"thread/resume"',
      "const FORBIDDEN_METHODS = [",
      '"agentSession/update"',
      '"agentSession/archiveMany"',
      "SQLITE3_BINARY",
      "launchElectronFixture",
      "closeElectronFixture",
      "runThreadArchivePhase",
      "assertThreadArchivePhase",
      "runThreadUnarchivePhase",
      "assertThreadUnarchivePhase",
      "findRolloutPaths",
      "threadArchiveSummary",
      "threadUnarchiveSummary",
      "archivedRolloutPaths",
      "restoredRolloutPaths",
      "archivedRolloutPaths.active.length === 0",
      "archivedRolloutPaths.archived.length === 1",
      "restoredRolloutPaths.active.length === 1",
      "restoredRolloutPaths.archived.length === 0",
      "seedThreadReadPageIsomorphicCanonicalThread",
      "runThreadReadPageIsomorphicReadPhase",
      "runThreadReadPageIsomorphicDomOracle",
      "LAST_PROJECT_ID_KEY",
      "APP_SIDEBAR_COLLAPSED_STORAGE_KEY",
      "databaseBootstrapRestart",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'APP_SERVER_BACKEND_MODE: "external"',
      "APP_SERVER_BACKEND_COMMAND",
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      'backendMode: "mock"',
      'call("agentSession/update"',
      'call("agentSession/archiveMany"',
    ],
  },
  {
    name: "Electron code artifact workbench fixture proves App Server current artifact path",
    file: "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
    snippets: [
      "[smoke:code-artifact-workbench-electron-fixture]",
      "import { _electron as electron }",
      "electron.launch({",
      '"--use-mock-keychain"',
      'APP_SERVER_BACKEND_MODE: "external"',
      "APP_SERVER_BACKEND_COMMAND: process.execPath",
      "writeFixtureBackend(",
      "function persistBackendLedgerEvidence(",
      "writeJsonFile(evidencePath",
      'LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"',
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI.supportsCommand",
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      '"thread/start"',
      '"turn/start"',
      '"thread/read"',
      '"thread/list"',
      'type: "artifact.snapshot"',
      'type: "turn.completed"',
      'kind: "backendEvents"',
      "backendEmittedEventTypes",
      "backendEmittedCurrentTerminal",
      "backendDidNotEmitLegacyTerminal",
      "Hello Lime Workbench",
      "openFixtureSessionFromSidebar",
      "openWorkbench",
      "workbenchOpened",
      "artifactPersisted",
      "appServerJsonRpcUsed",
      "externalFixtureBackendUsed",
      "liveProviderNotUsed",
      "noInvokeErrors",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
      'type: "turn.final_done"',
      '"agentSession/update"',
    ],
  },
  {
    name: "npm exposes explicit Electron current fixture smokes",
    file: "package.json",
    snippets: [
      '"smoke:agent-session-history-electron-fixture"',
      "scripts/electron/session-history-fixture-smoke.mjs",
      '"smoke:codex-import-continuation-electron-fixture"',
      "scripts/electron/codex-import-continuation-fixture-smoke.mjs",
      '"smoke:codex-import-click-through-electron-fixture"',
      "scripts/electron/codex-import-click-through-fixture-smoke.mjs",
      '"smoke:code-artifact-workbench-electron-fixture"',
      "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
    ],
  },
  {
    name: "Electron local history import click-through fixture launches current Desktop Host path",
    file: "scripts/electron/codex-import-click-through-fixture-smoke.mjs",
    snippets: [
      "[smoke:codex-import-click-through-fixture]",
      "import { _electron as electron }",
      "electron.launch({",
      '"--use-mock-keychain"',
      'APP_SERVER_BACKEND_MODE: "external"',
      "APP_SERVER_BACKEND_COMMAND: process.execPath",
      'LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"',
      "createClickThroughFixtureRuntimeEnv",
      "REQUIRED_BACKEND_METHODS",
      '"thread/read"',
      "导入细节还原",
      "waitForImportedSessionDetails",
      "sendFollowUpFromGui",
      "CODEX_IMPORT_CLICK_THROUGH_DONE",
      "backendMetadataImported",
      "backendCwd === IMPORTED_CWD",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
    ],
  },
  {
    name: "Electron local history import click-through fixture owns source data and backend methods",
    file: "scripts/electron/lib/local-history-import-click-through-fixture.mjs",
    snippets: [
      "writeFixtureBackend(",
      "session_index.jsonl",
      "CODEX_HOME: sourceRoot",
      '"conversationImport/source/scan"',
      '"conversationImport/thread/preview"',
      '"conversationImport/thread/commit"',
      '"thread/read"',
      '"turn/start"',
      "IMPORTED_REASONING_TEXT",
      "IMPORTED_CWD",
      "readBackendLedger(",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
    ],
  },
  {
    name: "Electron local history import click-through fixture owns sidebar dialog selectors",
    file: "scripts/electron/lib/local-history-import-click-through-gui.mjs",
    snippets: [
      "app-sidebar-import-conversation-button",
      "app-sidebar-conversation-import-dialog",
      "app-sidebar-conversation-import-confirm",
      'textarea[name="agent-chat-message"]',
      "waitForImportPreview",
      "confirmImport",
      "waitForImportedSessionDetails",
      "sendFollowUpFromGui",
      "collectImportedSessionVisualAudit",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
    ],
  },
  {
    name: "Electron local history import click-through fixture owns App Server bridge readiness checks",
    file: "scripts/electron/lib/local-history-import-smoke-utils.mjs",
    snippets: [
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI.supportsCommand",
      'export const APP_SERVER_HANDLE_JSON_LINES_COMMAND =\n  "app_server_handle_json_lines"',
      "invokeAppServerFromPage",
      'jsonrpc: "2.0"',
      "initializeAppServer",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
    ],
  },
  {
    name: "Electron session messages fixture proves App Server read detail messages after turn",
    file: "scripts/smoke/agent-session-messages-electron-fixture-smoke.mjs",
    snippets: [
      "[smoke:agent-session-messages-electron-fixture]",
      "import { _electron as electron }",
      "electron.launch({",
      '"--use-mock-keychain"',
      'APP_SERVER_BACKEND_MODE: "external"',
      "APP_SERVER_BACKEND_COMMAND: process.execPath",
      "APP_SERVER_BACKEND_ARGS: JSON.stringify",
      'LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"',
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI.supportsCommand",
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      '"initialize"',
      '"thread/start"',
      '"turn/start"',
      '"thread/read"',
      "writeFixtureBackend(",
      "readBackendLedger(",
      'input.kind === "turnStart"',
      'type: "message.delta"',
      "waitForReadModel",
      "readModelConverged",
      "readSnapshots",
      "detail?.messages_count",
      "detailMessagesLength",
      "contentTextFromMessage",
      "summary.detailMessagesCount === 2",
      "summary.detailMessagesLength === 2",
      "summary.userMessageText === USER_TEXT",
      "summary.assistantMessageText === ASSISTANT_TEXT",
      "用户消息未从 App Server detail.messages 恢复",
      "助手消息未从 message.delta 投影",
      "backendTurnStartSeen",
    ],
    absentSnippets: [
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
      "--allow-live-provider",
      "agent_runtime_",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "explicitMockFallback",
    ],
  },
  {
    name: "Rust v2 router rejects legacy turn shape and RuntimeCore denies hidden capabilities",
    files: [
      "lime-rs/crates/app-server/src/lib.rs",
      "lime-rs/crates/app-server/src/runtime/tests/turn_lifecycle.rs",
    ],
    snippets: [
      "turn_start_rejects_legacy_session_shape_before_runtime_dispatch",
      "capability_list_with_session_id_uses_stored_session_scope",
      "error_codes::INVALID_PARAMS",
      "error_codes::SESSION_NOT_FOUND",
      'assert!(error.error.message.contains("threadId"))',
      "start_turn_rejects_hidden_capability_id_without_persisting_turn",
      "RuntimeCoreError::CapabilityDenied(capability_id)",
      'assert_eq!(capability_id, "content.draft.generate")',
      "assert!(read.turns.is_empty())",
    ],
  },
  {
    name: "Rust runtime backend initializes current Agent runtime before action responses",
    file: "lime-rs/crates/app-server/src/runtime_backend/initialization_tests.rs",
    snippets: [
      "respond_action_initializes_agent_before_runtime_resume",
      "ExecutionBackend::respond_action",
      "ActionRespondRequest",
      "AgentSessionActionType::AskUser",
      "assert!(backend.agent_state.is_initialized().await)",
    ],
  },
  {
    name: "Rust runtime backend flow tests cover current turn and action response events",
    files: appServerRuntimeBackendFiles,
    snippets: [
      "handle_action_response",
      "respond_action_tool_confirmation_resumes_pending_agent_tool_future",
      '"action.resolved"',
      "run_agent_turn_with_policy",
      "AgentTurnExecutionRequest",
      '"turn.completed"',
    ],
  },
  {
    name: "Rust app-server facade reexports artifact/read protocol types",
    file: "lime-rs/crates/app-server/src/lib.rs",
    snippets: [
      "pub use app_server_protocol::ArtifactReadParams",
      "pub use app_server_protocol::ArtifactReadResponse",
      "pub use app_server_protocol::ArtifactContentStatus",
      "pub use app_server_protocol::ArtifactSummary",
      "pub use runtime::FilesystemArtifactContentProvider",
      "pub use app_server_protocol::METHOD_ARTIFACT_READ",
    ],
  },
  {
    name: "Rust runtime factory can inject capability source without host dependencies",
    file: "lime-rs/crates/app-server/src/runtime_factory.rs",
    snippets: [
      "pub fn mock_runtime_core_with_capability_source(",
      "pub fn mock_app_server_with_capability_source(",
      "capability_source: Arc<dyn CapabilitySource>",
      "RuntimeCore::with_backend_and_capability_source",
      "pub fn unavailable_runtime_core_with_capability_source(",
      "pub fn unavailable_app_server_with_capability_source(",
      "factory_builds_unavailable_runtime_with_injected_capability_source",
      "pub fn runtime_backend_core_with_capability_source(",
      "pub fn runtime_backend_core_with_db_and_capability_source(",
      "pub fn runtime_app_server_with_capability_source(",
      "artifact_content_provider: Arc<dyn ArtifactContentProvider>",
      "Arc::new(RuntimeBackend::with_execution_process_server(",
    ],
  },
  {
    name: "Rust app policy manifest builds standalone capability source",
    file: "lime-rs/crates/app-server/src/capability.rs",
    snippets: [
      "pub struct AppPolicyManifest",
      "pub struct AppPolicyCapability",
      "pub enum AppPolicyManifestError",
      "pub fn capability_source_from_app_policy_json(",
      "impl TryFrom<AppPolicyManifest> for CapabilityInventorySource",
      "app_policy_manifest_builds_scoped_capability_source",
      "app_policy_manifest_rejects_incomplete_capabilities",
    ],
  },
  {
    name: "Rust app-server facade reexports app policy source types",
    file: "lime-rs/crates/app-server/src/lib.rs",
    snippets: [
      "pub use capability::capability_source_from_app_policy_json",
      "pub use capability::AppPolicyCapability",
      "pub use capability::AppPolicyLoadError",
      "pub use capability::AppPolicyManifest",
      "pub use capability::AppPolicyManifestError",
    ],
  },
  {
    name: "Standalone App Server exposes explicit unavailable backend mode",
    file: "lime-rs/crates/app-server/src/runtime_factory.rs",
    snippets: [
      "pub enum AppServerBackendMode",
      "Unavailable",
      'Self::Unavailable => "unavailable"',
      '"unavailable" => Ok(Self::Unavailable)',
      "pub fn unavailable_runtime_core() -> RuntimeCore",
      "pub fn unavailable_app_server() -> AppServer",
      "factory_builds_unavailable_runtime_without_host_dependencies",
    ],
  },
  {
    name: "Standalone App Server CLI accepts unavailable backend mode",
    file: "lime-rs/crates/app-server/src/main.rs",
    snippets: [
      "let mut backend_mode = AppServerBackendMode::Unavailable",
      "parse_args_defaults_to_stdio_unavailable_backend",
      "(AppServerBackendMode::Unavailable, Some(capability_source))",
      "AppServerRuntimeFactory::unavailable_runtime_core_with_capability_source(",
      "AppServerRuntimeFactory::unavailable_runtime_core()",
      "--backend external|runtime|mock|unavailable",
      "--app-policy path",
      "app_policy_path: Option<String>",
      ".map(load_app_policy_source)",
      ".transpose()?",
      "parse_args_accepts_app_policy_path",
      "load_app_policy_source_reads_scoped_capabilities",
      "parse_args_accepts_explicit_unavailable_backend",
      "unsupported app-server backend mode: agent",
    ],
  },
  {
    name: "Standalone App Server CLI accepts external backend launch options",
    file: "lime-rs/crates/app-server/src/main.rs",
    snippets: [
      "(AppServerBackendMode::External, Some(capability_source))",
      "AppServerRuntimeFactory::external_runtime_core_with_capability_source(",
      "AppServerRuntimeFactory::external_runtime_core(config.external_backend_config()?)",
      "external_backend_config",
      "--backend-command is required when --backend external",
      "ExternalBackendConfig::new(command.clone())",
      "--backend-command",
      "--backend-arg",
      "--backend-timeout-ms",
      "invalid --backend-timeout-ms",
      "--backend external|runtime|mock|unavailable",
      "[--backend-command path]",
      "[--backend-arg value]",
      "[--backend-timeout-ms ms]",
      "parse_args_accepts_external_backend_command",
      "external_backend_requires_command",
    ],
  },
  {
    name: "Standalone App Server runtime exposes App Server RuntimeBackend mode",
    file: "lime-rs/crates/app-server/src/runtime_factory.rs",
    snippets: [
      "Runtime",
      'Self::Runtime => "runtime"',
      '"runtime" => Ok(Self::Runtime)',
      "pub fn runtime_backend_core() -> RuntimeCore",
      "pub fn runtime_backend_core_with_capability_source(",
      "pub fn runtime_app_server() -> AppServer",
      "factory_builds_runtime_backend_without_host_dependencies",
      "factory_builds_runtime_backend_with_injected_capability_source",
    ],
  },
  {
    name: "Standalone App Server external backend keeps command args and timeout explicit",
    file: "lime-rs/crates/app-server/src/external_backend.rs",
    snippets: [
      "pub struct ExternalBackendConfig",
      "pub command: String",
      "pub args: Vec<String>",
      "pub timeout_ms: u64",
      "pub const DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS",
      "pub fn with_args(",
      "pub fn with_timeout_ms(",
      "external app-server backend timed out after {timeout_ms}ms while {phase}",
      "stdout_lines.next_line()",
      "cleanup_external_backend_after_timeout(",
      "let _ = child.start_kill()",
      "let _ = child.wait().await",
      "external_backend_stderr_summary(stderr_task).await",
      "emit_external_backend_line(&line, sink)",
      "fn emit_external_backend_line(",
      "external_backend_config_keeps_command_and_args_separate",
      "external_backend_reads_jsonl_event_stream",
      "external_backend_timeout_kills_process_and_reports_stderr_while_reading_stdout",
      "external_backend_timeout_kills_process_and_reports_stderr_while_waiting_for_exit",
    ],
    absentSnippets: ["child.wait_with_output()"],
  },
  {
    name: "Standalone App Server current runtime backend delegates Claw Agent execution chain",
    files: [
      ...agentRuntimeBoundaryFiles,
      ...agentRequestToolPolicyFiles,
      "lime-rs/crates/app-server/src/lib.rs",
      ...appServerRuntimeBackendExecutionChainFiles,
    ],
    snippets: [
      "mod runtime_backend;",
      "pub use runtime_backend::RuntimeBackend",
      "pub struct RuntimeBackend",
      "impl ExecutionBackend for RuntimeBackend",
      "init_agent_with_db(",
      "AgentRuntimeState::new()",
      "session_loops.get_or_create(",
      "execute_backend_via_session_loop(",
      "direct_provider_config_from_request",
      "configure_provider_for_session(",
      "install_provider_for_session(agent_state, request.session_id, &runtime_config).await?",
      "create_configured_reply_provider(config, &self.provider_health)?",
      "provider_for_session(request.session_id)",
      "close_provider_session(session_id).await",
      "configure_model_route_provider_for_session_with_provider_and_credential_ref(",
      "pub struct SessionProviderConfig",
      "ProviderConfigurationRequest",
      "session_provider_config_to_runtime_provider_config(",
      "config.route_protocol = request.route_protocol.or(config.route_protocol)",
      "route_protocol_from_session_provider_config",
      "ProtocolKind::OpenaiResponses",
      "ModelProviderProtocol::Responses",
      "RuntimeProviderProtocol::Responses",
      "build_agent_session_config(",
      "AgentSessionConfigurationRequest",
      "SessionConfigBuilder",
      "build_agent_turn_context(",
      "AgentTurnContextConfigurationRequest",
      "set_agent_turn_output_schema(",
      "TurnContextOverride",
      "TurnOutputSchemaSource",
      "impl RuntimeToolExecutor for CurrentTurnToolExecutor",
      "fn execute<'a>(",
      "RuntimeToolExecutionRequest<'a>",
      "run_agent_turn_with_policy",
      "AgentTurnExecutionRequest",
      "stream_runtime_reply_with_policy(",
      "runtime_events_from_agent_event",
      "runtime_event_type_for_agent_event",
      '"item.started"',
      '"item.completed"',
      '"failureCategory"',
      "canonical_tool_item(event)",
      "canonical_arguments_value",
      "final_done_raw_runtime_event_does_not_map_to_current_terminal_event",
      "canonical_tool_start_without_arguments_emits_only_item",
      "canonical_tool_start_emits_only_typed_item_arguments",
      "canonical_tool_completion_preserves_output_and_metadata",
      "runtime_agent_failed_shell_tool_is_mirrored_to_coding_facts",
      "request_tool_policy_from_request",
      "RequestToolPolicyMode::Auto",
      "resolve_request_tool_policy_with_mode(web_search, search_mode)",
      "natural_language_news_turn_exposes_search_tool_surface_by_default",
      "explicit_auto_search_mode_uses_model_tool_choice",
      "direct_host_provider_config_allows_localhost_fixture_without_database_provider",
      "if request.api_key.is_none() && request.base_url.is_none() {",
      '"message.delta"',
      '"turn.completed"',
      "selection_from_explicit_preferences",
      "selection_from_host_provider_config",
      "selection_from_session_default",
      '"/providerSelector"',
      '"/modelName"',
      "session_extension_data_provider_routing_is_used_as_session_default",
      "let provider = session_default_provider(metadata)?;",
      "let model = session_default_model(metadata)?;",
      "RuntimeCoreError::pending_route_for_session(",
      "RuntimeCoreError::RouteRejected",
      'reason_code == "provider_and_model_missing"',
      'reason_code == "capability_snapshot_missing"',
    ],
    absentSnippets: [
      ".configure_provider(",
      "configure_provider_from_pool(",
      "impl From<SessionProviderConfig> for StateProviderConfig",
      ".complete(&system, &messages, &[])",
      "message_requires_fresh_web_search",
      "mode_default",
      "unwrap_or(mode_default)",
      "(None, None) if mode_default",
      "resolve_request_tool_policy_with_mode(web_search, search_mode, true)",
      "resolve_request_tool_policy_with_mode(web_search, search_mode, false)",
      "fresh_news_request_promotes_search_to_required",
      "selection_from_enabled_provider_catalog",
      "selection_from_cached_provider_models",
      "enabled_provider_custom_model",
      "cached_provider_models",
      "ModelRegistryService",
      "tauri::",
      "APP_SERVER_BACKEND_MODE=mock",
      '"backend": "mock"',
      "mockPriorityCommands",
    ],
  },
  {
    name: "App Server runtime backend delegates Agent streaming loop to lime-agent",
    file: "lime-rs/crates/app-server/src/runtime_backend.rs",
    snippets: ["run_agent_turn_with_policy(", "AgentTurnExecutionRequest"],
    absentSnippets: [
      "stream_reply_with_policy(",
      "create_cancel_token(",
      "remove_cancel_token(",
    ],
  },
  {
    name: "Request tool policy avoids freshness keyword hard-code for preflight",
    files: ["lime-rs/crates/agent/src/lib.rs", ...agentRequestToolPolicyFiles],
    snippets: [
      "pub fn resolve_request_tool_policy_with_mode(",
      "record_tool_item",
      "observed_item_lifecycle",
      "pub struct RequestToolPolicy",
      "pub struct WebSearchExecutionTracker",
      "validate_web_search_requirement",
    ],
    absentSnippets: [
      "message_requires_fresh_web_search",
      "pub fn message_requires_fresh_web_search(",
      "pub use request_tool_policy::message_requires_fresh_web_search",
      "merge_system_prompt_with_web_search_preflight_context, message_requires_fresh_web_search",
      "fresh_news_request_promotes_search_to_required",
    ],
  },
  {
    name: "App Server stdio streams direct v2 lifecycle before turn response",
    files: [
      "lime-rs/crates/app-server/src/lib.rs",
      "lime-rs/crates/app-server/src/processor/mod.rs",
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/runtime/turn_execution.rs",
      "lime-rs/crates/app-server-transport/src/lib.rs",
    ],
    snippets: [
      "pub use transport::start_stdio_connection",
      "type StreamedTransportMessage",
      "pub async fn handle_message_streaming",
      "handle_request_streaming",
      "start_turn_with_event_callback",
      "AppendingRuntimeEventSink",
      "fn emit_failure(&mut self, error: &RuntimeCoreError) -> Result<(), RuntimeCoreError>",
      '"turn.failed"',
      "mpsc::unbounded_channel::<StreamedTransportMessage>()",
      "json_lines_loop_streams_external_backend_events_before_turn_response",
      "json_lines_loop_streams_turn_failed_after_partial_external_backend_events",
      "assert_next_direct_delta_notification",
      'notification.method != "turn/completed"',
      'assert_eq!(params["turn"]["status"], "failed")',
      "external backend crashed after partial output",
    ],
  },
  {
    name: "App Server stdio uses transport lifecycle and per-connection writer queues",
    files: [
      "lime-rs/crates/app-server/src/lib.rs",
      "lime-rs/crates/app-server-transport/src/lib.rs",
      "lime-rs/crates/app-server-transport/src/transport/stdio.rs",
    ],
    snippets: [
      "pub use transport::start_stdio_connection",
      "start_stdio_connection(transport_event_tx, reader, writer).await?",
      "TransportEvent::ConnectionOpened",
      "TransportEvent::StdioClientInitialized",
      "TransportEvent::ConnectionClosed",
      "TransportEvent::IncomingMessage",
      "server.register_transport_writer(",
      "disconnect_sender",
      "server.unregister_transport_writer(connection_id)",
      "send_to_transport_connection",
      "QueuedOutgoingMessage::new(OutgoingMessage::from(message))",
      "enqueue_transport_outbound_message(",
      "disconnects: &TransportDisconnects",
      "initialized: &TransportInitialized",
      "tokio::spawn(async move",
      "writer.try_send(queued)",
      "stdio_connection_emits_lifecycle_events_and_writes_queue_messages",
    ],
    absentSnippets: [
      "broadcast::error::RecvError::Lagged",
      "try_send(QueuedOutgoingMessage",
    ],
  },
  {
    name: "App Server test client launch-stdio runs a real stdio smoke harness",
    files: [
      "lime-rs/crates/app-server-test-client/src/harness.rs",
      "lime-rs/crates/app-server-test-client/src/lib.rs",
      "lime-rs/crates/app-server-test-client/src/main.rs",
    ],
    snippets: [
      "LaunchStdio {",
      "extra_args: Vec<String>",
      "pub struct StdioSmokeReport",
      "pub fn run_stdio_smoke(",
      "config\n        .command()",
      ".spawn()",
      "read_response(&mut stdout, RequestId::Integer(1))",
      "read_response(&mut stdout, RequestId::Integer(2))",
      "wait_for_exit(child, Duration::from_secs(2))",
      "cleanup_child(&mut child)",
      "run_stdio_smoke(config, &lines)",
      "stdio_smoke_report_summary_is_stable",
    ],
  },
  {
    name: "Standalone App Server stdio smoke isolates persistent state",
    file: "scripts/app-server/stdio-smoke.mjs",
    snippets: [
      'mkdtemp(path.join(tmpdir(), "app-server-stdio-"))',
      'stdioSidecar(binaryPath, undefined, path.join(tempDir, "data"))',
      "nextResponseForRequest(",
      "rm(tempDir, { recursive: true, force: true })",
    ],
  },
  {
    name: "Standalone App Server external backend smoke proves controlled fixture event bridge",
    file: "scripts/app-server/external-backend-smoke.mjs",
    snippets: [
      "[smoke:app-server-external-backend] ok",
      "connectAppServerSidecar",
      'stdioSidecar(binaryPath, policyPath, path.join(tempDir, "data"))',
      'backendMode: "external"',
      "backendCommand: process.execPath",
      "backendArgs: [backendPath]",
      "content.draft.generate",
      "METHOD_AGENT_SESSION_EVENT",
      "connection.listCapabilities",
      "connection.startSession",
      "connection.startTurn",
      "connection.readSession",
      "connection.readArtifacts",
      "message.delta",
      "artifact.snapshot",
      "turn.completed",
      'assertEqual(readTurns.length, 1, "read turn count")',
      'assertEqual(readTurn.status, "completed", "read turn status")',
      "content-draft-smoke",
    ],
    absentSnippets: ["turn.final_done"],
  },
  {
    name: "npm exposes standalone external backend smoke",
    file: "package.json",
    snippets: [
      '"smoke:app-server-external-backend"',
      "scripts/app-server/external-backend-smoke.mjs",
      'npm --prefix \\"packages/app-server-client\\" run build',
    ],
  },
  {
    name: "RuntimeCore rolls back turn state when backend start fails",
    files: appServerRuntimeFiles,
    snippets: [
      "pub struct UnavailableBackend",
      "standalone app-server backend is not configured",
      "fn rollback_started_turn(",
      "stored.turns.retain(|turn| turn.turn_id != turn_id)",
      "unavailable_backend_rejects_turn_without_persisting_fake_turn",
    ],
  },
  {
    name: "Rust app-server host boundary guard covers capability source",
    file: "lime-rs/crates/app-server/tests/host_boundary_guard.rs",
    snippets: [
      "src/capability.rs",
      "app-server crate 不能直接依赖桌面宿主壳层",
      "app-server 公共后端边界只能暴露 RuntimeEvent",
    ],
  },
  {
    name: "Rust client exposes typed capability/list helper",
    file: "lime-rs/crates/app-server-client/src/lib.rs",
    snippets: [
      "pub fn list_capabilities(",
      "params: CapabilityListParams",
      "pub fn list_capabilities_default(&mut self) -> Result<JsonRpcRequest, ClientError>",
      "TypedRequest<CapabilityListParams>",
      "TypedRequest::new(METHOD_CAPABILITY_LIST, params)",
      "pub use app_server_protocol::app_server_method_catalog",
      "pub use app_server_protocol::is_app_server_request_method",
      "reexports_protocol_method_catalog_for_consumers",
    ],
  },
  {
    name: "Rust client exposes typed artifact/read helper",
    file: "lime-rs/crates/app-server-client/src/lib.rs",
    snippets: [
      "use app_server_protocol::ArtifactReadParams",
      "use app_server_protocol::METHOD_ARTIFACT_READ",
      "pub fn read_artifacts(",
      "typed::read_artifacts(params)",
      "TypedRequest<ArtifactReadParams>",
      "TypedRequest::new(METHOD_ARTIFACT_READ, params)",
      "read_artifacts_preserves_filter_and_stable_method",
    ],
  },
  {
    name: "Rust client exposes typed exact fs helpers",
    file: "lime-rs/crates/app-server-client/src/lib.rs",
    snippets: [
      "FsChangedNotification, FsCopyParams, FsCopyResponse, FsCreateDirectoryParams,",
      "FsReadDirectoryParams, FsReadDirectoryResponse, FsReadFileParams, FsReadFileResponse,",
      "FsRemoveParams, FsRemoveResponse, FsUnwatchParams, FsUnwatchResponse, FsWatchParams,",
      "METHOD_FS_GET_METADATA, METHOD_FS_READ_DIRECTORY, METHOD_FS_READ_FILE, METHOD_FS_REMOVE,",
      "METHOD_FS_UNWATCH, METHOD_FS_WATCH, METHOD_FS_WRITE_FILE",
      "pub fn read_file(",
      "typed::read_file(params)",
      "TypedRequest<FsReadFileParams>",
      "TypedRequest::new(METHOD_FS_READ_FILE, params)",
      "pub fn write_file(",
      "typed::write_file(params)",
      "TypedRequest<FsWriteFileParams>",
      "TypedRequest::new(METHOD_FS_WRITE_FILE, params)",
      "pub fn create_directory(",
      "typed::create_directory(params)",
      "TypedRequest<FsCreateDirectoryParams>",
      "TypedRequest::new(METHOD_FS_CREATE_DIRECTORY, params)",
      "pub fn get_metadata(",
      "typed::get_metadata(params)",
      "pub fn read_directory(",
      "typed::read_directory(params)",
      "pub fn remove(",
      "typed::remove(params)",
      "pub fn copy(",
      "typed::copy(params)",
      "pub fn watch(",
      "typed::watch(params)",
      "pub fn unwatch(",
      "typed::unwatch(params)",
      "fn fs_helpers_use_exact_v2_methods()",
    ],
  },
  {
    name: "TypeScript protocol exposes typed capability/list contract",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      "GENERATED_APP_SERVER_METHODS,",
      "GENERATED_APP_SERVER_REQUEST_SERIALIZATION_SCOPES,",
      'export type AppServerMethodKind = "request" | "notification"',
      "export type AppServerMethodSpec = {",
      "export const APP_SERVER_METHODS =",
      "GENERATED_APP_SERVER_METHODS satisfies readonly AppServerMethodSpec[]",
      "export type AppServerRequestSerializationScope =",
      "export type AppServerRequestSerializationScopeSpec = {",
      "export const APP_SERVER_REQUEST_SERIALIZATION_SCOPES =",
      "GENERATED_APP_SERVER_REQUEST_SERIALIZATION_SCOPES satisfies readonly AppServerRequestSerializationScopeSpec[]",
      "export function getAppServerRequestSerializationScope(",
      "export function isAppServerRequestMethod(method: string): boolean",
      "export function isAppServerNotificationMethod(method: string): boolean",
      "capabilityDenied: -32020",
      "export type CapabilityListParams = {",
      "appId?: string",
      "workspaceId?: string",
      "sessionId?: string",
      "cursor?: string",
      "limit?: number",
      "export type CapabilityDescriptor = {",
      "nextCursor?: string",
      "requestSerializationScopes: AppServerRequestSerializationScopeSpec[]",
    ],
    absentSnippets: [
      'export const METHOD_INITIALIZE = "initialize"',
      'export const METHOD_CAPABILITY_LIST = "capability/list"',
      "DEFAULT_LISTEN_URL",
      "DEFAULT_RELEASE_MANIFEST_NAME",
      "DEFAULT_PROTOCOL_SCHEMA_MANIFEST_NAME",
    ],
  },
  {
    name: "Generated TypeScript protocol exposes Rust-owned method constants and v2 typed envelopes",
    file: "packages/app-server-client/src/generated/protocol-types.ts",
    snippets: [
      'export const METHOD_INITIALIZE = "initialize";',
      'export const METHOD_CAPABILITY_LIST = "capability/list";',
      "export type AppServerClientRequest =",
      'method: "initialize";',
      "export type ClientRequest =",
      'method: "thread/start";',
      'method: "turn/start";',
      "export type ClientNotification =",
      'method: "initialized";',
      "export type ServerNotification =",
      'method: "turn/started";',
      'method: "turn/completed";',
      "export type ServerRequest =",
      'method: "mcpServer/elicitation/request";',
      "params: McpServerElicitationRequestParams;",
      'method: "item/commandExecution/requestApproval";',
      "params: CommandExecutionRequestApprovalParams;",
      'method: "item/fileChange/requestApproval";',
      "params: FileChangeRequestApprovalParams;",
      'method: "item/tool/requestUserInput";',
      "params: ToolRequestUserInputParams;",
      "export interface McpServerElicitationRequestResponse",
      "export interface FileChangeRequestApprovalResponse",
      "export interface ToolRequestUserInputResponse",
      "export interface McpToolCallResult",
      "export interface McpToolCallError",
      'type: "inputAudio";',
      "error: null | {",
      "result: null | {",
      'export type MessagePhase = "commentary" | "final_answer";',
    ],
    absentSnippets: [
      "export interface AppServerClientRequest",
      "method: AppServerRequestMethod;",
      "export type AppServerNotification =",
      "export interface McpServerElicitationResponse",
      'type: "audio";',
      'type: "localAudio";',
    ],
  },
  {
    name: "TypeScript client wraps typed capability/list helper with single-pump response routing",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      'export * from "./protocol.js"',
      'export * from "./request-client-methods.js"',
      'from "./protocol.js"',
      "METHOD_CAPABILITY_LIST",
      "export class AppServerRequestError extends Error",
      "readonly response: JsonRpcErrorResponse",
      "readonly notifications: JsonRpcNotification[]",
      "readonly messages: JsonRpcMessage[]",
      "new AppServerRequestError(",
      "method,",
      "#readPump: Promise<void> | null = null",
      "#pendingRequests = new Map<protocol.RequestId, PendingRequestRead>()",
      "notifications: [],",
      "messages: [message],",
      "listCapabilities(params: CapabilityListParams = {}): JsonRpcRequest",
      "this.client.listCapabilities(params)",
      "Promise<AppServerRequestResult<CapabilityListResponse>>",
    ],
    absentSnippets: ['export const METHOD_CAPABILITY_LIST = "capability/list"'],
  },
  {
    name: "Media task artifact video create is wired through App Server current clients",
    files: [
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/index.ts",
      "src/lib/api/appServer.ts",
      "src/lib/api/mediaTasks.ts",
      "src/lib/api/videoGeneration.ts",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v0/media.rs",
      "lime-rs/crates/app-server/src/processor/mod.rs",
      "lime-rs/crates/app-server/src/processor/media.rs",
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/media_task.rs",
    ],
    snippets: [
      "mediaTaskArtifact/video/create",
      "mediaTaskArtifact/image/complete",
      "METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE",
      "METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE",
      "MediaTaskArtifactVideoCreateParams",
      "MediaTaskArtifactCompletedImageInput",
      "MediaTaskArtifactImageCompleteParams",
      "createVideoMediaTaskArtifact",
      "completeImageMediaTaskArtifact",
      "createVideoGenerationTaskArtifact",
      "completeImageGenerationTaskArtifact",
      "create_video_media_task_artifact",
      "complete_image_media_task_artifact",
      "create_video_generation_task_artifact",
      "complete_image_generation_task_artifact",
      "MediaTaskType::VideoGenerate",
      "video_generation_model",
    ],
    absentSnippets: [
      '"create_video_generation_task"',
      '"get_video_generation_task"',
      '"list_video_generation_tasks"',
      '"cancel_video_generation_task"',
    ],
  },
  {
    name: "TypeScript protocol exposes typed agentSession/action/respond contract",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      "export const METHOD_AGENT_SESSION_ACTION_RESPOND =",
      '"agentSession/action/respond"',
      "export type AgentSessionActionType =",
      '"tool_confirmation"',
      '"ask_user"',
      '"elicitation"',
      "export type AgentSessionActionScope = {",
      "export type AgentSessionActionRespondParams = {",
      "export type AgentSessionActionRespondResponse = Record<string, never>",
    ],
  },
  {
    name: "TypeScript client wraps typed agentSession/action/respond helper",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "METHOD_AGENT_SESSION_ACTION_RESPOND",
      "respondAction(params: AgentSessionActionRespondParams): JsonRpcRequest",
      "this.client.respondAction(params)",
      "Promise<AppServerRequestResult<AgentSessionActionRespondResponse>>",
    ],
  },
  {
    name: "TypeScript protocol exposes typed artifact/read contract",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      'export const METHOD_ARTIFACT_READ = "artifact/read"',
      "export type ArtifactReadParams = {",
      "artifactRef?: string",
      "includeContent?: boolean",
      "export type ArtifactSummary = {",
      "artifactRef: string",
      "content?: string",
      "contentStatus: ArtifactContentStatus",
      "export type ArtifactReadResponse = {",
      "artifacts: ArtifactSummary[]",
    ],
  },
  {
    name: "TypeScript protocol exposes typed exact fs contract",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      'export const METHOD_FS_READ_FILE = "fs/readFile"',
      'export const METHOD_FS_WRITE_FILE = "fs/writeFile"',
      'export const METHOD_FS_CREATE_DIRECTORY = "fs/createDirectory"',
      'export const METHOD_FS_GET_METADATA = "fs/getMetadata"',
      'export const METHOD_FS_READ_DIRECTORY = "fs/readDirectory"',
      'export const METHOD_FS_REMOVE = "fs/remove"',
      'export const METHOD_FS_COPY = "fs/copy"',
      'export const METHOD_FS_WATCH = "fs/watch"',
      'export const METHOD_FS_UNWATCH = "fs/unwatch"',
      'export const METHOD_FS_CHANGED = "fs/changed"',
      "export interface FsReadFileParams",
      "dataBase64: string",
      "export interface FsWriteFileParams",
      "export interface FsCreateDirectoryParams",
      "export interface FsGetMetadataResponse",
      "export interface FsReadDirectoryResponse",
      "entries: FsReadDirectoryEntry[]",
      "export interface FsRemoveParams",
      "force?: boolean | null",
      "export interface FsCopyParams",
      "sourcePath: string",
      "destinationPath: string",
      "export interface FsWatchParams",
      "export interface FsChangedNotification",
      "changedPaths: string[]",
    ],
  },
  {
    name: "TypeScript client wraps typed artifact/read helper",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "METHOD_ARTIFACT_READ",
      "readArtifacts(params: ArtifactReadParams): JsonRpcRequest",
      "this.client.readArtifacts(params)",
      "Promise<AppServerRequestResult<ArtifactReadResponse>>",
    ],
  },
  {
    name: "TypeScript client wraps typed exact fs helpers",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "readFile(params: protocol.FsReadFileParams): protocol.JsonRpcRequest",
      "writeFile(params: protocol.FsWriteFileParams): protocol.JsonRpcRequest",
      "params: protocol.FsCreateDirectoryParams",
      "getMetadata(params: protocol.FsGetMetadataParams): protocol.JsonRpcRequest",
      "params: protocol.FsReadDirectoryParams",
      "remove(params: protocol.FsRemoveParams): protocol.JsonRpcRequest",
      "copy(params: protocol.FsCopyParams): protocol.JsonRpcRequest",
      "watch(params: protocol.FsWatchParams): protocol.JsonRpcRequest",
      "unwatch(params: protocol.FsUnwatchParams): protocol.JsonRpcRequest",
      "method: protocol.METHOD_FS_READ_FILE",
      "method: protocol.METHOD_FS_WRITE_FILE",
      "method: protocol.METHOD_FS_CREATE_DIRECTORY",
      "method: protocol.METHOD_FS_GET_METADATA",
      "method: protocol.METHOD_FS_READ_DIRECTORY",
      "method: protocol.METHOD_FS_REMOVE",
      "method: protocol.METHOD_FS_COPY",
      "method: protocol.METHOD_FS_WATCH",
      "method: protocol.METHOD_FS_UNWATCH",
      "this.client.readFile(params)",
      "this.client.writeFile(params)",
      "this.client.createDirectory(params)",
      "this.client.getMetadata(params)",
      "this.client.readDirectory(params)",
      "this.client.remove(params)",
      "this.client.copy(params)",
      "this.client.watch(params)",
      "this.client.unwatch(params)",
    ],
  },
  {
    name: "TypeScript protocol exposes typed agentSession derived export contracts",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      "export const METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT =",
      '"agentSession/replayCase/export"',
      "export const METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT =",
      '"agentSession/analysisHandoff/export"',
      "export const METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT =",
      '"agentSession/reviewDecisionTemplate/export"',
      "export const METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE =",
      '"agentSession/reviewDecision/save"',
      "export type AgentSessionReplayCaseExportParams = {",
      "export type AgentSessionReplayCaseExportResponse = {",
      "export type AgentSessionAnalysisHandoffExportParams = {",
      "export type AgentSessionAnalysisHandoffExportResponse = {",
      "export type AgentSessionReviewDecisionTemplateExportParams = {",
      "export type AgentSessionReviewDecisionSaveParams = {",
      "export type AgentSessionReviewDecisionTemplateExportResponse = {",
    ],
  },
  {
    name: "TypeScript client wraps typed agentSession derived export helpers",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT",
      "METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT",
      "METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT",
      "METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE",
      "exportReplayCase(params: AgentSessionReplayCaseExportParams): JsonRpcRequest",
      "exportAnalysisHandoff(",
      "exportReviewDecisionTemplate(",
      "this.client.exportReplayCase(params)",
      "this.client.exportAnalysisHandoff(params)",
      "this.client.exportReviewDecisionTemplate(params)",
      "this.client.saveReviewDecision(params)",
      "Promise<AppServerRequestResult<AgentSessionReplayCaseExportResponse>>",
      "Promise<\n    AppServerRequestResult<AgentSessionReviewDecisionTemplateExportResponse>",
    ],
    normalizedSnippets: [
      "saveReviewDecision(params: AgentSessionReviewDecisionSaveParams,): JsonRpcRequest",
      "Promise<AppServerRequestResult<AgentSessionAnalysisHandoffExportResponse>>",
    ],
  },
  {
    name: "Retired recent Team selection is absent from current session contracts",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v0/session_admin.rs",
      "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
      "lime-rs/crates/app-server/src/runtime/projection_store.rs",
      "lime-rs/crates/app-server/src/runtime/read_model/session_metadata.rs",
      "lime-rs/crates/agent-runtime/src/session_execution.rs",
      "lime-rs/crates/agent-runtime/src/session_recent.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "src/lib/api/agentExecutionRuntime.ts",
      "src/lib/api/agentRuntime/requestTypes.ts",
      "src/lib/api/agentRuntime/appServerSessionClient.ts",
    ],
    snippets: [],
    absentSnippets: [
      "recent_team_selection",
      "recentTeamSelection",
      "RecentTeamSelection",
    ],
  },
  {
    name: "TypeScript protocol exposes typed thread archive lifecycle contract",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      'export const METHOD_THREAD_ARCHIVE = "thread/archive"',
      'export const METHOD_THREAD_UNARCHIVE = "thread/unarchive"',
      "export interface ThreadArchiveParams {",
      "export type ThreadArchiveResponse = Record<string, unknown>",
      "export interface ThreadUnarchiveParams {",
      "export interface ThreadUnarchiveResponse {",
    ],
    absentSnippets: [
      '"agentSession/archiveMany"',
      "AgentSessionArchiveManyParams",
      "AgentSessionArchiveManyResponse",
    ],
  },
  {
    name: "Retired agent session mutation methods stay absent from current protocol",
    files: [
      ...protocolV0ModuleFiles,
      ...protocolV2ModuleFiles,
      ...schemaExportModuleFiles,
      "lime-rs/crates/app-server-protocol/schema/json/app_server_protocol.schemas.json",
      "lime-rs/crates/app-server-protocol/schema/json/manifest.json",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
    ],
    snippets: [],
    absentSnippets: [
      '"agentSession/archiveMany"',
      "AgentSessionArchiveManyParams",
      "AgentSessionArchiveManyResponse",
      '"agentSession/update"',
      "METHOD_AGENT_SESSION_UPDATE",
      "AgentSessionUpdateParams",
      "AgentSessionUpdateResponse",
    ],
  },
  {
    name: "TypeScript client wraps typed thread archive lifecycle helpers",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "METHOD_THREAD_ARCHIVE",
      "METHOD_THREAD_UNARCHIVE",
      "archiveThread(params: ThreadArchiveParams): JsonRpcRequest",
      "unarchiveThread(params: ThreadUnarchiveParams): JsonRpcRequest",
      "this.client.archiveThread(params)",
      "this.client.unarchiveThread(params)",
      "Promise<AppServerRequestResult<ThreadArchiveResponse>>",
      "Promise<AppServerRequestResult<ThreadUnarchiveResponse>>",
    ],
  },
  {
    name: "TypeScript protocol exposes typed thread/delete contract",
    file: "packages/app-server-client/src/generated/protocol-types.ts",
    snippets: [
      'export const METHOD_THREAD_DELETE = "thread/delete"',
      "export interface ThreadDeleteParams {",
      "threadId: string",
      "export type ThreadDeleteResponse = Record<string, unknown>;",
    ],
    absentSnippets: [
      "METHOD_AGENT_SESSION_DELETE",
      "AgentSessionDeleteParams",
      "AgentSessionDeleteResponse",
      '"agentSession/delete"',
    ],
  },
  {
    name: "skills/changed keeps a strict generated notification contract",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v2/methods.rs",
      "lime-rs/crates/app-server-protocol/src/protocol/v2/notification.rs",
      "lime-rs/crates/app-server-protocol/schema/json/v2/SkillsChangedNotification.json",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/server-notifications.ts",
    ],
    snippets: [
      'METHOD_SKILLS_CHANGED: &str = "skills/changed"',
      "pub struct SkillsChangedNotification {}",
      '"additionalProperties": false',
      'export const METHOD_SKILLS_CHANGED = "skills/changed"',
      "export type SkillsChangedNotification = Record<string, never>;",
      "skillsChangedServerNotification",
      "Object.keys(params).length !== 0",
    ],
  },
  {
    name: "media/read is the only production media sidecar read contract",
    files: [
      ...rustProtocolFiles,
      "lime-rs/crates/app-server/src/lib.rs",
      "lime-rs/crates/app-server/src/processor/agent_session.rs",
      "lime-rs/crates/app-server/src/processor/dispatch.rs",
      "lime-rs/crates/app-server/src/processor/dispatch/v2_ingress.rs",
      "lime-rs/crates/app-server/src/runtime/session_media_reader.rs",
      "lime-rs/crates/app-server/src/runtime/session_media_refs.rs",
      "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionMediaReadParams.json",
      "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionMediaReadResponse.json",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/connection-methods.ts",
      "src/lib/api/appServerConstants.ts",
      "src/lib/api/appServerTypes.ts",
      "src/lib/api/appServerClientMethods.ts",
      "src/lib/api/appServerClientMethodSpecs.ts",
      "src/components/agent/chat/components/MessageImageAttachments.tsx",
      "src/components/agent/chat/workspace/mediaReferencePreviewArtifacts.ts",
      "src/components/agent/chat/workspace/mediaReferencePreviewPagination.ts",
      "src/components/agent/chat/workspace/useWorkspaceMediaReferencePreviewRuntime.ts",
    ],
    allowMissingFiles: true,
    snippets: [],
    absentSnippets: [
      '"agentSession/media/read"',
      "METHOD_AGENT_SESSION_MEDIA_READ",
      "AgentSessionMediaReadParams",
      "AgentSessionMediaReadResponse",
      "readAgentSessionMedia",
      "buildAgentSessionMediaReadParams",
      "APP_SERVER_METHOD_AGENT_SESSION_MEDIA_READ",
    ],
  },
  {
    name: "TypeScript protocol and clients expose typed media/read contract",
    files: [
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/request-client.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "packages/app-server-client/src/connection-methods.ts",
      "src/lib/api/appServerConstants.ts",
      "src/lib/api/appServerTypes.ts",
      "src/lib/api/appServerClientMethods.ts",
      "src/lib/api/appServerClientMethodSpecs.ts",
    ],
    snippets: [
      'export const METHOD_MEDIA_READ = "media/read"',
      "export interface MediaReadParams",
      "threadId: string",
      "export interface MediaReadResponse",
      "readMedia(",
      "METHOD_MEDIA_READ",
      "APP_SERVER_METHOD_MEDIA_READ",
      "AppServerMediaReadParams",
      "AppServerMediaReadResponse",
    ],
  },
  {
    name: "media/read uses one typed range response path without raw transient events",
    files: [
      "lime-rs/crates/app-server-protocol/src/protocol/v2/media.rs",
      "lime-rs/crates/app-server/src/processor/agent_session.rs",
      "lime-rs/crates/app-server/src/runtime/session_media_reader.rs",
      "lime-rs/crates/app-server/src/runtime/sidecar_store.rs",
      "lime-rs/crates/app-server/src/lib.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/agent-runtime.ts",
      "packages/agent-runtime-client/src/runtimeClient.ts",
      "packages/agent-runtime-client/src/sessionGateway.ts",
      "src/components/agent/chat/workspace/mediaReferencePreviewArtifacts.ts",
      "src/components/agent/chat/workspace/useWorkspaceMediaReferencePreviewRuntime.ts",
      "src/lib/api/agentRuntime/appServerEventStreamProjection.ts",
      "src/lib/api/agentRuntime/eventSequenceGate.ts",
    ],
    snippets: [
      "pub struct MediaReadParams",
      "read_media_with_cancel",
      "read_bytes_range_verified_with_cancel",
      "createMediaReferenceChunkedObjectUrlPreviewArtifact",
    ],
    absentSnippets: [
      '"media.read.chunk"',
      '"media.read.completed"',
      "MediaReadEventNotification",
      "mediaReadEventNotification",
      "read_media_streaming_with_cancel",
      "stream_bytes_range_verified_with_cancel",
      "should_stream_transport_request",
      "subscribeMediaReferencePreviewReadProgress",
      "emitStreamingMediaReadProgress",
    ],
  },
  {
    name: "TypeScript client wraps typed thread/delete helper",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "METHOD_THREAD_DELETE",
      "deleteThread(params: ThreadDeleteParams): JsonRpcRequest",
      "this.client.deleteThread(params)",
      "Promise<AppServerRequestResult<ThreadDeleteResponse>>",
    ],
    absentSnippets: [
      "METHOD_AGENT_SESSION_DELETE",
      "deleteSession(params: AgentSessionDeleteParams)",
    ],
  },
  {
    name: "Managed Objective protocol and client surface stays deleted",
    files: [
      ...rustProtocolFiles,
      "lime-rs/crates/app-server/src/objective.rs",
      "lime-rs/crates/app-server/src/runtime/objectives.rs",
      "packages/app-server-client/src/protocol.ts",
      "packages/app-server-client/src/generated/protocol-types.ts",
      "packages/app-server-client/src/index.ts",
      "packages/app-server-client/src/request-client-methods.ts",
      "src/lib/api/appServerTypes.ts",
      "src/lib/api/agentRuntime/clientFactory.ts",
      "src/lib/api/agentRuntime/objectiveClient.ts",
      "src/lib/api/agentRuntime/sessionTypes.ts",
    ],
    allowMissingFiles: true,
    snippets: [],
    absentSnippets: [
      "agentSession/objective/",
      "AgentSessionObjective",
      "ManagedObjective",
      "AgentRuntimeObjectiveAppServerClient",
      "createObjectiveClient",
    ],
  },
  {
    name: "TypeScript connection methods derive from request client protocol specs",
    file: "packages/app-server-client/src/connection-methods.ts",
    snippets: [
      "APP_SERVER_REQUEST_CLIENT_METHODS.filter(",
      "CONNECTION_CLIENT_METHOD_EXCLUSIONS",
      'spec.kind === "request"',
      "clientMethod: spec.name",
      "method: spec.method",
      "params: spec.params",
    ],
    absentSnippets: ["method: protocol.METHOD_"],
  },
  {
    name: "TypeScript request client exposes the single wrapper protocol spec",
    file: "packages/app-server-client/src/request-client-methods.ts",
    snippets: [
      "export type AppServerRequestClientMethodSpec",
      "export const APP_SERVER_REQUEST_CLIENT_METHODS",
      "installAppServerRequestClientMethods(",
    ],
  },
  {
    name: "TypeScript capability/list tests lock helper and connection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "APP_SERVER_METHODS",
      "METHOD_CAPABILITY_LIST",
      "builds capability list requests with empty params",
      "exports app-server method catalog from checked-in Rust manifest",
      "METHOD_THREAD_DELETE",
      "assert.equal(deleteThread.method, METHOD_THREAD_DELETE)",
      "isAppServerRequestMethod(METHOD_TURN_START)",
      "isAppServerNotificationMethod(METHOD_AGENT_SESSION_EVENT)",
      "connection wraps capability list response",
      "connection keeps streamed notifications independent from request error context",
      "error instanceof AppServerRequestError",
      "assert.equal(error.notifications.length, 0)",
      'assert.equal(partial.params.event.type, "message.delta")',
      'assert.equal(failed.params.event.type, "turn.failed")',
      "assert.deepEqual(capabilities.params, {})",
      "assert.deepEqual(scopedCapabilities.params, {",
      'appId: "content-studio"',
      'workspaceId: "default"',
      'sessionId: "sess_external"',
      'cursor: "2"',
      "limit: 25",
      'assert.equal(result.result.nextCursor, "1")',
    ],
  },
  {
    name: "TypeScript action respond tests lock helper and connection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "METHOD_AGENT_SESSION_ACTION_RESPOND",
      "builds action respond requests for host action resolution",
      "connection wraps action respond response",
      'actionType: "tool_confirmation"',
      "assert.equal(response.method, METHOD_AGENT_SESSION_ACTION_RESPOND)",
      "assert.equal(sent[0].method, METHOD_AGENT_SESSION_ACTION_RESPOND)",
      'assert.equal(sent[0].params.actionScope.turnId, "turn_external")',
    ],
  },
  {
    name: "TypeScript artifact/read tests lock helper and connection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "METHOD_ARTIFACT_READ",
      "builds artifact read requests with optional content lookup",
      "connection wraps artifact read response",
      "assert.equal(artifacts.method, METHOD_ARTIFACT_READ)",
      "assert.equal(sent[0].method, METHOD_ARTIFACT_READ)",
      'artifactRef: "artifact-document:req-1"',
      "includeContent: true",
      'assert.equal(result.result.artifacts[0].content, "# Report")',
      'assert.equal(result.result.artifacts[0].contentStatus, "available")',
    ],
  },
  {
    name: "TypeScript exact fs tests lock helper and connection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "METHOD_FS_READ_FILE",
      "METHOD_FS_WRITE_FILE",
      "METHOD_FS_CREATE_DIRECTORY",
      "METHOD_FS_GET_METADATA",
      "METHOD_FS_READ_DIRECTORY",
      "METHOD_FS_REMOVE",
      "METHOD_FS_COPY",
      "METHOD_FS_WATCH",
      "METHOD_FS_UNWATCH",
      'test("builds exact fs requests"',
      "assert.equal(readFile.method, METHOD_FS_READ_FILE)",
      "assert.equal(writeFile.method, METHOD_FS_WRITE_FILE)",
      "assert.equal(createDirectory.method, METHOD_FS_CREATE_DIRECTORY)",
      "assert.equal(getMetadata.method, METHOD_FS_GET_METADATA)",
      "assert.equal(readDirectory.method, METHOD_FS_READ_DIRECTORY)",
      "assert.equal(remove.method, METHOD_FS_REMOVE)",
      "assert.equal(copy.method, METHOD_FS_COPY)",
      "assert.equal(watch.method, METHOD_FS_WATCH)",
      "assert.equal(unwatch.method, METHOD_FS_UNWATCH)",
      'test("connection wraps exact fs responses"',
      'assert.equal(readFileResult.result.dataBase64, "IyBMaW1l")',
      'assert.equal(directoryResult.result.entries[0].fileName, "README.md")',
      'assert.equal(watchResult.result.path, "/workspace")',
    ],
  },
  {
    name: "TypeScript derived export tests lock helper and connection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT",
      "METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT",
      "METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT",
      "METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE",
      "builds derived export requests",
      "assert.equal(replay.method, METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT)",
      "assert.equal(analysis.method, METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT)",
      "assert.equal(save.method, METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE)",
      "connection wraps derived agent session export responses",
      "connection.exportReplayCase",
      "connection.exportAnalysisHandoff",
      "connection.exportReviewDecisionTemplate",
      "connection.saveReviewDecision",
    ],
  },
  {
    name: "Renderer-safe App Server helper aliases capability/list protocol types",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "APP_SERVER_METHOD_CAPABILITY_LIST = protocol.METHOD_CAPABILITY_LIST",
      "export type AppServerCapabilityListParams = protocol.CapabilityListParams;",
      "export type AppServerCapabilityListResponse = protocol.CapabilityListResponse;",
      "listCapabilities(params?: appServer.AppServerCapabilityListParams)",
      '{ name: "listCapabilities", method: constants.APP_SERVER_METHOD_CAPABILITY_LIST',
    ],
  },
  {
    name: "Renderer-safe App Server helper aliases exact fs protocol types",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "export const APP_SERVER_METHOD_FS_READ_FILE = protocol.METHOD_FS_READ_FILE",
      "export const APP_SERVER_METHOD_FS_WRITE_FILE = protocol.METHOD_FS_WRITE_FILE",
      "protocol.METHOD_FS_CREATE_DIRECTORY",
      "protocol.METHOD_FS_GET_METADATA",
      "protocol.METHOD_FS_READ_DIRECTORY",
      "export const APP_SERVER_METHOD_FS_REMOVE = protocol.METHOD_FS_REMOVE",
      "export const APP_SERVER_METHOD_FS_COPY = protocol.METHOD_FS_COPY",
      "export const APP_SERVER_METHOD_FS_WATCH = protocol.METHOD_FS_WATCH",
      "export const APP_SERVER_METHOD_FS_UNWATCH = protocol.METHOD_FS_UNWATCH",
      "export const APP_SERVER_METHOD_FS_CHANGED = protocol.METHOD_FS_CHANGED",
      "export type AppServerFsReadFileParams = protocol.FsReadFileParams",
      "export type AppServerFsWriteFileParams = protocol.FsWriteFileParams",
      "protocol.FsCreateDirectoryParams",
      "export type AppServerFsGetMetadataParams = protocol.FsGetMetadataParams",
      "export type AppServerFsReadDirectoryParams = protocol.FsReadDirectoryParams",
      "export type AppServerFsRemoveParams = protocol.FsRemoveParams",
      "export type AppServerFsCopyParams = protocol.FsCopyParams",
      "export type AppServerFsWatchParams = protocol.FsWatchParams",
      "export type AppServerFsUnwatchParams = protocol.FsUnwatchParams",
      "export type AppServerFsChangedNotification = protocol.FsChangedNotification",
      "readFile(params: appServer.AppServerFsReadFileParams)",
      '{ name: "readFile", method: constants.APP_SERVER_METHOD_FS_READ_FILE',
      "writeFile(params: appServer.AppServerFsWriteFileParams)",
      '{ name: "writeFile", method: constants.APP_SERVER_METHOD_FS_WRITE_FILE',
      "createDirectory(params: appServer.AppServerFsCreateDirectoryParams)",
      '{ name: "createDirectory", method: constants.APP_SERVER_METHOD_FS_CREATE_DIRECTORY',
      "getMetadata(params: appServer.AppServerFsGetMetadataParams)",
      '{ name: "getMetadata", method: constants.APP_SERVER_METHOD_FS_GET_METADATA',
      "readDirectory(params: appServer.AppServerFsReadDirectoryParams)",
      '{ name: "readDirectory", method: constants.APP_SERVER_METHOD_FS_READ_DIRECTORY',
      "remove(params: appServer.AppServerFsRemoveParams)",
      '{ name: "remove", method: constants.APP_SERVER_METHOD_FS_REMOVE',
      "copy(params: appServer.AppServerFsCopyParams)",
      '{ name: "copy", method: constants.APP_SERVER_METHOD_FS_COPY',
      "watch(params: appServer.AppServerFsWatchParams)",
      '{ name: "watch", method: constants.APP_SERVER_METHOD_FS_WATCH',
      "unwatch(params: appServer.AppServerFsUnwatchParams)",
      '{ name: "unwatch", method: constants.APP_SERVER_METHOD_FS_UNWATCH',
    ],
  },
  {
    name: "Renderer file browser uses exact fs current path and no legacy fallback",
    file: "src/lib/api/fileBrowser.test.ts",
    snippets: [
      "appServerReadDirectoryMock",
      "appServerReadFileMock",
      "appServerWriteFileMock",
      "appServerCreateDirectoryMock",
      "appServerGetMetadataMock",
      "appServerCopyMock",
      "appServerRemoveMock",
      "应通过 App Server current 主链获取目录列表与文件预览",
      "文件预览应在 renderer 解码、截断文本并识别二进制",
      "应代理文件增删改命令",
      "文件写命令应透传 App Server RPC 错误",
      "创建目录时应原样传递 Windows 原生路径",
      "expect(appServerReadFileMock).toHaveBeenCalledWith(",
      "expect(appServerWriteFileMock).toHaveBeenCalledWith(",
      "expect(appServerCopyMock).toHaveBeenCalledWith(",
      "expect(appServerRemoveMock).toHaveBeenNthCalledWith(",
    ],
  },
  {
    name: "Renderer-safe App Server helper aliases action respond protocol types",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "export const APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND =",
      "protocol.METHOD_AGENT_SESSION_ACTION_RESPOND;",
      "export type AppServerAgentSessionActionType = protocol.AgentSessionActionType;",
      "export type AppServerAgentSessionActionRespondParams =\n  AgentSessionActionRespondParams;",
      "export type AppServerAgentSessionActionRespondResponse =\n  protocol.AgentSessionActionRespondResponse;",
      "respondAction(params: appServer.AppServerAgentSessionActionRespondParams)",
      '{ name: "respondAction", method: constants.APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND',
    ],
  },
  {
    name: "Renderer-safe App Server helper aliases turn cancel protocol types",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "export const APP_SERVER_METHOD_TURN_INTERRUPT =",
      "protocol.METHOD_TURN_INTERRUPT;",
      "export type AppServerAgentSessionTurnCancelParams =\n  protocol.AgentSessionTurnCancelParams;",
      "export type AppServerAgentSessionTurnCancelResponse =\n  protocol.AgentSessionTurnCancelResponse;",
      "cancelTurn(params: appServer.AppServerAgentSessionTurnCancelParams)",
      '{ name: "cancelTurn", method: constants.APP_SERVER_METHOD_TURN_INTERRUPT',
    ],
  },
  {
    name: "Renderer-safe App Server helper exposes typed turn steer",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "export const APP_SERVER_METHOD_TURN_STEER = protocol.METHOD_TURN_STEER;",
      "export type AppServerTurnSteerParams = protocol.TurnSteerParams;",
      "export type AppServerTurnSteerResponse = protocol.TurnSteerResponse;",
      "steerTurn(params: appServer.AppServerTurnSteerParams)",
      '{ name: "steerTurn", method: constants.APP_SERVER_METHOD_TURN_STEER',
    ],
  },
  {
    name: "Renderer Agent Runtime thread client requires centralized App Server bridge availability",
    files: [
      "src/lib/api/agentRuntime/threadClient.ts",
      "src/lib/api/agentRuntime/appServerEventStream.ts",
      "src/lib/api/agentRuntime/appServerEventStreamRouting.ts",
      "src/lib/api/agentRuntime/appServerEventStreamProjection.ts",
      "src/lib/api/agentRuntime/appServerV2Notification.ts",
      "src/lib/api/agentRuntime/appServerEventTimelineReaders.ts",
    ],
    snippets: [
      "isAppServerBridgeAvailable",
      "export type AgentRuntimeAppServerClient = Pick<",
      "appServerClient = new AppServerClient()",
      "isAppServerTurnLifecycleAvailable = defaultIsAppServerTurnLifecycleAvailable",
      "assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable)",
      "Agent Runtime requires the App Server current lifecycle channel",
      "appServerClient.startTurn(",
      "standardRuntimeClient.startTurn(params)",
      "createAppServerAgentRuntimeLifecycleClient(",
      "appServerClient.cancelTurn(",
      "appServerTurnCancelParamsFromRequest(request)",
      "request.turn_id",
      "appServerClient.startThreadCompaction(",
      "const params: ThreadResumeParams",
      "appServerClient.resumeThread(params)",
      "findPendingTypedServerRequestAction(",
      "replayedActionViewFromPendingAction(",
      "respondPendingTypedServerRequest(request)",
      "generic agentSession/action/respond is retired",
      "publishAppServerAgentSessionNotifications(",
      'from "./appServerEventStream"',
      "projectAppServerAgentEventPayload(",
      "APP_SERVER_METHOD_AGENT_SESSION_EVENT",
      "publishProcessedAgentRuntimeEvent(eventName, payload)",
      '"thread/started"',
      '"turn/started"',
      '"turn/completed"',
      '"item/started"',
      '"item/completed"',
      '"item/agentMessage/delta"',
      'type: "text_delta"',
      '"item_completed"',
      'type: "runtime_status"',
      'return "turn_completed"',
      'return "turn_failed"',
      'return "turn_canceled"',
    ],
    absentSnippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      "isElectronHostCommandAvailable(APP_SERVER_HANDLE_JSON_LINES_COMMAND)",
      "AGENT_RUNTIME_COMMANDS.submitTurn",
      "AGENT_RUNTIME_COMMANDS.respondAction",
      "appServerClient.respondAction(",
      "standardRuntimeClient.respondAction(",
      "appServerActionRespondParamsFromRequest",
      "AGENT_RUNTIME_COMMANDS.compactSession",
      "AGENT_RUNTIME_COMMANDS.resumeThread",
      "AGENT_RUNTIME_COMMANDS.removeQueuedTurn",
      "AGENT_RUNTIME_COMMANDS.promoteQueuedTurn",
      '"agent_runtime_compact_session"',
      '"agent_runtime_resume_thread"',
      '"agent_runtime_remove_queued_turn"',
      '"agent_runtime_promote_queued_turn"',
      "resumeAgentSessionThread",
      "resumeAgentRuntimeThread",
      "RuntimeResumeContract",
      "runtimeRequest: {",
      "appServerTurnStartParamsFromRequest",
      "appServerRuntimeRequestFromRequest",
      "turnConfig",
      "turn_config",
      "agentChatRequest",
      "agent_chat_request",
      "hostOptions",
    ],
  },
  {
    name: "Renderer App Server bridge availability centralizes legacy IPC command name",
    file: "src/lib/api/appServerBridgeAvailability.ts",
    snippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      "isElectronHostCommandAvailable(APP_SERVER_HANDLE_JSON_LINES_COMMAND)",
      "isDevBridgeAvailable()",
      "export function isAppServerBridgeAvailable(): boolean",
    ],
  },
  {
    name: "Renderer user input op owns one typed turn/start payload",
    files: [
      "src/lib/api/agentProtocolOps.ts",
      "src/lib/api/agentProtocol.d.ts",
    ],
    snippets: [
      "export interface AgentUserInputOp {",
      "turn: TurnStartParams;",
      "const { additionalContext: turnContext, ...turn } = op.turn",
    ],
    absentSnippets: [
      "AgentUserPreferences",
      "preferences?:",
      "providerConfig?:",
      "providerPreference?:",
      "webSearch?:",
      "searchMode?:",
      "systemPrompt?:",
      "workspaceId?:",
      "queueIfBusy?:",
      "skipPreSubmitResume?:",
    ],
  },
  {
    name: "Renderer submit builder emits typed turn fields instead of runtime preferences",
    file: "src/components/agent/chat/utils/buildUserInputSubmitOp.ts",
    snippets: [
      "turn: {",
      "threadId: currentThreadId",
      "input: buildTurnInput(",
      "compaction.shouldSubmitModel",
      "approvalPolicy: runtimePolicies.approvalPolicy",
      "permissions: permissionProfileIdFromAccessMode(effectiveAccessMode)",
      "const collaborationMode = buildCollaborationMode(",
      "metadata: compaction.metadata",
      "collaborationMode ? { collaborationMode } : {}",
      "? { additionalContext }",
    ],
    absentSnippets: [
      "preferences: {",
      "providerConfig: compaction.providerConfig",
      "providerPreference: compaction.shouldSubmitProviderPreference",
      "webSearch: compaction.shouldSubmitWebSearch",
      "executionStrategy: compaction.shouldSubmitExecutionStrategy",
      "autoContinue,",
      "systemPrompt,",
      "workspaceId,",
      "turnId,",
      "lowerCollaborationMode",
      "collaboration_mode",
      "sandboxPolicy: runtimePolicies.sandboxPolicy",
    ],
  },
  {
    name: "Renderer submit compaction owns only metadata sanitizing and typed model decisions",
    files: [
      "src/components/agent/chat/utils/submitOpRuntimeCompaction.ts",
      "src/components/agent/chat/utils/submitOpRuntimeCompaction.test.ts",
    ],
    snippets: [
      "export interface SubmitOpRuntimeCompactionResult {",
      "shouldSubmitModel: boolean",
      "sanitizeSubmitMetadata(options)",
      "shouldSubmitModel: shouldSubmitTurnModel(options, metadata)",
      "RETIRED_TOOL_PREFERENCE_PATH",
      "existsSync(resolve(process.cwd(), RETIRED_TOOL_PREFERENCE_PATH))",
    ],
    absentSnippets: [
      'from "./submitOpToolPreferenceCompaction"',
      "RuntimeProviderConfig",
      "providerConfig?:",
      "shouldSubmitProviderPreference",
      "shouldSubmitModelPreference",
      "shouldSubmitExecutionStrategy",
      "shouldSubmitWebSearch",
      "shouldSubmitThinking",
      "webSearchPreference",
      "thinkingPreference",
      "requestedWebSearch",
      "requestedThinking",
    ],
  },
  {
    name: "Renderer Agent Runtime thread client keeps current turn/start typed",
    file: "src/lib/api/agentRuntime/threadClient.test.ts",
    snippets: [
      "App Server submit 参数只保留 typed Turn 配置与业务 metadata",
      'model: "deepseek-v4-pro"',
      'effort: "high"',
      'approvalPolicy: "on-request"',
      'sandboxPolicy: "workspace-write"',
      "metadata: {",
      "outputSchema,",
    ],
    absentSnippets: [
      "runtimeRequest?: Record<string, unknown>",
      "queueIfBusy?: boolean",
      "skipPreSubmitResume?: boolean",
      'queuedTurnId: "queued-claw"',
      'providerPreference: "deepseek"',
      "providerConfig: {",
      "thinkingEnabled: true",
      "webSearch: true",
      'searchMode: "required"',
      'executionStrategy: "react"',
      "autoContinue: true",
      'systemPrompt: "保留 Claw 原始系统提示"',
      'workspaceId: "workspace-claw"',
      "hostOptions: {",
      "agentChatRequest",
      "agent_chat_request",
      "turn_config",
      "turnConfig",
    ],
  },
  {
    name: "Current Turn submission accepts only RuntimeRequest across renderer and smoke evidence",
    files: [
      "src/lib/api/agentProtocolOps.ts",
      "src/components/agent/chat/hooks/agentRuntimeAdapter.ts",
      "src/lib/api/agentRuntime/threadClient.ts",
      "scripts/claw-chat-ready-streaming-smoke.mjs",
      "scripts/lib/agent-runtime-smoke-core.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-backend-ledger.mjs",
    ],
    snippets: ["runtimeRequest", "AgentSessionTurnStartParams"],
    absentSnippets: [
      "agentChatRequest",
      "agent_chat_request",
      "turn_config",
      "turnConfig",
      "hostOptions",
    ],
  },
  {
    name: "Renderer Agent Runtime client factory forwards App Server lifecycle deps",
    file: "src/lib/api/agentRuntime/clientFactory.ts",
    snippets: [
      'import type { AppServerSessionRpcClient } from "./appServerSessionClient"',
      "type AgentRuntimeExportAppServerClient",
      "type AgentRuntimeWorkspaceSkillBindingsAppServerClient",
      "type AgentRuntimeThreadClientDeps",
      "export type AgentRuntimeAppServerClient =",
      'AgentRuntimeThreadClientDeps["appServerClient"] &',
      "AppServerSessionRpcClient &",
      "AgentRuntimeExportAppServerClient &",
      "AgentRuntimeWorkspaceSkillBindingsAppServerClient;",
      "appServerClient?: AgentRuntimeAppServerClient",
      'isAppServerTurnLifecycleAvailable?: AgentRuntimeThreadClientDeps["isAppServerTurnLifecycleAvailable"]',
      "appServerClient,",
      "isAppServerTurnLifecycleAvailable,",
      "...createExportClient({",
      "appServerClient,",
      "...createInventoryClient({",
      "appServerClient,",
      "...createSessionClient({",
      "appServerClient,",
      "...createThreadClient({",
      "appServerClient,",
      "isAppServerTurnLifecycleAvailable,",
    ],
  },
  {
    name: "Renderer Agent Runtime inventory client uses App Server workspace skill bindings",
    file: "src/lib/api/agentRuntime/inventoryClient.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_WORKSPACE_SKILL_BINDINGS_LIST",
      'from "../../../../packages/app-server-client/src/protocol"',
      "export type AgentRuntimeWorkspaceSkillBindingsAppServerClient = Pick<",
      "AppServerClient,",
      '"request"',
      "appServerClient = new AppServerClient()",
      "const workspaceRoot = request.workspaceRoot.trim()",
      'throw new Error(\n        "workspaceRoot is required to list App Server workspace skill bindings"',
      "appServerClient.request<AppServerWorkspaceSkillBindingsListResponse>",
      "METHOD_WORKSPACE_SKILL_BINDINGS_LIST",
      "workspaceSkillBindingsListParamsFromRequest(request, workspaceRoot)",
      "App Server workspaceSkillBindings/list did not return bindings",
      "return response.result.bindings",
    ],
    absentSnippets: [
      "APP_SERVER_METHOD_WORKSPACE_SKILL_BINDINGS_LIST",
      "AGENT_RUNTIME_COMMANDS.listWorkspaceSkillBindings",
      '"agent_runtime_list_workspace_skill_bindings"',
    ],
  },
  {
    name: "Renderer Capability Drafts registered skills discovery uses App Server current method",
    file: "src/lib/api/capabilityDrafts.ts",
    snippets: [
      'import { AppServerClient } from "@/lib/api/appServer"',
      "METHOD_WORKSPACE_REGISTERED_SKILLS_LIST",
      'from "../../../packages/app-server-client/src/protocol"',
      'type CapabilityDraftsAppServerClient = Pick<AppServerClient, "request">',
      "appServerClient: new AppServerClient() as CapabilityDraftsAppServerClient",
      "const workspaceRoot = request.workspaceRoot.trim()",
      'throw new Error(\n        "workspaceRoot is required to list App Server workspace registered skills"',
      "this.appServerClient.request<",
      "METHOD_WORKSPACE_REGISTERED_SKILLS_LIST",
      "App Server workspaceRegisteredSkills/list did not return skills",
      "return response.result.skills.map(normalizeWorkspaceRegisteredSkill)",
    ],
    absentSnippets: ['"capability_draft_list_registered_skills"'],
  },
  {
    name: "Renderer Agent Runtime tool inventory uses App Server current method",
    file: "src/lib/api/agentRuntime/inventoryClient.ts",
    snippets: [
      "async function getAgentRuntimeToolInventory(",
      "METHOD_AGENT_SESSION_TOOL_INVENTORY_READ",
      "appServerClient.request<AppServerToolInventoryReadResponse>",
      "toolInventoryParamsFromRequest(request)",
      "App Server agentSession/toolInventory/read did not return tool inventory",
    ],
    absentSnippets: [
      "APP_SERVER_METHOD_CAPABILITY_LIST",
      "listCapabilities(",
      "AGENT_RUNTIME_COMMANDS.getToolInventory",
      "agent_runtime_get_tool_inventory",
      "invokeCommand<AgentRuntimeToolInventory>",
    ],
  },
  {
    name: "Agent Runtime workspace skill bindings legacy command is retired from production command surfaces",
    files: [
      "electron/hostCommands.ts",
      "electron/ipcChannels.ts",
      "src/lib/dev-bridge/commandPolicy.ts",
      "src/lib/governance/agentCommandCatalog.json",
    ],
    snippets: [],
    absentSnippets: ['"agent_runtime_list_workspace_skill_bindings"'],
  },
  {
    name: "Agent Runtime retired subagent facade names stay out of production gateway policy",
    file: "src/lib/dev-bridge/commandPolicy.ts",
    snippets: [],
    absentSnippets: [
      '"agent_runtime_spawn_subagent"',
      '"agent_runtime_send_subagent_input"',
      '"agent_runtime_wait_subagents"',
      '"agent_runtime_resume_subagent"',
      '"agent_runtime_close_subagent"',
      'command.startsWith("agent_runtime_")',
    ],
  },
  {
    name: "Prompt-to-artifact smoke uses App Server current discovery and binding readiness",
    file: "scripts/prompt-to-artifact-smoke.mjs",
    snippets: [
      "writeWorkspaceSkillFixture",
      "registrationSurface",
      "direct-workspace-skill-fixture",
      "retiredCommandSurface",
      "dead/guard-only",
      '"app_server_handle_json_lines"',
      '"workspaceRegisteredSkills/list"',
      '"workspaceSkillBindings/list"',
    ],
    absentSnippets: [
      '"capability_draft_list_registered_skills"',
      '"capability_draft_create"',
      '"capability_draft_verify"',
      '"capability_draft_register"',
      '"agent_runtime_list_workspace_skill_bindings"',
    ],
  },
  {
    name: "Read-only HTTP API smoke no longer invokes retired Capability Draft authoring commands",
    file: "scripts/readonly-http-api-smoke.mjs",
    snippets: [
      "buildReadonlyHttpApiGeneratedFiles",
      "writeWorkspaceSkillFixture",
      "registrationSurface",
      "direct-workspace-skill-fixture",
      "retiredCommandSurface",
      "dead/guard-only",
      '"app_server_handle_json_lines"',
      '"workspaceRegisteredSkills/list"',
      '"workspaceSkillBindings/list"',
    ],
    absentSnippets: [
      '"capability_draft_create"',
      '"capability_draft_verify"',
      '"capability_draft_register"',
    ],
  },
  {
    name: "Renderer Agent Runtime session client uses App Server session facade",
    file: "src/lib/api/agentRuntime/sessionClient.ts",
    snippets: [
      "import {",
      "createAppServerSessionClient",
      "type AppServerSessionClient",
      "type AppServerSessionRpcClient",
      "const defaultAppServerSessionClient = createAppServerSessionClient()",
      "deps.appServerSessionClient ??",
      "createAppServerSessionClient({ appServerClient: deps.appServerClient })",
      "appServerSessionClient.createAgentRuntimeSession(",
      "appServerSessionClient.listAgentRuntimeSessions({",
      "appServerSessionClient.getAgentRuntimeSession(",
      "appServerSessionClient.updateAgentRuntimeThreadToolPreferences(",
      "appServerSessionClient.archiveAgentRuntimeSession(sessionId)",
      "appServerSessionClient.unarchiveAgentRuntimeSession(sessionId)",
      "async function deleteAgentRuntimeSession(sessionId: string): Promise<void>",
      "appServerSessionClient.deleteAgentRuntimeSession(sessionId)",
      'reason: "deleted"',
      '"deleted"',
      "AGENT_RUNTIME_SESSIONS_CHANGED_EVENT",
      "notifyAgentRuntimeSessionsChanged(",
    ],
    absentSnippets: [
      "AGENT_RUNTIME_COMMANDS.createSession",
      "AGENT_RUNTIME_COMMANDS.listSessions",
      "AGENT_RUNTIME_COMMANDS.getSession",
      "AGENT_RUNTIME_COMMANDS.updateSession",
      "AGENT_RUNTIME_COMMANDS.deleteSession",
      '"agent_runtime_create_session"',
      '"agent_runtime_list_sessions"',
      '"agent_runtime_get_session"',
      '"agent_runtime_update_session"',
      '"agent_runtime_delete_session"',
      "updateAgentRuntimeSession",
    ],
  },
  {
    name: "Renderer App Server session facade and history window use protocol method constants",
    files: [
      "src/lib/api/agentRuntime/appServerSessionClient.ts",
      "src/lib/api/agentRuntime/canonicalThreadHistoryWindow.ts",
    ],
    snippets: [
      "METHOD_THREAD_LIST,",
      "METHOD_THREAD_ITEMS_LIST,",
      "METHOD_THREAD_TURNS_LIST,",
      "export type AppServerSessionRpcClient = Pick<",
      '| "startSession"',
      '| "readThread"',
      '| "updateThreadSettings"',
      '| "archiveThread"',
      '| "unarchiveThread"',
      '| "deleteThread"',
      '| "request"',
      "client.request<AppServerThreadListResponse>",
      "METHOD_THREAD_LIST",
      "listCanonicalSessionOverviews(",
      "appServerThreadReadParams(threadId, false)",
      "appServerClient.readThread(",
      "appServerClient.updateThreadSettings(",
      "appServerClient.archiveThread(",
      "appServerClient.unarchiveThread(",
      "appServerClient.deleteThread(",
      '"thread/settings/update"',
      '"thread/archive"',
      '"thread/unarchive"',
      '"thread/delete"',
    ],
    absentSnippets: [
      "APP_SERVER_METHOD_THREAD_LIST",
      'const APP_SERVER_METHOD_THREAD_LIST = "thread/list"',
      "readAppServerAgentSessionReadResponse",
      "readSessionDetail",
      "appServerClient.readSession(",
      '"agent_runtime_list_sessions"',
      '"agent_runtime_get_session"',
      '"agent_runtime_create_session"',
      '"agent_runtime_update_session"',
      "appServerClient.updateSession(",
      "updateAgentRuntimeSession",
      "appServerSessionUpdateParamsFromRequest",
      "recentTeamSelection",
      "recent_team_selection",
      "archiveManySessions",
      "agentSession/archiveMany",
      "appServerClient.deleteSession(",
      '"agentSession/delete"',
      "request.archived",
    ],
  },
  {
    name: "Renderer App Server session facade tests reject legacy read envelopes",
    file: "src/lib/api/agentRuntime/appServerSessionClient.test.ts",
    snippets: [
      "get 应从 canonical Thread items 恢复消息并分离排队回合",
      "get 遇到旧 session envelope 时应显式拒绝，不恢复兼容解析",
      "appServerClient.readThread",
      'threadId: "session-codex"',
      "includeTurns: false",
      "thread/read did not return canonical session detail",
    ],
    absentSnippets: [
      "appServerClient.readSession",
      "readSessionResult",
      '"agent_runtime_get_session"',
      "invokeMockOnly",
      "mockPriorityCommands",
      "defaultMocks",
    ],
  },
  {
    name: "Renderer chat history hydration tests consume App Server read detail messages",
    file: "src/components/agent/chat/hooks/agentChatHistory.test.ts",
    snippets: [
      "App Server read detail.messages 当前形状应直接恢复用户与助手消息",
      "messages_count: 2",
      "history_cursor: {",
      "loaded_count: 2",
      "请整理 App Server 对话历史",
      "已从 App Server detail.messages 读取。",
      "hydrateSessionDetailMessages(",
      "session-app-server-messages-0",
      "session-app-server-messages-1",
      "contentParts: [",
    ],
    absentSnippets: [
      "agent_runtime_get_session",
      "invokeMockOnly",
      "mockPriorityCommands",
      "defaultMocks",
    ],
  },
  {
    name: "Renderer fresh Agent sessions persist provider model defaults before first turn",
    files: [
      "src/components/agent/chat/hooks/useAgentSession.ts",
      "src/components/agent/chat/hooks/useAgentChat.test.tsx",
      "src/components/agent/chat/hooks/useAgentChat.test/slashSkillExecution.case.tsx",
    ],
    snippets: [
      "function buildFreshSessionProviderModelMetadata(",
      "const providerSelector = providerType.trim()",
      "const modelName = model.trim()",
      "executionRuntime: {",
      "extensionData: {",
      '"lime_provider_routing.v0": {',
      "metadata: buildFreshSessionProviderModelMetadata(",
      "首条发送创建新会话时不应额外回写 provider/model 或 accessMode",
      "providerSelector: selectedProvider",
      "modelName: selectedModel",
      "mockUpdateAgentRuntimeThreadSettings).not.toHaveBeenCalled",
    ],
  },
  {
    name: "Renderer Agent Runtime thread lifecycle uses standard AgentRuntimeClient facade",
    file: "src/lib/api/agentRuntime/threadClient.ts",
    snippets: [
      'import type { AgentRuntimeClient as StandardAgentRuntimeClient } from "@limecloud/agent-runtime-client"',
      "export type AgentRuntimeLifecycleClient = Pick<",
      '"startTurn" | "steerTurn" | "cancelTurn" | "readThread"',
      "standardRuntimeClient?: AgentRuntimeLifecycleClient",
      "createAppServerAgentRuntimeLifecycleClient(appServerClient)",
      "async function getAgentRuntimeThreadRead(",
      "assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable)",
      "appServerClient.readThread({",
      "return projectAppServerThreadReadResult(response.result)",
      "async function readAgentRuntimeThread(",
      "includeTurns: true",
    ],
    absentSnippets: [
      "AGENT_RUNTIME_COMMANDS.getThreadRead",
      '"agent_runtime_get_thread_read"',
      '"startTurn" | "steerTurn" | "cancelTurn" | "respondAction" | "readThread"',
    ],
  },
  {
    name: "Renderer Agent UI projection store delegates event selectors to standard projection package",
    file: "src/components/agent/chat/projection/conversationProjectionStore.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "type ConversationAgentUiProjectionSlice = AgentUiProjectionEventStoreState",
      "createEmptyAgentUiProjectionEventStoreState()",
      "indexAgentUiProjectionEvents(nextEvents)",
      "selectAgentUiProjectionEventsForScopeFromStore(state.agentUi, filter)",
      "selectLatestAgentUiProjectionEventForArtifactFromStore(",
    ],
    absentSnippets: [
      "function hasAgentUiProjectionScopeFilter",
      "function matchesAgentUiProjectionScopeValue",
      "function matchesAgentUiProjectionScope(",
      "function normalizeAgentUiRunKey",
    ],
  },
  {
    name: "Renderer Agent UI projection summary delegates host-neutral selectors to standard projection package",
    file: "src/components/agent/chat/projection/agentUiProjectionSummary.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "summarizeAgentUiSubagentsSurfaceLanesBase",
      "summarizeAgentUiSubagentsSurfacesBase",
      "findLatestAgentUiProjectionEventForArtifact",
      "summarizeAgentUiProjectionEvents",
      "summarizeAgentUiSubagentsProjectionEvents",
    ],
    absentSnippets: [
      "export const AGENT_UI_ACTION_EVENT_TYPES = new Set",
      "export const AGENT_UI_TASK_EVENT_TYPES = new Set",
      "export const AGENT_UI_ARTIFACT_EVENT_TYPES = new Set",
      "export const AGENT_UI_DIAGNOSTIC_EVENT_TYPES = new Set",
      "export const AGENT_UI_EVIDENCE_EVENT_TYPES = new Set",
      "export const AGENT_UI_SUBAGENTS_SURFACES = new Set",
      "export const AGENT_UI_NOTABLE_EVENT_TYPES = new Set",
      "export function summarizeAgentUiProjectionEvents(",
      "export function summarizeAgentUiSubagentsProjectionEvents(",
      "export function findLatestAgentUiProjectionEventForArtifact(",
      "function normalizeLookupKey",
    ],
  },
  {
    name: "Agent UI contracts package keeps index as barrel exports",
    file: "packages/agent-ui-contracts/src/index.ts",
    snippets: [
      'export type * from "./events"',
      'export type * from "./graph"',
      'export type * from "./messages"',
      'export type * from "./projection"',
      'export type * from "./runtime"',
      'export type * from "./timeline"',
    ],
    absentSnippets: [
      "function ",
      "interface ",
      "const ",
      "import ",
      "AgentRuntimeExecutionEventKind",
      "AgentUiProjectionEvent",
      "AgentUiProjectionState",
    ],
  },
  {
    name: "Agent UI contracts package keeps type modules split by responsibility",
    files: [
      "packages/agent-ui-contracts/src/events.ts",
      "packages/agent-ui-contracts/src/runtime.ts",
      "packages/agent-ui-contracts/src/projection.ts",
      "packages/agent-ui-contracts/src/messages.ts",
      "packages/agent-ui-contracts/src/timeline.ts",
      "packages/agent-ui-contracts/src/graph.ts",
    ],
    snippets: [
      "export type AgentUiEventClass",
      "export interface AgentUiProjectionEvent",
      "export interface AgentRuntimeExecutionEvent",
      "export interface AgentRuntimeEventProjection",
      "actions?: AgentRuntimeActionProjection[]",
      "export interface AgentUiProjectionState",
      "export interface UIMessagePart",
      "export interface ProcessTimelineEntry",
      "export interface ExecutionGraphNode",
    ],
    absentSnippets: [
      'from "@/',
      'from "src/',
      'from "react"',
      "React",
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "EventSource",
      "new WebSocket",
      "fetch(",
      "XMLHttpRequest",
    ],
  },
  {
    name: "Agent UI contracts exclude retired runtime resume contract",
    files: [
      "packages/agent-ui-contracts/src/capabilities.ts",
      "packages/agent-ui-contracts/src/validation.ts",
      "packages/agent-ui-contracts/src/schemas.ts",
      "packages/agent-ui-contracts/tests/contracts.test.mjs",
    ],
    snippets: [
      "AgentRuntimeCapabilityManifest",
      "validateRuntimeCapabilityManifest",
      "AGENT_RUNTIME_CAPABILITY_MANIFEST_SCHEMA",
      '"hitl.actions"',
    ],
    absentSnippets: [
      "AgentRuntimeResumeMode",
      "AgentRuntimeResumeActionDecision",
      "AgentRuntimeResumeContract",
      "validateRuntimeResumeContract",
      "collectRuntimeResumeContractValidationIssues",
      "AGENT_RUNTIME_RESUME_CONTRACT_SCHEMA",
      "runtimeResumeContract",
      '"hitl.resume"',
      "lime-runtime-resume-contract/v0.1",
    ],
  },
  {
    name: "Capability projection does not recreate retired HITL resume",
    files: [
      "src/lib/api/agentRuntime/capabilityContract.ts",
      "lime-rs/crates/app-server/src/runtime/capabilities.rs",
    ],
    snippets: ['"hitl.actions"'],
    absentSnippets: [
      '"hitl.resume"',
      'id.includes("resume")',
      'id.contains("resume")',
    ],
  },
  {
    name: "Agent Runtime projection package keeps index as barrel exports",
    files: [
      "packages/agent-runtime-projection/src/index.ts",
      "packages/agent-runtime-projection/src/index.js",
      "packages/agent-runtime-projection/src/index.d.ts",
    ],
    snippets: [
      'from "./contracts.js"',
      'export * from "./actions.js"',
      'export * from "./artifactEvents.js"',
      'export * from "./contextEvents.js"',
      'export * from "./conversationEvents.js"',
      'export * from "./diagnosticEvents.js"',
      'export * from "./envelope.js"',
      'export * from "./eventStore.js"',
      'export * from "./hydrationEvents.js"',
      'export * from "./lifecycle.js"',
      'export * from "./normalization.js"',
      'export * from "./planApproval.js"',
      'export * from "./permissionEvents.js"',
      'export * from "./refs.js"',
      'export * from "./routing.js"',
      'export * from "./runtimeFacts.js"',
      'export * from "./summary.js"',
      'export * from "./threadItems.js"',
      'export * from "./toolEvents.js"',
      'export * from "./readModel.js"',
      'export * from "./uiState.js"',
    ],
    absentSnippets: [
      "function ",
      "interface ",
      "const ",
      "import ",
      "@limecloud/agent-ui-contracts",
    ],
  },
  {
    name: "Agent Runtime projection package keeps host-neutral modules split by responsibility",
    files: [
      "packages/agent-runtime-projection/src/actions.ts",
      "packages/agent-runtime-projection/src/artifactEvents.ts",
      "packages/agent-runtime-projection/src/contextEvents.ts",
      "packages/agent-runtime-projection/src/conversationEvents.ts",
      "packages/agent-runtime-projection/src/contracts.ts",
      "packages/agent-runtime-projection/src/diagnosticEvents.ts",
      "packages/agent-runtime-projection/src/envelope.ts",
      "packages/agent-runtime-projection/src/eventStore.ts",
      "packages/agent-runtime-projection/src/hydrationEvents.ts",
      "packages/agent-runtime-projection/src/lifecycle.ts",
      "packages/agent-runtime-projection/src/normalization.ts",
      "packages/agent-runtime-projection/src/planApproval.ts",
      "packages/agent-runtime-projection/src/permissionEvents.ts",
      "packages/agent-runtime-projection/src/refs.ts",
      "packages/agent-runtime-projection/src/routing.ts",
      "packages/agent-runtime-projection/src/runtimeFacts.ts",
      "packages/agent-runtime-projection/src/summary.ts",
      "packages/agent-runtime-projection/src/threadItems.ts",
      "packages/agent-runtime-projection/src/toolEvents.ts",
      "packages/agent-runtime-projection/src/readModel.ts",
      "packages/agent-runtime-projection/src/uiState.ts",
    ],
    snippets: [
      'from "@limecloud/agent-ui-contracts"',
      "buildAgentUiActionRequiredEvent",
      "buildAgentUiActionResolvedEvent",
      "buildAgentUiArtifactSnapshotEvent",
      "buildAgentUiContextTraceEvent",
      "buildAgentUiTurnContextEvents",
      "buildAgentUiMessageSnapshotEvent",
      "buildAgentUiTextDeltaEvent",
      "buildAgentUiReasoningDeltaEvent",
      "buildAgentUiWarningEvent",
      "buildAgentUiCostMetricEvent",
      "buildAgentUiHistoricalHydrationEvents",
      "buildAgentUiPlanApprovalRequiredEvent",
      "buildAgentUiPlanApprovalResolvedEvent",
      "buildAgentUiRuntimePermissionChangedEvent",
      "buildAgentUiRoutingStatusEvent",
      "buildAgentUiProjectionBase",
      "buildAgentUiThreadStartedEvent",
      "buildAgentUiRuntimeStatusEvent",
      "buildAgentUiRunStartedEvent",
      "buildAgentUiRunFinishedEvent",
      "buildAgentUiRunFailedEvent",
      "buildAgentUiModelChangeEvent",
      "buildAgentUiTaskProfileResolvedEvent",
      "buildAgentUiThreadItemBase",
      "buildAgentUiThreadItemActionEvent",
      "buildAgentUiThreadItemSubagentActivityEvent",
      "buildAgentUiThreadItemEvent",
      "buildAgentUiToolStartEvents",
      "buildAgentUiToolEndEvent",
      "buildAgentUiToolEndEvents",
      "buildAgentUiToolProgressEvent",
      "buildAgentUiToolOutputDeltaEvent",
      "buildAgentUiToolInputDeltaEvent",
      "extractAgentUiPlanApprovalProjection",
      "extractAgentUiPlanApprovalResponseProjection",
      "extractAgentUiTaskOwnerChangeProjection",
      "sequenceAgentUiProjectionEvents",
      "AgentUiProjectionEventStoreState",
      "definedString",
      "extractArtifactRefs",
      "buildRoutingDecisionPayload",
      "inferAgentUiRuntimeEntity",
      "resolveAgentUiThreadItemPhase",
      "summarizeAgentUiProjectionEvents",
      "projectAgentRuntimeReadModel",
      "projectAgentUiState",
    ],
    absentSnippets: [
      'from "@/',
      'from "src/',
      'from "../../src',
      'from "../../../src',
      "React",
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "EventSource",
      "new WebSocket",
      "fetch(",
      "XMLHttpRequest",
      "buildAgentUiThreadItemSubagentWorkerNotificationEvent",
    ],
  },
  {
    name: "Renderer Agent UI projection base delegates envelope and sequence to standard projection package",
    file: "src/components/agent/chat/projection/projectionBase.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiProjectionBase as buildStandardAgentUiProjectionBase",
      "sequenceAgentUiProjectionEvents",
      'typeof event.item?.type === "string"',
      "return buildStandardAgentUiProjectionBase(",
      "return sequenceAgentUiProjectionEvents(events, startSequence)",
    ],
    absentSnippets: [
      "function inferRuntimeEntityFromSource",
      "inferAgentUiRuntimeEntity",
      "definedString(context",
      "sessionId: definedString",
      "threadId: definedString",
      "runId: definedString",
      "turnId: definedString",
      "messageId: definedString",
      "taskId: definedString",
      "events.map((event, index)",
      "startSequence + index",
    ],
  },
  {
    name: "Renderer Agent UI conversation event projection delegates message and delta builders to standard projection package",
    file: "src/components/agent/chat/projection/conversationEventProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiMessageSnapshotEvent",
      "buildAgentUiTextDeltaEvent",
      "buildAgentUiReasoningDeltaEvent",
      "return buildAgentUiMessageSnapshotEvent(",
      "return buildAgentUiTextDeltaEvent(",
      "return buildAgentUiReasoningDeltaEvent(",
      "sourceType: event.type",
      "role: event.message.role",
      "partCount: event.message.content.length",
      "text: event.text",
      "chunkCount: event.chunks.length",
      "boundary: event.boundary",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "truncateText",
      'type: "messages.snapshot"',
      'type: "text.delta"',
      'type: "reasoning.delta"',
      'owner: "session"',
      'owner: "model"',
      'scope: "message"',
      'scope: "part"',
      'phase: "hydrating"',
      'phase: "producing"',
      'phase: "reasoning"',
      'surface: "conversation"',
      'surface: "inline_process"',
      'persistence: "snapshot"',
      'persistence: "transcript"',
      'persistence: "ephemeral_live"',
      "textLength:",
      "preview:",
    ],
  },
  {
    name: "Renderer Agent UI artifact projection delegates artifact and context builders to standard projection package",
    file: "src/components/agent/chat/projection/artifactProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiArtifactSnapshotEvent",
      "buildAgentUiContextTraceEvent",
      "return buildAgentUiArtifactSnapshotEvent(",
      "return buildAgentUiContextTraceEvent(",
      "sourceType: event.type",
      "artifactId: event.artifact.artifactId",
      "filePath: event.artifact.filePath",
      "content: event.artifact.content",
      "metadata: event.artifact.metadata",
      "steps: event.steps",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "truncateText",
      'type: "artifact.preview.ready"',
      'type: "artifact.updated"',
      'type: "context.changed"',
      'owner: "artifact"',
      'owner: "context"',
      'scope: "artifact"',
      'scope: "turn"',
      'phase: "completed"',
      'phase: "producing"',
      'phase: "preparing"',
      'surface: "artifact_workspace"',
      'surface: "runtime_status"',
      'persistence: "artifact_store"',
      'persistence: "snapshot"',
      "contentLength:",
      "metadataKeys:",
      "stepCount:",
      "latestStage:",
      "latestDetailPreview:",
      "artifactIds:",
      "artifactPaths:",
    ],
  },
  {
    name: "Renderer Agent UI turn context projection delegates context and permission builders to standard projection package",
    file: "src/components/agent/chat/projection/contextProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiTurnContextEvents",
      "return buildAgentUiTurnContextEvents(",
      "sessionId: event.session_id",
      "threadId: event.thread_id",
      "turnId: event.turn_id",
      "sourceType: event.type",
      "outputSchemaRuntime: event.output_schema_runtime",
      "contextSummary: buildTurnContextSummaryInput(event.context_summary)",
      "approvalPolicy: event.approval_policy",
      "sandboxPolicy: event.sandbox_policy",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      'type: "context.changed"',
      'type: "permission.changed"',
      'owner: "context"',
      'owner: "policy"',
      'scope: "turn"',
      'phase: "preparing"',
      'surface: "runtime_status"',
      'persistence: "snapshot"',
      "outputSchemaAvailable:",
      "contextSummaryAvailable:",
      "memoryBudget:",
      "missingContext:",
      "retrievalRefs:",
      "teamMemoryRefs:",
      "contextSourceIds:",
      "teamMemoryKeys:",
      "sourceEvent:",
    ],
  },
  {
    name: "Renderer Agent UI action projection delegates HITL builders to standard projection package",
    file: "src/components/agent/chat/projection/actionProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiActionRequiredEvent",
      "buildAgentUiActionResolvedEvent",
      "sourceType: event.type",
      "requestId: event.request_id",
      "actionType: event.action_type",
      "sessionId: event.scope?.session_id",
      "return buildAgentUiActionRequiredEvent(",
      "return buildAgentUiActionResolvedEvent(",
    ],
    absentSnippets: [
      "function actionControl",
      "function resolvedActionControl",
      "metadataKeys",
      "readBooleanField",
      "readRecord",
      "readStringField",
      "truncateText",
      'owner: "action"',
      'scope: "action_request"',
      'surface: "hitl"',
      "control: ",
    ],
    absentSnippets: ["buildAgentUiRuntimeTeamChangedEvent"],
  },
  {
    name: "Renderer Agent UI runtime lifecycle projection delegates lifecycle builders to standard projection package",
    file: "src/components/agent/chat/projection/runtimeLifecycleProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiThreadStartedEvent",
      "buildAgentUiRunStartedEvent",
      "buildAgentUiRunFinishedEvent",
      "buildAgentUiRunFailedEvent",
      "buildAgentUiRuntimeStatusEvent",
      "buildAgentUiModelChangeEvent",
      "buildAgentUiTaskProfileResolvedEvent",
      "return buildAgentUiThreadStartedEvent(",
      "return buildAgentUiRunStartedEvent(",
      "return buildAgentUiRunFinishedEvent(",
      "return buildAgentUiRunFailedEvent(",
      "buildAgentUiRuntimeStatusEvent(",
      "return buildAgentUiModelChangeEvent(",
      "return buildAgentUiTaskProfileResolvedEvent(",
      "sourceType: event.type",
      "threadId: event.thread_id",
      "threadId: event.turn.thread_id",
      "turnId: event.turn.id",
      "promptText: event.turn.prompt_text",
      "metadata: event.status.metadata",
      "model: event.model",
      "mode: event.mode",
      "kind: event.task_profile.kind",
      "permissionProfileKeys: event.task_profile.permissionProfileKeys ?? []",
    ],
    absentSnippets: [
      "buildAgentUiRuntimeTeamChangedEvent",
      "buildAgentUiProjectionBase",
      "buildTeamRuntimeFacts",
      'type: "team.changed"',
      "compactProjectionFields",
      "metadataKeys",
      "normalizeRuntimePhaseFromRuntimeStatusPhase",
      "normalizeRuntimeStatusFromRuntimePhase",
      "truncateText",
      'type: "session.opened"',
      'type: "run.started"',
      'type: "run.finished"',
      'type: "run.failed"',
      'type: "task.changed"',
      'owner: "session"',
      'owner: "runtime"',
      'owner: "task"',
      'scope: "thread"',
      'scope: "run"',
      'phase: "accepted"',
      'phase: "routing"',
      'surface: "session_tabs"',
      'surface: "runtime_status"',
      'surface: "task_capsule"',
      'persistence: "snapshot"',
      "runtimeStatus:",
      "latestTurnStatus:",
      "checkpointCount:",
    ],
  },
  {
    name: "Renderer Team control projection emits task and handoff facts only",
    file: "src/components/agent/chat/projection/teamControlProjection.ts",
    snippets: [
      "buildAgentUiTeamControlProjectionEvents",
      'type: "task.changed"',
      'type: "agent.handoff"',
      'sourceType: "team_control_projection"',
    ],
    absentSnippets: ['type: "team.changed"', 'owner: "team"', 'scope: "team"'],
  },
  {
    name: "Renderer Agent UI permission projection delegates runtime permission builders to standard projection package",
    file: "src/components/agent/chat/projection/permissionProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiRuntimePermissionChangedEvent",
      "return buildAgentUiRuntimePermissionChangedEvent(",
      "sourceType: event.type",
      "phase: event.status.phase",
      "metadata: event.status.metadata",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "definedString",
      "readStringArray",
      "normalizeRuntimePhaseFromRuntimeStatusPhase",
      "function hasPermissionProjectionMetadata",
      "function normalizePermissionPhase",
      'type: "permission.changed"',
      'owner: "policy"',
      'scope: "run"',
      'surface: "hitl"',
      'surface: "runtime_status"',
      'persistence: "snapshot"',
      'control: "approve"',
      "permissionStatus:",
      "confirmationStatus:",
      "requiredProfileKeys:",
      "askProfileKeys:",
      "blockingProfileKeys:",
      "declaredOnly:",
      "turnGating:",
      "sourcePhase:",
    ],
  },
  {
    name: "Renderer Agent UI plan approval projection remains a thin standard-package adapter",
    file: "src/components/agent/chat/projection/planApprovalProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiPlanApprovalRequiredEvent as buildPlanApprovalRequiredEvent",
      "buildAgentUiPlanApprovalResolvedEvent as buildPlanApprovalResolvedEvent",
      "extractAgentUiPlanApprovalProjection as extractPlanApprovalProjection",
      "extractAgentUiPlanApprovalResponseProjection as extractPlanApprovalResponseProjection",
      "AgentUiPlanApprovalProjection as PlanApprovalProjection",
      "AgentUiPlanApprovalResponseProjection as PlanApprovalResponseProjection",
    ],
    absentSnippets: [
      "readBooleanField",
      "readRecord",
      "readStringField",
      "truncateText",
      "function ",
      "interface ",
      'type: "action.required"',
      'type: "action.resolved"',
      'scope: "action_request"',
      'decisionKind: "plan_approval_request"',
      'decisionKind: "plan_approval_response"',
    ],
  },
  {
    name: "Renderer Agent UI diagnostic projection delegates warning and cost builders to standard projection package",
    file: "src/components/agent/chat/projection/diagnosticProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiWarningEvent",
      "buildAgentUiCostMetricEvent",
      "return buildAgentUiWarningEvent(",
      "return buildAgentUiCostMetricEvent(",
      "sourceType: event.type",
      "code: event.code",
      "message: event.message",
      "metricEvent: event.type",
      "costState: event.cost_state",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "truncateText",
      'type: "diagnostic.changed"',
      'type: "metric.changed"',
      'owner: "diagnostics"',
      'scope: "run"',
      'phase: "acting"',
      'surface: "diagnostics"',
      'persistence: "diagnostics_log"',
      "messagePreview:",
      "estimatedCostClass:",
      "estimatedTotalCost:",
      "cachedInputTokens:",
      "cacheCreationInputTokens:",
    ],
  },
  {
    name: "Renderer Agent UI historical hydration projection delegates snapshot builders to standard projection package",
    file: "src/components/agent/chat/projection/historicalMessageHydrationProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiHistoricalHydrationEvents",
      "return buildAgentUiHistoricalHydrationEvents(input, context)",
      "isHistoricalAssistantMessageHydrationCandidate",
      "buildHistoricalMarkdownHydrationTargets",
      "countDeferredHistoricalContentParts",
      "countDeferredHistoricalMarkdown",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      'type: "session.hydrated"',
      'type: "messages.snapshot"',
      'type: "diagnostic.changed"',
      'owner: "session"',
      'owner: "diagnostics"',
      'scope: "session"',
      'scope: "thread"',
      'phase: "hydrating"',
      'phase: "completed"',
      'surface: "session_tabs"',
      'surface: "conversation"',
      'surface: "diagnostics"',
      'persistence: "snapshot"',
      'persistence: "diagnostics_log"',
      "historical_hydration_stale_window",
      "diagnosticKey:",
    ],
  },
  {
    name: "Renderer Agent UI routing projection delegates run status builder to standard projection package",
    file: "src/components/agent/chat/projection/routingProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiRoutingStatusEvent",
      "return buildAgentUiRoutingStatusEvent(",
      "sourceType: event.type",
      "runtimeEvent: event.type",
      "routingDecision: event.routing_decision",
      "limitState: event.limit_state",
      "limitEvent: event.limit_event",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "buildRoutingDecisionPayload",
      "truncateText",
      'type: "run.status"',
      'owner: "runtime"',
      'scope: "run"',
      'phase: "routing"',
      'phase: "failed"',
      'surface: "runtime_status"',
      'persistence: "snapshot"',
      "limitStatus:",
      "singleCandidateOnly:",
      "providerLocked:",
      "settingsLocked:",
      "oemLocked:",
      "limitEventKind:",
      "messagePreview:",
      "retryable:",
    ],
  },
  {
    name: "Renderer Agent UI thread item projection delegates host-neutral builders to standard projection package",
    file: "src/components/agent/chat/projection/threadItemProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiThreadItemBase as buildStandardThreadItemBase",
      "buildAgentUiThreadItemEvent",
      "extractAgentUiTaskOwnerChangeProjection",
      "return buildStandardThreadItemBase(sourceType, item, context)",
      "return buildAgentUiThreadItemEvent(sourceType, item, context)",
      "const taskOwnerProjection = extractAgentUiTaskOwnerChangeProjection(",
      "extractPlanApprovalProjection(item.metadata)",
      "buildTaskOwnerChangeProjectionEvents(sourceType, item, context)",
    ],
    absentSnippets: [
      "function threadItemPhase",
      "function threadItemToolResultType",
      "function threadItemToolPhase",
      "function normalizeProjectionToolName",
      "function isTaskUpdateToolName",
      "readRecord",
      "readStringArrayField",
      "readStringField",
      "owner_change",
      "ownerChange",
      "updated_fields",
      "updatedFields",
      "metadataKeys",
      "extractArtifactRefs",
      'type: "plan.final"',
      'type: "reasoning.summary"',
      'type: "tool.result"',
      'type: "tool.progress"',
      'type: "artifact.preview.ready"',
      'type: "context.compaction.completed"',
      'type: "state.snapshot"',
      'type: "diagnostic.changed"',
      'type: "action.required"',
      'type: "action.resolved"',
      'type: "agent.changed"',
      'type: "worker.notification"',
      "buildAgentUiThreadItemSubagentWorkerNotificationEvent",
      'owner: "tool"',
      'scope: "tool_call"',
      'scope: "action_request"',
      'runtimeEntity: "subagent_turn"',
      "workerNotificationId:",
      'notificationKind: "worker_result"',
      "summaryPreview:",
      "promptPreview:",
      "hasResponse:",
      "commandPreview:",
      "outputPreview:",
      "errorPreview:",
    ],
  },
  {
    name: "Renderer Agent UI tool event projection delegates host-neutral builders to standard projection package",
    file: "src/components/agent/chat/projection/toolEventProjection.ts",
    snippets: [
      'from "@limecloud/agent-runtime-projection"',
      "buildAgentUiToolStartEvents",
      "buildAgentUiToolEndEvents",
      "buildAgentUiToolProgressEvent",
      "buildAgentUiToolOutputDeltaEvent",
      "buildAgentUiToolInputDeltaEvent",
      "return buildAgentUiToolStartEvents(",
      "return buildAgentUiToolEndEvents(",
      "return buildAgentUiToolProgressEvent(",
      "return buildAgentUiToolOutputDeltaEvent(",
      "return buildAgentUiToolInputDeltaEvent(",
      "sourceType: event.type",
      "toolCallId: event.tool_id",
      "toolName: event.tool_name",
      "input: event.arguments",
      "result: event.result",
      "progress: event.progress",
      "outputKind: event.output_kind",
      "accumulatedInput: event.accumulated_arguments",
    ],
    absentSnippets: [
      "buildAgentUiProjectionBase",
      "buildPlanApprovalRequiredEvent",
      "buildPlanApprovalResolvedEvent",
      "extractPlanApprovalProjection",
      "extractPlanApprovalResponseProjection",
      "extractArtifactRefs",
      "metadataKeys",
      "truncateText",
      "function buildToolEndEvent(",
      'type: "tool.started"',
      'type: "tool.args"',
      'type: "tool.result"',
      'type: "tool.failed"',
      'type: "tool.progress"',
      'type: "tool.output.delta"',
      'type: "tool.args.delta"',
      'owner: "tool"',
      'scope: "tool_call"',
      'surface: "tool_ui"',
      'persistence: "ephemeral_live"',
      'persistence: "archive"',
      "outputPreview:",
      "errorPreview:",
      "metadataKeys:",
      "diagnosticKeys:",
    ],
  },
  {
    name: "Agent Runtime UI package keeps index as barrel exports",
    file: "packages/agent-runtime-ui/src/index.ts",
    snippets: [
      'export type * from "./types.js"',
      'export * from "./messages.js"',
      'export * from "./processTimeline.js"',
      'export * from "./executionGraph.js"',
      'export * from "./runtimeFacts.js"',
      'export * from "./projectionView.js"',
    ],
    absentSnippets: [
      "function ",
      "interface ",
      "const ",
      "import ",
      "@limecloud/agent-ui-contracts",
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
    ],
  },
  {
    name: "Agent Runtime UI package keeps React primitives split by responsibility",
    files: [
      "packages/agent-runtime-ui/src/types.ts",
      "packages/agent-runtime-ui/src/labels.ts",
      "packages/agent-runtime-ui/src/messages.tsx",
      "packages/agent-runtime-ui/src/processTimeline.tsx",
      "packages/agent-runtime-ui/src/executionGraph.tsx",
      "packages/agent-runtime-ui/src/runtimeFacts.tsx",
      "packages/agent-runtime-ui/src/projectionView.tsx",
    ],
    snippets: [
      "AgentTimelineProps",
      "defaultMessageTitle",
      "UIMessagePartsView",
      "ProcessTimelineView",
      "ExecutionGraphView",
      "RuntimeFactCard",
      "AgentUiProjectionView",
    ],
    absentSnippets: [
      'from "@/',
      'from "src/',
      'from "../../src',
      'from "../../../src',
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "EventSource",
      "new WebSocket",
      "fetch(",
      "XMLHttpRequest",
    ],
  },
  {
    name: "Agent Runtime UI labels stay injectable and host-neutral",
    files: [
      "packages/agent-runtime-ui/src/types.ts",
      "packages/agent-runtime-ui/src/labels.ts",
      "packages/agent-runtime-ui/src/messages.tsx",
      "packages/agent-runtime-ui/src/processTimeline.tsx",
      "packages/agent-runtime-ui/src/executionGraph.tsx",
      "packages/agent-runtime-ui/src/runtimeFacts.tsx",
      "packages/agent-runtime-ui/src/projectionView.tsx",
    ],
    snippets: [
      "AgentUiProjectionViewLabels",
      "messagePartsAriaLabel",
      "processTimelineAriaLabel",
      "actionButtonLabel",
      "eventStatusLabel",
      "data-action-decision",
      "event.actions?.length",
      "Open model settings",
      "Message parts",
      "Process timeline",
      "Execution graph",
    ],
    absentSnippets: [
      "消息部分",
      "过程时间线",
      "工具调用",
      "待处理动作",
      "协作事实摘要",
      "打开模型设置",
      "补输入源",
    ],
  },
  {
    name: "Agent UI Runtime standard document is the current package and host integration fact source",
    files: [
      "internal/aiprompts/agent-ui-runtime-standard.md",
      "internal/aiprompts/README.md",
      "internal/aiprompts/agent-protocol-standards-map.md",
      "internal/prd/next/implementation-roadmap.md",
      "packages/agent-runtime-projection/README.md",
    ],
    snippets: [
      "agent-ui-runtime-standard.md",
      "@limecloud/agent-runtime-client",
      "@limecloud/agent-ui-contracts",
      "@limecloud/agent-runtime-projection",
      "@limecloud/agent-runtime-ui",
      "Standard Package Layering",
      "Source Layout",
      "src/envelope.ts",
      "src/normalization.ts",
      "src/refs.ts",
      "src/routing.ts",
      "src/runtimeFacts.ts",
      "buildAgentUiProjectionBase",
      "sequenceAgentUiProjectionEvents",
      "AgentRuntimeClient -> projectAgentUiState -> AgentUiProjectionView",
      "主聊天 projection 迁移分类",
      "projection package `index` 只能做 barrel exports",
      "barrel exports only",
      "src/components/agent/chat/projection/**",
      "agentUiEventProjection.ts",
      "conversationProjectionStore.ts",
      "agentUiProjectionSummary.ts",
      "不得新增第二套 runtime fact source",
    ],
    absentSnippets: [
      ...retiredTreeProjectionNames,
      "直接套用外部 SDK",
      "新增 Agent Runtime 能力可以落到 legacy desktop facade",
      "主聊天目录可以重新实现标准 projection selector",
    ],
  },
  {
    name: "Renderer Agent Runtime derived exports use App Server agentSession export methods",
    file: "src/lib/api/agentRuntime/exportClient.ts",
    snippets: [
      "APP_SERVER_METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT",
      "APP_SERVER_METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT",
      "APP_SERVER_METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT",
      "APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT",
      "APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE",
      '"exportHandoffBundle"',
      '"exportReplayCase"',
      '"exportAnalysisHandoff"',
      '"exportReviewDecisionTemplate"',
      '"saveReviewDecision"',
      "appServerClient.exportHandoffBundle({",
      "appServerClient.exportReplayCase({",
      "appServerClient.exportAnalysisHandoff({",
      "appServerClient.exportReviewDecisionTemplate({",
      "appServerClient.saveReviewDecision({",
      "sessionId: normalizedSessionId",
      "decisionStatus: request.decision_status",
      "decisionSummary: request.decision_summary",
      "chosenFixStrategy: request.chosen_fix_strategy",
      "riskLevel: request.risk_level",
    ],
    absentSnippets: [
      "AGENT_RUNTIME_COMMANDS.exportHandoffBundle",
      "AGENT_RUNTIME_COMMANDS.exportReplayCase",
      "AGENT_RUNTIME_COMMANDS.exportAnalysisHandoff",
      "AGENT_RUNTIME_COMMANDS.exportReviewDecisionTemplate",
      "AGENT_RUNTIME_COMMANDS.saveReviewDecision",
      '"agent_runtime_export_handoff_bundle"',
      '"agent_runtime_export_replay_case"',
      '"agent_runtime_export_analysis_handoff"',
      '"agent_runtime_export_review_decision_template"',
      '"agent_runtime_save_review_decision"',
    ],
  },
  {
    name: "Renderer Agent Runtime derived export tests lock App Server path and fail closed behavior",
    file: "src/lib/api/agentRuntime/exportClient.test.ts",
    snippets: [
      "analysis / replay / review current 导出应走 App Server",
      "appServerClient.exportAnalysisHandoff",
      "appServerClient.exportReplayCase",
      "appServerClient.exportReviewDecisionTemplate",
      "appServerClient.saveReviewDecision",
      "agentSession/analysisHandoff/export did not return runtime analysis handoff",
      "agentSession/replayCase/export did not return runtime replay case",
      "agentSession/reviewDecisionTemplate/export did not return runtime review decision template",
      "agentSession/reviewDecision/save did not return runtime review decision template",
      "缺少 sessionId 时派生导出应 fail closed",
    ],
    absentSnippets: [
      "analysis / replay / review compat 导出",
      "agent_runtime_export_analysis_handoff did not return",
      "agent_runtime_export_replay_case did not return",
      "agent_runtime_export_review_decision_template did not return",
      "agent_runtime_save_review_decision did not return",
    ],
  },
  {
    name: "Renderer Agent Runtime tests lock App Server lifecycle and direct v2 notification behavior",
    files: [
      "src/lib/api/agentRuntime/threadClient.test.ts",
      "src/lib/api/agentRuntime/serverRequestReplay.unit.test.ts",
      "src/lib/api/agentRuntime/appServerV2Notification.test.ts",
      "src/lib/api/agentRuntime/appServerEventStream.test.ts",
    ],
    snippets: [
      "App Server 可用时 submit 应进入 turn/start",
      "App Server 不可用时 submit 应 fail closed，不回退 legacy command",
      "App Server 可用且 turn_id 存在时 interrupt 应进入 turn/interrupt",
      "App Server 不可用时 interrupt 应 fail closed，不回退 legacy command",
      "缺少 turn_id 时 interrupt 应 fail closed，不回退 legacy command",
      "typed pending 不存在时 respond action 应 fail closed，不回退旧 action/respond",
      "App Server 不可用时 respond action 应 fail closed，不回退 legacy command",
      "projects direct lifecycle notifications into the existing GUI payloads",
      "routes drained direct notifications and closes on turn/completed",
      "拒绝 retired agentSession/event wrapper",
      "即使 wrapper 携带 canonicalEvent/typedEvent 也应 fail closed",
      "request projection 应在未传可选项时保持精简 App Server 参数",
      "replay request 无当前 typed server-request 时应 fail closed 且不调用旧 action/replay",
      "listenAgentRuntimeEvent",
      "appServerClient.startTurn",
      "appServerClient.cancelTurn",
      "appServerClient.startThreadCompaction",
      "appServerClient.resumeThread",
      "findPendingTypedServerRequestAction",
      "respondPendingTypedServerRequest",
      "同作用域 AskUser typed pending 应由 controller settle",
      "typed pending scope 不匹配时 fail closed",
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
      "expect(invokeCommand).not.toHaveBeenCalled()",
    ],
    absentSnippets: [
      "App Server 不可用时 submit 应保留 legacy fallback",
      "缺少 turn_id 时 interrupt 应回退 legacy command",
      "AGENT_RUNTIME_COMMANDS",
      "publishAgentRuntimeEvent",
      "resumeAgentSessionThread",
      "resumeAgentRuntimeThread",
      "RuntimeResumeContract",
    ],
  },
  {
    name: "Renderer Agent API facade locks lifecycle to Electron IPC App Server JSON-RPC",
    file: "src/lib/api/agent.test.ts",
    snippets: [
      'expect(call?.[0]).toBe("app_server_handle_json_lines")',
      "APP_SERVER_METHOD_TURN_START",
      "APP_SERVER_METHOD_THREAD_DELETE",
      "APP_SERVER_METHOD_THREAD_RESUME",
      "submitAgentRuntimeTurn 应经 Electron IPC 调 App Server turn/start",
      "replayAgentRuntimeRequest 无当前 typed pending 时应 fail closed",
      "respondAgentRuntimeAction 缺少 typed pending 时应 fail closed，不发旧 action/respond",
      "resumeThread 应经 Electron IPC 调 App Server thread/resume",
      "mockIsElectronHostCommandAvailable.mockReturnValue(true)",
      "expectAppServerRequest(1, APP_SERVER_METHOD_TURN_START",
      "expectAppServerRequest(2, APP_SERVER_METHOD_THREAD_DELETE",
      "generic agentSession/action/respond is retired",
      "expectAppServerRequest(1, APP_SERVER_METHOD_THREAD_RESUME",
    ],
    absentSnippets: [
      "submitAgentRuntimeTurn 应走统一 runtime submit 命令",
      "respondAgentRuntimeAction 应走统一 action 响应命令",
      '"agent_runtime_submit_turn"',
      '"agent_runtime_interrupt_turn"',
      '"agent_runtime_respond_action"',
      '"agent_runtime_export_evidence_pack"',
      "resumeAgentSessionThread",
      "resumeAgentRuntimeThread",
      "RuntimeResumeContract",
    ],
  },
  {
    name: "Renderer Agent Runtime client factory prevents lifecycle and queue bridgeInvoke fallback",
    file: "src/lib/api/agentRuntime/clientFactory.test.ts",
    snippets: [
      "queue/session control 应走 App Server current，且不再暴露 retired site adapter surface",
      "queue/session control 不应回退到 legacy bridgeInvoke",
      "appServerClient.resumeThread",
      'expect(client).not.toHaveProperty("siteListAdapters")',
      'expect(client).not.toHaveProperty("siteRunAdapter")',
      "expect(invoke).not.toHaveBeenCalled()",
      "turn lifecycle 应走 App Server client，不复用 legacy bridgeInvoke",
      "appServerClient.startTurn",
      "expect(bridgeInvoke).not.toHaveBeenCalled()",
    ],
    absentSnippets: [
      '"agent_runtime_submit_turn"',
      '"agent_runtime_interrupt_turn"',
      '"agent_runtime_respond_action"',
      "resumeAgentSessionThread",
      "resumeAgentRuntimeThread",
      "RuntimeResumeContract",
    ],
  },
  {
    name: "Agent runtime client package exposes session gateway adapter",
    files: [
      "packages/agent-runtime-client/package.json",
      "packages/agent-runtime-client/README.md",
      "packages/agent-runtime-client/src/index.ts",
      "packages/agent-runtime-client/src/sessionGateway.ts",
      "packages/agent-runtime-client/tests/client.test.mjs",
      "tsconfig.json",
      "vite.config.ts",
    ],
    snippets: [
      '"./sessionGateway"',
      "@limecloud/agent-runtime-client/sessionGateway",
      "./packages/agent-runtime-client/src/sessionGateway.ts",
      "./packages/app-server-client/src/browser.ts",
      "browser-safe 子路径",
      "createAgentRuntimeClientFromSessionGateway",
      "type AgentRuntimeLifecycleClient",
      "type AgentRuntimeSessionGateway",
      '"startTurn" | "steerTurn" | "readThread" | "cancelTurn" | "respondAction"',
      "readThread:",
      "callAgentRuntimeSessionGateway(gateway.readThread, params, options)",
      "if (options === undefined)",
      "createAgentRuntimeClientFromSessionGateway(...)` 只适配现有 session gateway",
      "readThread",
      "timeoutMs: 120_000",
    ],
    absentSnippets: [
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "plugin_runtime_",
      "agent_runtime_submit_turn",
      "fetch(",
      "XMLHttpRequest",
    ],
  },
  {
    name: "Desktop Host renderer invoke does not fall back to default mocks",
    file: "src/lib/desktop-host/core.ts",
    snippets: [
      "Frontend -> Electron Desktop Host IPC -> App Server JSON-RPC -> RuntimeCore / backend",
      "生产 invoke 不再回退 mock",
      "export async function invokeMockOnly<T = any>",
      "return invokeDefaultMock<T>(cmd, args)",
      "export async function invoke<T = any>",
      "const electronHost = getElectronHostBridge()",
      "return electronHost.invoke<T>(cmd, args)",
      "return await invokeViaHttp<T>(cmd, args)",
      "throw normalizeDevBridgeError(cmd, error)",
      "无法进入 App Server JSON-RPC 主链",
    ],
    absentSnippets: [
      "shouldPreferMockInBrowser",
      "shouldDisallowMockFallbackInBrowser",
      "fallback 使用",
    ],
  },
  {
    name: "Desktop Host tests lock production fail-closed and explicit mock-only boundary",
    file: "src/lib/desktop-host/core.test.ts",
    snippets: [
      "HTTP bridge 失败时 production invoke 直接抛出规范化错误",
      "无 Electron host 且无 HTTP bridge 时 production invoke fail-closed",
      "测试注册的配置 mock",
      "知识库 legacy 显式 mock 已退场",
      "工具库存 legacy 显式 mock 已退场",
      "显式 mock 入口可返回默认工作区数据已退场",
      "媒体任务 artifact 默认 mock 已退场并 fail closed",
      'invokeMockOnly("agent_runtime_get_tool_inventory"',
      '未注册命令 "agent_runtime_get_tool_inventory"',
      "expect(mocks.invokeViaHttp).not.toHaveBeenCalled()",
    ],
    absentSnippets: [
      "shouldPreferMockInBrowser",
      "mock 优先命令直接返回默认 mock",
      "bridge 失败且命令存在 mock 时回退默认 mock 数据",
      "fallback mock 应",
      "命令在 bridge 失败时应回退",
      "媒体任务显式 mock 应返回统一 task file 协议",
      "音频任务显式 mock 应返回 voice_generation task file 协议",
    ],
  },
  {
    name: "Desktop Host core no longer loads Agent Runtime default mocks",
    file: "src/lib/desktop-host/core.ts",
    snippets: [
      "生产 invoke 不再回退 mock",
      "export async function invokeMockOnly<T = any>",
      "const defaultMocks = await loadDefaultMocks()",
    ],
    absentSnippets: [
      'import("./agentRuntimeMocks")',
      'import("./agentRuntimeObjectiveMocks")',
      "agentRuntime.agentRuntimeMocks",
      "agentRuntimeObjective.resetAgentRuntimeObjectiveMocks",
      "agent_runtime_submit_turn:",
      "agent_runtime_interrupt_turn:",
      "agent_runtime_respond_action:",
    ],
  },
  {
    name: "Frontend integration matrix forbids legacy or mock turn lifecycle evidence",
    file: "internal/roadmap/appserver/frontend-integration-matrix.md",
    snippets: [
      "Frontend -> Electron Desktop Host bridge -> App Server JSON-RPC -> RuntimeCore / backend",
      "真正的后端事实源只有一个：App Server JSON-RPC",
      "submit / cancel / respond 已改为 App Server current-only",
      "当 `app_server_handle_json_lines` 不可用时 fail closed，不再回退 `agent_runtime_*`",
      "生产不能 mock，只有测试才 mock",
      "`scripts/check-command-contracts.mjs` 必须扫描 `src / electron / packages` 的生产源码",
      "禁止 `explicitMockFallback`、`invokeMockOnly`、`mockCommand`、`clearMocks`、`invokeExplicitMock`、`listenExplicitMock` 进入生产路径",
      "不能把 initialize smoke 或 hook 测试误判成完整聊天闭环",
      "`agent_runtime_submit_turn` / `agent_runtime_interrupt_turn` / `agent_runtime_respond_action`",
      "只能作为负向回归扫描对象，不能作为 Agent turn lifecycle 的完成证据",
      "`claw-chat-ready-streaming` 是用户显式发起的 live E2E 入口，不再要求 `--allow-live-provider` 二次授权",
      "证明 `WebSearch` / `WebFetch` 同一 turn 的真实工具事件与 read model 输出",
      "不能用 mock、无模型探针、Provider 自动探测或 renderer fallback 替代真实业务闭环",
      "handoff bundle / replay case / analysis handoff / review decision template / save review decision 已分别通过",
      "| canonical read model / derived exports",
      "| `wired`",
    ],
    absentSnippets: [
      "Electron truth adapter",
      "legacy facade 投影作为完成路线",
      "mock-only 成功作为完成证据",
    ],
  },
  {
    name: "Command contract scans production mock-only usage outside tests",
    file: "scripts/check-command-contracts.mjs",
    snippets: [
      'const sourceRoots = ["src"]',
      'const productionRuntimeRoots = ["src", "electron", "packages"]',
      "function collectFrontendCommandUsage()",
      "for (const root of sourceRoots)",
      "function collectProductionMockOnlyUsageFailures()",
      "for (const root of productionRuntimeRoots)",
      "function isAllowedTestMockFixtureSource(relativePath)",
      "生产源码不能调用测试 mock invoke 入口",
      "生产源码不能注册 renderer mock command",
      "生产源码不能清理测试 mock command",
      "生产源码不能调用显式 renderer mock fallback",
      "生产源码不能调用显式 renderer event mock fallback",
      "生产源码不能导入显式 renderer mock fallback",
      "生产源码不能导入 desktop-host 测试 mock 入口",
      "failures.push(...collectProductionMockOnlyUsageFailures())",
    ],
    absentSnippets: [
      'for (const root of productionRuntimeRoots) {\n    const absoluteRoot = path.join(repoRoot, root);\n    for (const relativePath of walkDirectory(absoluteRoot)) {\n      const absolutePath = path.join(repoRoot, relativePath);\n      const sourceCode = fs.readFileSync(absolutePath, "utf8");',
    ],
  },
  {
    name: "Execution plan records Agent turn lifecycle as current-only after P3.122",
    file: "internal/exec-plans/app-server-implementation-plan.md",
    snippets: [
      "P3.135 Agent turn lifecycle current-only 文档 / 契约守卫",
      "该诊断已由 P3.122 之后的 current-only 改造收口",
      "后续不得再把该旧路径作为可选实现或完成证据",
      "后续结论已由 P3.122 确定",
      "前端 Agent turn lifecycle 必须直接经 `src/lib/api/appServer.ts -> app_server_handle_json_lines -> App Server JSON-RPC`",
      "legacy facade 只可作为已退役兼容残留，不再作为二选一实现路径",
    ],
    absentSnippets: [
      "后续实现时优先二选一",
      "legacy facade 投影到 App Server method，禁止两条路同时长",
    ],
  },
  {
    name: "Execution plan records production mock and live-provider smoke gates",
    file: "internal/exec-plans/app-server-implementation-plan.md",
    snippets: [
      "P3.137 production mock / live-provider gate 收口",
      "生产不能 mock，只有测试才 mock",
      "`scripts/check-command-contracts.mjs` 的生产 mock-only 扫描已覆盖 `src / electron / packages`",
      "前端命令统计仍只扫描 `src`",
      "P3.308 Claw live E2E 二次授权 gate 清理",
      "`claw-chat-ready-streaming` 是用户显式发起的 live Provider / WebSearch-WebFetch E2E 入口",
      "不再要求 `--allow-live-provider` 二次授权",
      "P3.316 Claw live E2E 二次授权残留清理",
      "P3.316 live E2E 自动选型修正",
      "历史早期 P3.137/P3.194/P3.220 等记录只作为 historical evidence，不再作为 current 执行指令",
    ],
  },
  {
    name: "Claw chat streaming smoke proves GUI turn lifecycle through App Server JSON-RPC",
    file: "scripts/claw-chat-ready-streaming-smoke.mjs",
    snippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      "const APP_SERVER_METHOD_TURN_START",
      '"turn/start"',
      "const APP_SERVER_METHOD_TURN_INTERRUPT",
      '"turn/interrupt"',
      'const APP_SERVER_METHOD_THREAD_READ = "thread/read"',
      'const APP_SERVER_METHOD_AGENT_SESSION_EVENT = "agentSession/event"',
      "attachAppServerRequestMessages(",
      "attachAppServerResponsePayload(",
      "function findAppServerMethodRecord(",
      "function appServerMethodSeen(",
      "function isSmokeDiagnosticJsonRpcId(",
      "function appServerSessionReadAfterEventEvidence(",
      "async function readAppServerSession(",
      "async function readAppServerThreadRead(",
      "async function cancelAppServerTurn(",
      "等待长 turn App Server submit",
      "等待 App Server turn cancel invoke",
      "等待恢复 turn App Server submit",
      "DevBridge transport must be electron-host",
      "summary.longSubmitAppServer",
      "summary.interruptAppServer",
      "summary.followSubmitAppServer",
      "summary.appServerEvidence =",
      "appServerTurnStartSeen",
      "appServerTurnCancelSeen",
      "longTurnFastCompleteAccepted",
      "longTurnCanBeInterrupted",
      "appServerSessionReadSeen",
      "appServerEventSeen",
      "appServerSessionReadAfterEventSeen",
      'electronHostBridge: health?.transport === "electron-host"',
      "readAfterEvent: appServerReadAfterEvent",
      "interruptedTurnCanceled",
      "summary.assertions.appServerTurnStartSeen &&",
      "(summary.assertions.longTurnCanBeInterrupted ||",
      "summary.assertions.longTurnFastCompleteAccepted) &&",
      "summary.assertions.appServerSessionReadSeen &&",
      "summary.assertions.appServerEventSeen &&",
      "summary.assertions.appServerSessionReadAfterEventSeen &&",
      "summary.assertions.electronHostBridge &&",
    ],
    absentSnippets: [
      'item.cmd === "agent_runtime_submit_turn"',
      'item.cmd === "agent_runtime_interrupt_turn"',
      '"agent_runtime_promote_queued_turn"',
      '"agent_runtime_get_session"',
      '"agent_runtime_get_thread_read"',
      "interruptCommandSeen",
      "interruptedTurnAborted",
    ],
  },
  {
    name: "Claw chat streaming smoke is explicit live E2E without duplicate provider gate",
    file: "scripts/claw-chat-ready-streaming-smoke.mjs",
    snippets: [
      'logStage("wait-health")',
      'logStage("prepare-runtime")',
      '"modelProvider/list"',
      '"modelProvider/testChat"',
      "LIVE_WEB_TOOL_PROMPT",
      "REQUIRED_LIVE_WEB_TOOL_NAMES",
      "function modelLooksLightweight(",
      "function modelLooksExpensive(",
      "function modelLooksToolReliable(",
      "modelLooksToolReliable(value)",
      "modelLooksLightweight(model)",
      "modelLooksExpensive(model)",
      "modelPenalty",
      "@搜索 关键词:联网工具验证",
      "webSearch=true",
      'searchMode="required"',
      "liveWebExplicitSearchRequired",
    ],
    absentSnippets: [
      'from "./lib/live-provider-smoke-gate.mjs"',
      "assertLiveProviderSmokeAllowed({",
      "liveProviderSmokeAllowed(",
      "allowLiveProvider",
      "--allow-live-provider",
    ],
  },
  {
    name: "Social Workbench smoke uses App Server current session read model",
    file: "scripts/social-workbench-e2e-smoke.mjs",
    snippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      'const METHOD_THREAD_READ = "thread/read"',
      "async function readAgentSession(sessionId)",
      "function projectWorkbenchState(sessionRead, limit)",
      "function projectRunDetail(sessionRead, runId)",
      "await invokeAppServerJsonRpc(METHOD_THREAD_READ",
      "thread/read 未能投影运行详情",
    ],
    absentSnippets: [
      '"execution_run_get_theme_workbench_state"',
      '"execution_run_get"',
    ],
  },
  {
    name: "Electron Desktop Host records Connect deep links from startup and second-instance args",
    file: "electron/main.ts",
    snippets: [
      "function normalizeDeepLinkUrl(",
      'trimmed.startsWith("lime://")',
      "function collectDeepLinkUrls(",
      "function recordDeepLinkUrls(",
      "recordDeepLinkUrls(collectDeepLinkUrls(process.argv))",
      "recordDeepLinkUrls(collectDeepLinkUrls(argv))",
      "ELECTRON_E2E_USER_DATA_DIR",
      'app.setPath("userData", resolvedUserDataDir)',
    ],
    absentSnippets: [
      'const url = argv.find((arg) => arg.startsWith("lime://"))',
      "handle_deep_link",
      "handle_open_deep_link",
    ],
  },
  {
    name: "Connect deep link current smoke proves Electron preload to App Server JSON-RPC path",
    file: "scripts/connect-deep-link-current-smoke.mjs",
    snippets: [
      'import { _electron as electron } from "playwright"',
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      'const REQUIRED_APP_SERVER_METHOD = "connectDeepLink/resolve"',
      "const FORBIDDEN_CONNECT_COMMANDS = [",
      '"handle_deep_link"',
      '"handle_open_deep_link"',
      '"save_relay_api_key"',
      '"send_connect_callback"',
      '"list_relay_providers"',
      '"refresh_relay_registry"',
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI?.deepLink?.getCurrent",
      "await window.electronAPI.deepLink.getCurrent()",
      'args: ["--use-mock-keychain", ".", options.deepLinkUrl]',
      "ELECTRON_E2E_USER_DATA_DIR: tmpUserDataDir",
      "appServerHandleJsonLinesSeen",
      "connectDeepLinkResolveSeen",
      "forbiddenCommandsSeen.length === 0",
      "await cancelConnectDialog(page)",
    ],
    absentSnippets: [
      "invokeViaHttp",
      "chromium.launch",
      "agent_runtime_",
      "safeInvoke",
    ],
  },
  {
    name: "Connect deep link save smoke proves API Key save and callback current path",
    file: "scripts/connect-deep-link-save-current-smoke.mjs",
    snippets: [
      'import { _electron as electron } from "playwright"',
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      '"connectDeepLink/resolve"',
      '"connectRelayApiKey/save"',
      '"connectCallback/send"',
      "seedConnectRegistryCaches(",
      "registryPayload()",
      "HOME: home",
      "XDG_DATA_HOME: xdgDataHome",
      "APPDATA: appData",
      "LOCALAPPDATA: localAppData",
      "LIME_AGENT_RUNTIME_ROOT: agentRoot",
      "ELECTRON_E2E_USER_DATA_DIR: tmpUserDataDir",
      'args: ["--use-mock-keychain", ".", options.deepLinkUrl]',
      "LIME_ELECTRON_E2E",
      "callbackNetworkDeliveryVerified: false",
      "await clickConfirm(page)",
      "waitForSaveEvidence(page, options)",
      "dialogClosedAfterConfirm",
      "missingRequiredAppServerMethods.length === 0",
      "databaseFiles.length > 0",
      "const FORBIDDEN_CONNECT_COMMANDS = [",
      '"handle_deep_link"',
      '"handle_open_deep_link"',
      '"save_relay_api_key"',
      '"send_connect_callback"',
      '"list_relay_providers"',
      '"refresh_relay_registry"',
      "forbiddenCommandsSeen.length === 0",
    ],
    absentSnippets: [
      "invokeViaHttp",
      "chromium.launch",
      "agent_runtime_",
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
    ],
  },
  {
    name: "Connect open deep link smoke proves website entry current path",
    file: "scripts/connect-open-deep-link-current-smoke.mjs",
    snippets: [
      'import { _electron as electron } from "playwright"',
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      'const REQUIRED_APP_SERVER_METHOD = "connectOpenDeepLink/resolve"',
      "const EXPECTED_BANNER_TEXT =",
      "const EXPECTED_PROMPT_TEXT =",
      "const FORBIDDEN_CONNECT_COMMANDS = [",
      '"handle_deep_link"',
      '"handle_open_deep_link"',
      '"save_relay_api_key"',
      '"send_connect_callback"',
      '"list_relay_providers"',
      '"refresh_relay_registry"',
      "window.__LIME_ELECTRON__ === true",
      "window.electronAPI?.deepLink?.getCurrent",
      "await window.electronAPI.deepLink.getCurrent()",
      "websiteOpenBannerVisible",
      "websitePromptVisible",
      "connectOpenDeepLinkResolveSeen",
      "connectDialogVisible",
      "appServerHandleJsonLinesSeen",
      "forbiddenCommandsSeen.length === 0",
      "HOME: home",
      "XDG_DATA_HOME: xdgDataHome",
      "APPDATA: appData",
      "LOCALAPPDATA: localAppData",
      "LIME_AGENT_RUNTIME_ROOT: agentRoot",
      "ELECTRON_E2E_USER_DATA_DIR: tmpUserDataDir",
      'args: ["--use-mock-keychain", ".", options.deepLinkUrl]',
      "LIME_ELECTRON_E2E",
    ],
    absentSnippets: [
      "invokeViaHttp",
      "chromium.launch",
      "agent_runtime_",
      "safeInvoke",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "saveConnectRelayApiKey",
      "sendConnectCallback",
    ],
  },
  {
    name: "Agent runtime tool surface page smoke uses App Server current runtime methods",
    file: "scripts/agent-runtime/tool-surface-page-smoke.mjs",
    snippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      'const APP_SERVER_METHOD_THREAD_START = "thread/start"',
      'const APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE = "thread/settings/update"',
      'const APP_SERVER_METHOD_THREAD_READ = "thread/read"',
      'const APP_SERVER_METHOD_THREAD_LIST = "thread/list"',
      "const APP_SERVER_METHOD_TURN_START =",
      '"turn/start"',
      "const APP_SERVER_METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL =",
      '"item/commandExecution/requestApproval"',
      'const APP_SERVER_METHOD_SERVER_REQUEST_RESOLVED = "serverRequest/resolved"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST =",
      '"agentSession/fileCheckpoint/list"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET =",
      '"agentSession/fileCheckpoint/get"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF =",
      '"agentSession/fileCheckpoint/diff"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE =",
      '"agentSession/fileCheckpoint/restore"',
      "FORBIDDEN_AGENT_RUNTIME_CURRENT_METHOD_COMMANDS",
      "legacy_agent_runtime_current_method_command",
      "hasAppServerMethodCount(",
      "APP_SERVER_METHOD_TURN_START",
      "APP_SERVER_METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL",
      "APP_SERVER_METHOD_SERVER_REQUEST_RESOLVED",
      "buildCodeRuntimeApprovalServerRequest()",
      "isExpectedApprovalServerRequestResponse(",
      "APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ",
      "APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE",
      "findForbiddenAgentRuntimeCurrentMethodCommands(finalDiagnostics)",
    ],
    absentSnippets: [
      'command === "agent_runtime_get_thread_read"',
      'command === "agent_runtime_submit_turn"',
      'command === "agent_runtime_respond_action"',
      '"agentSession/action/respond"',
      'command === "agent_runtime_create_session"',
      'command === "agent_runtime_update_session"',
      'command === "agent_runtime_list_file_checkpoints"',
      'command === "agent_runtime_diff_file_checkpoint"',
      'command === "agent_runtime_restore_file_checkpoint"',
      '"agent_runtime_get_thread_read")',
      '"agent_runtime_submit_turn")',
      '"agent_runtime_respond_action")',
    ],
  },
  {
    name: "Code runtime fixture smoke uses App Server JSON-RPC current runtime path",
    file: "scripts/code-runtime-fixture-smoke.mjs",
    snippets: [
      'const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines"',
      'const APP_SERVER_METHOD_THREAD_START = "thread/start"',
      'const APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE = "thread/settings/update"',
      'const APP_SERVER_METHOD_TURN_START = "turn/start"',
      'const APP_SERVER_METHOD_THREAD_READ = "thread/read"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST =",
      '"agentSession/fileCheckpoint/list"',
      "const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF =",
      '"agentSession/fileCheckpoint/diff"',
      "async function invokeAppServer(",
      "APP_SERVER_METHOD_THREAD_START",
      "APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE",
      "APP_SERVER_METHOD_TURN_START",
      "APP_SERVER_METHOD_THREAD_READ",
      "APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST",
      "APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF",
      "function buildRuntimeRequest({",
      "runtimeRequest,",
      "providerConfig: fixture.provider.providerConfig",
      "usesAppServerJsonRpcSubmitTurn",
      "usesAppServerSessionRead",
      "usesAppServerFileCheckpointCurrent",
    ],
    absentSnippets: [
      '"agent_runtime_create_session"',
      '"agent_runtime_update_session"',
      '"agent_runtime_submit_turn"',
      '"agent_runtime_get_session"',
      '"agent_runtime_get_thread_read"',
      '"agent_runtime_export_evidence_pack"',
      '"agent_runtime_list_file_checkpoints"',
      '"agent_runtime_diff_file_checkpoint"',
      "usesCurrentRuntimeSubmitTurn",
      "usesFileCheckpointCompat",
      "turnConfig",
      "turn_config",
      "agentChatRequest",
      "agent_chat_request",
      "hostOptions",
    ],
  },
  {
    name: "Renderer Agent Runtime event source accepts App Server local fanout",
    file: "src/lib/api/agentRuntimeEvents.ts",
    snippets: [
      "export function publishAgentRuntimeEvent",
      "localRuntimeEventListeners",
      "listenLocalAgentRuntimeEvent(",
      "projectAgentRuntimeSequenceGatePayloads(",
      "handler({ payload: projectedPayload })",
      "const unlistenLocal = listenLocalAgentRuntimeEvent(",
      "const bridgeHandler: AgentRuntimeEventHandler<TPayload> = (event) => {",
      "const unlistenBridge = await listen(",
      "unlistenLocal();",
      "unlistenBridge();",
    ],
  },
  {
    name: "Renderer Agent Runtime event tests lock App Server local fanout",
    file: "src/lib/api/agentRuntimeEvents.test.ts",
    snippets: [
      "应把本地发布的 App Server runtime 事件投递给现有 listener",
      'publishAgentRuntimeEvent("agent_stream_message-1"',
      'type: "text_delta"',
      "App Server delta",
      "应在 Lime runtime event 网关阻断未配对的 App Server tool.result",
      "tool-orphan",
      "unlisten();",
    ],
  },
  {
    name: "Renderer Agent chat sync refreshes session detail from current App Server turn events",
    file: "src/components/agent/chat/hooks/useAgentRuntimeSyncEffects.ts",
    snippets: [
      "currentTurnEventName?: string | null",
      "const normalizedCurrentTurnEventName = currentTurnEventName?.trim() || null",
      "function resolveRefreshRequestForTurnEventPayload(",
      "function preferRuntimeRefreshRequest(",
      "deferredRuntimeRefreshRequestRef",
      'case "action_required":',
      'case "action_resolved":',
      'case "artifact_snapshot":',
      'case "runtime_status":',
      'case "turn_completed":',
      'case "turn_canceled":',
      'case "turn_failed":',
      "runtime.listenToTurnEvents(",
      "normalizedCurrentTurnEventName",
      "!refreshRequest",
      "refreshSessionDetail(targetSessionId, request)",
      "RUNTIME_SYNC_REFRESH_REQUESTS.sendSettled",
      "RUNTIME_SYNC_REFRESH_REQUESTS.event",
      "RUNTIME_SYNC_REFRESH_REQUESTS.terminalEvent",
    ],
  },
  {
    name: "Renderer Agent chat sync centralizes App Server bridge availability",
    file: "src/components/agent/chat/hooks/useAgentRuntimeSyncEffects.ts",
    snippets: [
      "isAppServerBridgeAvailable",
      "const APP_SERVER_BRIDGE_RUNTIME_POLL_MS = 1000",
      "const shouldUseAppServerBridgeRuntimePolling",
      "isAppServerBridgeAvailable()",
      "!hasRuntimeEventListenerCapability",
    ],
    absentSnippets: [
      "isDevBridgeAvailable(",
      "app_server_handle_json_lines",
      "DEV_BRIDGE_RUNTIME_POLL_MS",
    ],
  },
  {
    name: "Renderer Agent chat sync tests lock App Server event session-detail refresh boundary",
    file: "src/components/agent/chat/hooks/useAgentRuntimeSyncEffects.test.tsx",
    snippets: [
      "收到当前 turn 的 App Server runtime event 后应刷新会话详情",
      "当前 turn 的 text_delta 不应触发完整 read model 刷新",
      "App Server turn notification 应通过当前 stream event 触发会话详情刷新",
      "createThreadClient",
      "listenAgentRuntimeEvent(name, handler)",
      'method: "turn/completed"',
      'currentTurnEventName: "agent_stream_assistant-1"',
      'type: "runtime_status"',
      'type: "text_delta"',
      'type: "turn.completed"',
      'type: "turn.failed"',
      'type: "turn.canceled"',
      'terminalRefreshRequest("runtimeSync.event")',
      'runtimeSyncRefreshRequest("runtimeSync.sendSettled")',
      "expect(refreshSessionDetail).not.toHaveBeenCalled()",
    ],
  },
  {
    name: "Renderer Agent task status keeps turn.completed non-terminal while assistant awaits final response",
    file: "src/components/agent/chat/utils/agentTaskRuntime.ts",
    snippets: [
      "export function isAssistantAwaitingFinalResponse(",
      'latestAssistant.runtimeStatus?.phase !== "failed"',
      'latestAssistant.runtimeStatus?.phase !== "cancelled"',
      'case "completed":\n      if (isAssistantAwaitingFinalResponse(latestAssistant)) {\n        return "running";',
    ],
  },
  {
    name: "Renderer inputbar status reuses assistant final-response guard",
    file: "src/components/agent/chat/utils/inputbarRuntimeStatusLine.ts",
    snippets: [
      "isAssistantAwaitingFinalResponse",
      "function resolveVisibleCompletedAt(",
      'status === "completed" || status === "failed" || status === "aborted"',
      'case "completed":\n      if (isAssistantAwaitingFinalResponse(latestAssistant)) {\n        return "running";',
    ],
  },
  {
    name: "Renderer Agent task runtime tests lock turn.completed first-token waiting state",
    file: "src/components/agent/chat/utils/agentTaskRuntime.test.ts",
    snippets: [
      "turn_completed 没有可见输出时不应把等待首个输出的助手草稿标记为已完成",
      'phase: "routing"',
      'status: "completed" as const',
      'expect(taskModel?.status).toBe("running")',
      'expect(inputbarModel?.status).toBe("running")',
      "expect(inputbarModel?.completedAt).toBeNull()",
    ],
  },
  {
    name: "Renderer Agent stream handler projects canonical Item lifecycle through ConversationProjection",
    files: [
      "src/components/agent/chat/hooks/agentStreamRuntimeHandler.ts",
      "src/components/agent/chat/hooks/agentStreamRuntimeHandlerTypes.ts",
      "src/components/agent/chat/hooks/agentStreamRuntimeLifecycleEvents.ts",
      "src/components/agent/chat/hooks/agentStreamConversationProjection.ts",
    ],
    snippets: [
      "applyAgentStreamConversationProjection",
      "conversationProjectionOwner",
      "reconcileAgentStreamProjectionItems",
      "getThreadItems?: () => readonly AgentThreadItem[]",
      'case "item_completed":',
      "projectedItems:",
    ],
    absentSnippets: [
      "syncMessageToolCallFromThreadItem",
      "agentStreamToolItemMessageSync",
      "toolCallStateFromThreadItem",
    ],
  },
  {
    name: "Renderer MessageList gates legacy message.toolCalls behind missing process timeline",
    files: [
      "src/components/agent/chat/components/messageListItemProjection.ts",
      "src/components/agent/chat/components/messageListItemProjection.unit.test.ts",
      "src/components/agent/chat/components/messageListItemProjection.legacyTools.unit.test.ts",
    ],
    snippets: [
      "const hasProcessTimelineItems = hasTimelineProcessItems(rawTimelineItems)",
      "const shouldAllowLegacyToolCallsProcess =",
      "message.toolCalls",
      "timeline 已有工具 item 时不应再把 legacy message.toolCalls 作为第二套过程源",
      "无 timeline 时应继续允许 legacy message.toolCalls 作为兼容过程源",
      "turn_summary",
      "timeline 过程项未生成 tool_use part 时仍应禁用 legacy message.toolCalls",
    ],
    normalizedSnippets: [
      'constshouldAllowLegacyToolCallsProcess=message.role==="assistant"&&includeInlineProcessFlow&&!hasProcessTimelineItems;',
      "constconversationToolCalls=shouldAllowLegacyToolCallsProcess?message.toolCalls:undefined;",
      "preserveToolUseParts:!hasProcessTimelineItems,",
    ],
  },
  {
    name: "Renderer Agent stream handler tests canonical Item terminal ownership",
    file: "src/components/agent/chat/hooks/agentStreamRuntimeHandler.unit.test.ts",
    snippets: [
      "item_completed 只更新 canonical ThreadItem，不再改写 legacy Message 工具卡",
      'type: "item_completed"',
      'type: "tool_call"',
      'source: "item_lifecycle"',
      "getThreadItems: () => threadItems",
      'status: "completed"',
      'output: "权威评测摘要"',
    ],
  },
  {
    name: "Renderer Agent chat passes active stream event into runtime sync",
    file: "src/components/agent/chat/hooks/useAgentChat.ts",
    snippets: [
      "const currentStreamingEventNameRef = useRef<string | null>(null)",
      "currentTurnEventName: stream.activeStreamEventName",
      "refreshSessionDetail: session.refreshSessionDetail",
      "getThreadItems: () => session.threadItems",
    ],
  },
  {
    name: "Renderer-safe App Server helper aliases artifact/read protocol types",
    file: "src/lib/api/appServer.ts",
    snippets: [
      "METHOD_ARTIFACT_READ,",
      "type ArtifactReadParams,",
      "type ArtifactSummary,",
      "type ArtifactReadResponse,",
      "export const APP_SERVER_METHOD_ARTIFACT_READ = METHOD_ARTIFACT_READ;",
      "export type AppServerArtifactReadParams = ArtifactReadParams;",
      "export type AppServerArtifactSummary = ArtifactSummary;",
      "export type AppServerArtifactReadResponse = ArtifactReadResponse;",
      "async readArtifacts(",
      "APP_SERVER_METHOD_ARTIFACT_READ",
    ],
  },
  {
    name: "Renderer Agent Runtime artifact client reads timeline content through App Server artifact/read",
    file: "src/lib/api/agentRuntime/appServerArtifactClient.ts",
    snippets: [
      'Pick<AppServerClient, "readArtifacts">',
      "appServerArtifactReadParamsFromTimelineItem",
      "appServerArtifactReadParamsFromArtifactPreview",
      "hasAgentRuntimeArtifactPreviewScope",
      "readAgentRuntimeTimelineArtifactContent",
      "readAgentRuntimeArtifactPreviewContent",
      "appServerClient.readArtifacts(params)",
      "includeContent: true",
      "limit: 1",
      "projectTimelineArtifactContentFromAppServerSummaries",
      "projectArtifactPreviewContentFromAppServerSummaries",
      'selected.contentStatus !== "available"',
      "return null",
    ],
    absentSnippets: [
      "AGENT_RUNTIME_COMMANDS",
      "invokeAgentRuntimeCommand",
      "agent_runtime_",
      "invokeMockOnly",
    ],
  },
  {
    name: "Renderer Agent Runtime artifact client tests lock artifact/read fail-closed behavior",
    file: "src/lib/api/agentRuntime/appServerArtifactClient.test.ts",
    snippets: [
      "应从 timeline metadata 构造 artifact/read includeContent 请求",
      "缺少 sessionId 时应 fail closed，不生成 App Server 请求",
      "应从 Workbench artifact meta 构造 artifact/read includeContent 请求",
      "Workbench artifact 缺少 sessionId 时应 fail closed，不进入 App Server",
      "应通过 App Server artifact/read 读取 timeline artifact 正文",
      "应通过 App Server artifact/read 读取 Workbench artifact preview 正文",
      "App Server 未返回可用 content 时不伪造 artifact 正文",
      "Workbench artifact preview 的 contentStatus 不可用时不伪造正文",
      "hasAgentRuntimeArtifactPreviewScope",
      "projectArtifactPreviewContentFromAppServerSummaries",
      "readArtifacts: vi.fn().mockResolvedValue",
      'artifactRef: "artifact-report"',
      "includeContent: true",
      'contentStatus: "available"',
      'contentStatus: "unavailable"',
    ],
    absentSnippets: ["agent_runtime_"],
  },
  {
    name: "Renderer Artifact timeline card hydrates omitted content through App Server artifact client",
    file: "src/components/agent/chat/components/AgentThreadTimelineArtifactCard.tsx",
    snippets: [
      "readAgentRuntimeTimelineArtifactContent",
      "readTimelineArtifactContent = readAgentRuntimeTimelineArtifactContent",
      "const resolveOpenTarget = async",
      "const artifactContent = await readTimelineArtifactContent(item)",
      "artifactContent?.content.trim()",
      "void openTimelineTarget(",
      "void readTimelineArtifactContent(item)",
    ],
    absentSnippets: [
      "AGENT_RUNTIME_COMMANDS",
      "invokeAgentRuntimeCommand",
      "agent_runtime_",
      "invokeMockOnly",
    ],
  },
  {
    name: "Renderer Artifact timeline card tests lock App Server hydration on omitted content",
    file: "src/components/agent/chat/components/AgentThreadTimelineArtifactCard.test.tsx",
    snippets: [
      "首屏省略 artifact 正文时点击打开应通过 App Server artifact/read 补齐内容",
      "readTimelineArtifactContent",
      'session_id: "session-1"',
      'artifact_ref: "artifact-document:demo"',
      "这里是 App Server 返回的完整正文",
      "expect(readTimelineArtifactContent).toHaveBeenCalledWith",
      "expect(onOpenArtifactFromTimeline).toHaveBeenCalledWith",
    ],
    absentSnippets: ["agent_runtime_"],
  },
  {
    name: "Renderer-safe App Server excludes Lime-only evidence/export",
    files: rendererAppServerSplitSourceFiles,
    snippets: ["APP_SERVER_METHOD_ARTIFACT_READ"],
    absentSnippets: [
      "METHOD_EVIDENCE_EXPORT",
      "APP_SERVER_METHOD_EVIDENCE_EXPORT",
      "EvidenceExportParams",
      "EvidenceExportResponse",
      "EvidencePackSummary",
      "EvidencePackArtifact",
      "exportEvidence(",
    ],
  },
  {
    name: "Renderer-safe App Server tests lock capability/list session scope",
    file: "src/lib/api/appServer.test.ts",
    snippets: [
      "APP_SERVER_METHOD_CAPABILITY_LIST",
      "listCapabilities 应透传 sessionId scope",
      'sessionId: "session-1"',
      "method: APP_SERVER_METHOD_CAPABILITY_LIST",
    ],
  },
  {
    name: "Renderer-safe App Server tests lock action respond JSON-RPC path",
    file: "src/lib/api/appServer.test.ts",
    snippets: [
      "APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND",
      "respondAction 应通过 App Server JSON-RPC 响应 action.required",
      "method: APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND",
      'actionType: "tool_confirmation"',
      'eventName: "agentSession/event/session-1"',
    ],
  },
  {
    name: "Renderer-safe App Server tests lock artifact/read JSON-RPC path",
    file: "src/lib/api/appServer.test.ts",
    snippets: [
      "APP_SERVER_METHOD_ARTIFACT_READ",
      "readArtifacts 应通过 App Server JSON-RPC 读取 artifact summary/content",
      "method: APP_SERVER_METHOD_ARTIFACT_READ",
      'artifactRef: "artifact-report"',
      "includeContent: true",
      'expect(result.result.artifacts[0].content).toBe("# Report")',
      'expect(result.result.artifacts[0].contentStatus).toBe("available")',
    ],
  },
  {
    name: "Renderer workspace artifact preview reads scoped content through App Server artifact/read",
    file: "src/components/agent/chat/workspace/useWorkspaceArtifactPreviewActions.ts",
    snippets: [
      "hasAgentRuntimeArtifactPreviewScope",
      "readAgentRuntimeArtifactPreviewContent",
      "artifact?: Artifact",
      "hasAgentRuntimeArtifactPreviewScope(artifact, normalizedPath)",
      "const appServerContent = await readAgentRuntimeArtifactPreviewContent(",
      "App Server artifact 内容不可用",
      "handleHarnessLoadFilePreview(\n          artifactPath,\n          artifact,",
    ],
    absentSnippets: [
      "AppServerClient",
      "readArtifacts(",
      "new AppServerClient()",
      "mockReadArtifacts",
      "invokeMockOnly",
      "invokeAgentRuntimeCommand",
      "AGENT_RUNTIME_COMMANDS",
    ],
  },
  {
    name: "Renderer workspace artifact preview tests lock App Server current path and fail-closed behavior",
    file: "src/components/agent/chat/workspace/useWorkspaceArtifactPreviewActions.test.tsx",
    snippets: [
      "mockHasArtifactPreviewScope",
      "mockReadArtifactPreviewContent",
      "读取带 App Server scope 的 artifact 时应走 artifact/read current 主链",
      "带 App Server scope 的 artifact 内容不可用时不应回退旧文件预览",
      'sessionId: "session-1"',
      'turnId: "turn-1"',
      'artifactRef: "artifact-report"',
      "expect(mockReadArtifactPreviewContent).toHaveBeenCalledWith",
      'error: "App Server artifact 内容不可用"',
      "expect(readFilePreviewSpy).not.toHaveBeenCalled()",
    ],
  },
  {
    name: "Renderer artifact snapshot metadata carries App Server artifact/read scope",
    file: "src/components/agent/chat/hooks/agentStreamEventProcessorAuxiliary.ts",
    snippets: [
      "activeSessionId,",
      "sessionId: activeSessionId",
      "artifactId: data.artifact.artifactId",
      "artifactRef: data.artifact.artifactId || artifactPath",
      'source: "artifact_snapshot"',
    ],
  },
  {
    name: "TypeScript client exposes release manifest file helpers",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "export async function readReleaseManifest(",
      "path: string,",
      "Promise<AppServerReleaseManifest>",
      "export async function resolveSidecarFromReleaseManifestFile(",
      "await readReleaseManifest(manifestPath)",
      "resolveSidecarFromReleaseManifest(",
    ],
  },
  {
    name: "TypeScript manifest file tests lock independent app sidecar resolution",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "readReleaseManifest",
      "resolveSidecarFromReleaseManifestFile",
      "resolves sidecar config from release manifest file",
      "app-server.release.json",
      'resourcesPath: "/app/resources"',
      'binaryPathSource, "resources"',
    ],
  },
  {
    name: "TypeScript client consumes Rust protocol schema manifest",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "DEFAULT_PROTOCOL_SCHEMA_MANIFEST_NAME",
      "type AppServerProtocolSchemaManifest,",
      "type AppServerMethodSpec,",
      "type ProtocolSchemaGroup,",
      'export * from "./protocol.js"',
      'from "./protocol.js"',
      "export async function readProtocolSchemaManifest(",
      "export function assertCompatibleProtocolSchemaManifest(",
      "manifest.protocolVersion !== expectedProtocolVersion",
      "manifest.jsonRpc.version !== JSONRPC_VERSION",
      "normalizeMethodSpecs(manifest.methods)",
      "normalizeMethodSpecs(expectedMethods)",
      "app-server schema method catalog mismatch",
      "export function protocolSchemaFilePath(",
      "export function listProtocolSchemaFiles(",
      '(["jsonrpc", "v0", "v2"] as const).flatMap',
    ],
  },
  {
    name: "TypeScript client tests lock checked-in Rust schema consumption",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "DEFAULT_PROTOCOL_SCHEMA_MANIFEST_NAME",
      "readProtocolSchemaManifest",
      "assertCompatibleProtocolSchemaManifest",
      "defaultProtocolSchemaManifestPath",
      "protocolSchemaFilePath",
      "listProtocolSchemaFiles",
      "reads and validates protocol schema manifest metadata",
      "consumes checked-in Rust protocol schema manifest",
      "AgentSessionTurnStartParams",
      "ThreadStartParams",
      "TurnStartParams",
      "JsonRpcRequest",
      "schema method catalog mismatch",
    ],
    absentSnippets: ["EvidenceExportResponse", "evidence/export"],
  },
  {
    name: "Rust daemon probes stdio sidecar readiness before handing it to hosts",
    file: "lime-rs/crates/app-server-daemon/src/lib.rs",
    snippets: [
      "pub fn probe_readiness(",
      "pub struct SidecarReadinessReport",
      "pub enum SidecarReadinessError",
      "pub fn probe_sidecar_readiness(",
      "initialize_probe_request(client_version)",
      "initialized_notification()",
      "probe_info_from_initialize_response(&message)",
      "write_jsonrpc_line(stdin, &initialize)",
      "write_jsonrpc_line(stdin, &initialized)",
      "drop(child.stdin.take())",
      "wait_for_sidecar_exit(child, Duration::from_secs(2))",
      "drain_child_stderr(child, &stderr_log_file)",
      "if result.is_err()",
      "readiness_probe_drains_stderr_when_sidecar_exits_before_initialize_response",
      "cleanup_sidecar_child(&mut child)",
      "readiness_probe_rejects_sha256_mismatch_before_process_start",
      "readiness_probe_smokes_real_app_server_when_env_is_set",
    ],
  },
  {
    name: "Rust daemon sidecar launch config mirrors standalone backend options",
    files: [
      "lime-rs/crates/app-server-daemon/src/lib.rs",
      "lime-rs/crates/app-server-daemon/src/backend.rs",
    ],
    snippets: [
      "pub enum SidecarBackendMode",
      "External",
      "Mock",
      "Unavailable",
      "pub backend_mode: SidecarBackendMode",
      "pub backend_command: Option<String>",
      "pub backend_args: Vec<String>",
      "pub backend_timeout_ms: Option<u64>",
      "pub app_policy_path: Option<PathBuf>",
      'args.push("--backend".to_string())',
      "args.push(self.backend_mode.as_str().to_string())",
      'args.push("--backend-command".to_string())',
      'args.push("--backend-arg".to_string())',
      'args.push("--backend-timeout-ms".to_string())',
      'args.push("--app-policy".to_string())',
      "release_manifest_resolution_preserves_backend_launch_options",
      "sidecar_args_follow_standalone_backend_cli",
    ],
  },
  {
    name: "Rust daemon settings persist external backend launch options",
    file: "lime-rs/crates/app-server-daemon/src/settings.rs",
    snippets: [
      "pub backend_mode: Option<String>",
      "pub backend_command: Option<String>",
      "pub backend_args: Vec<String>",
      "pub backend_timeout_ms: Option<u64>",
      "pub app_policy_path: Option<PathBuf>",
      'value["backendCommand"]',
      'value["backendArgs"][0]',
      'value["backendTimeoutMs"]',
      "daemon_settings_round_trips_external_backend_launch_options",
      "resources/app-server/backend/content-backend.mjs",
    ],
  },
  {
    name: "Rust daemon applies settings to sidecar launch resolution",
    file: "lime-rs/crates/app-server-daemon/src/lib.rs",
    snippets: [
      "pub fn apply_daemon_settings(&mut self, settings: &DaemonSettings) -> Result<(), String>",
      "pub fn with_daemon_settings(mut self, settings: &DaemonSettings) -> Result<Self, String>",
      "pub fn with_daemon_state_paths(",
      "let settings = DaemonSettings::load(&state_paths.settings_file)?",
      "self.allow_env_override = settings.allow_env_override",
      "self.resource_relative_path = settings.resource_relative_path.clone()",
      "self.backend_mode = SidecarBackendMode::parse(backend_mode.trim())?",
      "self.backend_command = settings.backend_command.clone()",
      "self.backend_args = settings.backend_args.clone()",
      "self.backend_timeout_ms = settings.backend_timeout_ms",
      "self.app_policy_path = settings.app_policy_path.clone()",
      "pub fn resolve_sidecar_from_release_manifest_path_with_daemon_state(",
      "options.with_daemon_state_paths(state_paths)?",
      "daemon_settings_apply_to_sidecar_binary_options_and_manifest_resolution",
      "with_daemon_state_paths_loads_settings_file_into_manifest_resolution",
      "with_daemon_state_paths_missing_settings_uses_default_resolution",
      "daemon_settings_reject_unsupported_backend_mode_before_launch_resolution",
    ],
  },
  {
    name: "Rust daemon lifecycle keeps operation lock and pid backend explicit",
    files: [
      "lime-rs/crates/app-server-daemon/src/backend.rs",
      "lime-rs/crates/app-server-daemon/src/lifecycle.rs",
    ],
    snippets: [
      "pub fn is_supported(self) -> bool",
      "matches!(self, Self::Sidecar)",
      "pub fn unsupported_reason(self) -> Option<&'static str>",
      "pid backend is not supported until local socket lifecycle is enabled",
      "pub struct OperationLock",
      "File::options()",
      ".create_new(true)",
      "OperationLockError::AlreadyLocked",
      "impl Drop for OperationLock",
      "fs::remove_file(&self.path)",
      "pub fn acquire_operation_lock(",
      "pid_backend_is_explicitly_unsupported_until_local_socket_lifecycle_exists",
      "operation_lock_serializes_lifecycle_actions_and_releases_on_drop",
    ],
  },
  {
    name: "TypeScript client exposes sidecar lifecycle backoff",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "export type SidecarRestartPolicy = {",
      "export class AppServerSidecarLifecycle",
      "export function sidecarRestartDelayMs(",
      "export function shouldRestartSidecar(",
      "async #connectWithRetry(",
      "onRestartScheduled?: (event: SidecarRestartScheduledEvent) => void",
      "onRestarted?: (connected: ConnectedAppServerSidecar, attempt: number) => void",
    ],
  },
  {
    name: "TypeScript sidecar launch config supports standalone backend options",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "export const DEFAULT_STANDALONE_BACKEND_MODE",
      '= "unavailable"',
      'backendMode?: "external" | "runtime" | "mock" | "unavailable"',
      "backendCommand?: string",
      "backendArgs?: string[]",
      "backendTimeoutMs?: number",
      "appPolicyPath?: string",
      'backendMode?: SidecarLaunchConfig["backendMode"]',
      "backendCommand?: string",
      "backendArgs?: string[]",
      "backendTimeoutMs?: number",
      'backendMode: SidecarLaunchConfig["backendMode"] = DEFAULT_STANDALONE_BACKEND_MODE',
      "backendMode: DEFAULT_STANDALONE_BACKEND_MODE",
      "config.backendMode ?? DEFAULT_STANDALONE_BACKEND_MODE",
      'args.push("--backend-command", config.backendCommand)',
      'args.push("--backend-arg", backendArg)',
      'args.push("--backend-timeout-ms", String(config.backendTimeoutMs))',
      'args.push("--app-policy", config.appPolicyPath)',
      "backendMode: options.backendMode ?? DEFAULT_STANDALONE_BACKEND_MODE",
      "backendCommand: options.backendCommand",
      "backendArgs: options.backendArgs",
      "backendTimeoutMs: options.backendTimeoutMs",
      "appPolicyPath: options.appPolicyPath",
    ],
  },
  {
    name: "TypeScript sidecar launch tests lock unavailable mock and external args",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "DEFAULT_STANDALONE_BACKEND_MODE",
      "uses agent-style stdio sidecar launch args",
      "assert.equal(config.backendMode, DEFAULT_STANDALONE_BACKEND_MODE)",
      '"--stdio"',
      '"--backend"',
      '"unavailable"',
      'backendMode: "runtime"',
      '"runtime"',
      "policyConfig.appPolicyPath",
      '"--app-policy"',
      '"/tmp/content-studio.policy.json"',
      'backendMode: "mock"',
      '"mock"',
      'backendMode: "external"',
      'backendCommand: "/usr/local/bin/content-backend"',
      'backendArgs: ["--workspace", "/tmp/content-studio", "--json"]',
      "backendTimeoutMs: 30_000",
      '"--backend-command"',
      '"--backend-arg"',
      '"--backend-timeout-ms"',
      '"30000"',
      'appPolicyPath: "/app/content-studio.policy.json"',
      'backendArgs: ["--workspace", "/app/workspace"]',
      "backendTimeoutMs: 45_000",
    ],
  },
  {
    name: "Electron dev sidecars default to App Server runtime backend and only keep explicit external override",
    files: [
      "scripts/lib/electron-dev-sidecar.mjs",
      "scripts/lib/electron-dev-sidecar.test.mjs",
      "scripts/electron/run-dev.mjs",
    ],
    snippets: [
      "resolveDevAppServerBackendEnv",
      'defaultMode = "runtime"',
      "APP_SERVER_BACKEND_MODE: defaultMode",
      "APP_SERVER_BACKEND_MODE: requestedMode",
      'APP_SERVER_BACKEND_MODE: "external"',
      "APP_SERVER_BACKEND_COMMAND",
      "APP_SERVER_BACKEND_TIMEOUT_MS",
      "默认 dev App Server backend 使用 App Server 内部 runtime",
      "显式 unavailable 时 dev App Server backend 不接 external",
      "调用 cargo build 成组构建 app-server 与 code-mode host",
    ],
    absentSnippets: [
      "resolveDevAppServerAgentBackendBinary",
      "shouldUseDevAppServerExternalBackend",
      "appServerAgentBackendBinaryName",
      "localAppServerAgentBackendBinaryPath",
      "app-server-agent-backend",
      'APP_SERVER_BACKEND_MODE: "mock"',
      'backendMode: "mock"',
    ],
  },
  {
    name: "Windows test workflow builds both App Server sidecars with pinned Rust and shared linker inputs",
    file: ".github/workflows/build-windows-test.yml",
    snippets: [
      "dtolnay/rust-toolchain@e081816240890017053eacbb1bdf337761dc5582 # 1.95.0",
      "Configure Windows MSVC linker",
      "./scripts/lib/windows-msvc-linker.ps1 -Target x86_64-pc-windows-msvc",
      "Refresh sandboxed rusty_v8 artifacts",
      "node scripts/lib/rusty-v8-artifacts.mjs --github-env",
      "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
      "code-mode-host sidecars",
    ],
  },
  {
    name: "Windows quality links both App Server sidecar binaries with pinned Rust and shared inputs",
    file: ".github/workflows/quality.yml",
    snippets: [
      "dtolnay/rust-toolchain@e081816240890017053eacbb1bdf337761dc5582 # 1.95.0",
      "Configure Windows MSVC linker",
      "./scripts/lib/windows-msvc-linker.ps1 -Target x86_64-pc-windows-msvc",
      "Refresh sandboxed rusty_v8 artifacts",
      "node scripts/lib/rusty-v8-artifacts.mjs --github-env",
      "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
      "Build Windows app-server and code-mode-host sidecars",
      "cargo build --manifest-path lime-rs/Cargo.toml --target x86_64-pc-windows-msvc -p app-server --bin app-server -p tool-runtime --bin code-mode-host",
    ],
    absentSnippets: [
      "cargo check --manifest-path lime-rs/Cargo.toml -p app-server --bin app-server -p tool-runtime --bin code-mode-host",
    ],
  },
  {
    name: "Release workflow uses pinned Rust and shared Windows linker inputs",
    file: ".github/workflows/release.yml",
    snippets: [
      "dtolnay/rust-toolchain@e081816240890017053eacbb1bdf337761dc5582 # 1.95.0",
      "Configure Windows MSVC linker",
      "./scripts/lib/windows-msvc-linker.ps1 -Target x86_64-pc-windows-msvc",
      "Refresh sandboxed rusty_v8 artifacts",
      "node scripts/lib/rusty-v8-artifacts.mjs --github-env",
      "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
    ],
    absentSnippets: ["dtolnay/rust-toolchain@stable"],
  },
  {
    name: "Existing GitHub release is retargeted when a stable tag is rebuilt",
    file: ".github/workflows/release.yml",
    snippets: [
      'TARGET_REF="${{ github.event.inputs.source_ref || github.sha }}"',
      'gh release edit "$TAG"',
      '--target "$TARGET_REF"',
      '--notes-file "$NOTES_FILE"',
    ],
  },
  {
    name: "Rust Full configures sandboxed V8 artifacts before testing the workspace",
    file: ".github/workflows/quality.yml",
    snippets: [
      "quality-rust-full",
      "Refresh sandboxed rusty_v8 artifacts",
      "node scripts/lib/rusty-v8-artifacts.mjs --github-env",
      "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
      "Build standalone Code Mode host",
      "cargo build --manifest-path lime-rs/Cargo.toml -p tool-runtime --bin code-mode-host",
      "npm run test:rust",
    ],
  },
  {
    name: "GUI smoke refreshes sandboxed V8 artifacts after restoring the Rust cache",
    file: ".github/workflows/quality.yml",
    snippets: [
      "quality-gui-smoke",
      "Refresh sandboxed rusty_v8 artifacts",
      "node scripts/lib/rusty-v8-artifacts.mjs --github-env",
      "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
      "npm run verify:gui-smoke -- --timeout-ms 1800000",
    ],
  },
  {
    name: "Repository Rust toolchain stays pinned to the Windows V8 release toolchain",
    file: "rust-toolchain.toml",
    snippets: [
      'channel = "1.95.0"',
      'components = ["clippy", "rustfmt", "rust-src"]',
    ],
  },
  {
    name: "Windows MSVC builds use the static CRT required by the sandbox V8 artifact",
    file: "scripts/lib/rusty-v8-artifacts.mjs",
    snippets: [
      "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS",
      '"-C target-feature=+crt-static"',
    ],
    absentSnippets: ["/FORCE:MULTIPLE", "/NODEFAULTLIB"],
  },
  {
    name: "Windows MSVC linker setup exports the complete SDK environment and native linker",
    file: "scripts/lib/windows-msvc-linker.ps1",
    snippets: [
      "VsDevCmd.bat",
      "-arch=x64 -host_arch=x64",
      '"INCLUDE"',
      '"LIB"',
      '"LIBPATH"',
      '"UCRTVersion"',
      '"WindowsSdkDir"',
      'throw "VsDevCmd.bat did not export $RequiredVariable"',
      '"bin\\Hostx64\\x64\\link.exe"',
      '"CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=$Linker"',
    ],
    absentSnippets: ["/FORCE:MULTIPLE", "/NODEFAULTLIB"],
  },
  {
    name: "Rust changed-scope selection treats the root toolchain as workspace-wide",
    file: "scripts/lib/rust-test-scope-core.mjs",
    snippets: [
      'relPath === "rust-toolchain.toml"',
      'relPath === "rust-toolchain"',
      'relPath === "lime-rs/rust-toolchain.toml"',
    ],
  },
  {
    name: "Retired App Server agent backend crate stays deleted",
    file: "lime-rs/Cargo.lock",
    snippets: ['name = "app-server"'],
    absentSnippets: ['name = "app-server-agent-backend"'],
  },
  {
    name: "Electron App Server host preserves external backend env for resources manifest",
    file: "electron/appServerHost.ts",
    snippets: [
      'resolveRuntimeBackendLaunchOptions("runtime")',
      "...resolveRuntimeBackendLaunchOptions(defaultBackendMode)",
      'const APP_SERVER_TURN_START_METHOD = "turn/start"',
      "APP_SERVER_BACKEND_TIMEOUT_GRACE_MS",
      "process.env.APP_SERVER_BACKEND_TIMEOUT_MS",
      "process.env.APP_SERVER_BACKEND_COMMAND?.trim()",
      "parseBackendArgs(process.env.APP_SERVER_BACKEND_ARGS)",
      "parsePositiveInteger(",
      'normalized === "mock"',
      "throw new Error(",
    ],
    normalizedSnippets: [
      "resolveAppServerRequestTimeoutMs(proxiedMessage.message.method,request.timeoutMs,)",
    ],
    absentSnippets: [
      'backendMode: "mock"',
      'backendMode: "unavailable",',
      "#requestStreamingTurnStart",
      "#readCanonicalTurnIdentity",
      "#recentTurnStartedIdentity",
      "APP_SERVER_STREAMING_TURN_ACK_GRACE_MS",
      "APP_SERVER_STREAMING_TURN_IDENTITY_READ_RETRY_MS",
      "streamingTurnStartAcceptedResponse",
      "turnIdentityFromThreadRead",
      "requestUntilFirstNotificationOrResponse",
    ],
  },
  {
    name: "TypeScript client exposes one-step packaged sidecar lifecycle",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "DEFAULT_RELEASE_MANIFEST_NAME",
      'export * from "./protocol.js"',
      'from "./protocol.js"',
      "export type PackagedSidecarLifecycleOptions",
      "export type StartedPackagedAppServerSidecar",
      "export function defaultReleaseManifestPath(",
      "export async function startPackagedAppServerSidecar(",
      "allowEnvOverride: options.allowEnvOverride ?? false",
      "const lifecycle = new AppServerSidecarLifecycle(",
    ],
  },
  {
    name: "TypeScript sidecar lifecycle tests lock crash restart behavior",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "calculates sidecar restart backoff policy",
      "sidecar lifecycle restarts once after crash",
      "sidecar lifecycle retries initial handshake failure",
      "new AppServerSidecarLifecycle(",
      "restartPolicy: {",
      "onRestartScheduled(event)",
      "onRestarted(connected, attempt)",
      "onRestartFailed(event)",
    ],
  },
  {
    name: "TypeScript packaged lifecycle tests lock resources manifest entrypoint",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "starts packaged sidecar lifecycle from resources manifest",
      "DEFAULT_RELEASE_MANIFEST_NAME",
      "defaultReleaseManifestPath(resourcesPath)",
      "startPackagedAppServerSidecar(",
      'binaryPathSource, "resources"',
      "initialized-packaged",
    ],
  },
  {
    name: "Sidecar lifecycle smoke verifies packaged manifest flow and unavailable backend fail-closed behavior",
    file: "scripts/app-server/sidecar-lifecycle-smoke.mjs",
    snippets: [
      "startPackagedAppServerSidecar",
      "defaultReleaseManifestPath",
      "resourcesPath",
      'dataDir: path.join(tempDir, "data")',
      'backendMode: "unavailable"',
      "connection.listCapabilities",
      "connection.startSession",
      "connection.startTurn",
      "expectStartTurnFailClosed",
      "standalone app-server backend is not configured",
      "backend=unavailable",
      "turn=fail-closed",
      "[smoke:app-server-sidecar-lifecycle] ok",
    ],
    absentSnippets: [
      "AppServerAgentEventRouter",
      "const idleNotificationPromise = connection.nextNotification(5_000)",
      "projectedEvents.length",
      "turn.accepted",
      'backendMode: "mock"',
    ],
  },
  {
    name: "Root scripts expose sidecar lifecycle smoke",
    file: "package.json",
    snippets: [
      '"smoke:app-server-sidecar-lifecycle"',
      "node scripts/app-server/sidecar-lifecycle-smoke.mjs",
    ],
  },
  {
    name: "Packaged sidecar failure smoke drains streamed failure through read model",
    file: "scripts/app-server/packaged-external-backend-failure-smoke.mjs",
    snippets: [
      "[smoke:app-server-packaged-external-backend-failure] ok",
      "startPackagedAppServerSidecar",
      'backendMode: "external"',
      "backendCommand: process.execPath",
      "backendArgs: [backendPath]",
      'dataDir: path.join(tempDir, "data")',
      "copyElectronAppServerRuntimeLibraries",
      "writeFailingExternalBackend(backendPath)",
      "packaged external backend crashed after partial output",
      'historyMode: "paginated"',
      'model: "fixture-model"',
      'modelProvider: "fixture-provider"',
      "collectRuntimeNotificationsUntilFailure(",
      "connection.nextNotification(remainingMs)",
      'notification.method === "item/agentMessage/delta"',
      'notification.method === "turn/completed"',
      "const clientFailure = assertDirectFailureNotifications(",
      "connection.readThread",
      "{ threadId, includeTurns: true }",
      "readResult.result.thread.turns",
      'assertEqual(readTurns.length, 1, "read failed turn count")',
      'assertEqual(readTurn.status, "failed", "read failed turn status")',
      "read failed turn ${turnId} is missing completedAt",
      'clientNotifications=${clientNotifications.map((notification) => notification.method).join(",")}',
      "readTurnStatus=${readTurn.status}",
    ],
    absentSnippets: [
      'backendMode: "mock"',
      "AppServerRequestError",
      "METHOD_AGENT_SESSION_EVENT",
      "connection.readSession",
    ],
  },
  {
    name: "Root package gate runs packaged external backend failure smoke",
    file: "package.json",
    snippets: [
      '"smoke:app-server-packaged-external-backend-failure"',
      "scripts/app-server/packaged-external-backend-failure-smoke.mjs",
      'npm --prefix \\"packages/app-server-client\\" run build',
      '"electron:package:dir"',
      "node scripts/electron/run-package-dir.mjs",
      '"electron:verify:package"',
      "scripts/electron/verify-package-resources.mjs && npm run smoke:app-server-packaged-external-backend-failure",
    ],
  },
  {
    name: "Electron directory package uses Forge package",
    file: "scripts/electron/run-package-dir.mjs",
    snippets: [
      '"electron-forge"',
      '"package"',
      '"--platform"',
      '"--arch"',
      'shell: process.platform === "win32"',
    ],
  },
  {
    name: "Rust turn start params expose caller supplied turn_id",
    files: rustProtocolFiles,
    snippets: [
      "pub struct AgentSessionTurnStartParams",
      "pub turn_id: Option<String>",
      "pub queue_if_busy: bool",
      "pub skip_pre_submit_resume: bool",
    ],
  },
  {
    name: "RuntimeCore owns single active turn instead of relying on renderer queue flag",
    files: [
      "lime-rs/crates/app-server-protocol/src/jsonrpc_lite.rs",
      "lime-rs/crates/app-server/src/runtime.rs",
      "lime-rs/crates/app-server/src/runtime/tests/queue.rs",
      "lime-rs/crates/app-server/src/runtime/tests/sessions.rs",
      "lime-rs/crates/app-server/src/lib.rs",
    ],
    snippets: [
      "pub const TURN_ALREADY_ACTIVE",
      "TurnAlreadyActive",
      "turn already active",
      "second_active_turn_without_queue_fails_closed",
      "turn_start_rejects_parallel_active_turn_without_queue_flag",
    ],
  },
  {
    name: "Rust runtime options expose Turn envelope and typed RuntimeRequest",
    files: rustProtocolFiles,
    snippets: [
      "pub struct RuntimeOptions",
      "pub event_name: Option<String>",
      "pub queued_turn_id: Option<String>",
      "pub runtime_request: Option<RuntimeRequest>",
      "pub expected_output: Option<serde_json::Value>",
      "pub structured_output: Option<StructuredOutputContract>",
      "pub output_schema: Option<serde_json::Value>",
      "pub struct RuntimeRequest",
      "pub provider_preference: Option<String>",
      "pub model_preference: Option<String>",
      "pub metadata: Option<serde_json::Value>",
    ],
    absentSnippets: ["pub host_options: Option<serde_json::Value>"],
  },
  {
    name: "TypeScript turn start params mirror Rust wire fields",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      "export interface TurnStartParams {",
      "threadId: string",
      "input: UserInput[]",
      "clientUserMessageId?: null | string",
    ],
  },
  {
    name: "TypeScript client exposes canonical v2 thread resume only",
    files: [
      "packages/app-server-client/src/index.ts",
      "packages/app-server-client/src/protocol.ts",
    ],
    snippets: [
      "resumeThread(params: protocol.ThreadResumeParams)",
      'name: "resumeThread"',
      "method: protocol.METHOD_THREAD_RESUME",
      "params: protocol.ThreadResumeParams",
      "AppServerRequestResult<protocol.ThreadResumeResponse>",
      'export const METHOD_THREAD_RESUME = "thread/resume"',
    ],
    absentSnippets: [
      "resumeAgentSessionThread",
      "resumeAgentRuntimeThread",
      "AgentSessionThreadResumeParams",
      "AgentSessionThreadResumeResponse",
      "RuntimeResumeContract",
      "RuntimeResumeActionDecision",
    ],
  },
  {
    name: "TypeScript request builder test locks canonical thread resume shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "const resume = client.resumeThread({",
      'threadId: "thread_1"',
      "excludeTurns: true",
      "initialTurnsPage: {",
      "assert.equal(resume.method, METHOD_THREAD_RESUME)",
      "assert.deepEqual(resume.params, {",
    ],
    absentSnippets: [
      "resumeAgentSessionThread",
      "RuntimeResumeContract",
      'sessionId: "sess_1",\n    resumeContract:',
    ],
  },
  {
    name: "TypeScript runtime options mirror Rust wire fields",
    file: "packages/app-server-client/src/protocol.ts",
    snippets: [
      "export type RuntimeOptions = {",
      "eventName?: string",
      "queuedTurnId?: string",
      "runtimeRequest?: RuntimeRequest",
      "expectedOutput?: unknown",
      "structuredOutput?: StructuredOutputContract",
      "outputSchema?: unknown",
    ],
    absentSnippets: ["hostOptions?: unknown"],
  },
  {
    name: "TypeScript request builder test locks caller supplied turnId",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      'threadId: "thread_external"',
      'assert.equal(turn.params.threadId, "thread_external")',
      'assert.deepEqual(turn.params.input, [{ type: "text", text: "draft" }])',
      'assert.equal(turn.params.model, "gpt-5-codex")',
    ],
  },
  {
    name: "TypeScript client exposes direct lifecycle projection router",
    file: "packages/app-server-client/src/index.ts",
    snippets: [
      "agentRuntimeLifecycleNotification",
      'export * from "./protocol.js"',
      'from "./protocol.js"',
      "export class AppServerAgentEventRouter",
      "async dispatch(message: JsonRpcMessage): Promise<boolean>",
    ],
    absentSnippets: ["AgentRuntimeEventListener", "subscribeEvents("],
  },
  {
    name: "TypeScript event router tests lock direct lifecycle projection shape",
    file: "packages/app-server-client/tests/client.test.mjs",
    snippets: [
      "AppServerAgentEventRouter",
      "agentRuntimeLifecycleNotification(notification)",
      "routes direct lifecycle notifications without wrapper projection",
      "connection buffers request responses read by idle notification loop",
      "assert.equal(await router.dispatch(notification), true)",
      "assert.deepEqual(lifecycleRouted, [",
    ],
  },
  {
    name: "TypeScript README documents independent app event projection",
    file: "packages/app-server-client/README.md",
    snippets: [
      "AppServerAgentEventRouter",
      'mainWindow.webContents.send("agent:lifecycle", notification)',
      "await eventRouter.dispatch(await connection.nextNotification())",
      "project events into their own renderer state",
    ],
  },
  {
    name: "Runtime client readThread stays on canonical thread/read",
    files: [
      "packages/app-server-client/src/agent-runtime.ts",
      "packages/agent-runtime-client/src/runtimeClient.ts",
      "packages/agent-runtime-client/src/sessionGateway.ts",
    ],
    snippets: ["connection.readThread", "gateway.readThread"],
    absentSnippets: ["readSession", '"thread/read"'],
  },
  {
    name: "Runtime client root exports canonical thread read types only",
    file: "packages/agent-runtime-client/src/index.ts",
    snippets: ["type ThreadReadParams", "type ThreadReadResponse"],
    absentSnippets: ["AgentSessionReadParams", "AgentSessionReadResponse"],
  },
  {
    name: "Standard runtime client excludes Lime-only evidence/export",
    files: [
      "packages/agent-runtime-client/src/index.ts",
      "packages/agent-runtime-client/src/runtimeClient.ts",
      "packages/agent-runtime-client/src/sessionGateway.ts",
    ],
    snippets: ["readThread"],
    absentSnippets: [
      "EvidenceExportParams",
      "EvidenceExportResponse",
      "exportEvidence",
      "evidence/export",
    ],
  },
];

const failures = [];

for (const check of checks) {
  const files = expandContractFiles(check.files ?? [check.file]);
  const location = files.join(", ");
  const existingFiles = files.filter((file) =>
    fs.existsSync(path.join(repoRoot, file)),
  );
  if (check.allowMissingFiles && existingFiles.length === 0) {
    continue;
  }
  const missingFiles = files.filter((file) => !existingFiles.includes(file));
  if (missingFiles.length > 0 && !check.allowMissingFiles) {
    failures.push(
      `${check.name}: missing file(s) ${missingFiles.join(", ")} in ${location}`,
    );
    continue;
  }
  const content = existingFiles
    .map((file) => fs.readFileSync(path.join(repoRoot, file), "utf8"))
    .join("\n");
  const requiredContent = requiredContractContent(existingFiles, content);
  for (const snippet of check.snippets) {
    if (!contractContentIncludes(requiredContent, snippet)) {
      failures.push(
        `${check.name}: missing ${JSON.stringify(snippet)} in ${location}`,
      );
    }
  }
  const normalizedContent = normalizeContractSnippet(requiredContent);
  for (const snippet of check.normalizedSnippets ?? []) {
    if (!normalizedContent.includes(normalizeContractSnippet(snippet))) {
      failures.push(
        `${check.name}: missing normalized ${JSON.stringify(snippet)} in ${location}`,
      );
    }
  }
  for (const snippet of check.absentSnippets ?? []) {
    if (content.includes(snippet)) {
      failures.push(
        `${check.name}: forbidden ${JSON.stringify(snippet)} in ${location}`,
      );
    }
  }
  let orderedSearchOffset = 0;
  for (const snippet of check.orderedSnippets ?? []) {
    const foundIndex = content.indexOf(snippet, orderedSearchOffset);
    if (foundIndex < 0) {
      failures.push(
        `${check.name}: ordered snippet ${JSON.stringify(snippet)} missing after offset ${orderedSearchOffset} in ${location}`,
      );
      break;
    }
    orderedSearchOffset = foundIndex + snippet.length;
  }
}

checkRetiredExecutionProcessSurface();
checkRetiredFileSystemSurface();
checkRetiredAgentRuntimeSessionFacadeSurface();
checkAgentRuntimeThinGatewayContracts();
checkRetiredPluginRuntimeRendererFacade();
checkRetiredAgentRuntimeMockFiles();
checkRetiredAgentRuntimeCommandManifestFiles();
checkRetiredAgentRuntimeAdapterFiles();
checkRetiredAgentSessionUpdateSurface();
checkRetiredRendererProjectionFiles();
checkRetiredAgentStreamToolMessageSynthesisSurface();
checkRetiredRendererQueuedTurnProjectionSurface();
checkRetiredRendererQueuedTurnSecondaryProjectionSurface();
checkRetiredAgentUiResumeContractFiles();
checkRetiredAgentRuntimeLegacyQueueSurface();
checkRetiredSkillExecutionSurfaceFiles();
checkRetiredAgentRuntimeToolInventoryMockFiles();
checkRetiredAgentRuntimeEvidenceExportFacadeSurface();
checkRetiredAgentRuntimeThreadReadFacadeSurface();
checkRetiredAgentRuntimeSubmitTurnFacadeSurface();
checkRetiredAgentRuntimeInterruptTurnFacadeSurface();
checkRetiredAgentRuntimeRespondActionFacadeSurface();
checkRendererQueuedTurnWriteSurface();
checkRetiredPublicQueuedTurnSurface();
checkActiveAipromptsDoNotPromoteRetiredAgentRuntimeCommands();
checkScriptsDoNotCallRetiredAgentRuntimeCommands();
checkMcpRuntimeCurrentContracts({ repoRoot, failures });
checkWorkspaceRightSurfaceCurrentContracts({ repoRoot, failures });
checkKnowledgeBuilderRuntimeCurrentContracts();
checkRetiredAppServerAgentBackendCrate();
checkRetiredRuntimeCoreMapperSurface();
checkRetiredAgentRuntimeClientShells();
checkRetiredToolWireSurface();
checkOrchestratorHostBoundary();
checkCanonicalRendererSequenceGate();
checkAgentUiPackageCanonicalNaming();

if (failures.length > 0) {
  console.error("[app-server-client-contract] failed");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(`[app-server-client-contract] ok (${checks.length + 35} checks)`);

function checkAgentUiPackageCanonicalNaming() {
  const packageManifestFiles = fs
    .readdirSync(path.join(repoRoot, "packages"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => `packages/${entry.name}/package.json`)
    .filter((relativePath) => fs.existsSync(path.join(repoRoot, relativePath)));

  const files = [
    ...agentUiPackageNamingGuardFiles.filter((relativePath) =>
      fs.existsSync(path.join(repoRoot, relativePath)),
    ),
    ...packageManifestFiles,
  ];
  const contentByFile = new Map(
    files.map((relativePath) => [
      relativePath,
      fs.readFileSync(path.join(repoRoot, relativePath), "utf8"),
    ]),
  );

  const combinedContent = Array.from(contentByFile.values()).join("\n");
  for (const packageName of canonicalAgentUiPackages) {
    if (!combinedContent.includes(packageName)) {
      failures.push(
        `Agent UI package canonical naming: missing canonical package ${packageName}`,
      );
    }
  }

  for (const [relativePath, content] of contentByFile.entries()) {
    for (const packageName of retiredAgentUiPackageNames) {
      if (content.includes(packageName)) {
        failures.push(
          `Agent UI package canonical naming: ${relativePath} must not reference retired package name ${packageName}`,
        );
      }
    }
  }
}

function checkKnowledgeBuilderRuntimeCurrentContracts() {
  const currentFiles = [
    "lime-rs/crates/app-server/src/runtime.rs",
    "lime-rs/crates/app-server/src/local_data_source.rs",
    "lime-rs/crates/app-server/src/knowledge_builder_runtime.rs",
  ];
  const content = currentFiles
    .map((file) => fs.readFileSync(path.join(repoRoot, file), "utf8"))
    .join("\n");
  const forbidden = [
    "builderRuntime is not available in App Server current path",
    "builder_runtime_requested(",
    "knowledge_cmd",
    "execute_knowledge_builder_skill",
  ];
  for (const snippet of forbidden) {
    if (content.includes(snippet)) {
      failures.push(
        `Knowledge builderRuntime current contract: forbidden ${JSON.stringify(
          snippet,
        )} in ${currentFiles.join(", ")}`,
      );
    }
  }
}

function checkRetiredAppServerAgentBackendCrate() {
  const retiredCratePath = "lime-rs/crates/app-server-agent-backend";
  if (fs.existsSync(path.join(repoRoot, retiredCratePath))) {
    failures.push(
      `retired App Server agent backend crate must stay deleted: ${retiredCratePath}`,
    );
  }
}

function checkRetiredRuntimeCoreMapperSurface() {
  const retiredPaths = [
    "lime-rs/crates/runtime-core/src/llm_protocol/mapper",
    "lime-rs/crates/runtime-core/src/llm_protocol/types.rs",
    "lime-rs/crates/runtime-core/src/llm_protocol/events.rs",
    "lime-rs/crates/runtime-core/src/llm_protocol/tests.rs",
    "lime-rs/crates/model-provider/src/lowering/anthropic_messages.rs",
    "lime-rs/crates/model-provider/src/lowering/gemini.rs",
    "lime-rs/crates/model-provider/src/lowering/ollama_chat.rs",
    "lime-rs/crates/model-provider/src/lowering/openai_chat.rs",
    "lime-rs/crates/model-provider/src/lowering/openai_responses.rs",
  ];
  for (const relativePath of retiredPaths) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(
        `retired provider dual-algebra path must stay deleted: ${relativePath}`,
      );
    }
  }

  const protocolFiles = [
    "lime-rs/crates/runtime-core/src/llm_protocol.rs",
    "lime-rs/crates/runtime-core/src/lib.rs",
  ];
  for (const relativePath of protocolFiles) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const snippet of [
      "mod mapper",
      "pub use mapper",
      "build_provider_wire_request",
      "runtime_event_from_llm_event",
      "ProviderWireRequest",
      "LlmRuntimeEvent",
    ]) {
      if (content.includes(snippet)) {
        failures.push(
          `retired runtime-core mapper export must stay absent: ${relativePath} contains ${JSON.stringify(snippet)}`,
        );
      }
    }
  }

  const currentOllamaProtocolFiles = [
    "lime-rs/crates/app-server-protocol/src/protocol/v0/model.rs",
    "lime-rs/crates/runtime-core/src/model_route.rs",
    "lime-rs/crates/model-provider/src/runtime_provider.rs",
    "lime-rs/crates/agent/src/provider_configuration.rs",
    "lime-rs/crates/app-server/src/runtime_backend/model_route_contract.rs",
    "lime-rs/crates/app-server/src/runtime_backend/model_routing.rs",
  ];
  for (const relativePath of currentOllamaProtocolFiles) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const snippet of ["OllamaChat", "ollama_chat"]) {
      if (content.includes(snippet)) {
        failures.push(
          `retired Ollama chat protocol must stay absent: ${relativePath} contains ${JSON.stringify(snippet)}`,
        );
      }
    }
  }

  const loweringRoot = path.join(
    repoRoot,
    "lime-rs/crates/model-provider/src/lowering",
  );
  for (const relativePath of walkSourceFiles(loweringRoot)) {
    if (!relativePath.endsWith(".rs")) {
      continue;
    }
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const snippet of [
      "LlmRequest",
      "ProviderWireRequest",
      "build_provider_wire_request",
      "build_responses_image_generation_wire_request",
    ]) {
      if (content.includes(snippet)) {
        failures.push(
          `retired provider dual algebra must stay absent: ${relativePath} contains ${JSON.stringify(snippet)}`,
        );
      }
    }
  }
}

function checkRetiredToolWireSurface() {
  const retiredPaths = [
    "lime-rs/crates/agent/src/agent_tools/workspace_patch_runtime_adapter.rs",
    "lime-rs/crates/agent/src/agent_tools/tool_lifecycle.rs",
    "lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs",
    "lime-rs/crates/tool-runtime/src/tool_batch.rs",
    "lime-rs/crates/agent/src/tool_output_truncation.rs",
    "lime-rs/crates/app-server/src/runtime/tests/external_events/actions.rs",
    "lime-rs/crates/app-server/src/runtime/tests/external_events/owner_terminal.rs",
    "lime-rs/crates/app-server/src/runtime/tests/external_events/tool_lifecycle.rs",
    "lime-rs/crates/app-server/src/backend_event.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent/execution.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent/execution_tests.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent/projection.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent/tests.rs",
    "lime-rs/crates/tool-runtime/src/collab_agent/validation.rs",
  ];
  for (const relativePath of retiredPaths) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(`retired raw tool wire must stay deleted: ${relativePath}`);
    }
  }

  for (const relativePath of [
    "lime-rs/crates/app-server/src/runtime/event_store.rs",
    "lime-rs/crates/app-server/src/runtime/tests/external_events/canonical_tool_items.rs",
  ]) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    if (!content.includes('"tool_end"')) {
      failures.push(
        `retired tool_end alias must stay fail-closed: ${relativePath} has no explicit guard`,
      );
    }
  }

  const currentFiles = [
    {
      path: "lime-rs/crates/app-server/src/runtime/event_store.rs",
      forbidden: [
        '"tool.started" =>',
        '"tool.result" =>',
        '"tool.failed" =>',
        "is_imported_tool_wire_payload",
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/lib.rs",
      forbidden: ["mod backend_event;", "runtime_event_type_from_backend_type"],
    },
    {
      path: "lime-rs/crates/core/src/agent/types.rs",
      forbidden: [
        "pub enum StreamEvent {",
        "pub struct ToolExecutionResult {",
        "pub struct StreamResult {",
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/projection_item_events.rs",
      forbidden: ['"tool.started"', '"tool.result"', '"tool.failed"'],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/tests/external_events/canonical_tool_items.rs",
      forbidden: [
        "append_external_runtime_events_allows_explicit_import_tool_compat",
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime_backend/tool_events.rs",
      forbidden: ['"tool.args"', "runtime_tool_args_event_payload"],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/tool_item_projection.rs",
      forbidden: [
        "LegacyToolEvent",
        "legacy_tool_event_from_event",
        "apply_legacy_tool_event",
        "find_legacy_match_for_current",
        "find_legacy_index",
        "from_legacy_event",
        "merge_legacy_event",
        "has_current_item",
        '"tool.started"',
        '"tool.result"',
        '"tool.failed"',
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/tool_item_projection/extract.rs",
      forbidden: [
        "LegacyToolEvent",
        "legacy_tool_event_from_event",
        "legacy_tool_id",
        "is_imported_legacy_tool_event",
        "normalize_tool_name_for_id",
        '"tool.started"',
        '"tool.result"',
        '"tool.failed"',
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/output_refs.rs",
      forbidden: [
        "largest_legacy_tool_output",
        "nested_result_output",
        "nested_runtime_event_result_output",
        "truncate_result_output_field",
        "truncate_runtime_event_result_output_field",
      ],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/thread_item_projection.rs",
      forbidden: ['"tool.started"', '"tool.result"', '"tool.failed"'],
    },
    {
      path: "lime-rs/crates/app-server/src/runtime/tests/coding_events/output_snapshots.rs",
      forbidden: [
        '"tool.started"',
        '"tool.result"',
        '"tool.failed"',
        '"tool_end"',
      ],
    },
    {
      path: "lime-rs/crates/agent/src/agent_tools/mod.rs",
      forbidden: ["pub mod tool_orchestrator;", "mod tool_lifecycle;"],
    },
    {
      path: "lime-rs/crates/agent/src/lib.rs",
      forbidden: ["mod tool_output_truncation;"],
    },
    {
      path: "lime-rs/crates/agent/src/protocol.rs",
      forbidden: ["ToolStart {", "ToolEnd {"],
    },
    {
      path: "lime-rs/crates/tool-runtime/src/lib.rs",
      forbidden: ["pub mod tool_batch;", "pub mod collab_agent;"],
    },
    {
      path: "lime-rs/crates/agent/src/agent_tools/catalog.rs",
      forbidden: [
        'name: "Agent",',
        'name: "SendMessage",',
        'name: "TeamCreate",',
        'name: "TeamDelete",',
        'name: "ListPeers",',
        '=> "Agent"',
        '=> "SendMessage"',
        '=> "TeamCreate"',
        '=> "TeamDelete"',
        '=> "ListPeers"',
      ],
    },
    {
      path: "lime-rs/crates/core/src/tool_calling.rs",
      forbidden: [
        'canonical_name: "Agent",',
        'canonical_name: "SendMessage",',
        'canonical_name: "TeamCreate",',
        'canonical_name: "TeamDelete",',
        'canonical_name: "ListPeers",',
      ],
    },
    {
      path: "lime-rs/crates/tool-runtime/src/turn_tool_surface.rs",
      forbidden: [
        "SUBAGENT_TEAMMATE_ALLOWED_TOOL_NAMES",
        "SUBAGENT_ALLOWED_NATIVE_TOOL_NAMES",
        "SUBAGENT_ALLOWED_COORDINATION_TOOL_NAMES",
        "runtime_turn_tool_exposure_allows_tool_name(",
      ],
    },
  ];
  for (const currentFile of currentFiles) {
    const content = fs.readFileSync(
      path.join(repoRoot, currentFile.path),
      "utf8",
    );
    for (const snippet of currentFile.forbidden) {
      if (content.includes(snippet)) {
        failures.push(
          `retired raw tool wire export must stay absent: ${currentFile.path} contains ${JSON.stringify(snippet)}`,
        );
      }
    }
  }

  const nativeOverlayProduction = fs
    .readFileSync(
      path.join(repoRoot, "lime-rs/crates/tool-runtime/src/native_overlay.rs"),
      "utf8",
    )
    .split("#[cfg(test)]", 1)[0];
  for (const retiredName of [
    "Agent",
    "SendMessage",
    "TeamCreate",
    "TeamDelete",
    "ListPeers",
  ]) {
    if (nativeOverlayProduction.includes(`"${retiredName}"`)) {
      failures.push(
        `retired Team tool must stay out of the native registry allowlist: ${retiredName}`,
      );
    }
  }

  for (const bridgeSurface of [
    {
      path: "lime-rs/crates/tool-runtime/src/mcp_connection.rs",
      forbidden: [
        "async fn list_resources(",
        "async fn read_resource(",
        "async fn list_prompts(",
        "async fn get_prompt(",
        "fn get_info(",
      ],
    },
    {
      path: "lime-rs/crates/tool-runtime/src/mcp_connection/registry.rs",
      forbidden: [
        "pub struct McpConnectionSummary",
        "pub async fn supports_resources(",
        "pub async fn summaries(",
        "pub async fn dispatch(",
        "pub async fn list_prompts(",
        "pub async fn get_prompt(",
      ],
    },
    {
      path: "lime-rs/crates/mcp/src/bridge_client.rs",
      forbidden: [
        "pub fn server_info(",
        "pub async fn list_resources(",
        "pub async fn read_resource(",
        "pub async fn list_prompts(",
        "pub async fn get_prompt(",
      ],
    },
    {
      path: "lime-rs/crates/agent/src/mcp_bridge.rs",
      forbidden: [
        "async fn list_resources(",
        "async fn read_resource(",
        "async fn list_prompts(",
        "async fn get_prompt(",
        "fn get_info(",
      ],
    },
  ]) {
    const content = fs.readFileSync(
      path.join(repoRoot, bridgeSurface.path),
      "utf8",
    );
    for (const snippet of bridgeSurface.forbidden) {
      if (content.includes(snippet)) {
        failures.push(
          `MCP sampling-step bridge must not own live management surface: ${bridgeSurface.path} contains ${JSON.stringify(snippet)}`,
        );
      }
    }
  }

  const mcpClientHandler = fs.readFileSync(
    path.join(repoRoot, "lime-rs/crates/mcp/src/client.rs"),
    "utf8",
  );
  if (mcpClientHandler.includes("enable_sampling()")) {
    failures.push(
      "MCP client must not advertise sampling without a typed createMessage owner: lime-rs/crates/mcp/src/client.rs contains enable_sampling()",
    );
  }

  const canonicalToolConsumerFiles = [
    "lime-rs/crates/app-server/src/runtime/provider_history.rs",
    "lime-rs/crates/app-server/src/runtime/context_compaction.rs",
    "lime-rs/crates/app-server/src/runtime/output_refs.rs",
    "lime-rs/crates/app-server/src/runtime/thread_item_projection/media_result.rs",
    "lime-rs/crates/app-server/src/runtime/thread_item_projection/coding_items.rs",
    "lime-rs/crates/app-server/src/runtime/thread_item_projection/control_items.rs",
    "lime-rs/crates/app-server/src/runtime/thread_item_projection/helpers.rs",
    "lime-rs/crates/app-server/src/runtime/thread_item_projection/materializer.rs",
    ...collectRustFiles(
      "lime-rs/crates/app-server/src/runtime/thread_item_projection/materializer",
    ),
  ];
  for (const relativePath of canonicalToolConsumerFiles) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    const productionContent = content.split("#[cfg(test)]", 1)[0];
    for (const retiredEventType of [
      '"tool.started"',
      '"tool.result"',
      '"tool.failed"',
      '"tool.completed"',
    ]) {
      if (productionContent.includes(retiredEventType)) {
        failures.push(
          `canonical Tool consumer must not parse retired lifecycle: ${relativePath} contains ${retiredEventType}`,
        );
      }
    }
  }
}

function checkOrchestratorHostBoundary() {
  const roots = [
    path.join(repoRoot, "electron"),
    path.join(repoRoot, "src/lib/dev-bridge"),
  ];
  const forbiddenSnippets = [
    "orchestrator.skills.enabled",
    "orchestrator.mcp.enabled",
    "discover_orchestrator_skills",
    "OrchestratorSkill",
    "codex_apps",
  ];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const relativePath of walkSourceFiles(root)) {
      if (
        !/\.(?:d\.ts|ts|tsx|mjs)$/u.test(relativePath) ||
        relativePath.includes(".test") ||
        relativePath.includes(".spec")
      ) {
        continue;
      }
      const content = fs.readFileSync(
        path.join(repoRoot, relativePath),
        "utf8",
      );
      for (const snippet of forbiddenSnippets) {
        if (content.includes(snippet)) {
          failures.push(
            `Orchestrator current owner must remain App Server/runtime/skills/MCP; ${relativePath} must not contain ${JSON.stringify(snippet)}`,
          );
        }
      }
    }
  }
}

function checkRetiredAgentRuntimeClientShells() {
  for (const relativePath of [
    "src/lib/api/agentRuntime.ts",
    "src/lib/api/agentRuntime.d.ts",
    "src/lib/api/agentRuntime/index.ts",
    "src/lib/api/agentRuntime/index.d.ts",
    "src/lib/api/agentRuntime/types.ts",
    "src/lib/api/agentRuntime/types.d.ts",
    "src/lib/api/agentRuntime/mediaClient.ts",
    "src/lib/api/agentRuntime/mediaClient.d.ts",
    "src/lib/api/agentRuntime/subagentClient.ts",
    "src/lib/api/agentRuntime/subagentClient.d.ts",
  ]) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(
        `retired Agent Runtime client shell or root barrel must stay deleted: ${relativePath}`,
      );
    }
  }
}

function checkRetiredPluginRuntimeRendererFacade() {
  const retiredFiles = [
    "electron/pluginRuntimeTaskHost.ts",
    "electron/pluginShellHost.ts",
    "src/lib/api/pluginRuntime.ts",
    "src/lib/api/plugins.ts",
    "src/lib/api/pluginsTypes.ts",
    "src/lib/api/pluginsResultGuards.ts",
    "src/features/plugin/types.ts",
    "src/features/plugin/types.d.ts",
  ];
  for (const relativePath of retiredFiles) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(
        `retired Plugin runtime/lifecycle facade must stay deleted: ${relativePath}`,
      );
    }
  }
  for (const relativePath of [
    "src/features/plugin/runtime",
    "src/features/plugin/sdk",
    "src/features/plugin/shell",
    "src/features/plugin/publish",
    "src/features/plugin/readiness",
    "src/features/plugin/runtime-profile",
    "src/features/plugin/manifest",
  ]) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push(
        `retired Plugin implementation directory must stay deleted: ${relativePath}`,
      );
    }
  }
}

function checkCanonicalRendererSequenceGate() {
  const relativePath = "src/lib/api/agentRuntime/eventSequenceGate.ts";
  const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
  for (const snippet of ["AgentRuntimeEventPipeline", "processSync("]) {
    if (!content.includes(snippet)) {
      failures.push(
        `canonical renderer sequence gate missing ${JSON.stringify(snippet)} in ${relativePath}`,
      );
    }
  }
  for (const snippet of [
    "AgentRuntimeEventAdapter",
    "withEvent",
    "currentCompatibilityAdapter",
    "currentToolCompletedFanoutAdapter",
  ]) {
    if (content.includes(snippet)) {
      failures.push(
        `canonical renderer sequence gate must not restore raw lifecycle adaptation: ${relativePath} contains ${JSON.stringify(snippet)}`,
      );
    }
  }
}

function checkAgentRuntimeThinGatewayContracts() {
  const sourceRoot = path.join(repoRoot, "src/lib/api/agentRuntime");
  for (const relativePath of walkSourceFiles(sourceRoot)) {
    if (!isAgentRuntimeGatewaySource(relativePath)) {
      continue;
    }
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const {
      snippet,
      reason,
    } of agentRuntimeThinGatewayForbiddenSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `agentRuntime thin gateway: ${relativePath} forbidden ${JSON.stringify(
            snippet,
          )}; ${reason}`,
        );
      }
    }
  }
}

function walkSourceFiles(root) {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const absolutePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkSourceFiles(absolutePath));
      continue;
    }
    if (!entry.isFile()) {
      continue;
    }
    files.push(path.relative(repoRoot, absolutePath));
  }
  return files.sort();
}

function checkRendererQueuedTurnWriteSurface() {
  for (const relativeRoot of rendererQueuedTurnWriteSurfaceRoots) {
    const sourceRoot = path.join(repoRoot, relativeRoot);
    for (const relativePath of walkSourceFiles(sourceRoot)) {
      if (
        !/\.(?:d\.ts|ts|tsx)$/u.test(relativePath) ||
        relativePath.includes(".test") ||
        relativePath.includes(".spec") ||
        /(?:^|\/)__tests__(?:\/|$)/u.test(relativePath)
      ) {
        continue;
      }
      const content = fs.readFileSync(
        path.join(repoRoot, relativePath),
        "utf8",
      );
      for (const snippet of rendererQueuedTurnWriteSurfaceForbiddenSnippets) {
        if (content.includes(snippet)) {
          failures.push(
            `Renderer queued-turn write surface must stay deleted: ${relativePath} contains ${JSON.stringify(snippet)}`,
          );
        }
      }
    }
  }
}

function checkRetiredPublicQueuedTurnSurface() {
  for (const file of retiredPublicQueuedTurnSchemaFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired public queued-turn schema must stay deleted: ${file}`,
      );
    }
  }
  for (const file of [
    ...retiredRendererQueuedTurnFiles,
    ...retiredPendingSteerFixtureFiles,
  ]) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired public queued-turn file must stay deleted: ${file}`,
      );
    }
  }
  for (const { file, snippets } of retiredPublicQueuedTurnSurfaceSpecs) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired public queued-turn guard missing production file: ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of snippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired public queued-turn surface: ${file} must not contain ${JSON.stringify(snippet)}`,
        );
      }
    }
  }
  const agentRuntimeScriptsRoot = path.join(repoRoot, "scripts/agent-runtime");
  for (const relativePath of walkScriptFiles(agentRuntimeScriptsRoot)) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const snippet of [
      ...retiredPendingSteerScenarios,
      "INPUTBAR_PENDING_STEER",
      "inputbarPendingSteer",
      "InputbarPendingSteer",
    ]) {
      if (content.includes(snippet)) {
        failures.push(
          `retired pending-steer fixture surface: ${relativePath} must not contain ${JSON.stringify(snippet)}`,
        );
      }
    }
  }
}

function checkRetiredExecutionProcessSurface() {
  for (const file of retiredExecutionProcessFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired executionProcess surface must stay deleted: ${file}`,
      );
    }
  }
  for (const spec of retiredExecutionProcessSurfaceSpecs) {
    const files = spec.files ?? [spec.file];
    for (const file of files) {
      const absolutePath = path.join(repoRoot, file);
      if (!fs.existsSync(absolutePath)) {
        failures.push(
          `retired executionProcess guard missing production file: ${file}`,
        );
        continue;
      }
      const content = fs.readFileSync(absolutePath, "utf8");
      for (const snippet of spec.snippets) {
        if (content.includes(snippet)) {
          failures.push(
            `retired executionProcess surface: ${file} must not contain ${JSON.stringify(snippet)}`,
          );
        }
      }
    }
  }
}

function checkRetiredFileSystemSurface() {
  for (const file of retiredFileSystemFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(`retired fileSystem surface must stay deleted: ${file}`);
    }
  }
  for (const spec of retiredFileSystemSurfaceSpecs) {
    const files = spec.files ?? [spec.file];
    for (const file of files) {
      const absolutePath = path.join(repoRoot, file);
      if (!fs.existsSync(absolutePath)) {
        failures.push(
          `retired fileSystem guard missing production file: ${file}`,
        );
        continue;
      }
      const content = fs.readFileSync(absolutePath, "utf8");
      for (const snippet of spec.snippets) {
        if (content.includes(snippet)) {
          failures.push(
            `retired fileSystem surface: ${file} must not contain ${JSON.stringify(snippet)}`,
          );
        }
      }
    }
  }
}

function isAgentRuntimeGatewaySource(relativePath) {
  return (
    relativePath.endsWith(".ts") &&
    !relativePath.endsWith(".d.ts") &&
    !relativePath.endsWith(".test.ts") &&
    !relativePath.endsWith(".generated.ts")
  );
}

function checkRetiredAgentRuntimeSessionFacadeSurface() {
  for (const file of retiredAgentRuntimeSessionFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired session facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeSessionFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired session facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredSkillExecutionSurfaceFiles() {
  for (const file of retiredSkillExecutionSurfaceFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Skill execution surface must stay deleted: ${file}`,
      );
    }
  }
}

function checkRetiredAgentRuntimeMockFiles() {
  for (const file of retiredAgentRuntimeMockFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Agent Runtime mock file must stay deleted: ${file}`,
      );
    }
  }
}

function checkRetiredAgentRuntimeCommandManifestFiles() {
  for (const file of retiredAgentRuntimeCommandManifestFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Agent Runtime command manifest surface must stay deleted: ${file}`,
      );
    }
  }
}

function checkRetiredAgentRuntimeAdapterFiles() {
  for (const file of retiredAgentRuntimeAdapterFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(`retired Agent runtime adapter must stay deleted: ${file}`);
    }
  }
}

function checkRetiredAgentSessionUpdateSurface() {
  const retiredSchemaFiles = [
    "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionUpdateParams.json",
    "lime-rs/crates/app-server-protocol/schema/json/v0/AgentSessionUpdateResponse.json",
  ];
  const productionFiles = [
    "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
    "lime-rs/crates/app-server-protocol/src/protocol/v0/session_admin.rs",
    "lime-rs/crates/app-server-protocol/src/protocol/v0/client_request.rs",
    "lime-rs/crates/app-server-protocol/src/protocol/v0/schema_types.rs",
    "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
    "lime-rs/crates/app-server-protocol/src/schema_export/registry.rs",
    "lime-rs/crates/app-server-protocol/schema/json/app_server_protocol.schemas.json",
    "lime-rs/crates/app-server-protocol/schema/json/manifest.json",
    "lime-rs/crates/app-server/src/processor/agent_session.rs",
    "lime-rs/crates/app-server/src/processor/dispatch.rs",
    "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
    "lime-rs/crates/app-server/src/runtime/projection_store.rs",
    "packages/app-server-client/src/connection-methods.ts",
    "packages/app-server-client/src/generated/protocol-types.ts",
    "packages/app-server-client/src/protocol.ts",
    "packages/app-server-client/src/request-client-methods.ts",
    "packages/app-server-client/src/request-client.ts",
    "src/lib/api/appServer.ts",
    "src/lib/api/agentExecutionRuntime.ts",
    "src/lib/api/agentRuntime/appServerSessionClient.ts",
    "src/lib/api/agentRuntime/requestTypes.ts",
    "src/lib/api/agentRuntime/sessionClient.ts",
    "src/lib/dev-bridge/commandPolicy.ts",
    "src/lib/governance/agentCommandCatalog.json",
  ];
  const retiredSnippets = [
    '"agentSession/update"',
    "METHOD_AGENT_SESSION_UPDATE",
    "AgentSessionUpdateParams",
    "AgentSessionUpdateResponse",
    "updateAgentRuntimeSession",
    "appServerClient.updateSession(",
    "update_session_current",
    "update_session_overview",
  ];

  for (const file of retiredSchemaFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired agentSession/update schema must stay deleted: ${file}`,
      );
    }
  }

  for (const file of productionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired agentSession/update surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredRendererProjectionFiles() {
  for (const file of retiredRendererProjectionFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(`retired Renderer projection must stay deleted: ${file}`);
    }
  }
}

function checkRetiredAgentStreamToolMessageSynthesisSurface() {
  for (const file of retiredAgentStreamToolMessageSynthesisFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Item-to-Message tool synthesis must stay deleted: ${file}`,
      );
    }
  }

  const agentChatRoot = path.join(repoRoot, "src/components/agent/chat");
  for (const relativePath of walkSourceFiles(agentChatRoot)) {
    if (
      !/\.[cm]?[jt]sx?$/u.test(relativePath) ||
      /\.(?:test|spec)\.[cm]?[jt]sx?$/u.test(relativePath)
    ) {
      continue;
    }
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const snippet of retiredAgentStreamToolMessageSynthesisSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired Item-to-Message tool synthesis: ${relativePath} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredRendererQueuedTurnProjectionSurface() {
  for (const file of retiredRendererQueuedTurnProjectionProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired Renderer queued-turn projection guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredRendererQueuedTurnProjectionSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired Renderer queued-turn projection surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredRendererQueuedTurnSecondaryProjectionSurface() {
  for (const {
    file,
    snippets,
  } of retiredRendererQueuedTurnSecondaryProjectionSpecs) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired Renderer queued-turn secondary projection guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of snippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired Renderer queued-turn secondary projection surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentUiResumeContractFiles() {
  for (const file of retiredAgentUiResumeContractFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Agent UI resume contract must stay deleted: ${file}`,
      );
    }
  }
}

function checkRetiredAgentRuntimeLegacyQueueSurface() {
  for (const file of retiredAgentRuntimeLegacyQueueFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Agent Runtime legacy queue file must stay deleted: ${file}`,
      );
    }
  }

  for (const file of retiredAgentRuntimeLegacyQueueSurfaceFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired Agent Runtime legacy queue guard: missing expected file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeLegacyQueueSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired Agent Runtime legacy queue surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentRuntimeToolInventoryMockFiles() {
  for (const file of retiredAgentRuntimeToolInventoryMockFiles) {
    if (fs.existsSync(path.join(repoRoot, file))) {
      failures.push(
        `retired Agent Runtime tool inventory mock must stay deleted: ${file}`,
      );
    }
  }
}

function checkRetiredAgentRuntimeEvidenceExportFacadeSurface() {
  for (const file of retiredAgentRuntimeEvidenceExportFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired evidence export facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeEvidenceExportFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired evidence export facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentRuntimeThreadReadFacadeSurface() {
  for (const file of retiredAgentRuntimeThreadReadFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired thread read facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeThreadReadFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired thread read facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentRuntimeSubmitTurnFacadeSurface() {
  for (const file of retiredAgentRuntimeSubmitTurnFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired submit turn facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeSubmitTurnFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired submit turn facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentRuntimeInterruptTurnFacadeSurface() {
  for (const file of retiredAgentRuntimeInterruptTurnFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired interrupt turn facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeInterruptTurnFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired interrupt turn facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkRetiredAgentRuntimeRespondActionFacadeSurface() {
  for (const file of retiredAgentRuntimeRespondActionFacadeProductionFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `retired respond action facade surface guard: missing expected production file ${file}`,
      );
      continue;
    }
    const content = fs.readFileSync(absolutePath, "utf8");
    for (const snippet of retiredAgentRuntimeRespondActionFacadeSnippets) {
      if (content.includes(snippet)) {
        failures.push(
          `retired respond action facade surface: ${file} must not contain ${JSON.stringify(
            snippet,
          )}`,
        );
      }
    }
  }
}

function checkActiveAipromptsDoNotPromoteRetiredAgentRuntimeCommands() {
  for (const file of activeAgentRuntimeMarkdownFiles) {
    const absolutePath = path.join(repoRoot, file);
    if (!fs.existsSync(absolutePath)) {
      failures.push(
        `active markdown retired agent_runtime guard: missing expected file ${file}`,
      );
      continue;
    }

    const lines = fs.readFileSync(absolutePath, "utf8").split(/\r?\n/u);
    lines.forEach((line, index) => {
      if (!/agent_runtime_[A-Za-z0-9_*]+/u.test(line)) {
        return;
      }
      if (allowedRetiredAgentRuntimeDocContextPattern.test(line)) {
        return;
      }
      failures.push(
        `active markdown retired agent_runtime guard: ${file}:${index + 1} must mark agent_runtime_* as retired/guard/history/migration-only, got ${JSON.stringify(
          line.trim(),
        )}`,
      );
    });

    lines.forEach((line, index) => {
      if (!/agent_runtime_[A-Za-z0-9_*]+/u.test(line)) {
        return;
      }
      if (!forbiddenAgentRuntimeCurrentDocContextPattern.test(line)) {
        return;
      }
      if (allowedRetiredAgentRuntimeDocContextPattern.test(line)) {
        return;
      }
      failures.push(
        `active markdown retired agent_runtime current wording: ${file}:${index + 1} must not describe agent_runtime_* as current, got ${JSON.stringify(
          line.trim(),
        )}`,
      );
    });
  }
}

function collectMarkdownFiles(relativeRoot) {
  const root = path.join(repoRoot, relativeRoot);
  if (!fs.existsSync(root)) {
    return [];
  }
  const files = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const absolutePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (
        entry.name === "node_modules" ||
        entry.name === "dist" ||
        entry.name === "target"
      ) {
        continue;
      }
      files.push(
        ...collectMarkdownFiles(
          path.relative(repoRoot, absolutePath).replaceAll("\\", "/"),
        ),
      );
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith(".md")) {
      continue;
    }
    files.push(path.relative(repoRoot, absolutePath).replaceAll("\\", "/"));
  }
  return files.sort();
}

function checkScriptsDoNotCallRetiredAgentRuntimeCommands() {
  const scriptsRoot = path.join(repoRoot, "scripts");
  for (const relativePath of walkScriptFiles(scriptsRoot)) {
    const content = fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
    for (const pattern of retiredAgentRuntimeScriptCallPatterns) {
      const match = pattern.exec(content);
      if (match) {
        failures.push(
          `retired agent_runtime script call guard: ${relativePath} must not call retired agent_runtime_* command via ${JSON.stringify(
            match[0],
          )}; keep legacy names only as negative guards or history evidence`,
        );
      }
    }
  }
}

function walkScriptFiles(root) {
  if (!fs.existsSync(root)) {
    return [];
  }
  const files = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const absolutePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (
        entry.name === "node_modules" ||
        entry.name === "dist" ||
        entry.name === "target"
      ) {
        continue;
      }
      files.push(...walkScriptFiles(absolutePath));
      continue;
    }
    if (!entry.isFile()) {
      continue;
    }
    const relativePath = path.relative(repoRoot, absolutePath);
    if (!/\.(?:cjs|js|mjs)$/u.test(relativePath)) {
      continue;
    }
    if (/\.(?:test|spec)\.(?:cjs|js|mjs)$/u.test(relativePath)) {
      continue;
    }
    files.push(relativePath.replaceAll("\\", "/"));
  }
  return files.sort();
}

function descriptorBlock(content, marker) {
  const startIndex = content.indexOf(marker);
  if (startIndex < 0) {
    return null;
  }
  const endIndex = content.indexOf("\n  },", startIndex);
  if (endIndex < 0) {
    return null;
  }
  return content.slice(startIndex, endIndex + "\n  },".length);
}

function assertBlockIncludes(block, snippet, description) {
  if (!block.includes(snippet)) {
    failures.push(`${description}: missing ${JSON.stringify(snippet)}`);
  }
}

function assertBlockExcludes(block, snippet, description) {
  if (block.includes(snippet)) {
    failures.push(`${description}: forbidden ${JSON.stringify(snippet)}`);
  }
}
