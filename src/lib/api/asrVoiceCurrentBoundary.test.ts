import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { beforeAll, describe, expect, it } from "vitest";
import { readAppServerApiSources } from "../../test/appServerApiSources";

const RETIRED_VOICE_INPUT_CONFIG_FACADE_COMMANDS = [
  "get_voice_input_config",
  "save_voice_input_config",
];
const CURRENT_VOICE_MODEL_READ_COMMANDS = [
  "voice_models_list_catalog",
  "voice_models_get_install_state",
];
const CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS = [
  "voice_models_delete",
  "voice_models_download",
];
const CURRENT_VOICE_MODEL_DEFAULT_METHOD = "voiceModel/default/set";
const CURRENT_VOICE_MODEL_DEFAULT_CLIENT_HELPER = "setDefaultVoiceModel";
const CURRENT_VOICE_MODEL_TEST_TRANSCRIBE_METHOD =
  "voiceModel/testTranscribeFile";
const CURRENT_VOICE_MODEL_TEST_TRANSCRIBE_CLIENT_HELPER =
  "testTranscribeVoiceModelFile";
const CURRENT_VOICE_TRANSCRIPTION_METHOD = "voiceTranscription/transcribeAudio";
const CURRENT_VOICE_TRANSCRIPTION_CLIENT_HELPER = "transcribeVoiceAudio";
const CURRENT_VOICE_TRANSCRIPTION_FRONTEND_HELPER = "transcribeVoiceInputAudio";
const CURRENT_VOICE_POLISH_METHOD = "voiceTranscription/polishText";
const CURRENT_VOICE_POLISH_CLIENT_HELPER = "polishVoiceText";
const CURRENT_VOICE_POLISH_FRONTEND_HELPER = "polishVoiceInputText";
const CURRENT_ASR_CREDENTIAL_METHODS = [
  "voiceAsrCredential/list",
  "voiceAsrCredential/create",
  "voiceAsrCredential/update",
  "voiceAsrCredential/delete",
  "voiceAsrCredential/default/set",
  "voiceAsrCredential/test",
];
const CURRENT_ASR_CREDENTIAL_CLIENT_HELPERS = [
  "listVoiceAsrCredentials",
  "createVoiceAsrCredential",
  "updateVoiceAsrCredential",
  "deleteVoiceAsrCredential",
  "setDefaultVoiceAsrCredential",
  "testVoiceAsrCredential",
];
const CURRENT_VOICE_INSTRUCTION_METHODS = [
  "voiceInstruction/list",
  "voiceInstruction/save",
  "voiceInstruction/delete",
];
const CURRENT_VOICE_INSTRUCTION_CLIENT_HELPERS = [
  "listVoiceInstructions",
  "saveVoiceInstruction",
  "deleteVoiceInstruction",
];
const RETIRED_ASR_CREDENTIAL_FACADE_COMMANDS = [
  "get_asr_credentials",
  "add_asr_credential",
  "update_asr_credential",
  "delete_asr_credential",
  "set_default_asr_credential",
  "test_asr_credential",
];
const RETIRED_VOICE_INSTRUCTION_FACADE_COMMANDS = [
  "get_voice_instructions",
  "save_voice_instruction",
  "delete_voice_instruction",
];
const RETIRED_VOICE_REALTIME_FACADE_COMMANDS = [
  "transcribe_audio",
  "polish_voice_text",
  "output_voice_text",
  "start_recording",
  "stop_recording",
  "get_recording_snapshot",
  "get_recording_segment",
  "cancel_recording",
  "get_recording_status",
];
const RETIRED_VOICE_REALTIME_FRONTEND_HELPERS = [
  "transcribeAudio",
  "polishVoiceText",
  "outputVoiceText",
  "startRecording",
  "stopRecording",
  "getRecordingSnapshot",
  "getRecordingSegment",
  "cancelRecording",
  "getRecordingStatus",
];
const RETIRED_VOICE_REALTIME_GUI_FALSE_ENTRY_LITERALS = [
  "guide-voice-input",
  "home-guide-voice-input",
  "guide-voice",
  "home-guide-voice",
  "voice-input",
  "listenForVoiceShortcut",
];
const RETIRED_VOICE_MODEL_DEFAULT_FACADE_COMMAND = "voice_models_set_default";
const RETIRED_VOICE_MODEL_TEST_TRANSCRIBE_FACADE_COMMAND =
  "voice_models_test_transcribe_file";

function readRepoFile(path: string): string {
  return readFileSync(resolve(cwd(), path), "utf8");
}

function repoFileExists(path: string): boolean {
  return existsSync(resolve(cwd(), path));
}

function readRepoFileIfExists(path: string): string {
  return repoFileExists(path) ? readRepoFile(path) : "";
}

function expectLegacyRustFileDeleted(path: string): void {
  expect(repoFileExists(path), `${path} should stay deleted`).toBe(false);
}

function listProductionTsFiles(rootPath: string): string[] {
  const absoluteRoot = resolve(cwd(), rootPath);
  if (!existsSync(absoluteRoot)) {
    return [];
  }

  return readdirSync(absoluteRoot, { withFileTypes: true }).flatMap((entry) => {
    const relativePath = `${rootPath}/${entry.name}`;
    if (
      entry.name === "__tests__" ||
      entry.name === "test" ||
      entry.name === "tests"
    ) {
      return [];
    }
    if (entry.isDirectory()) {
      return listProductionTsFiles(relativePath);
    }
    if (
      !/\.(ts|tsx)$/.test(entry.name) ||
      /\.(test|spec)\.(ts|tsx)$/.test(entry.name)
    ) {
      return [];
    }
    return [relativePath];
  });
}

function listProductionGuiTsFiles(): string[] {
  return [
    ...listProductionTsFiles("src/components"),
    ...listProductionTsFiles("src/hooks"),
    ...listProductionTsFiles("src/lib"),
  ];
}

let productionGuiSourceByPath = new Map<string, string>();

function getProductionGuiSource(path: string): string {
  return productionGuiSourceByPath.get(path) ?? readRepoFile(path);
}

function expectStringLiteralsAbsent(source: string, literals: string[]): void {
  for (const literal of literals) {
    expect(source).not.toContain(`"${literal}"`);
    expect(source).not.toContain(`'${literal}'`);
  }
}

function readStringSetLiteral(source: string, setName: string): Set<string> {
  const escapedSetName = setName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(
    new RegExp(
      `const\\s+${escapedSetName}\\s*=\\s+new\\s+Set(?:<string>)?\\(\\[([\\s\\S]*?)\\]\\);`,
    ),
  );
  expect(
    match?.[1],
    `${setName} should be declared as a string Set`,
  ).toBeDefined();
  return new Set(
    Array.from(match![1].matchAll(/["']([^"']+)["']/g), ([, value]) => value),
  );
}

function expectStringSetExcludes(
  source: string,
  setName: string,
  commands: string[],
): void {
  const values = readStringSetLiteral(source, setName);
  for (const command of commands) {
    expect(
      values.has(command),
      `${setName} should not include ${command}`,
    ).toBe(false);
  }
}

function expectStringSetIncludes(
  source: string,
  setName: string,
  commands: string[],
): void {
  const values = readStringSetLiteral(source, setName);
  for (const command of commands) {
    expect(values.has(command), `${setName} should include ${command}`).toBe(
      true,
    );
  }
}

function findAsrProviderNamedImports(
  source: string,
  helpers: string[],
): string[] {
  const imported = new Set<string>();
  const importPattern =
    /import\s+(?:type\s+)?\{([\s\S]*?)\}\s+from\s+["'][^"']*asrProvider["'];/g;
  for (const match of source.matchAll(importPattern)) {
    const names = match[1]
      .split(",")
      .map((name) =>
        name
          .replace(/^type\s+/, "")
          .split(/\s+as\s+/)[0]
          .trim(),
      )
      .filter(Boolean);
    for (const name of names) {
      if (helpers.includes(name)) {
        imported.add(name);
      }
    }
  }
  return [...imported].sort();
}

function expectRustRunnerDoesNotRegister(
  source: string,
  commands: string[],
): void {
  for (const command of commands) {
    expect(source).not.toContain(`::${command}`);
  }
}

function readElectronSources(): string {
  return [
    readRepoFile("electron/ipcChannels.ts"),
    readRepoFile("electron/hostCommands.ts"),
  ].join("\n");
}

function readDevBridgeAndMockSources(): string {
  return [
    readRepoFile("src/lib/dev-bridge/commandPolicy.ts"),
    readRepoFile("src/lib/dev-bridge/mockPriorityCommands.ts"),
    readRepoFile("src/lib/desktop-host/voiceMocks.ts"),
  ].join("\n");
}

function readLegacyVoiceSources(): string {
  const paths = [
    readRepoFileIfExists("lime-rs/src/app/runner.rs"),
    readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
    readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
  ];
  return paths.join("\n");
}

function readAppServerAsrCredentialSources(): string {
  return [
    readRepoFile("packages/app-server-client/src/protocol.ts"),
    readRepoFile("packages/app-server-client/src/index.ts"),
    readAppServerApiSources(),
    readRepoFile(
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
    ),
    readRepoFile("lime-rs/crates/app-server-protocol/src/protocol/v0/voice.rs"),
    readRepoFile(
      "lime-rs/crates/app-server/src/local_data_source/voice_asr_credentials.rs",
    ),
    readRepoFile(
      "lime-rs/crates/app-server/src/local_data_source/voice_instructions.rs",
    ),
    readRepoFile(
      "lime-rs/crates/app-server/src/local_data_source/voice_text_processing.rs",
    ),
  ].join("\n");
}

function readAppServerVoiceInstructionSources(): string {
  return [
    readRepoFile("packages/app-server-client/src/protocol.ts"),
    readRepoFile("packages/app-server-client/src/index.ts"),
    readAppServerApiSources(),
    readRepoFile(
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
    ),
    readRepoFile("lime-rs/crates/app-server-protocol/src/protocol/v0/voice.rs"),
    readRepoFile(
      "lime-rs/crates/app-server/src/local_data_source/voice_instructions.rs",
    ),
  ].join("\n");
}

describe("ASR / Voice current boundary", () => {
  beforeAll(() => {
    const productionFiles = listProductionGuiTsFiles().filter(
      (path) => path !== "src/lib/api/asrProvider.ts",
    );
    productionGuiSourceByPath = new Map(
      productionFiles.map((path) => [path, readRepoFile(path)]),
    );
  }, 20_000);

  it("Voice Input config 应固定走 app config current 网关", () => {
    const asrProviderSource = readRepoFile("src/lib/api/asrProvider.ts");
    const restrictedSources = [
      asrProviderSource,
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readLegacyVoiceSources(),
    ].join("\n");

    expect(asrProviderSource).toContain("getVoiceInputConfig");
    expect(asrProviderSource).toContain("saveVoiceInputConfig");
    expect(asrProviderSource).toContain("getConfig()");
    expect(asrProviderSource).toContain("saveConfig(");
    expect(asrProviderSource).toContain("voice_input");
    expectStringLiteralsAbsent(
      restrictedSources,
      RETIRED_VOICE_INPUT_CONFIG_FACADE_COMMANDS,
    );
  });

  it("Voice Model catalog / install-state 应固定为 Electron Host current 读链", () => {
    const voiceModelsSource = readRepoFile("src/lib/api/voiceModels.ts");
    const electronSources = readElectronSources();
    const commandPolicySource = readRepoFile(
      "src/lib/dev-bridge/commandPolicy.ts",
    );
    const mockPrioritySource = readRepoFile(
      "src/lib/dev-bridge/mockPriorityCommands.ts",
    );
    const voiceMocksSource = readRepoFile("src/lib/desktop-host/voiceMocks.ts");
    const runnerSource = readRepoFileIfExists("lime-rs/src/app/runner.rs");
    const dispatcherSource = readRepoFileIfExists(
      "lime-rs/src/dev_bridge/dispatcher.rs",
    );

    for (const command of CURRENT_VOICE_MODEL_READ_COMMANDS) {
      expect(voiceModelsSource).toContain(`"${command}"`);
      expect(electronSources).toContain(`"${command}"`);
    }

    expectStringSetExcludes(
      commandPolicySource,
      "bridgeTruthCommands",
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expectStringSetExcludes(
      commandPolicySource,
      "electronHostNoMockFallbackCommands",
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expectStringSetExcludes(
      mockPrioritySource,
      "mockPriorityCommands",
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expectStringLiteralsAbsent(
      voiceMocksSource,
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expectRustRunnerDoesNotRegister(
      runnerSource,
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expectStringLiteralsAbsent(
      dispatcherSource,
      CURRENT_VOICE_MODEL_READ_COMMANDS,
    );
    expect(dispatcherSource).not.toContain("mod voice;");
    expect(dispatcherSource).not.toContain("voice::try_handle");
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expect(repoFileExists("lime-rs/src/dev_bridge/dispatcher/voice.rs")).toBe(
      false,
    );
    expect(repoFileExists("lime-rs/src/commands/voice_model_cmd.rs")).toBe(
      false,
    );
  });

  it("Voice Model delete 应固定为 Electron Host current 删除壳能力", () => {
    const voiceModelsSource = readRepoFile("src/lib/api/voiceModels.ts");
    const electronSources = readElectronSources();
    const commandPolicySource = readRepoFile(
      "src/lib/dev-bridge/commandPolicy.ts",
    );
    const mockPrioritySource = readRepoFile(
      "src/lib/dev-bridge/mockPriorityCommands.ts",
    );
    const voiceMocksSource = readRepoFile("src/lib/desktop-host/voiceMocks.ts");
    const runnerSource = readRepoFileIfExists("lime-rs/src/app/runner.rs");
    const dispatcherSource = readRepoFileIfExists(
      "lime-rs/src/dev_bridge/dispatcher.rs",
    );

    for (const command of CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS) {
      expect(voiceModelsSource).toContain(`"${command}"`);
      expect(electronSources).toContain(`"${command}"`);
    }

    expectStringSetExcludes(
      commandPolicySource,
      "bridgeTruthCommands",
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expectStringSetIncludes(
      commandPolicySource,
      "electronHostNoMockFallbackCommands",
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expectStringSetExcludes(
      mockPrioritySource,
      "mockPriorityCommands",
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expectStringLiteralsAbsent(
      voiceMocksSource,
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expectRustRunnerDoesNotRegister(
      runnerSource,
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expectStringLiteralsAbsent(
      dispatcherSource,
      CURRENT_VOICE_MODEL_DESKTOP_HOST_SIDE_EFFECT_COMMANDS,
    );
    expect(dispatcherSource).not.toContain("mod voice;");
    expect(dispatcherSource).not.toContain("voice::try_handle");
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expect(repoFileExists("lime-rs/src/dev_bridge/dispatcher/voice.rs")).toBe(
      false,
    );
    expect(repoFileExists("lime-rs/src/commands/voice_model_cmd.rs")).toBe(
      false,
    );
  });

  it("Voice Model set-default 应固定走 App Server voiceModel/default/set current 写链", () => {
    const voiceModelsSource = readRepoFile("src/lib/api/voiceModels.ts");
    const appServerSources = readAppServerAsrCredentialSources();
    const restrictedSources = [
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expect(voiceModelsSource).toContain("createAppServerClient()");
    expect(voiceModelsSource).toContain(
      `${CURRENT_VOICE_MODEL_DEFAULT_CLIENT_HELPER}({`,
    );
    expect(voiceModelsSource).toContain(`"voice_models_get_install_state"`);
    expect(voiceModelsSource).not.toMatch(
      /safeInvoke(?:<[^>]+>)?\s*\(\s*["']voice_models_set_default["']/,
    );
    expect(appServerSources).toContain(
      `"${CURRENT_VOICE_MODEL_DEFAULT_METHOD}"`,
    );
    expect(appServerSources).toContain(
      CURRENT_VOICE_MODEL_DEFAULT_CLIENT_HELPER,
    );
    expectStringLiteralsAbsent(restrictedSources, [
      RETIRED_VOICE_MODEL_DEFAULT_FACADE_COMMAND,
    ]);
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expect(repoFileExists("lime-rs/src/commands/voice_model_cmd.rs")).toBe(
      false,
    );
  });

  it("Voice Model test-transcribe 应固定走 App Server current 且不回流旧命令面", () => {
    const voiceModelsSource = readRepoFile("src/lib/api/voiceModels.ts");
    const appServerSources = readAppServerAsrCredentialSources();
    const restrictedSources = [
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFileIfExists("lime-rs/src/commands/mod.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expect(voiceModelsSource).toContain("testTranscribeVoiceModelFile");
    expect(voiceModelsSource).toContain("createAppServerClient()");
    expect(voiceModelsSource).toContain(
      `.${CURRENT_VOICE_MODEL_TEST_TRANSCRIBE_CLIENT_HELPER}({`,
    );
    expect(appServerSources).toContain(
      `"${CURRENT_VOICE_MODEL_TEST_TRANSCRIBE_METHOD}"`,
    );
    expect(appServerSources).toContain(
      CURRENT_VOICE_MODEL_TEST_TRANSCRIBE_CLIENT_HELPER,
    );
    expect(voiceModelsSource).not.toMatch(
      /safeInvoke(?:<[^>]+>)?\s*\(\s*["']voice_models_test_transcribe_file["']/,
    );
    expectStringLiteralsAbsent(restrictedSources, [
      RETIRED_VOICE_MODEL_TEST_TRANSCRIBE_FACADE_COMMAND,
    ]);
    expect(readElectronSources()).not.toContain(
      `"${RETIRED_VOICE_MODEL_TEST_TRANSCRIBE_FACADE_COMMAND}"`,
    );
    expect(readElectronSources()).not.toContain(
      `'${RETIRED_VOICE_MODEL_TEST_TRANSCRIBE_FACADE_COMMAND}'`,
    );
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expectLegacyRustFileDeleted("lime-rs/src/commands/mod.rs");
    expect(repoFileExists("lime-rs/src/commands/voice_model_cmd.rs")).toBe(
      false,
    );
  });

  it("ASR 凭证 CRUD 应固定走 App Server voiceAsrCredential current 主链", () => {
    const asrProviderSource = readRepoFile("src/lib/api/asrProvider.ts");
    const voiceModelsSource = readRepoFile("src/lib/api/voiceModels.ts");
    const appServerSources = readAppServerAsrCredentialSources();

    expect(asrProviderSource).toContain("createAppServerClient()");
    expect(asrProviderSource).toContain("asrProviderToAppServer");
    expect(asrProviderSource).toContain("asrProviderFromAppServer");
    expect(voiceModelsSource).toContain("getAsrCredentials()");

    for (const method of CURRENT_ASR_CREDENTIAL_METHODS) {
      expect(appServerSources).toContain(`"${method}"`);
    }
    for (const helper of CURRENT_ASR_CREDENTIAL_CLIENT_HELPERS) {
      expect(appServerSources).toContain(helper);
      expect(asrProviderSource).toContain(`.${helper}(`);
    }
    expectStringLiteralsAbsent(
      asrProviderSource,
      RETIRED_ASR_CREDENTIAL_FACADE_COMMANDS,
    );
  });

  it("Voice instructions 应固定走 App Server voiceInstruction current 主链", () => {
    const asrProviderSource = readRepoFile("src/lib/api/asrProvider.ts");
    const appServerSources = readAppServerVoiceInstructionSources();

    expect(asrProviderSource).toContain("createAppServerClient()");
    expect(asrProviderSource).toContain(
      "APP_SERVER_METHOD_VOICE_INSTRUCTION_LIST",
    );
    expect(asrProviderSource).toContain(
      "APP_SERVER_METHOD_VOICE_INSTRUCTION_SAVE",
    );
    expect(asrProviderSource).toContain(
      "APP_SERVER_METHOD_VOICE_INSTRUCTION_DELETE",
    );

    for (const method of CURRENT_VOICE_INSTRUCTION_METHODS) {
      expect(appServerSources).toContain(`"${method}"`);
    }
    for (const helper of CURRENT_VOICE_INSTRUCTION_CLIENT_HELPERS) {
      expect(appServerSources).toContain(helper);
      expect(asrProviderSource).toContain(`.${helper}(`);
    }
    expectStringLiteralsAbsent(
      asrProviderSource,
      RETIRED_VOICE_INSTRUCTION_FACADE_COMMANDS,
    );
  });

  it("旧 ASR 凭证 facade 不应回到 Electron、DevBridge、mock 或 legacy Rust", () => {
    const restrictedSources = [
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFileIfExists("lime-rs/src/commands/mod.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expectStringLiteralsAbsent(
      restrictedSources,
      RETIRED_ASR_CREDENTIAL_FACADE_COMMANDS,
    );
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expectLegacyRustFileDeleted("lime-rs/src/commands/mod.rs");
    expectLegacyRustFileDeleted("lime-rs/src/commands/asr_cmd.rs");
    expectLegacyRustFileDeleted("lime-rs/src/voice/commands.rs");
  });

  it("旧 Voice instruction facade 不应回到 Electron、DevBridge、mock、catalog 或 legacy Rust", () => {
    const restrictedSources = [
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFileIfExists("lime-rs/src/commands/mod.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expectStringLiteralsAbsent(
      restrictedSources,
      RETIRED_VOICE_INSTRUCTION_FACADE_COMMANDS,
    );
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expectLegacyRustFileDeleted("lime-rs/src/commands/mod.rs");
    expectLegacyRustFileDeleted("lime-rs/src/voice/commands.rs");
  });

  it("旧实时语音转写 / 录音 facade 不应回到前端、Electron、DevBridge、mock、catalog 或 legacy Rust", () => {
    const asrProviderSource = readRepoFile("src/lib/api/asrProvider.ts");
    const restrictedSources = [
      asrProviderSource,
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expect(asrProviderSource).toContain("failClosedRetiredVoiceInputCommand");
    expect(asrProviderSource).toContain(
      "旧实时语音转写、润色、输出与录音控制入口已退役",
    );
    expectStringLiteralsAbsent(
      restrictedSources,
      RETIRED_VOICE_REALTIME_FACADE_COMMANDS,
    );
    expectLegacyRustFileDeleted("lime-rs/src/app/runner.rs");
    expectLegacyRustFileDeleted("lime-rs/src/dev_bridge/dispatcher.rs");
    expectLegacyRustFileDeleted("lime-rs/src/voice/commands.rs");
  });

  it("输入框语音转写应固定走 App Server voiceTranscription current 主链", () => {
    const asrProviderSource = readRepoFile("src/lib/api/asrProvider.ts");
    const appServerSources = readAppServerAsrCredentialSources();
    const restrictedSources = [
      readElectronSources(),
      readDevBridgeAndMockSources(),
      readRepoFileIfExists("lime-rs/src/app/runner.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher.rs"),
      readRepoFileIfExists("lime-rs/src/dev_bridge/dispatcher/voice.rs"),
      readRepoFile("src/lib/governance/agentCommandCatalog.json"),
    ].join("\n");

    expect(asrProviderSource).toContain(
      CURRENT_VOICE_TRANSCRIPTION_FRONTEND_HELPER,
    );
    expect(asrProviderSource).toContain(CURRENT_VOICE_POLISH_FRONTEND_HELPER);
    expect(asrProviderSource).toContain("createAppServerClient()");
    expect(asrProviderSource).toContain(
      `.${CURRENT_VOICE_TRANSCRIPTION_CLIENT_HELPER}(`,
    );
    expect(asrProviderSource).toContain(
      `.${CURRENT_VOICE_POLISH_CLIENT_HELPER}(`,
    );
    expect(appServerSources).toContain(
      `"${CURRENT_VOICE_TRANSCRIPTION_METHOD}"`,
    );
    expect(appServerSources).toContain(`"${CURRENT_VOICE_POLISH_METHOD}"`);
    expect(appServerSources).toContain(
      CURRENT_VOICE_TRANSCRIPTION_CLIENT_HELPER,
    );
    expect(appServerSources).toContain(CURRENT_VOICE_POLISH_CLIENT_HELPER);
    expectStringLiteralsAbsent(
      restrictedSources,
      RETIRED_VOICE_REALTIME_FACADE_COMMANDS,
    );
  });

  it("App Server 应编入默认本地 SenseVoice 转写能力", () => {
    const appServerCargoToml = readRepoFile(
      "lime-rs/crates/app-server/Cargo.toml",
    );

    expect(appServerCargoToml).toMatch(
      /lime-services\s*=\s*\{[^}]*features\s*=\s*\[[^\]]*"local-sensevoice"/,
    );
  });

  it("生产 GUI 不应重新 import 实时语音 fail-closed wrapper", () => {
    const violations = [...productionGuiSourceByPath].flatMap(
      ([path, source]) => {
        const imports = findAsrProviderNamedImports(
          source,
          RETIRED_VOICE_REALTIME_FRONTEND_HELPERS,
        );
        return imports.map((name) => `${path}: ${name}`);
      },
    );

    expect(violations).toEqual([]);
  });

  it("生产 GUI 不应重新暴露实时语音默认入口", () => {
    const violations = [...productionGuiSourceByPath].flatMap(([path]) => {
      const source = getProductionGuiSource(path);
      return RETIRED_VOICE_REALTIME_GUI_FALSE_ENTRY_LITERALS.flatMap(
        (literal) =>
          source.includes(`"${literal}"`) || source.includes(`'${literal}'`)
            ? [`${path}: ${literal}`]
            : [],
      );
    });

    expect(violations).toEqual([]);
  });
});
