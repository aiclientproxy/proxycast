import { getElectronHostBridge } from "@/lib/electron-host";

function isTestEnvironment(): boolean {
  return Boolean(import.meta.env?.MODE === "test" || import.meta.env?.VITEST);
}

function assertTestShellFixture(apiName: string): void {
  if (isTestEnvironment()) {
    return;
  }
  throw new Error(
    `[Mock] ${apiName} 只能在测试环境使用；生产 Shell 能力必须进入 Electron Desktop Host IPC。`,
  );
}

export async function open(path: string, _openWith?: string): Promise<void> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.shell) {
    return electronHost.shell.open(path, _openWith);
  }

  assertTestShellFixture("shell.open");
}
