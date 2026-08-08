import { getElectronHostBridge } from "@/lib/electron-host";

type ShortcutHandler = () => void;

const shortcuts = new Map<string, ShortcutHandler>();

function isTestEnvironment(): boolean {
  return Boolean(import.meta.env?.MODE === "test" || import.meta.env?.VITEST);
}

function assertTestShortcutFixture(apiName: string): void {
  if (isTestEnvironment()) {
    return;
  }
  throw new Error(
    `[Mock] ${apiName} 只能在测试环境使用；生产全局快捷键能力必须进入 Electron Desktop Host IPC。`,
  );
}

export async function register(
  shortcut: string,
  handler: ShortcutHandler,
): Promise<void> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.globalShortcut) {
    await electronHost.globalShortcut.register(shortcut);
    shortcuts.set(shortcut, handler);
    return;
  }

  assertTestShortcutFixture("globalShortcut.register");
  console.log(`[Mock] Global shortcut registered: ${shortcut}`);
  shortcuts.set(shortcut, handler);
}

export async function unregister(shortcut: string): Promise<void> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.globalShortcut) {
    await electronHost.globalShortcut.unregister(shortcut);
    shortcuts.delete(shortcut);
    return;
  }

  assertTestShortcutFixture("globalShortcut.unregister");
  console.log(`[Mock] Global shortcut unregistered: ${shortcut}`);
  shortcuts.delete(shortcut);
}

export async function unregisterAll(): Promise<void> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.globalShortcut) {
    await electronHost.globalShortcut.unregisterAll();
    shortcuts.clear();
    return;
  }

  assertTestShortcutFixture("globalShortcut.unregisterAll");
  console.log("[Mock] All global shortcuts unregistered");
  shortcuts.clear();
}

export async function isRegistered(shortcut: string): Promise<boolean> {
  const electronHost = getElectronHostBridge();
  if (electronHost?.globalShortcut) {
    return electronHost.globalShortcut.isRegistered(shortcut);
  }

  assertTestShortcutFixture("globalShortcut.isRegistered");
  return shortcuts.has(shortcut);
}

export function triggerShortcut(shortcut: string) {
  assertTestShortcutFixture("globalShortcut.triggerShortcut");
  const handler = shortcuts.get(shortcut);
  if (handler) {
    console.log(`[Mock] Triggering shortcut: ${shortcut}`);
    handler();
  } else {
    console.warn(`[Mock] Shortcut not found: ${shortcut}`);
  }
}
