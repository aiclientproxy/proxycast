/* global process */
import { stat } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { app, BrowserWindow, shell } from "./electronRuntime";

type HostArgs = Record<string, unknown> | null | undefined;
type ElectronKnownPathName = "home" | "desktop" | "documents" | "downloads";

type FileManagerLocation = {
  id: string;
  label: string;
  path: string;
  kind: string;
};
type FileManagerLocationCandidate = Omit<FileManagerLocation, "path"> & {
  path: string | null;
};
type OpenFilePreviewWindowResult = {
  opened: true;
  reused: boolean;
  url: string;
  title: string;
};

export class FileShellHost {
  async openFilePreviewWindow(
    args: HostArgs,
  ): Promise<OpenFilePreviewWindowResult> {
    const request = readRequest(args);
    const targetPath = readRequiredAbsolutePath(request, "path");
    const requestedTitle = readString(request, "title");
    const title = requestedTitle || path.basename(targetPath) || targetPath;
    const url = pathToFileURL(targetPath).toString();
    return openFilePreviewBrowserWindow(url, title);
  }

  async revealInFinder(args: HostArgs): Promise<Record<string, never>> {
    const request = readRequest(args);
    const targetPath = readRequiredString(request, "path");

    if (await isDirectory(targetPath)) {
      const errorMessage = await shell.openPath(targetPath);
      if (errorMessage) {
        throw new Error(errorMessage);
      }
      return {};
    }

    shell.showItemInFolder(targetPath);
    return {};
  }

  async openWithDefaultApp(args: HostArgs): Promise<Record<string, never>> {
    const request = readRequest(args);
    const targetPath = readRequiredString(request, "path");
    const errorMessage = await shell.openPath(targetPath);
    if (errorMessage) {
      throw new Error(errorMessage);
    }
    return {};
  }

  async getFileIconDataUrl(args: HostArgs): Promise<string | null> {
    const request = readRequest(args);
    const targetPath = readRequiredString(request, "path");
    try {
      const icon = await app.getFileIcon(targetPath, { size: "normal" });
      if (icon.isEmpty()) {
        return null;
      }
      return icon.toDataURL() || null;
    } catch {
      return null;
    }
  }

  getHomeDir(): string {
    const homePath = readElectronPath("home");
    if (!homePath) {
      throw new Error("无法获取主目录");
    }
    return homePath;
  }

  async getFileManagerLocations(): Promise<FileManagerLocation[]> {
    const locations: FileManagerLocation[] = [];
    const seenPaths = new Set<string>();
    const homePath = readElectronPath("home");

    await appendFileManagerLocation(locations, seenPaths, {
      id: "home",
      label: "个人",
      kind: "home",
      path: homePath,
    });
    await appendFileManagerLocation(locations, seenPaths, {
      id: "desktop",
      label: "桌面",
      kind: "desktop",
      path: readElectronPath("desktop"),
    });
    await appendFileManagerLocation(locations, seenPaths, {
      id: "documents",
      label: "文档",
      kind: "documents",
      path: readElectronPath("documents"),
    });
    await appendFileManagerLocation(locations, seenPaths, {
      id: "downloads",
      label: "下载",
      kind: "downloads",
      path: readElectronPath("downloads"),
    });

    if (process.platform === "darwin") {
      await appendFileManagerLocation(locations, seenPaths, {
        id: "applications",
        label: "应用程序",
        kind: "applications",
        path: "/Applications",
      });
      await appendFileManagerLocation(locations, seenPaths, {
        id: "user-applications",
        label: "用户应用程序",
        kind: "applications",
        path: homePath ? path.join(homePath, "Applications") : null,
      });
    }

    if (process.platform === "win32") {
      await appendFileManagerLocation(locations, seenPaths, {
        id: "start-menu-programs",
        label: "应用程序",
        kind: "applications",
        path: process.env.APPDATA
          ? path.join(
              process.env.APPDATA,
              "Microsoft",
              "Windows",
              "Start Menu",
              "Programs",
            )
          : null,
      });
      await appendFileManagerLocation(locations, seenPaths, {
        id: "common-start-menu-programs",
        label: "公共应用程序",
        kind: "applications",
        path: process.env.PROGRAMDATA
          ? path.join(
              process.env.PROGRAMDATA,
              "Microsoft",
              "Windows",
              "Start Menu",
              "Programs",
            )
          : null,
      });
      await appendFileManagerLocation(locations, seenPaths, {
        id: "program-files",
        label: "Program Files",
        kind: "applications",
        path: process.env.ProgramFiles || null,
      });
      await appendFileManagerLocation(locations, seenPaths, {
        id: "program-files-x86",
        label: "Program Files (x86)",
        kind: "applications",
        path: process.env["ProgramFiles(x86)"] || null,
      });
    }

    return locations;
  }
}

function openFilePreviewBrowserWindow(
  url: string,
  title: string,
): OpenFilePreviewWindowResult {
  const existing = BrowserWindow.getAllWindows().find(
    (window) => window.webContents.getURL() === url,
  );
  const targetWindow =
    existing ??
    new BrowserWindow({
      width: 1280,
      height: 860,
      minWidth: 860,
      minHeight: 560,
      title,
      show: false,
      backgroundColor: "#f8fafc",
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });

  if (!existing) {
    void targetWindow.loadURL(url);
    targetWindow.once("ready-to-show", () => {
      targetWindow.show();
    });
  } else {
    targetWindow.show();
  }
  targetWindow.focus();

  return {
    opened: true,
    reused: Boolean(existing),
    url,
    title,
  };
}

function readRequest(value: unknown): Record<string, unknown> {
  return readRecord(value, "request") ?? toRecord(value) ?? {};
}

function readRecord(
  value: unknown,
  key: string,
): Record<string, unknown> | null {
  const record = toRecord(value);
  if (!record) {
    return null;
  }
  const next = record[key];
  return next && typeof next === "object" && !Array.isArray(next)
    ? (next as Record<string, unknown>)
    : null;
}

function readRequiredString(value: unknown, key: string): string {
  const next = readString(value, key);
  if (!next) {
    throw new Error(`Missing required string field: ${key}`);
  }
  return next;
}

function readRequiredAbsolutePath(value: unknown, key: string): string {
  const next = readRequiredString(value, key);
  if (!path.isAbsolute(next)) {
    throw new Error(`${key} 必须是绝对路径`);
  }
  return next;
}

function readString(value: unknown, key: string): string | null {
  const record = toRecord(value);
  const next = record?.[key];
  if (typeof next !== "string") {
    return null;
  }
  const trimmed = next.trim();
  return trimmed || null;
}

function readElectronPath(name: ElectronKnownPathName): string | null {
  try {
    const next = app.getPath(name);
    return next.trim() ? next : null;
  } catch {
    return null;
  }
}

async function isDirectory(targetPath: string): Promise<boolean> {
  try {
    return (await stat(targetPath)).isDirectory();
  } catch {
    return false;
  }
}

async function appendFileManagerLocation(
  locations: FileManagerLocation[],
  seenPaths: Set<string>,
  location: FileManagerLocationCandidate,
): Promise<void> {
  const normalizedPath = location.path?.trim() ?? "";
  if (!normalizedPath || seenPaths.has(normalizedPath)) {
    return;
  }
  try {
    const metadata = await stat(normalizedPath);
    if (!metadata.isDirectory()) {
      return;
    }
  } catch {
    return;
  }
  seenPaths.add(normalizedPath);
  locations.push({ ...location, path: normalizedPath });
}

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
