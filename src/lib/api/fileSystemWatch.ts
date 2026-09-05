import {
  fsChangedServerNotification,
  type FsChangedServerNotification,
} from "@limecloud/app-server-client";
import { AppServerClient } from "./appServerClient";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusSubscription,
} from "./appServerEventBus";
import type { AppServerJsonRpcNotification } from "./appServerTypes";

type FileSystemWatchClient = Pick<AppServerClient, "watch" | "unwatch">;
type NotificationSubscriber = (
  subscription: AppServerEventBusSubscription,
) => () => void;

export interface FileSystemWatchOptions {
  client?: FileSystemWatchClient;
  subscribeNotifications?: NotificationSubscriber;
  watchId?: string;
}

export type FileSystemWatchStop = () => Promise<void>;
export type FileSystemChangedHandler = (
  notification: FsChangedServerNotification["params"],
) => void;

let nextWatchId = 1;

export async function startFileSystemWatch(
  path: string,
  onChanged: FileSystemChangedHandler,
  options: FileSystemWatchOptions = {},
): Promise<FileSystemWatchStop> {
  const normalizedPath = path.trim();
  if (!isAbsolutePath(normalizedPath)) {
    throw new Error("fs/watch.path must be an absolute path");
  }

  const watchId = options.watchId?.trim() || createWatchId();
  if (!watchId) {
    throw new Error("fs/watch.watchId must not be empty");
  }

  const client = options.client ?? new AppServerClient();
  const subscribe =
    options.subscribeNotifications ?? subscribeAppServerNotifications;
  const unsubscribe = subscribe({
    getDrainOptions: () => ({
      activeIntervalMs: 25,
      intervalMs: 250,
      limit: 100,
    }),
    onNotifications: (notifications) => {
      for (const notification of notifications) {
        const changed = readFileSystemChangedNotification(notification);
        if (changed?.params.watchId === watchId) {
          onChanged(changed.params);
        }
      }
    },
  });

  try {
    await client.watch({ watchId, path: normalizedPath });
  } catch (error) {
    unsubscribe();
    throw error;
  }

  let stopped = false;
  return async () => {
    if (stopped) {
      return;
    }
    stopped = true;
    unsubscribe();
    await client.unwatch({ watchId });
  };
}

export function readFileSystemChangedNotification(
  notification: AppServerJsonRpcNotification,
): FsChangedServerNotification | undefined {
  return fsChangedServerNotification(notification);
}

function createWatchId(): string {
  const id = nextWatchId;
  nextWatchId += 1;
  return `file-manager-${id}`;
}

function isAbsolutePath(path: string): boolean {
  return path.startsWith("/") || /^[A-Za-z]:[\\/]/.test(path);
}
