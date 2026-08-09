import { commandExecOutputDeltaServerNotification } from "../../../packages/app-server-client/src/server-notifications";
import {
  AppServerClient,
  type AppServerCommandExecOutputDeltaNotification,
  type AppServerCommandExecParams,
  type AppServerCommandExecResponse,
  type AppServerCommandExecResizeParams,
  type AppServerCommandExecResizeResponse,
  type AppServerCommandExecTerminateParams,
  type AppServerCommandExecTerminateResponse,
  type AppServerCommandExecWriteParams,
  type AppServerCommandExecWriteResponse,
} from "./appServer";
import { subscribeAppServerNotifications } from "./appServerEventBus";

export type CommandExecClient = Pick<
  AppServerClient,
  | "execCommand"
  | "writeCommandExec"
  | "resizeCommandExec"
  | "terminateCommandExec"
>;

export type CommandExecOutputDelta =
  AppServerCommandExecOutputDeltaNotification;

export async function execCommand(
  params: AppServerCommandExecParams,
  client: CommandExecClient = new AppServerClient(),
): Promise<AppServerCommandExecResponse> {
  return (await client.execCommand(validateExecParams(params))).result;
}

export async function writeCommandExec(
  params: AppServerCommandExecWriteParams,
  client: CommandExecClient = new AppServerClient(),
): Promise<AppServerCommandExecWriteResponse> {
  return (await client.writeCommandExec(requiredProcessId(params))).result;
}

export async function resizeCommandExec(
  params: AppServerCommandExecResizeParams,
  client: CommandExecClient = new AppServerClient(),
): Promise<AppServerCommandExecResizeResponse> {
  requiredProcessId(params);
  if (params.size.rows <= 0 || params.size.cols <= 0) {
    throw new Error("command/exec terminal size must be greater than 0");
  }
  return (await client.resizeCommandExec(params)).result;
}

export async function terminateCommandExec(
  params: AppServerCommandExecTerminateParams,
  client: CommandExecClient = new AppServerClient(),
): Promise<AppServerCommandExecTerminateResponse> {
  return (await client.terminateCommandExec(requiredProcessId(params))).result;
}

export function subscribeCommandExecOutput(
  processId: string,
  handler: (delta: CommandExecOutputDelta) => void,
): () => void {
  const normalizedProcessId = processId.trim();
  if (!normalizedProcessId) {
    throw new Error("command/exec processId is required");
  }
  return subscribeAppServerNotifications({
    onNotifications(notifications) {
      for (const notification of notifications) {
        const delta = commandExecOutputDeltaServerNotification(notification);
        if (delta?.params.processId === normalizedProcessId) {
          handler(delta.params);
        }
      }
    },
    getDrainOptions: () => ({
      activeIntervalMs: 25,
      intervalMs: 250,
      limit: 100,
    }),
  });
}

function validateExecParams(
  params: AppServerCommandExecParams,
): AppServerCommandExecParams {
  if (!Array.isArray(params.command) || params.command.length === 0) {
    throw new Error("command/exec command must not be empty");
  }
  if (params.cwd && !isAbsolutePath(params.cwd)) {
    throw new Error("command/exec cwd must be an absolute path");
  }
  if (
    params.size &&
    (!params.tty || params.size.rows <= 0 || params.size.cols <= 0)
  ) {
    throw new Error("command/exec size requires a positive tty size");
  }
  return params;
}

function requiredProcessId<T extends { processId: string }>(params: T): T {
  if (!params.processId.trim()) {
    throw new Error("command/exec processId is required");
  }
  return params;
}

function isAbsolutePath(value: string): boolean {
  return value.startsWith("/") || /^[A-Za-z]:[\\/]/.test(value);
}
