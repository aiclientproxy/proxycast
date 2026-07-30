import {
  ERROR_CODES,
  isJsonRpcRequest,
  METHOD_CURRENT_TIME_READ,
  type AppServerConnection,
  type CurrentTimeReadResponse,
  type JsonRpcMessage,
} from "@limecloud/app-server-client";

type CurrentTimeConnection = Pick<
  AppServerConnection,
  "respondServerRequest" | "rejectServerRequest"
>;

export function tryHandleCurrentTimeRead(
  connection: CurrentTimeConnection,
  message: JsonRpcMessage,
  now: () => number = Date.now,
): boolean {
  if (
    !isJsonRpcRequest(message) ||
    message.method !== METHOD_CURRENT_TIME_READ
  ) {
    return false;
  }

  const params = asRecord(message.params);
  if (
    typeof params?.threadId !== "string" ||
    params.threadId.trim().length === 0
  ) {
    connection.rejectServerRequest(message.id, {
      code: ERROR_CODES.invalidParams,
      message: "currentTime/read requires a non-empty threadId",
    });
    return true;
  }

  const currentTimeAt = Math.floor(now() / 1_000);
  if (!Number.isSafeInteger(currentTimeAt)) {
    connection.rejectServerRequest(message.id, {
      code: ERROR_CODES.runtimeError,
      message: "host clock is outside the supported Unix time range",
    });
    return true;
  }

  connection.respondServerRequest<CurrentTimeReadResponse>(message.id, {
    currentTimeAt,
  });
  return true;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
