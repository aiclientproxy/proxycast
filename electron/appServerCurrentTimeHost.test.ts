import { describe, expect, it, vi } from "vitest";
import type {
  AppServerConnection,
  JsonRpcMessage,
} from "@limecloud/app-server-client";
import { tryHandleCurrentTimeRead } from "./appServerCurrentTimeHost";

type CurrentTimeConnection = Pick<
  AppServerConnection,
  "respondServerRequest" | "rejectServerRequest"
>;

function connection(): CurrentTimeConnection {
  return {
    respondServerRequest: vi.fn(),
    rejectServerRequest: vi.fn(),
  } as unknown as CurrentTimeConnection;
}

describe("tryHandleCurrentTimeRead", () => {
  it("returns whole Unix seconds through the exact request id", () => {
    for (const id of ["clock-1", 17]) {
      const host = connection();
      expect(
        tryHandleCurrentTimeRead(
          host,
          {
            id,
            method: "currentTime/read",
            params: { threadId: "thread-clock" },
          },
          () => 1_783_860_000_123,
        ),
      ).toBe(true);
      expect(host.respondServerRequest).toHaveBeenCalledWith(id, {
        currentTimeAt: 1_783_860_000,
      });
      expect(host.rejectServerRequest).not.toHaveBeenCalled();
    }
  });

  it("does not consume unrelated server requests", () => {
    const host = connection();
    const message: JsonRpcMessage = {
      id: "approval-1",
      method: "item/commandExecution/requestApproval",
      params: {},
    };

    expect(tryHandleCurrentTimeRead(host, message)).toBe(false);
    expect(host.respondServerRequest).not.toHaveBeenCalled();
    expect(host.rejectServerRequest).not.toHaveBeenCalled();
  });

  it.each([
    undefined,
    null,
    {},
    { threadId: "" },
    { threadId: "   " },
    { threadId: 7 },
  ])("rejects invalid params without forwarding them", (params) => {
    const host = connection();

    expect(
      tryHandleCurrentTimeRead(host, {
        id: "clock-invalid",
        method: "currentTime/read",
        ...(params === undefined ? {} : { params }),
      }),
    ).toBe(true);
    expect(host.rejectServerRequest).toHaveBeenCalledWith(
      "clock-invalid",
      expect.objectContaining({ code: -32602 }),
    );
    expect(host.respondServerRequest).not.toHaveBeenCalled();
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_VALUE])(
    "rejects an unsupported host clock value",
    (clock) => {
      const host = connection();

      expect(
        tryHandleCurrentTimeRead(
          host,
          {
            id: "clock-invalid-range",
            method: "currentTime/read",
            params: { threadId: "thread-clock" },
          },
          () => clock,
        ),
      ).toBe(true);
      expect(host.rejectServerRequest).toHaveBeenCalledWith(
        "clock-invalid-range",
        expect.objectContaining({ code: -32000 }),
      );
      expect(host.respondServerRequest).not.toHaveBeenCalled();
    },
  );
});
