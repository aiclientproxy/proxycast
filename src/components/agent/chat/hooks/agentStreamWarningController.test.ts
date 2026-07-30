import { describe, expect, it, vi } from "vitest";
import { WORKSPACE_PATH_AUTO_CREATED_WARNING_CODE } from "./agentChatCoreUtils";
import {
  applyAgentStreamWarningToastAction,
  buildAgentStreamWarningPlan,
  buildAgentStreamWarningToastAction,
} from "./agentStreamWarningController";
import {
  ARTIFACT_DOCUMENT_REPAIRED_WARNING_CODE,
  SKILL_LOAD_FAILED_WARNING_CODE,
  SKILL_NOT_AVAILABLE_WARNING_CODE,
} from "./runtimeWarningPresentation";

describe("agentStreamWarningController", () => {
  it("应忽略 workspace 自动创建 warning", () => {
    expect(
      buildAgentStreamWarningPlan({
        activeSessionId: "session-a",
        alreadyWarned: false,
        code: WORKSPACE_PATH_AUTO_CREATED_WARNING_CODE,
        message: "已自动创建",
      }),
    ).toEqual({
      shouldMarkWarned: false,
      toast: null,
      warningKey: null,
    });
  });

  it("已提示过的 warning 不应重复 toast", () => {
    expect(
      buildAgentStreamWarningPlan({
        activeSessionId: "session-a",
        alreadyWarned: true,
        code: "code-a",
        message: "提醒",
      }),
    ).toEqual({
      shouldMarkWarned: false,
      toast: null,
      warningKey: "session-a:code-a",
    });
  });

  it("应为普通 warning 构造 toast plan 并标记 warned", () => {
    expect(
      buildAgentStreamWarningPlan({
        activeSessionId: "session-a",
        alreadyWarned: false,
        message: "普通提醒",
      }),
    ).toEqual({
      shouldMarkWarned: true,
      toast: {
        level: "warning",
        message: "普通提醒",
      },
      warningKey: "session-a:普通提醒",
    });
  });

  it("不需要 toast 的 warning 仍应标记 warned", () => {
    expect(
      buildAgentStreamWarningPlan({
        activeSessionId: "session-a",
        alreadyWarned: false,
        code: ARTIFACT_DOCUMENT_REPAIRED_WARNING_CODE,
      }),
    ).toEqual({
      shouldMarkWarned: true,
      toast: null,
      warningKey: `session-a:${ARTIFACT_DOCUMENT_REPAIRED_WARNING_CODE}`,
    });
  });

  it("Skill warning 应随当前界面语言呈现", () => {
    const previousLocale = document.documentElement.lang;
    document.documentElement.lang = "en-US";
    try {
      expect(
        buildAgentStreamWarningPlan({
          activeSessionId: "session-a",
          alreadyWarned: false,
          code: SKILL_NOT_AVAILABLE_WARNING_CODE,
        }).toast?.message,
      ).toBe("The selected Skill is not available.");
      expect(
        buildAgentStreamWarningPlan({
          activeSessionId: "session-a",
          alreadyWarned: false,
          code: SKILL_LOAD_FAILED_WARNING_CODE,
        }).toast?.message,
      ).toBe("The selected Skill could not be loaded.");
    } finally {
      document.documentElement.lang = previousLocale;
    }
  });

  it("协议 notification drift 应使用本地化文案并保留 method", () => {
    const previousLocale = document.documentElement.lang;
    document.documentElement.lang = "en-US";
    try {
      expect(
        buildAgentStreamWarningPlan({
          activeSessionId: "session-a",
          alreadyWarned: false,
          code: "unknown_app_server_notification:future/itemChanged",
          message: "future/itemChanged",
        }).toast?.message,
      ).toBe(
        "This version does not recognize the runtime notification future/itemChanged. A protocol diagnostic was recorded.",
      );
      expect(
        buildAgentStreamWarningPlan({
          activeSessionId: "session-a",
          alreadyWarned: false,
          code: "unprojected_app_server_notification:hook/started",
          message: "hook/started",
        }).toast?.message,
      ).toBe(
        "The runtime notification hook/started is not connected to the current interface yet. A protocol diagnostic was recorded.",
      );
    } finally {
      document.documentElement.lang = previousLocale;
    }
  });

  it("应从 warning plan 构造 toast action", () => {
    expect(
      buildAgentStreamWarningToastAction({
        level: "info",
        message: "已恢复",
      }),
    ).toEqual({
      level: "info",
      message: "已恢复",
    });
    expect(buildAgentStreamWarningToastAction(null)).toBeNull();
  });

  it("应按 toast action level 调用对应 dispatcher", () => {
    const dispatcher = {
      error: vi.fn(),
      info: vi.fn(),
      warning: vi.fn(),
    };

    applyAgentStreamWarningToastAction(
      { level: "info", message: "提示" },
      dispatcher,
    );
    applyAgentStreamWarningToastAction(
      { level: "error", message: "错误" },
      dispatcher,
    );
    applyAgentStreamWarningToastAction(
      { level: "warning", message: "警告" },
      dispatcher,
    );
    applyAgentStreamWarningToastAction(null, dispatcher);

    expect(dispatcher.info).toHaveBeenCalledWith("提示");
    expect(dispatcher.error).toHaveBeenCalledWith("错误");
    expect(dispatcher.warning).toHaveBeenCalledWith("警告");
  });
});
