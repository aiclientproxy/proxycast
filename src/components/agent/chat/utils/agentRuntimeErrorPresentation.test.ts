import { describe, expect, it } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { MODEL_INPUT_CAPABILITY_GAP_ERROR_PREFIX } from "@/lib/model/modelCapabilitySendGate";

import {
  MODEL_SELECTION_REQUIRED_ERROR_MESSAGE,
  resolveAgentRuntimeErrorPresentation,
  resolveAgentRuntimeSubmitErrorMessage,
} from "./agentRuntimeErrorPresentation";

describe("agentRuntimeErrorPresentation", () => {
  it("普通错误应保留原始文案", () => {
    expect(resolveAgentRuntimeErrorPresentation("模型执行失败")).toEqual({
      displayMessage: "模型执行失败",
      toastMessage: "响应错误: 模型执行失败",
    });
  });

  it("鉴权失效错误应转换为友好提示", () => {
    expect(
      resolveAgentRuntimeErrorPresentation(
        "Request failed: 401 unauthorized, token expired",
      ),
    ).toEqual({
      displayMessage:
        "当前 Provider 鉴权未通过，请前往设置 -> AI 服务商检查 API Key、Base URL 或授权配置后重试。",
      toastMessage:
        "当前 Provider 鉴权未通过，请前往设置 -> AI 服务商检查 API Key、Base URL 或授权配置后重试。",
    });
  });

  it("SendMessage schema 错误应转换为模型通道配置提示", () => {
    expect(
      resolveAgentRuntimeErrorPresentation(
        "Agent provider execution failed: Request failed: Bad request (400): Invalid schema for function 'SendMessage': In context=('properties', 'message', 'oneOf', '2'), array schema missing items",
      ),
    ).toEqual({
      displayMessage:
        "当前模型通道返回了不兼容的工具 schema，请前往设置 -> AI 服务商检查 Provider 配置或切换模型后重试。",
      toastMessage:
        "当前模型通道返回了不兼容的工具 schema，请前往设置 -> AI 服务商检查 Provider 配置或切换模型后重试。",
    });
  });

  it("402 与余额不足错误应转换为额度提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "Agent provider execution failed: Request failed with status 402 Payment Required: Insufficient Balance",
      ),
    ).toEqual({
      displayMessage:
        "当前模型通道返回了计费或额度类错误，请检查该 Provider/模型通道的计费、配额或授权状态，或切换到其他可用模型后重试。",
      toastMessage:
        "当前模型通道返回了计费或额度类错误，请检查该 Provider/模型通道的计费、配额或授权状态，或切换到其他可用模型后重试。",
    });
  });

  it("503 与 Service Unavailable 应转换为模型通道暂不可用提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "Agent provider execution failed: Server error: Server error (503 Service Unavailable): Service temporarily unavailable",
      ),
    ).toEqual({
      displayMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
      toastMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
    });
  });

  it("Provider 404 NotFound 应转换为模型通道暂不可用提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        'execution backend error: Agent provider execution failed: Request failed: Resource not found (404): ***.NotFoundError: NotFoundError: OpenAIException - {"detail":"Not Found"}',
      ),
    ).toEqual({
      displayMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
      toastMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
    });
  });

  it("JSON-RPC 内部错误应转换为短提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "-32603: -32002: runtime error\n\nTroubleshooting: inspect provider logs",
      ),
    ).toEqual({
      displayMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
      toastMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
    });
  });

  it("运行时连接失败应转换为短提示并隐藏内部 host", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "Request failed: failed to connect to token-plan-cn.xiaomimimo.com",
      ),
    ).toEqual({
      displayMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
      toastMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
    });
  });

  it("运行时工具生命周期诊断应转换为短提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "execution backend error: agent runtime tool lifecycle validation failed: tool_args_without_start event_id=evt_1 tool_call_id=call_1",
      ),
    ).toEqual({
      displayMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
      toastMessage:
        "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。",
    });
  });

  it("模型输入模态 capability gap 应转换为本地化提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        `${MODEL_INPUT_CAPABILITY_GAP_ERROR_PREFIX}:missing_input_modalities:image`,
      ),
    ).toEqual({
      displayMessage:
        "当前模型不支持本次输入的媒体类型，请切换到支持图片或文件输入的模型后再发送。",
      toastMessage:
        "当前模型不支持本次输入的媒体类型，请切换到支持图片或文件输入的模型后再发送。",
    });
  });

  it("缺少完整模型选择时应转换为可恢复提示", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        MODEL_SELECTION_REQUIRED_ERROR_MESSAGE,
      ),
    ).toEqual({
      displayMessage:
        "请先在输入框底部选择可用模型，或完成服务商登录后再发送。",
      toastMessage: "请先在输入框底部选择可用模型，或完成服务商登录后再发送。",
    });

    expect(
      resolveAgentRuntimeErrorPresentation(
        "turn/start failed: App Server runtime backend requires provider/model selection.",
      ),
    ).toEqual({
      displayMessage:
        "请先在输入框底部选择可用模型，或完成服务商登录后再发送。",
      toastMessage: "请先在输入框底部选择可用模型，或完成服务商登录后再发送。",
    });
  });

  it("模型路由不可执行时应隐藏内部路由错误", async () => {
    await changeLimeLocale("zh-CN");

    expect(
      resolveAgentRuntimeErrorPresentation(
        "runtime model route is not executable",
      ),
    ).toEqual({
      displayMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
      toastMessage:
        "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
    });
  });

  it("提交失败文案统一使用发送失败前缀", () => {
    expect(resolveAgentRuntimeSubmitErrorMessage("network down")).toBe(
      "发送失败: network down",
    );
    expect(
      resolveAgentRuntimeSubmitErrorMessage(
        "runtime model route is not executable",
      ),
    ).toBe(
      "当前模型通道暂时不可用，请稍后重试；如果持续失败，请检查 Provider 状态或切换到其他可用模型。",
    );
  });

  it("普通错误里包含 402 字符串时不应误判为额度不足", () => {
    expect(
      resolveAgentRuntimeErrorPresentation(
        "requestKey 2026042402 failed: 模型通道暂时不可用",
      ),
    ).toEqual({
      displayMessage: "requestKey 2026042402 failed: 模型通道暂时不可用",
      toastMessage:
        "响应错误: requestKey 2026042402 failed: 模型通道暂时不可用",
    });
  });
});
