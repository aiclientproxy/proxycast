/**
 * @file Deep Link 事件处理 Hook
 * @description 监听 Deep Link URL，管理 Connect 弹窗状态
 * @module hooks/useDeepLink
 *
 * _Requirements: 5.1, 5.2, 5.3, 5.4, 7.1, 7.4_
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import {
  resolveConnectDeepLink,
  resolveOpenDeepLink,
  saveConnectRelayApiKey,
  type ConnectPayload,
  type DeepLinkResult,
  type RelayInfo,
} from "@/lib/api/connect";
import { getCurrent, onOpenUrl } from "@/lib/desktop-host/deep-link";
import { hasDesktopHostInvokeCapability } from "@/lib/desktop-runtime";
import {
  completeOemCloudDesktopOAuthLogin,
  parseOemCloudDesktopOAuthCallbackUrl,
} from "@/lib/oemCloudDesktopAuth";
import {
  dispatchOemCloudPaymentReturn,
  parseOemCloudPaymentReturnUrl,
} from "@/lib/oemCloudPaymentReturn";
import {
  claimStoredOemCloudReferralInvite,
  handleOemCloudReferralInviteUrl,
  type OemCloudReferralClaimResult,
} from "@/lib/oemCloudReferralClaim";
import { resolveOemCloudRuntimeContext } from "@/lib/api/oemCloudRuntime";
import { resolveOemLimeHubProviderName } from "@/lib/oemLimeHubProvider";
import {
  showDeepLinkError,
  showApiKeySaveError,
} from "@/lib/utils/connectError";
import { toast } from "sonner";
import { useConnectCallback } from "./useConnectCallback";

export type {
  ConnectPayload,
  DeepLinkResult,
  RelayInfo,
} from "@/lib/api/connect";

/**
 * Connect 错误
 */
export interface ConnectError {
  code: string;
  message: string;
}

/**
 * useDeepLink Hook 返回值
 */
interface UseDeepLinkReturn {
  /** 解析后的 Deep Link payload */
  connectPayload: ConnectPayload | null;
  /** 中转商信息（如果在注册表中找到） */
  relayInfo: RelayInfo | null;
  /** 是否为已验证的中转商 */
  isVerified: boolean;
  /** 弹窗是否打开 */
  isDialogOpen: boolean;
  /** 是否正在保存 */
  isSaving: boolean;
  /** 错误信息 */
  error: ConnectError | null;
  /** 确认添加 API Key */
  handleConfirm: () => Promise<void>;
  /** 取消添加 */
  handleCancel: () => void;
  /** 清除错误 */
  clearError: () => void;
}

interface UseDeepLinkOptions {
  onOpenWebsiteDeepLink?: (payload: OpenDeepLinkPayload) => void;
}

export interface OpenDeepLinkPayload {
  kind: "skill" | "prompt";
  slug: string;
  source?: string | null;
  version?: string | null;
  action?: "open" | "install" | null;
}

interface OpenDeepLinkResult {
  payload: OpenDeepLinkPayload;
}

/**
 * Deep Link 事件处理 Hook
 *
 * 监听 Electron Desktop Host deep link URL，管理 Connect 弹窗状态。
 *
 * ## 功能
 *
 * - 监听 Electron Desktop Host deep link URL（Requirements 5.1）
 * - 触发 Connect_Dialog 打开（Requirements 5.2）
 * - 提供解析后的 Deep Link 参数（Requirements 5.3）
 * - 关闭时清理临时状态（Requirements 5.4）
 *
 * ## 使用示例
 *
 * ```tsx
 * function App() {
 *   const {
 *     connectPayload,
 *     relayInfo,
 *     isDialogOpen,
 *     handleConfirm,
 *     handleCancel,
 *   } = useDeepLink();
 *
 *   return (
 *     <ConnectConfirmDialog
 *       open={isDialogOpen}
 *       relay={relayInfo}
 *       apiKey={connectPayload?.key ?? ''}
 *       keyName={connectPayload?.name}
 *       onConfirm={handleConfirm}
 *       onCancel={handleCancel}
 *     />
 *   );
 * }
 * ```
 *
 * @returns Hook 返回值
 */
export function useDeepLink(options?: UseDeepLinkOptions): UseDeepLinkReturn {
  const { t } = useTranslation("common");
  const onOpenWebsiteDeepLink = options?.onOpenWebsiteDeepLink;

  // 状态
  const [connectPayload, setConnectPayload] = useState<ConnectPayload | null>(
    null,
  );
  const [relayInfo, setRelayInfo] = useState<RelayInfo | null>(null);
  const [isVerified, setIsVerified] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<ConnectError | null>(null);

  const handledUrlSetRef = useRef<Set<string>>(new Set());

  // 统计回调 hook
  const { sendSuccessCallback, sendCancelledCallback, sendErrorCallback } =
    useConnectCallback();

  /**
   * 处理 Connect Deep Link 解析结果
   * _Requirements: 5.1, 5.2, 5.3_
   */
  const handleDeepLinkEvent = useCallback((result: DeepLinkResult) => {
    console.log("[useDeepLink] 收到 Connect Deep Link 解析结果:", result);

    // 设置状态
    setConnectPayload(result.payload);
    setRelayInfo(result.relay_info);
    setIsVerified(result.is_verified);
    setError(null);

    // 打开弹窗
    // _Requirements: 5.2_
    setIsDialogOpen(true);
  }, []);

  const showReferralClaimResult = useCallback(
    (result: OemCloudReferralClaimResult) => {
      if (result.status === "claimed") {
        toast.success(t("common.deepLink.referral.claimed.title"), {
          description: t("common.deepLink.referral.claimed.description"),
        });
        return;
      }

      if (result.status === "pending_login") {
        toast.info(t("common.deepLink.referral.saved.title"), {
          description: t("common.deepLink.referral.saved.description"),
        });
        return;
      }

      if (result.status === "tenant_mismatch") {
        toast.error(t("common.deepLink.referral.tenantMismatch.title"), {
          description: t("common.deepLink.referral.tenantMismatch.description"),
        });
      }
    },
    [t],
  );

  const tryClaimStoredReferralInvite = useCallback(async () => {
    try {
      const result = await claimStoredOemCloudReferralInvite();
      showReferralClaimResult(result);
    } catch (error) {
      const message =
        error instanceof Error && error.message.trim()
          ? error.message.trim()
          : t("common.deepLink.referral.claimFailed.fallback");
      toast.error(t("common.deepLink.referral.claimFailed.title"), {
        description: message,
      });
    }
  }, [showReferralClaimResult, t]);

  const handleOauthCallbackUrl = useCallback(
    async (url: string) => {
      const payload = parseOemCloudDesktopOAuthCallbackUrl(url);
      if (!payload) {
        return false;
      }

      if (payload.error) {
        toast.error(t("common.deepLink.oauth.incomplete.title"), {
          description: payload.error,
        });
        return true;
      }

      try {
        await completeOemCloudDesktopOAuthLogin(payload);
        const providerName = resolveOemLimeHubProviderName(
          resolveOemCloudRuntimeContext(),
        );
        toast.success(t("common.deepLink.oauth.success.title"), {
          description: t("common.deepLink.oauth.success.description", {
            providerName,
          }),
        });
        await tryClaimStoredReferralInvite();
      } catch (error) {
        const message =
          error instanceof Error && error.message.trim()
            ? error.message.trim()
            : t("common.deepLink.oauth.syncFailed.fallback");
        toast.error(t("common.deepLink.oauth.failed.title"), {
          description: message,
        });
      }

      return true;
    },
    [tryClaimStoredReferralInvite, t],
  );

  const handleOpenDeepLinkEvent = useCallback(
    (result: OpenDeepLinkResult) => {
      console.log("[useDeepLink] 收到 lime://open Deep Link:", result);
      onOpenWebsiteDeepLink?.(result.payload);
    },
    [onOpenWebsiteDeepLink],
  );

  const processDeepLinkUrl = useCallback(
    async (url: string) => {
      const normalizedUrl = String(url || "").trim();
      if (!normalizedUrl) {
        return;
      }

      if (handledUrlSetRef.current.has(normalizedUrl)) {
        return;
      }
      handledUrlSetRef.current.add(normalizedUrl);

      if (handledUrlSetRef.current.size > 32) {
        const oldestHandledUrl = handledUrlSetRef.current.values().next().value;
        if (oldestHandledUrl) {
          handledUrlSetRef.current.delete(oldestHandledUrl);
        }
      }

      if (await handleOauthCallbackUrl(normalizedUrl)) {
        return;
      }

      try {
        const referralClaimResult =
          await handleOemCloudReferralInviteUrl(normalizedUrl);
        if (referralClaimResult.status !== "ignored") {
          showReferralClaimResult(referralClaimResult);
          return;
        }
      } catch (error) {
        const message =
          error instanceof Error && error.message.trim()
            ? error.message.trim()
            : t("common.deepLink.referral.claimFailed.fallback");
        toast.error(t("common.deepLink.referral.claimFailed.title"), {
          description: message,
        });
        return;
      }

      const paymentReturn = parseOemCloudPaymentReturnUrl(normalizedUrl);
      if (paymentReturn) {
        dispatchOemCloudPaymentReturn(paymentReturn);
        toast.success(t("common.deepLink.payment.returned.title"), {
          description: t("common.deepLink.payment.returned.description"),
        });
        return;
      }

      if (normalizedUrl.startsWith("lime://open")) {
        try {
          const result = await resolveOpenDeepLink(normalizedUrl);
          handleOpenDeepLinkEvent(result);
        } catch (err) {
          console.error("[useDeepLink] 处理官网 Deep Link 失败:", err);
          const connectError = err as ConnectError;
          showDeepLinkError(connectError.message, connectError.code);
        }
        return;
      }

      if (!normalizedUrl.startsWith("lime://connect")) {
        return;
      }

      try {
        const result = await resolveConnectDeepLink(normalizedUrl);
        handleDeepLinkEvent(result);
      } catch (err) {
        console.error("[useDeepLink] 处理 Deep Link 失败:", err);
        const connectError = err as ConnectError;
        showDeepLinkError(connectError.message, connectError.code);
      }
    },
    [
      handleDeepLinkEvent,
      handleOpenDeepLinkEvent,
      handleOauthCallbackUrl,
      showReferralClaimResult,
      t,
    ],
  );

  /**
   * 确认添加 API Key
   * _Requirements: 5.4, 5.3 (统计回调)_
   */
  const handleConfirm = useCallback(async () => {
    if (!connectPayload) {
      console.warn("[useDeepLink] handleConfirm: connectPayload 为空");
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      // 调用后端保存 API Key（添加到 API Key Provider 系统）
      const result = await saveConnectRelayApiKey({
        relayId: connectPayload.relay,
        apiKey: connectPayload.key,
        name: connectPayload.name,
      });

      console.log(
        "[useDeepLink] API Key 保存成功，Provider ID:",
        result.provider_id,
        "Key ID:",
        result.key_id,
      );

      // 发送成功回调（异步，不阻塞）
      // _Requirements: 5.3_
      sendSuccessCallback(
        connectPayload.relay,
        connectPayload.key,
        connectPayload.ref_code,
      );

      // 关闭弹窗并清理状态
      // _Requirements: 5.4_
      setIsDialogOpen(false);
      setConnectPayload(null);
      setRelayInfo(null);
      setIsVerified(false);
    } catch (err) {
      console.error("[useDeepLink] 保存 API Key 失败:", err);
      // 设置错误，但不关闭弹窗
      // _Requirements: 7.4_
      const connectError = err as ConnectError;
      setError(connectError);
      // 显示 Toast 错误提示
      showApiKeySaveError(connectError.message);

      // 发送错误回调
      // _Requirements: 5.3_
      sendErrorCallback(
        connectPayload.relay,
        connectPayload.key,
        connectError.code,
        connectError.message,
        connectPayload.ref_code,
      );
    } finally {
      setIsSaving(false);
    }
  }, [connectPayload, sendSuccessCallback, sendErrorCallback]);

  /**
   * 取消添加
   * _Requirements: 5.4, 5.3 (统计回调)_
   */
  const handleCancel = useCallback(() => {
    console.log("[useDeepLink] 用户取消添加");

    // 发送取消回调（异步，不阻塞）
    // _Requirements: 5.3_
    if (connectPayload) {
      sendCancelledCallback(
        connectPayload.relay,
        connectPayload.key,
        connectPayload.ref_code,
      );
    }

    // 关闭弹窗并清理状态
    // _Requirements: 5.4_
    setIsDialogOpen(false);
    setConnectPayload(null);
    setRelayInfo(null);
    setIsVerified(false);
    setError(null);
  }, [connectPayload, sendCancelledCallback]);

  /**
   * 清除错误
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // 监听 Electron Desktop Host Deep Link URL
  // _Requirements: 5.1, 7.1_
  useEffect(() => {
    // 浏览器开发模式下 deep-link 事件不属于当前聊天主链，
    // 跳过这些 SSE 监听以避免占满 DevBridge 连接槽位。
    if (!hasDesktopHostInvokeCapability()) {
      return;
    }

    let mounted = true;
    let unlistenDeepLink: (() => void) | null = null;

    const setupListener = async () => {
      try {
        // 监听 Electron Desktop Host 派发的 Deep Link URL 事件。
        // _Requirements: 5.1_
        unlistenDeepLink = await onOpenUrl(async (urls) => {
          if (!mounted) return;

          console.log("[useDeepLink] 收到 Deep Link URL:", urls);

          for (const url of urls) {
            await processDeepLinkUrl(url);
          }
        });

        if (mounted) {
          console.log("[useDeepLink] 已注册 Electron Deep Link URL 监听器");

          const currentUrls = await getCurrent();
          if (mounted && Array.isArray(currentUrls) && currentUrls.length > 0) {
            console.log("[useDeepLink] 读取启动时 Deep Link URL:", currentUrls);
            for (const url of currentUrls) {
              await processDeepLinkUrl(url);
            }
          }
          await tryClaimStoredReferralInvite();
        } else {
          // 如果组件已卸载，立即取消监听
          if (unlistenDeepLink) unlistenDeepLink();
        }
      } catch (err) {
        console.error("[useDeepLink] 注册监听器失败:", err);
      }
    };

    setupListener();

    // 清理函数
    return () => {
      mounted = false;
      if (unlistenDeepLink) {
        unlistenDeepLink();
      }
      console.log("[useDeepLink] 已取消 Deep Link 监听器");
    };
  }, [
    handleDeepLinkEvent,
    handleOauthCallbackUrl,
    processDeepLinkUrl,
    tryClaimStoredReferralInvite,
    onOpenWebsiteDeepLink,
  ]);

  return {
    connectPayload,
    relayInfo,
    isVerified,
    isDialogOpen,
    isSaving,
    error,
    handleConfirm,
    handleCancel,
    clearError,
  };
}
