import { parseBootstrap } from "./oemCloudControlPlaneBootstrapParsers";
import {
  parseAccessToken,
  parseActiveAccessTokenResponse,
  parseBillingDashboard,
  parseCloudActivation,
  parseCreateAccessTokenResponse,
  parseCreditAccount,
  parseCreditsDashboard,
  parseCreditTopupOrder,
  parseEntitlementPlan,
  parseOrder,
  parsePaymentCheckoutResponse,
  parsePaymentConfig,
  parseRotateAccessTokenResponse,
  parseSubscription,
  parseTopupPackage,
  parseUsageDashboard,
} from "./oemCloudControlPlaneBillingParsers";
import {
  parseCurrentSession,
  parseDesktopAuthSessionStartResponse,
  parseDesktopAuthSessionStatusResponse,
  parseProviderModelItem,
  parseProviderOfferDetail,
  parseProviderOfferSummary,
  parseProviderPreference,
  parsePublicAuthCatalog,
  parseSceneSkillPreference,
} from "./oemCloudControlPlaneCoreParsers";
import {
  ensureRuntime,
  requestControlPlane,
} from "./oemCloudControlPlaneRuntime";
import {
  parseReferralClaimResponse,
  parseReferralDashboard,
} from "./oemCloudControlPlaneReferralParsers";
import type {
  ClaimClientReferralPayload,
  ClientPasswordLoginPayload,
  CreateClientAccessTokenPayload,
  CreateClientCreditTopupOrderPayload,
  CreateClientDesktopAuthSessionPayload,
  CreateClientOrderPayload,
  CreatePaymentCheckoutPayload,
  OemCloudAccessToken,
  OemCloudActivationResponse,
  OemCloudActiveAccessTokenResponse,
  OemCloudBillingDashboard,
  OemCloudBootstrapResponse,
  OemCloudCreateAccessTokenResponse,
  OemCloudCreditAccount,
  OemCloudCreditTopupOrder,
  OemCloudCreditsDashboard,
  OemCloudCurrentSession,
  OemCloudDesktopAuthSessionStartResponse,
  OemCloudDesktopAuthSessionStatusResponse,
  OemCloudEntitlementPlan,
  OemCloudOrder,
  OemCloudPaymentCheckoutResponse,
  OemCloudPaymentConfig,
  OemCloudProviderModelItem,
  OemCloudProviderOfferDetail,
  OemCloudProviderOfferSummary,
  OemCloudProviderPreference,
  OemCloudPublicAuthCatalog,
  OemCloudReferralClaimResponse,
  OemCloudReferralDashboard,
  OemCloudRotateAccessTokenResponse,
  OemCloudSceneSkillPreference,
  OemCloudSubscription,
  OemCloudTopupPackage,
  OemCloudUsageDashboard,
  SendAuthEmailCodeResponse,
  SendClientAuthEmailCodePayload,
  UpdateClientProviderPreferencePayload,
  UpdateClientSceneSkillPreferencePayload,
  VerifyClientAuthEmailCodePayload,
} from "./oemCloudControlPlaneTypes";

export { OemCloudControlPlaneError } from "./oemCloudControlPlaneRuntime";
export type * from "./oemCloudControlPlaneTypes";

export function getConfiguredOemCloudTarget() {
  const runtime = ensureRuntime();
  return {
    baseUrl: runtime.baseUrl,
    tenantId: runtime.tenantId,
  };
}

export async function sendClientAuthEmailCode(
  tenantId: string,
  payload: SendClientAuthEmailCodePayload,
): Promise<SendAuthEmailCodeResponse> {
  return requestControlPlane<SendAuthEmailCodeResponse>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/auth/email-code/send`,
    {
      method: "POST",
      payload,
    },
  );
}

export async function createClientDesktopAuthSession(
  tenantId: string,
  payload: CreateClientDesktopAuthSessionPayload,
): Promise<OemCloudDesktopAuthSessionStartResponse> {
  return parseDesktopAuthSessionStartResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/desktop/auth-sessions`,
      {
        method: "POST",
        payload,
      },
    ),
  );
}

export async function pollClientDesktopAuthSession(
  deviceCode: string,
): Promise<OemCloudDesktopAuthSessionStatusResponse> {
  return parseDesktopAuthSessionStatusResponse(
    await requestControlPlane<unknown>(
      `/v1/public/desktop/auth-sessions/${encodeURIComponent(deviceCode)}/poll`,
      {
        method: "POST",
      },
    ),
  );
}

export async function getPublicAuthCatalog(
  tenantId: string,
): Promise<OemCloudPublicAuthCatalog> {
  return parsePublicAuthCatalog(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/auth-catalog`,
    ),
  );
}

export async function verifyClientAuthEmailCode(
  tenantId: string,
  payload: VerifyClientAuthEmailCodePayload,
): Promise<OemCloudCurrentSession> {
  return parseCurrentSession(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/auth/email-code/verify`,
      {
        method: "POST",
        payload,
      },
    ),
  );
}

export async function loginClientByPassword(
  tenantId: string,
  payload: ClientPasswordLoginPayload,
): Promise<OemCloudCurrentSession> {
  return parseCurrentSession(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/auth/password/login`,
      {
        method: "POST",
        payload,
      },
    ),
  );
}

export async function getClientBootstrap(
  tenantId: string,
): Promise<OemCloudBootstrapResponse> {
  return parseBootstrap(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/bootstrap`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientSession(
  tenantId: string,
): Promise<OemCloudCurrentSession> {
  return parseCurrentSession(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/session`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientSceneSkillPreferences(
  tenantId: string,
): Promise<OemCloudSceneSkillPreference> {
  return parseSceneSkillPreference(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/scene-skill-preferences`,
      {
        auth: true,
      },
    ),
  );
}

export async function updateClientSceneSkillPreferences(
  tenantId: string,
  payload: UpdateClientSceneSkillPreferencePayload,
): Promise<OemCloudSceneSkillPreference> {
  return parseSceneSkillPreference(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/scene-skill-preferences`,
      {
        method: "PUT",
        payload,
        auth: true,
      },
    ),
  );
}

export async function listClientProviderOffers(
  tenantId: string,
): Promise<OemCloudProviderOfferSummary[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/provider-offers`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parseProviderOfferSummary)
    : [];
}

export async function getClientProviderOffer(
  tenantId: string,
  providerKey: string,
): Promise<OemCloudProviderOfferDetail> {
  return parseProviderOfferDetail(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/provider-offers/${encodeURIComponent(providerKey)}`,
      {
        auth: true,
      },
    ),
  );
}

export async function listClientProviderOfferModels(
  tenantId: string,
  providerKey: string,
): Promise<OemCloudProviderModelItem[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/provider-offers/${encodeURIComponent(providerKey)}/models`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parseProviderModelItem)
    : [];
}

export async function getClientProviderPreference(
  tenantId: string,
): Promise<OemCloudProviderPreference> {
  return parseProviderPreference(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/provider-preferences`,
      {
        auth: true,
      },
    ),
  );
}

export async function updateClientProviderPreference(
  tenantId: string,
  payload: UpdateClientProviderPreferencePayload,
): Promise<OemCloudProviderPreference> {
  return parseProviderPreference(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/provider-preferences`,
      {
        method: "PUT",
        payload,
        auth: true,
      },
    ),
  );
}

export async function getClientCloudActivation(
  tenantId: string,
): Promise<OemCloudActivationResponse> {
  return parseCloudActivation(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/cloud-activation`,
      {
        auth: true,
      },
    ),
  );
}

export async function listClientPaymentConfigs(
  tenantId: string,
): Promise<OemCloudPaymentConfig[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/payment-configs`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parsePaymentConfig)
    : [];
}

export async function listClientPlans(
  tenantId: string,
): Promise<OemCloudEntitlementPlan[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/plans`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parseEntitlementPlan)
    : [];
}

export async function getClientSubscription(
  tenantId: string,
): Promise<OemCloudSubscription> {
  return parseSubscription(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/subscription`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientCreditAccount(
  tenantId: string,
): Promise<OemCloudCreditAccount> {
  return parseCreditAccount(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/credits`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientCreditsDashboard(
  tenantId: string,
): Promise<OemCloudCreditsDashboard> {
  return parseCreditsDashboard(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/credits/dashboard`,
      {
        auth: true,
      },
    ),
  );
}

export async function listClientTopupPackages(
  tenantId: string,
): Promise<OemCloudTopupPackage[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/topup-packages`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parseTopupPackage)
    : [];
}

export async function getClientUsageDashboard(
  tenantId: string,
): Promise<OemCloudUsageDashboard> {
  return parseUsageDashboard(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/usage/dashboard`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientBillingDashboard(
  tenantId: string,
): Promise<OemCloudBillingDashboard> {
  return parseBillingDashboard(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/billing/dashboard`,
      {
        auth: true,
      },
    ),
  );
}

export async function getClientReferralDashboard(
  tenantId: string,
): Promise<OemCloudReferralDashboard> {
  return parseReferralDashboard(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/referral`,
      {
        auth: true,
      },
    ),
  );
}

export async function claimClientReferral(
  tenantId: string,
  payload: ClaimClientReferralPayload,
): Promise<OemCloudReferralClaimResponse> {
  return parseReferralClaimResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/referrals/claim`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function getClientActiveAccessToken(
  tenantId: string,
): Promise<OemCloudActiveAccessTokenResponse> {
  return parseActiveAccessTokenResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/access-tokens/active`,
      {
        auth: true,
      },
    ),
  );
}

export async function listClientAccessTokens(
  tenantId: string,
): Promise<OemCloudAccessToken[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/access-tokens`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items)
    ? payload.items.map(parseAccessToken)
    : [];
}

export async function createClientAccessToken(
  tenantId: string,
  payload: CreateClientAccessTokenPayload,
): Promise<OemCloudCreateAccessTokenResponse> {
  return parseCreateAccessTokenResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/access-tokens`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function rotateClientAccessToken(
  tenantId: string,
  tokenId: string,
): Promise<OemCloudRotateAccessTokenResponse> {
  return parseRotateAccessTokenResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/access-tokens/${encodeURIComponent(tokenId)}/rotate`,
      {
        method: "POST",
        auth: true,
      },
    ),
  );
}

export async function revokeClientAccessToken(
  tenantId: string,
  tokenId: string,
): Promise<OemCloudAccessToken> {
  return parseAccessToken(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/access-tokens/${encodeURIComponent(tokenId)}/revoke`,
      {
        method: "POST",
        auth: true,
      },
    ),
  );
}

export async function listClientOrders(
  tenantId: string,
): Promise<OemCloudOrder[]> {
  const payload = await requestControlPlane<{ items?: unknown[] }>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/orders`,
    {
      auth: true,
    },
  );
  return Array.isArray(payload.items) ? payload.items.map(parseOrder) : [];
}

export async function getClientOrder(
  tenantId: string,
  orderId: string,
): Promise<OemCloudOrder> {
  return parseOrder(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/orders/${encodeURIComponent(orderId)}`,
      {
        auth: true,
      },
    ),
  );
}

export async function createClientOrder(
  tenantId: string,
  payload: CreateClientOrderPayload,
): Promise<OemCloudOrder> {
  return parseOrder(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/orders`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function createClientOrderCheckout(
  tenantId: string,
  orderId: string,
  payload: CreatePaymentCheckoutPayload = {},
): Promise<OemCloudPaymentCheckoutResponse> {
  return parsePaymentCheckoutResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/orders/${encodeURIComponent(orderId)}/checkout`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function getClientCreditTopupOrder(
  tenantId: string,
  orderId: string,
): Promise<OemCloudCreditTopupOrder> {
  return parseCreditTopupOrder(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/credit-topup-orders/${encodeURIComponent(orderId)}`,
      {
        auth: true,
      },
    ),
  );
}

export async function createClientCreditTopupOrder(
  tenantId: string,
  payload: CreateClientCreditTopupOrderPayload,
): Promise<OemCloudCreditTopupOrder> {
  return parseCreditTopupOrder(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/credit-topup-orders`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function createClientCreditTopupOrderCheckout(
  tenantId: string,
  orderId: string,
  payload: CreatePaymentCheckoutPayload = {},
): Promise<OemCloudPaymentCheckoutResponse> {
  return parsePaymentCheckoutResponse(
    await requestControlPlane<unknown>(
      `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/credit-topup-orders/${encodeURIComponent(orderId)}/checkout`,
      {
        method: "POST",
        payload,
        auth: true,
      },
    ),
  );
}

export async function logoutClient(tenantId: string): Promise<void> {
  await requestControlPlane<unknown>(
    `/v1/public/tenants/${encodeURIComponent(tenantId)}/client/logout`,
    {
      method: "POST",
      auth: true,
    },
  );
}
