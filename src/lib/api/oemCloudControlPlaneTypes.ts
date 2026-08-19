import type { OemCloudCurrentSessionLike } from "@/lib/oemCloudSession";
import type {
  ModelAliasSource,
  ModelDeploymentSource,
  ModelManagementPlane,
  ModelModality,
  ModelRuntimeFeature,
  ModelTaskFamily,
} from "@/lib/types/modelRegistry";

export type OemCloudProviderSource = "local" | "oem_cloud";
export type OemCloudProviderOfferState =
  | "available_logged_out"
  | "available_subscribe_required"
  | "available_ready"
  | "available_quota_low"
  | "blocked"
  | "unavailable";
export type OemCloudPartnerHubAccessMode = "session" | "hub_token" | "api_key";
export type OemCloudPartnerHubConfigMode = "managed" | "hybrid" | "developer";
export type OemCloudPartnerHubModelsSource = "hub_catalog" | "manual";

export interface OemCloudTenant {
  id: string;
  name: string;
  slug: string;
}

export interface OemCloudUser {
  id: string;
  email: string;
  displayName: string;
  avatarUrl?: string;
  username?: string;
  passwordConfigured: boolean;
  roles: string[];
}

export interface OemCloudUserSession {
  id: string;
  tenantId: string;
  userId: string;
  provider: string;
  roles: string[];
  issuedAt: string;
  expiresAt: string;
}

export interface OemCloudAuthCatalogProvider {
  provider: string;
  displayName: string;
  authorizeUrl?: string;
  redirectUri?: string;
  scopes: string[];
  enabled: boolean;
  loginHint?: string;
}

export type OemCloudAuthStartupTrigger = "none" | "oauth";

export interface OemCloudAuthPolicy {
  required: boolean;
  startupTrigger: OemCloudAuthStartupTrigger;
  primaryProvider?: string;
}

export interface OemCloudPublicAuthCatalog {
  providers: OemCloudAuthCatalogProvider[];
  authPolicy: OemCloudAuthPolicy;
}

export interface OemCloudCurrentSession extends Omit<
  OemCloudCurrentSessionLike,
  "tenant" | "user" | "session"
> {
  token?: string;
  tenant: OemCloudTenant;
  user: OemCloudUser;
  session: OemCloudUserSession;
}

export interface OemCloudFeatureFlags {
  oauthLoginEnabled: boolean;
  emailCodeLoginEnabled: boolean;
  passwordLoginEnabled: boolean;
  profileEditable: boolean;
  hubTokensEnabled: boolean;
  billingEnabled: boolean;
  referralEnabled: boolean;
  gatewayEnabled: boolean;
}

export interface OemCloudProviderPreference {
  tenantId: string;
  userId: string;
  providerSource: OemCloudProviderSource;
  providerKey: string;
  defaultModel?: string;
  needsValidation: boolean;
  lastValidatedAt?: string;
  updatedAt: string;
}

export interface OemCloudSceneSkillTemplate {
  id: string;
  title: string;
  description?: string;
  prompt: string;
}

export interface OemCloudCustomScene {
  id?: string;
  title: string;
  summary?: string;
  linkedEntryId: string;
  placeholder?: string;
  templates: OemCloudSceneSkillTemplate[];
  enabled?: boolean;
}

export interface OemCloudSceneSkillPreference {
  tenantId: string;
  userId: string;
  orderedEntryIds: string[];
  hiddenEntryIds: string[];
  customScenes: OemCloudCustomScene[];
  updatedAt?: string;
}

export interface UpdateClientSceneSkillPreferencePayload {
  orderedEntryIds: string[];
  hiddenEntryIds: string[];
  customScenes: OemCloudCustomScene[];
}

export interface OemCloudProviderOfferSummary {
  providerKey: string;
  displayName: string;
  source: OemCloudProviderSource;
  state: OemCloudProviderOfferState;
  logoUrl?: string;
  description?: string;
  supportUrl?: string;
  visible: boolean;
  loggedIn: boolean;
  accountStatus: string;
  subscriptionStatus: string;
  quotaStatus: string;
  canInvoke: boolean;
  defaultModel?: string;
  effectiveAccessMode: OemCloudPartnerHubAccessMode;
  apiKeyModeEnabled: boolean;
  tenantOverrideApplied: boolean;
  configMode: OemCloudPartnerHubConfigMode;
  modelsSource: OemCloudPartnerHubModelsSource;
  developerAccessVisible: boolean;
  availableModelCount: number;
  fallbackToLocalAllowed: boolean;
  currentPlan?: string;
  creditsSummary?: string;
  statusReason?: string;
  tags?: string[];
}

export interface OemCloudProviderOfferAccess {
  offerId: string;
  accessMode: OemCloudPartnerHubAccessMode;
  sessionTokenRef?: string;
  hubTokenRef?: string;
  hubTokenEnabled: boolean;
  lastIssuedAt?: string;
}

export interface OemCloudProviderOfferDetail extends OemCloudProviderOfferSummary {
  loginHint?: string;
  subscribeHint?: string;
  unavailableHint?: string;
  access: OemCloudProviderOfferAccess;
}

export interface OemCloudProviderModelItem {
  id: string;
  offerId: string;
  modelId: string;
  displayName: string;
  description?: string;
  abilities: string[];
  task_families?: ModelTaskFamily[];
  input_modalities?: ModelModality[];
  output_modalities?: ModelModality[];
  runtime_features?: ModelRuntimeFeature[];
  deployment_source?: ModelDeploymentSource;
  management_plane?: ModelManagementPlane;
  canonical_model_id?: string;
  provider_model_id?: string;
  alias_source?: ModelAliasSource | null;
  recommended: boolean;
  status: string;
  sort: number;
  upstreamMapping?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudGatewayConfig {
  basePath?: string;
  llmBaseUrl?: string;
  openAIBaseUrl?: string;
  anthropicBaseUrl?: string;
  chatCompletionsPath?: string;
  authorizationHeader?: string;
  authorizationScheme?: string;
  tenantHeader?: string;
}

export interface OemCloudBootstrapResponse {
  session: OemCloudCurrentSession;
  app: {
    id: string;
    key: string;
    name: string;
    slug: string;
    category: string;
    description?: string;
    status: string;
    distributionChannels: string[];
  };
  authPolicy: OemCloudAuthPolicy;
  providerOffersSummary: OemCloudProviderOfferSummary[];
  providerPreference: OemCloudProviderPreference;
  skillCatalog?: unknown;
  serviceSkillCatalog?: unknown;
  sceneCatalog?: Array<{ id: string }>;
  features: OemCloudFeatureFlags;
  gateway?: OemCloudGatewayConfig;
  referral?: OemCloudReferralDashboard | null;
}

export interface SendClientAuthEmailCodePayload {
  identifier: string;
}

export interface SendAuthEmailCodeResponse {
  sent: boolean;
  maskedEmail: string;
  expiresInSeconds: number;
}

export interface VerifyClientAuthEmailCodePayload {
  identifier: string;
  code: string;
  displayName?: string;
  username?: string;
}

export interface ClientPasswordLoginPayload {
  identifier: string;
  password: string;
}

export type OemCloudDesktopAuthSessionStatus =
  | "pending_login"
  | "pending_consent"
  | "approved"
  | "denied"
  | "cancelled"
  | "consumed"
  | "expired";

export interface CreateClientDesktopAuthSessionPayload {
  clientId: string;
  provider?: string;
  desktopRedirectUri?: string;
}

export interface OemCloudDesktopAuthSessionStartResponse {
  authSessionId: string;
  deviceCode: string;
  tenantId: string;
  clientId: string;
  clientName: string;
  provider?: string;
  desktopRedirectUri?: string;
  status: OemCloudDesktopAuthSessionStatus;
  expiresInSeconds: number;
  pollIntervalSeconds: number;
  authorizeUrl: string;
}

export interface OemCloudDesktopAuthSessionStatusResponse {
  deviceCode: string;
  tenantId: string;
  clientId: string;
  clientName: string;
  provider?: string;
  desktopRedirectUri?: string;
  status: OemCloudDesktopAuthSessionStatus;
  expiresInSeconds: number;
  pollIntervalSeconds: number;
  sessionToken?: string;
  sessionExpiresAt?: string;
}

export interface UpdateClientProviderPreferencePayload {
  providerSource: OemCloudProviderSource;
  providerKey: string;
  defaultModel?: string;
}

export type OemCloudBillingCycle =
  | "monthly"
  | "yearly"
  | "one_time"
  | (string & {});

export interface OemCloudPlanBillingCycle {
  key: OemCloudBillingCycle;
  label: string;
  priceCents: number;
  credits: number;
  autoRenew: boolean;
  originalPriceCents?: number;
  discountPercent?: number;
}

export interface OemCloudPlanQuotaSummary {
  key: string;
  label: string;
  value: string;
  hint?: string;
}

export interface OemCloudPlanFeatureSection {
  key: string;
  title: string;
  description?: string;
  items: string[];
}

export interface OemCloudEntitlementPlan {
  id: string;
  tenantId: string;
  templateId?: string;
  key: string;
  name: string;
  description?: string;
  tagline?: string;
  badge?: string;
  priceMonthly: number;
  creditsMonthly: number;
  features: string[];
  status: string;
  recommended: boolean;
  sortOrder?: number;
  yearlyDiscountPercent?: number;
  oneTimeDiscountPercent?: number;
  billingCycles: OemCloudPlanBillingCycle[];
  quotaSummaries: OemCloudPlanQuotaSummary[];
  featureSections: OemCloudPlanFeatureSection[];
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudSubscription {
  id: string;
  tenantId: string;
  userId?: string;
  planId: string;
  planKey: string;
  planName?: string;
  status: string;
  billingCycle?: OemCloudBillingCycle;
  currentPeriodStart: string;
  currentPeriodEnd: string;
  renewalAt?: string;
  autoRenew: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudCreditAccount {
  tenantId: string;
  userId?: string;
  balance: number;
  reserved: number;
  currency: string;
  updatedAt: string;
  lastTopUp?: string;
  lastSource?: string;
}

export interface OemCloudPaymentConfig {
  id: string;
  tenantId: string;
  provider: string;
  displayName: string;
  merchantIdMasked?: string;
  currency?: string;
  environment?: string;
  notifyUrl: string;
  returnUrl: string;
  enabled: boolean;
  methods: OemCloudPaymentMethodConfig[];
  providerOptions: Record<string, string>;
  credentialMasks: Record<string, string>;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudPaymentMethodConfig {
  key: string;
  displayName: string;
  paymentType?: string;
  paymentName?: string;
  icon?: string;
  enabled: boolean;
}

export interface OemCloudTopupPackage {
  id: string;
  tenantId: string;
  key: string;
  name: string;
  credits: number;
  priceCents: number;
  bonusCredits?: number;
  validDays: number;
  recommended: boolean;
  sortOrder?: number;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudCreditWallet {
  id: string;
  tenantId: string;
  userId: string;
  packageId?: string;
  packageName?: string;
  sourceType: string;
  sourceId?: string;
  grantedCredits: number;
  usedCredits: number;
  remainingCredits: number;
  status: string;
  effectiveAt: string;
  expiresAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudCreditTopupOrder {
  id: string;
  tenantId: string;
  userId: string;
  packageId?: string;
  packageName: string;
  creditsGranted: number;
  amountCents: number;
  paymentChannel: string;
  paymentMethod?: string;
  paymentReference?: string;
  checkoutUrl?: string;
  providerOrderId?: string;
  providerSessionId?: string;
  providerStatus?: string;
  status: string;
  paidAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudUsageRecord {
  id: string;
  tenantId: string;
  userId: string;
  createdAt: string;
  usageType: string;
  triggerType: string;
  usageTag?: string;
  model: string;
  tokens: number;
  credits: number;
  durationMs: number;
  status: string;
}

export interface OemCloudMonthlyUsageSummary {
  freeCreditsUsed: number;
  freeCreditsLimit: number;
  topupCreditsUsed: number;
  topupCreditsLimit: number;
  subscriptionCreditsUsed: number;
  subscriptionCreditsLimit: number;
}

export interface OemCloudBillingSummary {
  currency: string;
  nextPaymentAmountCents: number;
  currentPeriodStart?: string;
  currentPeriodEnd?: string;
  renewalAt?: string;
  autoRenew: boolean;
  lastPaidAt?: string;
  totalSpentCents: number;
}

export interface OemCloudOrder {
  id: string;
  tenantId: string;
  userId: string;
  planId: string;
  planKey: string;
  planName: string;
  amountCents: number;
  creditsGranted: number;
  paymentChannel: string;
  paymentMethod?: string;
  billingCycle?: OemCloudBillingCycle;
  paymentReference?: string;
  checkoutUrl?: string;
  providerOrderId?: string;
  providerSessionId?: string;
  providerStatus?: string;
  status: string;
  paidAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudCreditsDashboard {
  creditAccount: OemCloudCreditAccount;
  subscription: OemCloudSubscription | null;
  topupPackages: OemCloudTopupPackage[];
  creditWallets: OemCloudCreditWallet[];
  creditOrders: OemCloudCreditTopupOrder[];
}

export interface OemCloudUsageDashboard {
  usageRecords: OemCloudUsageRecord[];
  monthlySummary: OemCloudMonthlyUsageSummary;
}

export interface OemCloudBillingDashboard {
  billingSummary: OemCloudBillingSummary;
  subscription: OemCloudSubscription | null;
  currentPlan: OemCloudEntitlementPlan | null;
  orders: OemCloudOrder[];
}

export interface OemCloudAccessToken {
  id: string;
  tenantId: string;
  userId: string;
  name: string;
  tokenMasked: string;
  tokenPrefix?: string;
  scopes: string[];
  allowedModels: string[];
  maxTokensPerRequest?: number;
  requestsPerMinute?: number;
  tokensPerMinute?: number;
  monthlyCreditLimit?: number;
  status: string;
  lastUsedAt?: string;
  createdAt: string;
  updatedAt: string;
  expiresAt: string;
}

export interface OemCloudActiveAccessTokenResponse {
  hasActive: boolean;
  token: OemCloudAccessToken | null;
}

export interface CreateClientAccessTokenPayload {
  name: string;
  scopes?: string[];
  allowedModels?: string[];
  maxTokensPerRequest?: number;
  requestsPerMinute?: number;
  tokensPerMinute?: number;
  monthlyCreditLimit?: number;
}

export interface OemCloudCreateAccessTokenResponse {
  token: OemCloudAccessToken;
  rawToken?: string;
  apiKey?: string;
}

export interface OemCloudRotateAccessTokenResponse {
  previousToken: OemCloudAccessToken;
  newToken: OemCloudAccessToken;
  rawToken?: string;
  apiKey?: string;
}

export interface CreateClientOrderPayload {
  planId: string;
  paymentChannel: string;
  paymentMethod?: string;
  billingCycle?: OemCloudBillingCycle;
}

export interface CreatePaymentCheckoutPayload {
  paymentMethod?: string;
  successUrl?: string;
  cancelUrl?: string;
}

export interface OemCloudPaymentCheckoutResponse {
  orderKind: string;
  orderId: string;
  paymentChannel: string;
  paymentMethod?: string;
  paymentReference?: string;
  checkoutUrl?: string;
  status: string;
}

export interface CreateClientCreditTopupOrderPayload {
  packageId?: string;
  customCredits?: number;
  paymentChannel: string;
  paymentMethod?: string;
}

export type OemCloudReadinessStatus =
  | "no_payment_channel"
  | "no_plan_or_credits"
  | "payment_pending"
  | "no_api_key"
  | "no_models"
  | "ready"
  | "quota_low"
  | "subscription_expired"
  | "blocked";

export interface OemCloudReadinessStep {
  key: string;
  label: string;
  description?: string;
  done: boolean;
  action?: string;
}

export interface OemCloudReadiness {
  status: OemCloudReadinessStatus;
  title: string;
  description?: string;
  nextAction?: string;
  canInvoke: boolean;
  blockers: string[];
  steps: OemCloudReadinessStep[];
}

export interface OemCloudPaymentAction {
  kind: "plan_order" | "credit_topup_order" | (string & {});
  orderId: string;
  title: string;
  paymentChannel: string;
  paymentReference?: string;
  amountCents: number;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudActivationResponse {
  gateway: OemCloudGatewayConfig;
  llmBaseUrl: string;
  openAIBaseUrl: string;
  anthropicBaseUrl: string;
  readiness: OemCloudReadiness;
  pendingPayment: OemCloudPaymentAction | null;
  paymentConfigs: OemCloudPaymentConfig[];
  plans: OemCloudEntitlementPlan[];
  subscription: OemCloudSubscription | null;
  creditAccount: OemCloudCreditAccount | null;
  creditsDashboard: OemCloudCreditsDashboard | null;
  topupPackages: OemCloudTopupPackage[];
  usageDashboard: OemCloudUsageDashboard | null;
  billingDashboard: OemCloudBillingDashboard | null;
  providerOffers: OemCloudProviderOfferSummary[];
  selectedOffer: OemCloudProviderOfferDetail | null;
  providerModels: OemCloudProviderModelItem[];
  providerPreference: OemCloudProviderPreference | null;
  accessTokens: OemCloudAccessToken[];
  activeAccessToken: OemCloudActiveAccessTokenResponse | null;
  orders: OemCloudOrder[];
  creditTopupOrders: OemCloudCreditTopupOrder[];
}

export interface OemCloudReferralCode {
  id: string;
  tenantId: string;
  userId: string;
  code: string;
  landingUrl: string;
  channel?: string;
  status: string;
  disabledReason?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OemCloudReferralPolicy {
  tenantId?: string;
  enabled: boolean;
  rewardCredits: number;
  referrerRewardCredits: number;
  inviteeRewardCredits: number;
  claimWindowDays: number;
  autoClaimEnabled: boolean;
  allowManualClaimFallback: boolean;
  landingPageHeadline?: string;
  landingPageRules?: string;
  riskReviewEnabled: boolean;
  updatedAt?: string;
}

export interface OemCloudReferralSummary {
  totalInvites: number;
  successfulInvites: number;
  totalRewardCredits: number;
  referrerRewardCreditsTotal: number;
  inviteeRewardCreditsTotal: number;
}

export interface OemCloudReferralInviteRelation {
  eventId?: string;
  code?: string;
  referrerUserId?: string;
  referrerEmail?: string;
  referrerName?: string;
  inviteeRewardCredits?: number;
  claimedAt?: string;
}

export interface OemCloudReferralShare {
  brandName: string;
  code: string;
  landingUrl: string;
  downloadUrl: string;
  shareText: string;
  headline?: string;
  rules?: string;
}

export interface OemCloudReferralDashboard {
  code: OemCloudReferralCode;
  policy: OemCloudReferralPolicy;
  summary: OemCloudReferralSummary;
  events: unknown[];
  rewards: unknown[];
  invitedBy: OemCloudReferralInviteRelation;
  share: OemCloudReferralShare;
}

export interface ClaimClientReferralPayload {
  code: string;
  inviteeEmail?: string;
  inviteeName?: string;
  claimMethod?: "auto" | "manual" | (string & {});
  entrySource?: "link" | "code_input" | (string & {});
  landingPath?: string;
  capturedAt?: string;
}

export interface OemCloudReferralClaimResponse {
  event?: unknown;
  reward?: unknown;
  rewards: unknown[];
  creditAccount: OemCloudCreditAccount | null;
  accountLedgers: unknown[];
}
