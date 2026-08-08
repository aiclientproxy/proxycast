import type { OemCloudBootstrapResponse } from "./oemCloudControlPlaneTypes";
import {
  parseAuthPolicy,
  parseCurrentSession,
  parseFeatureFlags,
  parseGatewayConfig,
  parseProviderOfferSummary,
  parseProviderPreference,
} from "./oemCloudControlPlaneCoreParsers";
import { parseReferralDashboard } from "./oemCloudControlPlaneReferralParsers";
import {
  OemCloudControlPlaneError,
  isRecord,
  normalizeStringArray,
  normalizeText,
} from "./oemCloudControlPlaneRuntime";

export function parseBootstrap(value: unknown): OemCloudBootstrapResponse {
  if (!isRecord(value)) {
    throw new OemCloudControlPlaneError("bootstrap 格式非法");
  }

  return {
    session: parseCurrentSession(value.session),
    app: {
      id:
        normalizeText(value.app && isRecord(value.app) ? value.app.id : "") ??
        "",
      key:
        normalizeText(value.app && isRecord(value.app) ? value.app.key : "") ??
        "",
      name:
        normalizeText(value.app && isRecord(value.app) ? value.app.name : "") ??
        "",
      slug:
        normalizeText(value.app && isRecord(value.app) ? value.app.slug : "") ??
        "",
      category:
        normalizeText(
          value.app && isRecord(value.app) ? value.app.category : "",
        ) ?? "",
      description:
        normalizeText(
          value.app && isRecord(value.app) ? value.app.description : "",
        ) ?? undefined,
      status:
        normalizeText(
          value.app && isRecord(value.app) ? value.app.status : "",
        ) ?? "",
      distributionChannels: normalizeStringArray(
        value.app && isRecord(value.app) ? value.app.distributionChannels : [],
      ),
    },
    authPolicy: parseAuthPolicy(value.authPolicy),
    providerOffersSummary: Array.isArray(value.providerOffersSummary)
      ? value.providerOffersSummary.map(parseProviderOfferSummary)
      : [],
    providerPreference: parseProviderPreference(value.providerPreference),
    skillCatalog: value.skillCatalog,
    serviceSkillCatalog: value.serviceSkillCatalog,
    siteAdapterCatalog: value.siteAdapterCatalog ?? value.site_adapter_catalog,
    sceneCatalog: Array.isArray(value.sceneCatalog)
      ? value.sceneCatalog
          .filter((item) => isRecord(item) && normalizeText(item.id))
          .map((item) => ({
            id: normalizeText((item as Record<string, unknown>).id) ?? "",
          }))
      : [],
    features: parseFeatureFlags(value.features),
    gateway: isRecord(value.gateway)
      ? parseGatewayConfig(value.gateway)
      : undefined,
    referral: isRecord(value.referral)
      ? parseReferralDashboard(value.referral)
      : null,
  };
}
