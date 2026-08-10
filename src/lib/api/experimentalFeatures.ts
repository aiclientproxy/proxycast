import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
  METHOD_EXPERIMENTAL_FEATURE_LIST,
  type ExperimentalFeatureEnablementSetResponse,
  type ExperimentalFeatureListResponse,
} from "@limecloud/app-server-client";
import type { ExperimentalFeatures } from "./experimentalFeatureTypes";

export type {
  WebMcpConfig,
  ExperimentalFeatures,
  ToolCallingConfig,
} from "./experimentalFeatureTypes";
export { DEFAULT_EXPERIMENTAL_FEATURES } from "./experimentalFeatureTypes";

export async function getExperimentalConfig(
  appServerClient: Pick<AppServerClient, "request"> = new AppServerClient(),
): Promise<ExperimentalFeatures> {
  const response = await appServerClient.request<ExperimentalFeatureListResponse>(
    METHOD_EXPERIMENTAL_FEATURE_LIST,
    {},
  );
  const feature = response.result.data.find(
    (candidate) => candidate.name === "webmcp",
  );
  if (!feature || typeof feature.enabled !== "boolean") {
    throw new Error(
      "App Server experimentalFeature/list did not return the webmcp feature",
    );
  }
  return { webmcp: { enabled: feature.enabled } };
}

export async function saveExperimentalConfig(
  config: ExperimentalFeatures,
  appServerClient: Pick<AppServerClient, "request"> = new AppServerClient(),
): Promise<void> {
  const response = await appServerClient.request<
    ExperimentalFeatureEnablementSetResponse
  >(METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET, {
    enablement: { webmcp: Boolean(config.webmcp?.enabled) },
  });
  const enablement = response.result.enablement as Record<string, unknown>;
  if (enablement.webmcp !== Boolean(config.webmcp?.enabled)) {
    throw new Error(
      "App Server experimentalFeature/enablement/set did not apply webmcp",
    );
  }
}
