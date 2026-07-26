export const MODEL_VISIBILITIES = [
  "list",
  "hide",
  "none",
] as const;

export type ModelVisibility = (typeof MODEL_VISIBILITIES)[number];

export const MODEL_SERVICE_TIER_DEFAULT_REQUEST_VALUE = "default";

export interface ModelServiceTier {
  id: string;
  name: string;
  description: string;
}

export interface ModelPickerPolicyInput {
  visibility?: unknown;
  service_tiers?: unknown;
  serviceTiers?: unknown;
  default_service_tier?: unknown;
  defaultServiceTier?: unknown;
}

export interface ModelPickerPolicy {
  visibility: ModelVisibility;
  show_in_picker: boolean;
  service_tiers: ModelServiceTier[];
  supported_service_tier_ids: string[];
  default_service_tier: string | null;
}

const MODEL_VISIBILITY_SET = new Set<string>(MODEL_VISIBILITIES);

function firstPresent<T>(
  input: ModelPickerPolicyInput,
  keys: Array<keyof ModelPickerPolicyInput>,
): T | undefined {
  for (const key of keys) {
    const value = input[key];
    if (value !== undefined && value !== null) {
      return value as T;
    }
  }
  return undefined;
}

function normalizeToken(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}

function normalizeVisibility(value: unknown): ModelVisibility {
  const normalized = normalizeToken(value)?.toLowerCase();
  if (!normalized || !MODEL_VISIBILITY_SET.has(normalized)) {
    return "none";
  }
  return normalized as ModelVisibility;
}

function normalizeServiceTier(value: unknown): ModelServiceTier | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const source = value as Record<string, unknown>;
  const id = normalizeToken(source.id);
  if (!id) {
    return null;
  }
  return {
    id,
    name: normalizeToken(source.name) ?? id,
    description: normalizeToken(source.description) ?? "",
  };
}

function normalizeServiceTiers(value: unknown): ModelServiceTier[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const tiers: ModelServiceTier[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    const tier = normalizeServiceTier(item);
    if (!tier || seen.has(tier.id)) {
      continue;
    }
    seen.add(tier.id);
    tiers.push(tier);
  }
  return tiers;
}

export function buildModelPickerPolicy(
  input: ModelPickerPolicyInput | null | undefined,
): ModelPickerPolicy {
  const source = input ?? {};
  const visibility = normalizeVisibility(source.visibility);
  const serviceTiers = normalizeServiceTiers(
    firstPresent(source, ["service_tiers", "serviceTiers"]),
  );

  return {
    visibility,
    show_in_picker: visibility === "list",
    service_tiers: serviceTiers,
    supported_service_tier_ids: serviceTiers.map((tier) => tier.id),
    default_service_tier: normalizeToken(
      firstPresent(source, ["default_service_tier", "defaultServiceTier"]),
    ),
  };
}

export function buildCatalogModelPickerPolicy(
  hidden: boolean,
  input: Omit<ModelPickerPolicyInput, "visibility"> = {},
): ModelPickerPolicy {
  return buildModelPickerPolicy({
    ...input,
    visibility: hidden ? "hide" : "list",
  });
}

export function resolveModelServiceTierForRequest(
  policy: Pick<ModelPickerPolicy, "supported_service_tier_ids">,
  requestedServiceTier: unknown,
): string | null {
  const serviceTier = normalizeToken(requestedServiceTier);
  if (
    !serviceTier ||
    serviceTier === MODEL_SERVICE_TIER_DEFAULT_REQUEST_VALUE
  ) {
    return null;
  }
  return policy.supported_service_tier_ids.includes(serviceTier)
    ? serviceTier
    : null;
}
