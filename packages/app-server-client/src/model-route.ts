const MODEL_ROUTE_PREFIX = "route:";

export interface ModelRoute {
  providerId: string;
  modelId: string;
}

function decodeBase64Url(value: string): string | null {
  try {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
    const binary = globalThis.atob(padded);
    const bytes = Uint8Array.from(binary, (character) =>
      character.charCodeAt(0),
    );
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

function encodeBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return globalThis
    .btoa(binary)
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

export function encodeModelRouteSelector(route: ModelRoute): string {
  const providerId = route.providerId.trim();
  const modelId = route.modelId.trim();
  if (!providerId || !modelId) {
    throw new TypeError("model route requires non-empty providerId and modelId");
  }
  return `${MODEL_ROUTE_PREFIX}${encodeBase64Url(providerId)}.${encodeBase64Url(modelId)}`;
}

export function decodeModelRouteSelector(selector: string): ModelRoute | null {
  if (!selector.startsWith(MODEL_ROUTE_PREFIX)) {
    return null;
  }
  const [providerPart, modelPart, ...extra] = selector
    .slice(MODEL_ROUTE_PREFIX.length)
    .split(".");
  if (!providerPart || !modelPart || extra.length > 0) {
    return null;
  }
  const providerId = decodeBase64Url(providerPart)?.trim();
  const modelId = decodeBase64Url(modelPart)?.trim();
  if (!providerId || !modelId) {
    return null;
  }
  return { providerId, modelId };
}
