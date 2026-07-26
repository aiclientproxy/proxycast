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
