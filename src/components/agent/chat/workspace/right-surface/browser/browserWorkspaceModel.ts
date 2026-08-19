import type { EmbeddedBrowserBounds } from "@/lib/api/embeddedBrowser";

const DOMAIN_PATTERN =
  /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?::\d{1,5})?(?:[/?#].*)?$/i;
const LOCAL_PATTERN =
  /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::\d{1,5})?(?:[/?#].*)?$/i;

export const DEFAULT_BROWSER_URL = "https://www.google.com/";
export const MIN_BROWSER_ZOOM = 0.5;
export const MAX_BROWSER_ZOOM = 3;
export const BROWSER_ZOOM_STEP = 0.1;

export function normalizeBrowserAddress(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return DEFAULT_BROWSER_URL;
  }
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {
    // Continue with domain/search normalization.
  }
  if (DOMAIN_PATTERN.test(trimmed) || LOCAL_PATTERN.test(trimmed)) {
    return new URL(`https://${trimmed}`).href;
  }
  return `https://www.google.com/search?q=${encodeURIComponent(trimmed)}`;
}

export function resolveBrowserAddressValue(url: string): string {
  try {
    return new URL(url).href;
  } catch {
    return url;
  }
}

export function clampBrowserZoom(value: number): number {
  return (
    Math.round(
      Math.min(MAX_BROWSER_ZOOM, Math.max(MIN_BROWSER_ZOOM, value)) * 100,
    ) / 100
  );
}

export function resolveElementBounds(
  element: HTMLElement,
): EmbeddedBrowserBounds {
  const rect = element.getBoundingClientRect();
  return {
    x: Math.round(rect.left),
    y: Math.round(rect.top),
    width: Math.max(0, Math.round(rect.width)),
    height: Math.max(0, Math.round(rect.height)),
  };
}

export function browserBoundsEqual(
  left: EmbeddedBrowserBounds | null,
  right: EmbeddedBrowserBounds,
): boolean {
  return (
    Boolean(left) &&
    left?.x === right.x &&
    left.y === right.y &&
    left.width === right.width &&
    left.height === right.height
  );
}
