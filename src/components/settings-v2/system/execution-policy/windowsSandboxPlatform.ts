export function isWindowsDesktopPlatform(
  platform = typeof navigator === "undefined" ? "" : navigator.platform,
  userAgent = typeof navigator === "undefined" ? "" : navigator.userAgent,
): boolean {
  return /win/i.test(platform) || /windows/i.test(userAgent);
}
