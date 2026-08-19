import type { ParsedBrowserWorkbenchCommand } from "../utils/browserWorkbenchCommand";

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

export function buildBrowserControlLaunchRequestMetadata(
  existingMetadata: Record<string, unknown> | undefined,
  parsedCommand: ParsedBrowserWorkbenchCommand,
): Record<string, unknown> {
  const existingHarness = asRecord(existingMetadata?.harness) || {};

  return {
    ...(existingMetadata || {}),
    harness: {
      ...existingHarness,
      browser_requirement: parsedCommand.browserRequirement,
      browser_requirement_reason: parsedCommand.browserRequirementReason,
      browser_launch_url: parsedCommand.launchUrl,
      browser_user_step_required:
        parsedCommand.browserRequirement === "required_with_user_step",
    },
  };
}
