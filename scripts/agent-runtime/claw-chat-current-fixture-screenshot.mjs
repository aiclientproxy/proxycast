const EVIDENCE_SCREENSHOT_TIMEOUT_MS = 15_000;

export async function captureEvidenceScreenshot({ page, path: targetPath }) {
  try {
    await page.screenshot({
      path: targetPath,
      fullPage: true,
      timeout: EVIDENCE_SCREENSHOT_TIMEOUT_MS,
    });
    return {
      path: targetPath,
      mode: "full-page",
      fallbackUsed: false,
      fullPageError: null,
    };
  } catch (error) {
    await page.screenshot({
      path: targetPath,
      fullPage: false,
      timeout: EVIDENCE_SCREENSHOT_TIMEOUT_MS,
    });
    return {
      path: targetPath,
      mode: "viewport",
      fallbackUsed: true,
      fullPageError: formatError(error),
    };
  }
}

function formatError(error) {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }
  return String(error);
}
