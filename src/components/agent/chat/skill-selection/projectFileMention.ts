export interface ProjectFileMentionSelection {
  cursorPos: number;
  value: string;
}

export function formatProjectFileMentionPath(path: string): string {
  const needsQuotes = /\s/.test(path);
  return needsQuotes && !path.includes('"') ? `"${path}"` : path;
}

export function replaceProjectFileMentionToken(params: {
  value: string;
  tokenStart: number;
  tokenEnd: number;
  path: string;
}): ProjectFileMentionSelection {
  const tokenStart = Math.max(
    0,
    Math.min(params.tokenStart, params.value.length),
  );
  const tokenEnd = Math.max(
    tokenStart,
    Math.min(params.tokenEnd, params.value.length),
  );
  const leadingText = params.value.slice(0, tokenStart);
  const trailingText = params.value.slice(tokenEnd);
  const insertedPath = formatProjectFileMentionPath(params.path);
  const leadingSeparator =
    leadingText.length > 0 && !/[\s\n]$/.test(leadingText) ? " " : "";
  const needsTrailingSeparator =
    trailingText.length === 0 || !/^[\t ]/.test(trailingText);
  const trailingSeparator = needsTrailingSeparator ? " " : "";
  const insertedText = `${leadingSeparator}${insertedPath}${trailingSeparator}`;
  const value = `${leadingText}${insertedText}${trailingText}`;
  const cursorPos =
    leadingText.length +
    insertedText.length +
    (!needsTrailingSeparator && trailingText.length > 0 ? 1 : 0);

  return { value, cursorPos };
}
