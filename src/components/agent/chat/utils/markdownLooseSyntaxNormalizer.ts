const MARKDOWN_FENCE_OPEN_PATTERN = /^(\s*)(`{3,}|~{3,})\s*([^\s`]*)?.*$/;
const PIPE_TABLE_DELIMITER_CELL_PATTERN = /^:?-{3,}:?$/;

function transformOutsideMarkdownFences(
  markdown: string,
  transform: (text: string) => string,
): string {
  const outputLines: string[] = [];
  let pendingOutsideLines: string[] = [];
  let activeFence: { marker: "`" | "~"; markerLength: number } | null = null;

  const flushOutsideLines = () => {
    if (pendingOutsideLines.length === 0) {
      return;
    }
    outputLines.push(transform(pendingOutsideLines.join("\n")));
    pendingOutsideLines = [];
  };

  for (const line of markdown.split("\n")) {
    const fenceMatch = MARKDOWN_FENCE_OPEN_PATTERN.exec(line);
    if (!activeFence && fenceMatch) {
      flushOutsideLines();
      const markerRun = fenceMatch[2] || "";
      activeFence = {
        marker: markerRun.startsWith("~") ? "~" : "`",
        markerLength: markerRun.length,
      };
      outputLines.push(line);
      continue;
    }

    if (activeFence) {
      outputLines.push(line);
      if (fenceMatch) {
        const markerRun = fenceMatch[2] || "";
        const marker = markerRun.startsWith("~") ? "~" : "`";
        if (
          marker === activeFence.marker &&
          markerRun.length >= activeFence.markerLength &&
          line.trim() === markerRun
        ) {
          activeFence = null;
        }
      }
      continue;
    }

    pendingOutsideLines.push(line);
  }

  flushOutsideLines();
  return outputLines.join("\n");
}

function parsePipeTableCells(row: string): string[] {
  const trimmed = row.trim();
  if (!trimmed.includes("|")) {
    return [];
  }

  const withoutLeadingPipe = trimmed.startsWith("|")
    ? trimmed.slice(1)
    : trimmed;
  const withoutEdgePipes = withoutLeadingPipe.endsWith("|")
    ? withoutLeadingPipe.slice(0, -1)
    : withoutLeadingPipe;

  return withoutEdgePipes.split("|").map((cell) => cell.trim());
}

function isPipeTableDelimiterLine(
  line: string,
  expectedWidth: number,
): boolean {
  const cells = parsePipeTableCells(line);
  return (
    expectedWidth >= 2 &&
    cells.length >= expectedWidth &&
    cells.every((cell) => PIPE_TABLE_DELIMITER_CELL_PATTERN.test(cell))
  );
}

function isPipeTableRowLine(line: string): boolean {
  return parsePipeTableCells(line).filter(Boolean).length >= 2;
}

function normalizePipeTableSpacing(text: string): string {
  const lines = text.split("\n");
  const outputLines: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    const nextLine = lines[index + 1] ?? "";
    const headerCells = parsePipeTableCells(line);
    const isTableStart =
      headerCells.filter(Boolean).length >= 2 &&
      isPipeTableDelimiterLine(nextLine, headerCells.length);

    if (!isTableStart) {
      outputLines.push(line);
      continue;
    }

    const previousOutputLine = outputLines[outputLines.length - 1] ?? "";
    if (previousOutputLine.trim()) {
      outputLines.push("");
    }

    outputLines.push(line, nextLine);
    index += 1;

    while (
      index + 1 < lines.length &&
      isPipeTableRowLine(lines[index + 1] || "")
    ) {
      index += 1;
      outputLines.push(lines[index] || "");
    }

    const followingLine = lines[index + 1] ?? "";
    if (followingLine.trim()) {
      outputLines.push("");
    }
  }

  return outputLines.join("\n");
}

function normalizeTrailingHeadingMarkers(text: string): string {
  return text
    .split("\n")
    .map((line) => {
      const match = /^(\s{0,3})([^#\n|`]{2,72}?)(#{2,6})\s*$/u.exec(line);
      if (!match) {
        return line;
      }

      const [, indent, rawTitle, marker] = match;
      const title = rawTitle.trim();
      if (
        !title ||
        /^(?:[-*+]\s|\d+\.\s|>)/.test(title) ||
        /[。！？.!?；;:：,，]$/.test(title)
      ) {
        return line;
      }

      return `${indent}${marker} ${title}`;
    })
    .join("\n");
}

function normalizeHeadingMarkers(text: string): string {
  return normalizeTrailingHeadingMarkers(text)
    .replace(/([^\n#])(?=#{2,6}(?!#)[\u3400-\u9fffA-Za-z0-9])/gu, "$1\n\n")
    .replace(/(^|\n)(#{1,6})(?!#)(?=\S)/g, "$1$2 ");
}

function normalizeStrongMarkerSpacing(text: string): string {
  return text.replace(/\*\*[ \t]*([^*\n]*?\S)[ \t]+\*\*/g, "**$1**");
}

function normalizeLooseMarkdownText(text: string): string {
  return normalizePipeTableSpacing(
    normalizeStrongMarkerSpacing(normalizeHeadingMarkers(text)),
  ).replace(/\n{3,}/g, "\n\n");
}

export function normalizeLooseMarkdownSyntax(markdown: string): string {
  if (!markdown.trim()) {
    return markdown;
  }

  return transformOutsideMarkdownFences(markdown, normalizeLooseMarkdownText);
}
