const PLAIN_TEXT_LANGUAGES = new Set(["text", "plaintext", "plain", "txt"]);
export const FLOW_ARROW_ONLY_PATTERN = /^(↓|⬇|⇣|↧|->|=>|→|↘|v)$/u;
const CODE_SIGNAL_PATTERN =
  /[{}[\];=]|\b(const|let|var|function|class|return|import|export|interface|type|async|await)\b/;
const LANGUAGE_CLASS_PATTERN = /\blanguage-([^\s]+)/i;
const FLOW_VIEW_LANGUAGES = new Set(["flow"]);
const CODE_LANGUAGE_ALIASES: Record<string, string> = {
  "c#": "csharp",
  "c++": "cpp",
  js: "javascript",
  md: "markdown",
  objc: "objectivec",
  "objective-c": "objectivec",
  plain: "text",
  plaintext: "text",
  ps1: "powershell",
  py: "python",
  rb: "ruby",
  rs: "rust",
  sh: "bash",
  shell: "bash",
  text: "text",
  ts: "typescript",
  txt: "text",
  yml: "yaml",
  zsh: "bash",
};
const COMPACT_PIPE_TABLE_SEPARATOR_PATTERN = /\|\|[ \t:|-]{3,}\|\|/;
const MARKDOWN_FENCE_LINE_PATTERN = /^\s*(`{3,}|~{3,})/;
const MARKDOWN_FENCE_OPEN_PATTERN = /^(\s*)(`{3,}|~{3,})\s*([^\s`]*)?.*$/;
const INLINE_COLLAPSED_FENCE_PATTERN =
  /```([A-Za-z0-9_+.#-]+)(?=[^\r\nA-Za-z0-9_+.#-])([\s\S]*?)```/g;
const COLLAPSED_MARKDOWN_TRAILING_TABLE_TEXT_PATTERN =
  /\|(?=[\u3400-\u9fffA-Za-z][^|\n]{0,24}[：:])/g;
const COLLAPSED_SPACED_HEADING_PATTERN = /[^\n#]#{2,6}\s+\S/g;
const COLLAPSED_ORDERED_LIST_PATTERN = /(?:[：:]\s*|\s)[1-9]\d{0,1}\.\s+\S/g;
const COLLAPSED_BULLET_LIST_PATTERN = /(?:[：:]\s*|\s)[-*+]\s+\S/g;
const MARKDOWN_HEADING_LINE_PATTERN = /^#{1,6}\s+\S/;
const MARKDOWN_ORDERED_LIST_LINE_PATTERN = /^[1-9]\d{0,1}\.\s+\S/;
const MARKDOWN_UNORDERED_LIST_LINE_PATTERN = /^[-*+]\s+\S/;
const PARTIAL_COLLAPSED_MARKDOWN_LINE_PATTERN =
  /(?:[^\n]\*\*[^*\n]{1,32}[：:]\*\*|[A-Za-z0-9\u3400-\u9fff。！？.!?)）】》”’\]][-+]\s*(?:\*\*)?[\u3400-\u9fffA-Za-z]|[\u3400-\u9fff，。！？；：,.!?;:)）】》”’\]][1-9]\d{0,1}\.\s+\S)/u;
const INLINE_FOLLOW_UP_ORDERED_MARKERS_PATTERN =
  /[2-9]\d{0,1}\.\s+\S[^\n]*[3-9]\d{0,1}\.\s+\S/u;
const COLLAPSED_HEADING_PROSE_BODY_PATTERN = /[。！？.!?]/u;
const COLLAPSED_HEADING_TITLE_FORBIDDEN_PATTERN =
  /[#>*`_[\]()|：:。！？.!?,，；;]/u;
const COLLAPSED_HEADING_BODY_LEADING_FORBIDDEN_PATTERN =
  /^[#>*`_[\]()|：:。！？.!?,，；;]/u;

export function normalizeCodeLanguage(language: string): string {
  const normalized = language.trim().toLowerCase();
  if (!normalized) {
    return "text";
  }

  return CODE_LANGUAGE_ALIASES[normalized] ?? normalized;
}

export function extractCodeLanguageToken(className: string): string {
  const match = LANGUAGE_CLASS_PATTERN.exec(className);
  return (match?.[1] ?? "text").trim().toLowerCase() || "text";
}

export function resolveCodePresentationMode(
  language: string,
  codeContent: string,
): "syntax" | "plain" | "flow" {
  const normalizedLanguage = normalizeCodeLanguage(language);
  const trimmed = codeContent.trim();
  if (!trimmed) {
    return "plain";
  }
  if (FLOW_VIEW_LANGUAGES.has(normalizedLanguage)) {
    return "flow";
  }

  const lines = trimmed
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const bulletRows = lines.filter((line) => /^[-*]\s+/.test(line)).length;
  const hasCodeSignals = CODE_SIGNAL_PATTERN.test(trimmed);

  if (
    PLAIN_TEXT_LANGUAGES.has(normalizedLanguage) ||
    (!hasCodeSignals && (bulletRows >= 2 || lines.length >= 4))
  ) {
    return "plain";
  }

  return "syntax";
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

function formatPipeTableRow(cells: string[]): string {
  return `| ${cells.join(" | ")} |`;
}

function normalizeCellsForTableWidth(cells: string[], width: number): string[] {
  if (cells.length === width) {
    return cells;
  }

  if (cells.length > width) {
    return [...cells.slice(0, width - 1), cells.slice(width - 1).join(" | ")];
  }

  return [...cells, ...Array.from({ length: width - cells.length }, () => "")];
}

function normalizeCompactPipeTableLine(line: string): string {
  const separatorMatch = COMPACT_PIPE_TABLE_SEPARATOR_PATTERN.exec(line);
  if (!separatorMatch || typeof separatorMatch.index !== "number") {
    return line;
  }

  const leadingWhitespace = line.match(/^\s*/)?.[0] ?? "";
  const headerSource = line.slice(0, separatorMatch.index).trim();
  if (!headerSource.startsWith("|")) {
    return line;
  }

  const headerCells = parsePipeTableCells(headerSource);
  const nonEmptyHeaderCells = headerCells.filter(Boolean);
  if (headerCells.length < 2 || nonEmptyHeaderCells.length < 2) {
    return line;
  }

  const rowSource = line
    .slice(separatorMatch.index + separatorMatch[0].length)
    .trim();
  const bodyRows = rowSource
    .split(/\s*\|\|\s*/)
    .map(parsePipeTableCells)
    .filter((cells) => cells.filter(Boolean).length >= 2)
    .map((cells) => normalizeCellsForTableWidth(cells, headerCells.length));

  if (bodyRows.length === 0) {
    return line;
  }

  const tableLines = [
    formatPipeTableRow(headerCells),
    formatPipeTableRow(headerCells.map(() => "---")),
    ...bodyRows.map(formatPipeTableRow),
  ];

  return tableLines
    .map((tableLine) => `${leadingWhitespace}${tableLine}`)
    .join("\n");
}

function isCompactPipeTableContinuationLine(line: string): boolean {
  const trimmed = line.trim();
  if (
    !trimmed ||
    MARKDOWN_HEADING_LINE_PATTERN.test(trimmed) ||
    MARKDOWN_FENCE_LINE_PATTERN.test(trimmed)
  ) {
    return false;
  }

  return (
    trimmed.includes("|") &&
    parsePipeTableCells(trimmed).filter(Boolean).length >= 2
  );
}

function collectCompactPipeTableLine(
  lines: string[],
  startIndex: number,
): { line: string; endIndex: number } {
  let line = lines[startIndex] ?? "";
  let endIndex = startIndex;
  let cursor = startIndex + 1;

  while (cursor < lines.length) {
    const currentLine = lines[cursor] ?? "";
    const nextLine = lines[cursor + 1] ?? "";

    if (
      currentLine.trim() === "" &&
      isCompactPipeTableContinuationLine(nextLine)
    ) {
      line = `${line.trimEnd()} ${nextLine.trim()}`;
      endIndex = cursor + 1;
      cursor += 2;
      continue;
    }

    if (isCompactPipeTableContinuationLine(currentLine)) {
      line = `${line.trimEnd()} ${currentLine.trim()}`;
      endIndex = cursor;
      cursor += 1;
      continue;
    }

    break;
  }

  return { line, endIndex };
}

function countPatternMatches(value: string, pattern: RegExp): number {
  return value.match(pattern)?.length ?? 0;
}

function collectTextOutsideMarkdownFences(markdown: string): string {
  const outsideLines: string[] = [];
  let activeFence: { marker: "`" | "~"; markerLength: number } | null = null;

  for (const line of markdown.split("\n")) {
    const fenceMatch = MARKDOWN_FENCE_OPEN_PATTERN.exec(line);
    if (!activeFence && fenceMatch) {
      const markerRun = fenceMatch[2] || "";
      activeFence = {
        marker: markerRun.startsWith("~") ? "~" : "`",
        markerLength: markerRun.length,
      };
      continue;
    }

    if (activeFence) {
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

    outsideLines.push(line);
  }

  return outsideLines.join("\n");
}

function shouldNormalizeCollapsedMarkdownBlocks(markdown: string): boolean {
  const trimmed = markdown.trim();
  if (!trimmed) {
    return false;
  }

  const outsideText = collectTextOutsideMarkdownFences(trimmed).trim();
  const hasInlineCollapsedFence = INLINE_COLLAPSED_FENCE_PATTERN.test(trimmed);
  INLINE_COLLAPSED_FENCE_PATTERN.lastIndex = 0;
  if (!outsideText && !hasInlineCollapsedFence) {
    return false;
  }

  const scanText = outsideText || trimmed;
  const lines = scanText.split("\n");
  const hasSparseLineBreaks =
    lines.length <= 2 || scanText.length / Math.max(lines.length, 1) > 800;
  const hasPartiallyCollapsedLine = lines.some(
    (line) =>
      line.length >= 72 && PARTIAL_COLLAPSED_MARKDOWN_LINE_PATTERN.test(line),
  );

  const collapsedSpacedHeadingCount = countPatternMatches(
    scanText,
    COLLAPSED_SPACED_HEADING_PATTERN,
  );
  const collapsedOrderedListCount = countPatternMatches(
    scanText,
    COLLAPSED_ORDERED_LIST_PATTERN,
  );
  const collapsedBulletListCount = countPatternMatches(
    scanText,
    COLLAPSED_BULLET_LIST_PATTERN,
  );
  const markerCount = [
    /(^|[^#])#{1,6}(?!#)\S/.test(scanText),
    collapsedSpacedHeadingCount > 0,
    /---#{1,6}(?!#)\S/.test(scanText),
    /[：:]\s*[-*+]\s+\S|`-\s*\S/.test(scanText),
    /[：:]\s*[1-9]\d{0,1}\.\s+\S/.test(scanText),
    collapsedOrderedListCount >= 2,
    collapsedBulletListCount >= 2,
    COMPACT_PIPE_TABLE_SEPARATOR_PATTERN.test(scanText),
    hasInlineCollapsedFence,
  ].filter(Boolean).length;

  INLINE_COLLAPSED_FENCE_PATTERN.lastIndex = 0;
  return (
    hasPartiallyCollapsedLine ||
    (hasSparseLineBreaks &&
      (markerCount >= 2 ||
        collapsedSpacedHeadingCount >= 2 ||
        collapsedOrderedListCount >= 2 ||
        collapsedBulletListCount >= 2))
  );
}

function normalizeCollapsedFenceBody(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) {
    return "";
  }

  return trimmed
    .replace(/(\/(?:Users|home|var|tmp|opt|Applications)\b)/g, "\n$1")
    .replace(/([├└]──)/gu, "\n$1")
    .replace(/^\n/, "");
}

function normalizeInlineCollapsedCodeFences(markdown: string): string {
  return markdown.replace(
    INLINE_COLLAPSED_FENCE_PATTERN,
    (_match, language: string, body: string) => {
      const rawLanguage = language.trim();
      const normalizedBody = normalizeCollapsedFenceBody(body);
      const isTreeCode = /[├└]──/u.test(normalizedBody);
      const normalizedLanguage = isTreeCode ? "text" : rawLanguage || "text";
      const codeBody = isTreeCode
        ? `${rawLanguage}${normalizedBody}`
        : normalizedBody;
      return `\n\n\`\`\`${normalizedLanguage}\n${codeBody}\n\`\`\`\n\n`;
    },
  );
}

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

function normalizeUnclosedStrongLabels(markdown: string): string {
  return markdown
    .split("\n")
    .map((line) => {
      let output = "";
      let cursor = 0;
      let strongIsOpen = false;

      while (cursor < line.length) {
        if (!line.startsWith("**", cursor)) {
          output += line[cursor] || "";
          cursor += 1;
          continue;
        }

        output += "**";
        cursor += 2;
        if (strongIsOpen) {
          strongIsOpen = false;
          continue;
        }

        const labelMatch = /^([^*]{1,32}[：:])/u.exec(line.slice(cursor));
        const nextMarkerIndex = line.indexOf("**", cursor);
        const hasNearbyClosingMarker =
          nextMarkerIndex >= cursor && nextMarkerIndex - cursor <= 120;
        if (labelMatch && !hasNearbyClosingMarker) {
          output += `${labelMatch[1]}**`;
          cursor += labelMatch[1].length;
          if (line[cursor] === "*" && line[cursor + 1] !== "*") {
            cursor += 1;
          }
          continue;
        }

        strongIsOpen = true;
      }

      return output;
    })
    .join("\n");
}

function protectInlineCodeSpans(markdown: string): {
  markdown: string;
  restore: (value: string) => string;
} {
  const inlineCodeSpans: string[] = [];
  const protectedMarkdown = markdown.replace(/(`+)([^`\n]*?)\1/g, (match) => {
    const index = inlineCodeSpans.push(match) - 1;
    return `\uE000${index}\uE001`;
  });

  return {
    markdown: protectedMarkdown,
    restore: (value) =>
      value.replace(/\uE000(\d+)\uE001/g, (_match, index: string) => {
        return inlineCodeSpans[Number(index)] || _match;
      }),
  };
}

function normalizeCollapsedMarkdownTextBlocks(markdown: string): string {
  const withBlockBoundaries = markdown
    .replace(/---(?=#{1,6}(?!#)\s*\S)/g, "\n\n---\n\n")
    .replace(/([^\n#])(?=#{2,6}\s+\S)/g, "$1\n\n")
    .replace(/(^|[^A-Za-z0-9#\n])(?=#{1,6}(?!#)\S)/g, "$1\n\n")
    .replace(/(^|\n)(#{1,6})(?!#)(?=\S)/g, "$1$2 ")
    .replace(/(^|\n)(#{1,6}\s+\d+\.)(?=\S)/g, "$1$2 ")
    .replace(
      /(^|\n)(#{1,6}\s+[^\n*#]{2,64}?)(?=\*\*[\u3400-\u9fffA-Za-z0-9][^*\n]{0,32}[：:])/gu,
      "$1$2\n\n",
    )
    .replace(/(^|\n)(#{1,6}\s+[^\n#>]{2,90}?)(?=>\s*\S)/gu, "$1$2\n\n")
    .replace(
      /(^|\n)(#{1,6}\s+[^\n`#*]{2,64}?)(?=`[^`\n])/gu,
      (_match, prefix: string, heading: string) => {
        const headingText = heading.replace(/^#{1,6}\s+/, "").trim();
        if (/^[1-9]\d{0,1}\.$/.test(headingText)) {
          return `${prefix}${heading}`;
        }
        return `${prefix}${heading}\n\n`;
      },
    )
    .replace(
      /(^|\n)(#{1,6}\s+[^\n|*#]{2,90}?)(?=\|[^\n]*\|\|[ \t:|-]{3,}\|\|)/gu,
      "$1$2\n\n",
    )
    .replace(/([：:])(\|[^\n]*?\|\|[ \t:|-]{3,}\|\|)/g, "$1\n\n$2")
    .replace(COLLAPSED_MARKDOWN_TRAILING_TABLE_TEXT_PATTERN, "|\n\n");

  const protectedInlineCode = protectInlineCodeSpans(
    normalizeCompactPipeTables(withBlockBoundaries),
  );
  const normalized = normalizeUnclosedStrongLabels(protectedInlineCode.markdown)
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9。！？.!?])(\*\*[^*\n]{1,32}[：:]\*\*)/gu,
      "\n\n$1",
    )
    .replace(
      /(^|\n)(\*\*[^*\n]{1,32}[：:]\*\*)[ \t]*(?=[\u3400-\u9fffA-Za-z0-9])/gu,
      "$1$2\n\n",
    )
    .replace(
      /(^|\n)([-*+]\s+)(\*\*[^*\n]{1,32}[：:]\*\*)[ \t]*(?=[\u3400-\u9fffA-Za-z0-9])/gu,
      "$1$2$3 ",
    )
    .replace(
      /(^|\n)(#{1,6}\s+[^\n#]{2,90}?)(?<=[\u3400-\u9fff）】》”’])([-*+])\s*(?=(?:\*\*)?(?:[\u3400-\u9fffA-Za-z0-9]|\[))/gu,
      "$1$2\n\n$3 ",
    )
    .replace(/([：:])\s*([1-9]\d{0,1}\.\s+)/g, "$1\n\n$2")
    .replace(
      /(?<=[\u3400-\u9fff，。！？；：,.!?;:)）】》”’\]])([1-9]\d{0,1}\.\s+\S)/gu,
      "\n\n$1",
    )
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9，。！？；：,.!?;:)）】》”’\]])\s+([1-9]\d{0,1}\.\s+\S)/gu,
      "\n\n$1",
    )
    .replace(/([：:])\s*([-+])\s*/g, "$1\n$2 ")
    .replace(/([：:])\s*(\*)(?!\*)\s*/g, "$1\n$2 ")
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9。！？.!?)）】》”’\]])(-|(?<!\*)\*(?!\*))\s+(?=(?:\*\*)?[\u3400-\u9fffA-Za-z][^\n]{2,})/gu,
      "\n$1 ",
    )
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9)）】》”’\]])(-|(?<!\*)\*(?!\*))(?=[\u3400-\u9fffA-Za-z][^，。！？；：,.!?;:\n]{0,24}[：:])/gu,
      "\n$1 ",
    )
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9。！？.!?)）】》”’\]])(-)(?=[\u3400-\u9fff])/gu,
      "\n$1 ",
    )
    .replace(
      /(^|\n)([-*+]\s+)(\*\*[^*\n]{1,32}[：:]\*\*)[ \t]*(?=[\u3400-\u9fffA-Za-z0-9])/gu,
      "$1$2$3 ",
    )
    .replace(
      /(?<=[\u3400-\u9fffA-Za-z0-9)）】》”’\]])\s+((?:-|(?<!\*)\*(?!\*))\s+\S)/gu,
      "\n$1",
    );
  const normalizedWithInlineCode = protectedInlineCode
    .restore(normalized)
    .replace(/`-\s*/g, "`\n- ")
    .replace(/([：:])```/g, "$1\n\n```");

  const withHeadingProseBoundaries = normalizeCollapsedHeadingProseBoundaries(
    normalizedWithInlineCode.replace(
      /(^|\n)(#{1,6}\s+[^\n*#]{2,90}?)(?=\*\*(?:[1-9]\d{0,1}\.)[^*\n])/gu,
      "$1$2\n\n",
    ),
  );

  const recoveredBlocks = normalizeRecoveredListNesting(
    withHeadingProseBoundaries
      .replace(/(^|\n)(#{1,6}\s+[^\n*#]{2,90}?)(?=\*\*(?!\*))/gu, "$1$2\n\n")
      .replace(
        /(^|\n)([^*\n]{1,24})(\*\*(?:[1-9]\d{0,1}\.)[^*\n]{1,120}\*\*)/gu,
        "$1$2\n\n$3",
      )
      .replace(
        /(?<=[\u3400-\u9fffA-Za-z0-9。！？；;）】》”’])\s*(\*\*(?:[1-9]\d{0,1}\.)[^*\n]{1,120}\*\*)/gu,
        "\n\n$1",
      )
      .replace(/(\*\*[^*\n]{2,120}\*\*)\s*([-+])(?=\s*\*\*)/gu, "$1\n\n$2 ")
      .replace(/\n{3,}/g, "\n\n"),
  ).trim();

  return normalizeBrokenPipeTableRows(recoveredBlocks).trim();
}

function normalizeBrokenPipeTableRows(markdown: string): string {
  const outputLines: string[] = [];

  for (const line of markdown.split("\n")) {
    const previousLine = outputLines[outputLines.length - 1];
    const isPlusWrappedTableCell = Boolean(
      previousLine &&
      /^\s*\|/.test(previousLine) &&
      !previousLine.trimEnd().endsWith("|") &&
      /^\s*\+\s+\S/.test(line),
    );
    const isWrappedTableRow = Boolean(
      previousLine &&
      /^\s*\|/.test(previousLine) &&
      !previousLine.trimEnd().endsWith("|") &&
      line.trimEnd().endsWith("|") &&
      !/^\s*(?:\||#{1,6}\s|[-*]\s|\d+\.\s|```|~~~)/.test(line),
    );

    if (isPlusWrappedTableCell || isWrappedTableRow) {
      outputLines[outputLines.length - 1] =
        `${previousLine.trimEnd()} ${line.trim()}`;
      continue;
    }

    outputLines.push(line);
  }

  return outputLines.join("\n");
}

type WordSegment = {
  segment: string;
  index: number;
  isWordLike?: boolean;
};

type WordSegmenter = {
  segment(input: string): Iterable<WordSegment>;
};

type IntlWithWordSegmenter = typeof Intl & {
  Segmenter?: new (
    locale?: string | string[],
    options?: { granularity?: "word" },
  ) => WordSegmenter;
};

function getWordSegmentBoundaries(value: string): number[] {
  const SegmenterConstructor = (Intl as IntlWithWordSegmenter).Segmenter;
  if (!SegmenterConstructor) {
    return [];
  }

  const segmenter = new SegmenterConstructor(["zh", "en"], {
    granularity: "word",
  });
  const boundaries: number[] = [];

  for (const segment of segmenter.segment(value)) {
    const isWordLike =
      segment.isWordLike ?? /[\u3400-\u9fffA-Za-z0-9]/u.test(segment.segment);
    if (!isWordLike) {
      continue;
    }
    boundaries.push(segment.index + segment.segment.length);
  }

  return boundaries;
}

function findCollapsedHeadingProseBoundary(value: string): number | null {
  const trimmed = value.trim();
  if (
    trimmed.length < 28 ||
    !COLLAPSED_HEADING_PROSE_BODY_PATTERN.test(trimmed) ||
    /^(?:[1-9]\d{0,1}\.\s|[-*+]\s|>)/.test(trimmed)
  ) {
    return null;
  }

  for (const boundary of getWordSegmentBoundaries(trimmed).slice(1, 5)) {
    const title = trimmed.slice(0, boundary).trim();
    const body = trimmed.slice(boundary).trimStart();
    if (
      title.length >= 4 &&
      title.length <= 16 &&
      body.length >= 18 &&
      !COLLAPSED_HEADING_TITLE_FORBIDDEN_PATTERN.test(title) &&
      !COLLAPSED_HEADING_BODY_LEADING_FORBIDDEN_PATTERN.test(body) &&
      COLLAPSED_HEADING_PROSE_BODY_PATTERN.test(body)
    ) {
      return boundary;
    }
  }

  return null;
}

function normalizeCollapsedHeadingProseBoundaries(markdown: string): string {
  return markdown
    .split("\n")
    .map((line) => {
      const match = /^(\s{0,3}#{1,6}\s+)(\S[^\n]*)$/u.exec(line);
      if (!match) {
        return line;
      }

      const [, prefix, headingText] = match;
      const boundary = findCollapsedHeadingProseBoundary(headingText);
      if (boundary == null) {
        return line;
      }

      return `${prefix}${headingText.slice(0, boundary).trimEnd()}\n\n${headingText.slice(boundary).trimStart()}`;
    })
    .join("\n");
}

function normalizeRecoveredListNesting(markdown: string): string {
  const outputLines: string[] = [];
  let activeOrderedItem = false;

  for (const line of markdown.split("\n")) {
    const trimmed = line.trim();

    if (!trimmed || MARKDOWN_HEADING_LINE_PATTERN.test(trimmed)) {
      activeOrderedItem = false;
      outputLines.push(line);
      continue;
    }

    if (MARKDOWN_ORDERED_LIST_LINE_PATTERN.test(trimmed)) {
      activeOrderedItem = true;
      outputLines.push(trimmed);
      continue;
    }

    if (
      activeOrderedItem &&
      MARKDOWN_UNORDERED_LIST_LINE_PATTERN.test(trimmed)
    ) {
      outputLines.push(`   ${trimmed}`);
      continue;
    }

    activeOrderedItem = false;
    outputLines.push(line);
  }

  return outputLines.join("\n");
}

export function normalizeCollapsedMarkdownBlocks(markdown: string): string {
  if (!shouldNormalizeCollapsedMarkdownBlocks(markdown)) {
    return markdown;
  }

  const withCodeFences = normalizeInlineCollapsedCodeFences(markdown);
  return transformOutsideMarkdownFences(
    withCodeFences,
    normalizeCollapsedMarkdownTextBlocks,
  );
}

export function normalizeInlineFollowUpListMarkers(markdown: string): string {
  if (!INLINE_FOLLOW_UP_ORDERED_MARKERS_PATTERN.test(markdown)) {
    return markdown;
  }

  return transformOutsideMarkdownFences(markdown, (text) =>
    text.replace(
      /(^|\n)([^\n]*[2-9]\d{0,1}\.\s+\S[^\n]*[3-9]\d{0,1}\.\s+\S[^\n]*)/gu,
      (_match, prefix: string, line: string) => {
        const trimmedLine = line.trim();
        if (/^1\.\s+/.test(trimmedLine)) {
          return `${prefix}${trimmedLine.replace(/\s+([2-9]\d{0,1}\.\s+)/g, "\n$1")}`;
        }
        return `${prefix}- ${trimmedLine.replace(/\s*[2-9]\d{0,1}\.\s+/g, "\n- ")}`;
      },
    ),
  );
}

export function normalizeCompactPipeTables(markdown: string): string {
  if (!COMPACT_PIPE_TABLE_SEPARATOR_PATTERN.test(markdown)) {
    return markdown;
  }

  let activeFenceMarker: "`" | "~" | null = null;
  const lines = markdown.split("\n");
  const outputLines: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    const fenceMatch = MARKDOWN_FENCE_LINE_PATTERN.exec(line);
    if (fenceMatch) {
      const marker = fenceMatch[1]?.startsWith("~") ? "~" : "`";
      activeFenceMarker = activeFenceMarker === marker ? null : marker;
      outputLines.push(line);
      continue;
    }

    if (activeFenceMarker) {
      outputLines.push(line);
      continue;
    }

    if (COMPACT_PIPE_TABLE_SEPARATOR_PATTERN.test(line)) {
      const collected = collectCompactPipeTableLine(lines, index);
      const normalizedLine = normalizeCompactPipeTableLine(collected.line);
      if (normalizedLine !== collected.line) {
        outputLines.push(normalizedLine);
        index = collected.endIndex;
        continue;
      }
    }

    outputLines.push(normalizeCompactPipeTableLine(line));
  }

  return outputLines.join("\n");
}

function isMarkdownFenceInfo(info: string | undefined): boolean {
  const normalized = (info || "").trim().toLowerCase();
  return (
    normalized === "md" || normalized === "markdown" || normalized === "gfm"
  );
}

function isMarkdownTableDelimiterCell(cell: string): boolean {
  return /^:?-{3,}:?$/.test(cell.trim());
}

function stripBlockquotePrefix(line: string): string {
  return line.replace(/^\s*>\s?/, "");
}

function containsMarkdownTable(markdown: string): boolean {
  const lines = markdown.split("\n").map(stripBlockquotePrefix);

  for (let index = 0; index < lines.length - 1; index += 1) {
    const headerCells = parsePipeTableCells(lines[index] || "");
    const delimiterCells = parsePipeTableCells(lines[index + 1] || "");
    if (
      headerCells.filter(Boolean).length >= 2 &&
      delimiterCells.length >= headerCells.length &&
      delimiterCells.every(isMarkdownTableDelimiterCell)
    ) {
      return true;
    }
  }

  return false;
}

export function normalizeMarkdownTableFences(markdown: string): string {
  if (!markdown.includes("```") && !markdown.includes("~~~")) {
    return markdown;
  }

  const lines = markdown.split("\n");
  const outputLines: string[] = [];
  let activeFence: {
    marker: "`" | "~";
    markerLength: number;
    openLine: string;
    isMarkdown: boolean;
    contentLines: string[];
  } | null = null;

  for (const line of lines) {
    const fenceMatch = MARKDOWN_FENCE_OPEN_PATTERN.exec(line);
    if (!activeFence) {
      if (fenceMatch) {
        const markerRun = fenceMatch[2] || "";
        const marker = markerRun.startsWith("~") ? "~" : "`";
        activeFence = {
          marker,
          markerLength: markerRun.length,
          openLine: line,
          isMarkdown: isMarkdownFenceInfo(fenceMatch[3]),
          contentLines: [],
        };
        continue;
      }

      outputLines.push(line);
      continue;
    }

    if (fenceMatch) {
      const markerRun = fenceMatch[2] || "";
      const marker = markerRun.startsWith("~") ? "~" : "`";
      const isClosingFence =
        marker === activeFence.marker &&
        markerRun.length >= activeFence.markerLength &&
        line.trim() === markerRun;

      if (isClosingFence) {
        const content = activeFence.contentLines.join("\n");
        if (activeFence.isMarkdown && containsMarkdownTable(content)) {
          outputLines.push(...activeFence.contentLines);
        } else {
          outputLines.push(
            activeFence.openLine,
            ...activeFence.contentLines,
            line,
          );
        }
        activeFence = null;
        continue;
      }
    }

    activeFence.contentLines.push(line);
  }

  if (activeFence) {
    outputLines.push(activeFence.openLine, ...activeFence.contentLines);
  }

  return outputLines.join("\n");
}
