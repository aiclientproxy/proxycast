import type {
  ComposerControllerSnapshot,
  ComposerDocument,
  ComposerDraftSnapshot,
  ComposerHistoryDirection,
  ComposerIntent,
  ComposerMentionBinding,
  ComposerPasteOptions,
  ComposerPathReference,
  ComposerPendingPaste,
  ComposerSubmitResult,
  ComposerSubmitTarget,
  ComposerAttachment,
} from "./types";

export const LARGE_PASTE_CHAR_THRESHOLD = 1_000;
const MAX_LOCAL_HISTORY_ENTRIES = 50;

export interface ComposerDocumentInput {
  text?: string;
  selectionStart?: number;
  selectionEnd?: number;
  textElements?: readonly unknown[];
  mentionBindings?: readonly ComposerMentionBinding[];
  attachments?: readonly ComposerAttachment[];
  pathReferences?: readonly ComposerPathReference[];
  pendingPastes?: readonly ComposerPendingPaste[];
  inputCapabilityRoute?: unknown;
}

type ComposerListener = () => void;

function normalizeText(value: string): string {
  return value.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function clampSelection(value: number | undefined, text: string): number {
  const position = Number.isFinite(value) ? Number(value) : text.length;
  return Math.max(0, Math.min(position, text.length));
}

function cloneDocument(document: ComposerDocument): ComposerDocument {
  return {
    text: document.text,
    selectionStart: document.selectionStart,
    selectionEnd: document.selectionEnd,
    textElements: [...document.textElements],
    mentionBindings: document.mentionBindings.map((binding) => ({
      ...binding,
    })),
    attachments: document.attachments.map((attachment) => ({ ...attachment })),
    pathReferences: document.pathReferences.map((reference) => ({
      ...reference,
    })),
    pendingPastes: document.pendingPastes.map((paste) => ({ ...paste })),
    ...(document.inputCapabilityRoute === undefined
      ? {}
      : { inputCapabilityRoute: document.inputCapabilityRoute }),
  };
}

function buildDocument(input: ComposerDocumentInput = {}): ComposerDocument {
  const text = normalizeText(input.text ?? "");
  const selectionStart = clampSelection(input.selectionStart, text);
  const selectionEnd = clampSelection(
    input.selectionEnd ?? selectionStart,
    text,
  );
  return {
    text,
    selectionStart: Math.min(selectionStart, selectionEnd),
    selectionEnd: Math.max(selectionStart, selectionEnd),
    textElements: [...(input.textElements ?? [])],
    mentionBindings: [...(input.mentionBindings ?? [])].map((binding) => ({
      ...binding,
    })),
    attachments: [...(input.attachments ?? [])].map((attachment) => ({
      ...attachment,
    })),
    pathReferences: [...(input.pathReferences ?? [])].map((reference) => ({
      ...reference,
    })),
    pendingPastes: [...(input.pendingPastes ?? [])].map((paste) => ({
      ...paste,
    })),
    ...(input.inputCapabilityRoute === undefined
      ? {}
      : { inputCapabilityRoute: input.inputCapabilityRoute }),
  };
}

function documentsEqual(
  left: ComposerDocument,
  right: ComposerDocument,
): boolean {
  return (
    left.text === right.text &&
    left.selectionStart === right.selectionStart &&
    left.selectionEnd === right.selectionEnd &&
    JSON.stringify(left.textElements) === JSON.stringify(right.textElements) &&
    JSON.stringify(left.mentionBindings) ===
      JSON.stringify(right.mentionBindings) &&
    JSON.stringify(left.attachments) === JSON.stringify(right.attachments) &&
    JSON.stringify(left.pathReferences) ===
      JSON.stringify(right.pathReferences) &&
    JSON.stringify(left.pendingPastes) ===
      JSON.stringify(right.pendingPastes) &&
    JSON.stringify(left.inputCapabilityRoute) ===
      JSON.stringify(right.inputCapabilityRoute)
  );
}

function persistedHistoryEntriesEqual(
  left: ComposerDocument,
  right: ComposerDocument,
): boolean {
  return (
    left.text === right.text &&
    JSON.stringify(left.textElements) === JSON.stringify(right.textElements) &&
    JSON.stringify(left.mentionBindings) ===
      JSON.stringify(right.mentionBindings) &&
    JSON.stringify(left.attachments) === JSON.stringify(right.attachments) &&
    JSON.stringify(left.pathReferences) ===
      JSON.stringify(right.pathReferences) &&
    JSON.stringify(left.pendingPastes) ===
      JSON.stringify(right.pendingPastes) &&
    JSON.stringify(left.inputCapabilityRoute) ===
      JSON.stringify(right.inputCapabilityRoute)
  );
}

export class ComposerController {
  private document: ComposerDocument;
  private revision = 0;
  private lastIntent: ComposerIntent | null = null;
  private pasteSequence = 0;
  private readonly historyEntries: ComposerDraftSnapshot[] = [];
  private historyCursor: number | null = null;
  private historyBaseDraft: ComposerDraftSnapshot | null = null;
  private snapshot: ComposerControllerSnapshot;
  private readonly listeners = new Set<ComposerListener>();

  constructor(initialDocument: ComposerDocumentInput = {}) {
    this.document = buildDocument(initialDocument);
    this.snapshot = this.createSnapshot();
  }

  getSnapshot = (): ComposerControllerSnapshot => this.snapshot;

  subscribe = (listener: ComposerListener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getDocument(): ComposerDocument {
    return cloneDocument(this.document);
  }

  captureDraft(): ComposerDraftSnapshot {
    return this.getDocument();
  }

  setText(value: string, selection?: { start?: number; end?: number }): void {
    const text = normalizeText(value);
    const selectionStart =
      selection?.start === undefined
        ? clampSelection(this.document.selectionStart, text)
        : clampSelection(selection.start, text);
    const selectionEnd =
      selection?.end === undefined
        ? selection?.start === undefined
          ? clampSelection(this.document.selectionEnd, text)
          : selectionStart
        : clampSelection(selection.end, text);
    this.updateDocument({
      ...this.document,
      text,
      selectionStart,
      selectionEnd,
    });
  }

  setSelection(start: number, end = start): void {
    this.updateDocument({
      ...this.document,
      selectionStart: clampSelection(start, this.document.text),
      selectionEnd: clampSelection(end, this.document.text),
    });
  }

  setTextElements(textElements: readonly unknown[]): void {
    this.updateDocument({ ...this.document, textElements });
  }

  setMentionBindings(mentionBindings: readonly ComposerMentionBinding[]): void {
    this.updateDocument({ ...this.document, mentionBindings });
  }

  setAttachments(attachments: readonly ComposerAttachment[]): void {
    this.updateDocument({ ...this.document, attachments });
  }

  setPathReferences(pathReferences: readonly ComposerPathReference[]): void {
    this.updateDocument({ ...this.document, pathReferences });
  }

  setInputCapabilityRoute(inputCapabilityRoute: unknown): void {
    this.updateDocument({ ...this.document, inputCapabilityRoute });
  }

  replaceDocument(document: ComposerDocumentInput): void {
    const next = buildDocument(document);
    this.updateDocument(next);
  }

  restoreDraft(draft: ComposerDocumentInput): void {
    this.recordIntent("edit");
    this.replaceDocument(draft);
  }

  clear(): void {
    this.recordIntent("clear");
    this.updateDocument(buildDocument());
  }

  ingestPaste(
    value: string,
    _options: ComposerPasteOptions = {},
  ): {
    handled: boolean;
    insertedText: string;
    placeholder?: string;
  } {
    const pasted = normalizeText(value);
    if (!pasted) {
      return { handled: false, insertedText: "" };
    }

    const start = this.document.selectionStart;
    if (pasted.length > LARGE_PASTE_CHAR_THRESHOLD) {
      const placeholder = this.nextPastePlaceholder(pasted.length);
      this.replaceSelection(placeholder, {
        placeholder,
        text: pasted,
      });
      return { handled: true, insertedText: placeholder, placeholder };
    }

    this.replaceSelection(pasted);
    return {
      handled: true,
      insertedText: this.document.text.slice(start, start + pasted.length),
    };
  }

  expandPendingPastes(text = this.document.text): string {
    return this.document.pendingPastes.reduce(
      (expanded, paste) => expanded.split(paste.placeholder).join(paste.text),
      text,
    );
  }

  submit(
    target: ComposerSubmitTarget,
    options: { allowEmpty?: boolean } = {},
  ): ComposerSubmitResult {
    const intent = target as ComposerIntent;
    this.recordIntent(intent);
    const draft = {
      ...this.captureDraft(),
      text: this.expandPendingPastes(),
    };
    const hasContent =
      options.allowEmpty === true ||
      draft.text.trim().length > 0 ||
      draft.attachments.length > 0 ||
      draft.pathReferences.length > 0;
    return hasContent || target === "interrupt"
      ? { kind: "accepted", intent, target, draft, revision: this.revision }
      : { kind: "empty", intent, target, draft, revision: this.revision };
  }

  commit(
    receipt: Extract<ComposerSubmitResult, { kind: "accepted" }>,
  ): boolean {
    if (receipt.target === "interrupt" || receipt.revision !== this.revision) {
      return false;
    }
    this.recordSubmission(receipt.draft);
    this.updateDocument(buildDocument());
    return true;
  }

  getHistory(): readonly ComposerDraftSnapshot[] {
    return this.historyEntries.map((entry) => cloneDocument(entry));
  }

  replaceHistory(entries: readonly ComposerDocumentInput[]): void {
    this.historyEntries.length = 0;
    this.appendHistoryEntries(entries);
    this.resetHistoryNavigation();
  }

  mergeHistory(entries: readonly ComposerDocumentInput[]): void {
    const incoming = this.buildHistoryEntries(entries);
    if (incoming.length === 0) {
      return;
    }
    const current = this.historyEntries.map((entry) => cloneDocument(entry));
    let matchedStart = -1;
    let matchedEnd = -1;
    let matchedSize = 0;
    for (let size = Math.min(incoming.length, current.length); size > 0; size -= 1) {
      const incomingTail = incoming.slice(-size);
      for (let start = current.length - size; start >= 0; start -= 1) {
        if (
          incomingTail.every(
            (entry, index) => {
              const existing = current[start + index];
              return (
                existing !== undefined &&
                persistedHistoryEntriesEqual(entry, existing)
              );
            },
          )
        ) {
          matchedStart = start;
          matchedEnd = start + size;
          matchedSize = size;
          break;
        }
      }
      if (matchedSize > 0) {
        break;
      }
    }
    this.historyEntries.length = 0;
    this.appendHistoryEntries([
      ...(matchedStart > 0 ? current.slice(0, matchedStart) : []),
      ...incoming,
      ...(matchedEnd >= 0 ? current.slice(matchedEnd) : current),
    ]);
    this.resetHistoryNavigation();
  }

  private appendHistoryEntries(entries: readonly ComposerDocumentInput[]): void {
    for (const entry of entries) {
      const draft = buildDocument(entry);
      if (!draft.text.trim()) {
        continue;
      }
      const previous = this.historyEntries.at(-1);
      if (!previous || !documentsEqual(previous, draft)) {
        this.historyEntries.push(draft);
      }
    }
    if (this.historyEntries.length > MAX_LOCAL_HISTORY_ENTRIES) {
      this.historyEntries.splice(
        0,
        this.historyEntries.length - MAX_LOCAL_HISTORY_ENTRIES,
      );
    }
  }

  private buildHistoryEntries(
    entries: readonly ComposerDocumentInput[],
  ): ComposerDraftSnapshot[] {
    const result: ComposerDraftSnapshot[] = [];
    for (const entry of entries) {
      const draft = buildDocument(entry);
      if (!draft.text.trim()) {
        continue;
      }
      const previous = result.at(-1);
      if (!previous || !documentsEqual(previous, draft)) {
        result.push(draft);
      }
    }
    return result;
  }

  recallHistory(
    direction: ComposerHistoryDirection,
  ): ComposerDraftSnapshot | null {
    if (this.historyEntries.length === 0) {
      return null;
    }

    if (direction === "previous") {
      if (this.historyCursor === null) {
        this.historyBaseDraft = this.captureDraft();
        this.historyCursor = this.historyEntries.length;
      }
      if (this.historyCursor === 0) {
        return null;
      }
      this.historyCursor -= 1;
      const entry = this.historyEntries[this.historyCursor];
      if (!entry) {
        return null;
      }
      this.applyHistoryDraft(entry);
      return this.captureDraft();
    }

    if (this.historyCursor === null) {
      return null;
    }
    if (this.historyCursor >= this.historyEntries.length - 1) {
      this.historyCursor = null;
      const baseDraft = this.historyBaseDraft;
      this.historyBaseDraft = null;
      if (!baseDraft) {
        return null;
      }
      this.applyHistoryDraft(baseDraft);
      return this.captureDraft();
    }

    this.historyCursor += 1;
    const entry = this.historyEntries[this.historyCursor];
    if (!entry) {
      return null;
    }
    this.applyHistoryDraft(entry);
    return this.captureDraft();
  }

  private replaceSelection(
    text: string,
    pendingPaste?: ComposerPendingPaste,
  ): void {
    const start = this.document.selectionStart;
    const end = this.document.selectionEnd;
    const nextText = `${this.document.text.slice(0, start)}${text}${this.document.text.slice(end)}`;
    this.updateDocument({
      ...this.document,
      text: nextText,
      selectionStart: start + text.length,
      selectionEnd: start + text.length,
      pendingPastes: pendingPaste
        ? [...this.document.pendingPastes, pendingPaste]
        : this.document.pendingPastes,
    });
  }

  private recordSubmission(draft: ComposerDraftSnapshot): void {
    const entry = buildDocument({
      ...draft,
      text: this.expandPendingPastes(draft.text),
    });
    if (
      !entry.text.trim() &&
      entry.attachments.length === 0 &&
      entry.pathReferences.length === 0
    ) {
      return;
    }
    const previous = this.historyEntries.at(-1);
    if (previous && documentsEqual(previous, entry)) {
      this.resetHistoryNavigation();
      return;
    }
    this.historyEntries.push(cloneDocument(entry));
    if (this.historyEntries.length > MAX_LOCAL_HISTORY_ENTRIES) {
      this.historyEntries.splice(
        0,
        this.historyEntries.length - MAX_LOCAL_HISTORY_ENTRIES,
      );
    }
    this.resetHistoryNavigation();
  }

  private applyHistoryDraft(draft: ComposerDraftSnapshot): void {
    this.lastIntent = "edit";
    this.updateDocument(draft, false);
  }

  private resetHistoryNavigation(): void {
    this.historyCursor = null;
    this.historyBaseDraft = null;
  }

  private nextPastePlaceholder(length: number): string {
    this.pasteSequence += 1;
    return `[Pasted text ${this.pasteSequence}: ${length} characters]`;
  }

  private recordIntent(intent: ComposerIntent): void {
    if (this.lastIntent === intent) {
      return;
    }
    this.lastIntent = intent;
    this.revision += 1;
    this.snapshot = this.createSnapshot();
    this.emit();
  }

  private updateDocument(document: ComposerDocument, resetHistory = true): void {
    const next = buildDocument(document);
    if (documentsEqual(this.document, next)) {
      return;
    }
    if (resetHistory) {
      this.resetHistoryNavigation();
    }
    this.document = next;
    this.revision += 1;
    this.snapshot = this.createSnapshot();
    this.emit();
  }

  private createSnapshot(): ComposerControllerSnapshot {
    return {
      document: cloneDocument(this.document),
      revision: this.revision,
      lastIntent: this.lastIntent,
    };
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}
