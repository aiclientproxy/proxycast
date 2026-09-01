import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import {
  ComposerController,
  type ComposerDocumentInput,
} from "./ComposerController";
import type {
  ComposerAttachment,
  ComposerMentionBinding,
  ComposerPathReference,
  ComposerSubmitTarget,
} from "./types";

export interface UseComposerControllerOptions {
  initialDocument?: ComposerDocumentInput;
  externalDocument?: ComposerDocumentInput;
  onDocumentChange?: (
    document: ReturnType<ComposerController["getDocument"]>,
  ) => void;
}

export function useComposerController({
  initialDocument,
  externalDocument,
  onDocumentChange,
}: UseComposerControllerOptions = {}) {
  const controllerRef = useRef<ComposerController | null>(null);
  if (!controllerRef.current) {
    controllerRef.current = new ComposerController(initialDocument);
  }
  const controller = controllerRef.current;
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );

  useEffect(() => {
    if (externalDocument) {
      controller.replaceDocument(externalDocument);
    }
  }, [controller, externalDocument]);

  const notify = useCallback(() => {
    onDocumentChange?.(controller.getDocument());
  }, [controller, onDocumentChange]);
  const setText = useCallback(
    (value: string) => {
      controller.setText(value);
      notify();
    },
    [controller, notify],
  );
  const setSelection = useCallback(
    (start: number, end = start) => {
      controller.setSelection(start, end);
      notify();
    },
    [controller, notify],
  );
  const setAttachments = useCallback(
    (attachments: readonly ComposerAttachment[]) => {
      controller.setAttachments(attachments);
      notify();
    },
    [controller, notify],
  );
  const setPathReferences = useCallback(
    (references: readonly ComposerPathReference[]) => {
      controller.setPathReferences(references);
      notify();
    },
    [controller, notify],
  );
  const setMentionBindings = useCallback(
    (bindings: readonly ComposerMentionBinding[]) => {
      controller.setMentionBindings(bindings);
      notify();
    },
    [controller, notify],
  );
  const submit = useCallback(
    (target: ComposerSubmitTarget, allowEmpty = false) =>
      controller.submit(target, { allowEmpty }),
    [controller],
  );
  const recallHistory = useCallback(
    (direction: "previous" | "next") => controller.recallHistory(direction),
    [controller],
  );
  const getHistory = useCallback(() => controller.getHistory(), [controller]);

  return {
    controller,
    snapshot,
    setText,
    setSelection,
    setAttachments,
    setPathReferences,
    setMentionBindings,
    submit,
    recallHistory,
    getHistory,
  };
}
