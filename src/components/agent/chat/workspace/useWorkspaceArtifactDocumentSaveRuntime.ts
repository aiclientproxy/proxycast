import { useCallback } from "react";
import { resolveArtifactProtocolFilePath } from "@/lib/artifact-protocol";
import { saveAgentRuntimeArtifactDocumentSnapshot } from "@/lib/api/agentRuntime/appServerArtifactClient";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { Artifact } from "@/lib/artifact/types";
import type { WriteArtifactContext } from "../types";
import { buildArtifactDocumentSaveEvidenceWriteContext } from "./workspaceArtifactDocumentSaveEvidence";

export type WorkspaceArtifactWriteFile = (
  content: string,
  fileName: string,
  context?: WriteArtifactContext,
) => Promise<unknown> | unknown;

export function useWorkspaceArtifactDocumentSaveRuntime({
  handleWriteFile,
}: {
  handleWriteFile: WorkspaceArtifactWriteFile;
}) {
  return useCallback(
    async (artifact: Artifact, document: ArtifactDocumentV1) => {
      const filePath = resolveArtifactProtocolFilePath(artifact);
      const serializedDocument = JSON.stringify(document, null, 2);

      await Promise.resolve(
        handleWriteFile(serializedDocument, filePath, {
          artifactId: artifact.id,
          source: "message_content",
          status: "complete",
          artifact: {
            ...artifact,
            content: serializedDocument,
            status: "complete",
            meta: {
              ...artifact.meta,
              artifactDocument: document,
              language: "json",
              filePath:
                typeof artifact.meta.filePath === "string" &&
                artifact.meta.filePath.trim()
                  ? artifact.meta.filePath
                  : filePath,
              filename:
                typeof artifact.meta.filename === "string" &&
                artifact.meta.filename.trim()
                  ? artifact.meta.filename
                  : artifact.title,
            },
            updatedAt: Date.now(),
          },
          metadata: {
            writePhase: "persisted",
            previewText: document.summary || document.title,
            lastUpdateSource: "message_content",
          },
        }),
      );

      const saveResult = await saveAgentRuntimeArtifactDocumentSnapshot(
        artifact,
        document,
      );
      if (saveResult.status !== "written") {
        return;
      }

      await Promise.resolve(
        handleWriteFile(
          serializedDocument,
          filePath,
          buildArtifactDocumentSaveEvidenceWriteContext({
            artifact,
            document,
            evidence: saveResult.evidence,
            serializedDocument,
          }),
        ),
      );
    },
    [handleWriteFile],
  );
}
