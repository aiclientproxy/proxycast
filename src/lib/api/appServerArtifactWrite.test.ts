import { describe, expect, it, vi } from "vitest";
import { AppServerClient } from "./appServerClient";

describe("AppServerClient artifact/write", () => {
  it("routes typed writes through the App Server JSON-RPC gateway", async () => {
    const client = new AppServerClient();
    const request = vi.spyOn(client, "request").mockResolvedValue({} as never);
    const params = {
      threadId: "thread_1",
      turnId: "turn_2",
      artifact: {
        artifactRef: "artifact_doc_3",
        artifactDocumentId: "doc_3",
        path: "drafts/article.json",
        kind: "artifact_document",
        content: "{}",
      },
    };

    await client.writeArtifact(params);

    expect(request).toHaveBeenCalledWith("artifact/write", params, undefined);
  });
});
