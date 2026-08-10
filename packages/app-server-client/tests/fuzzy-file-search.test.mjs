import assert from "node:assert/strict";
import { test } from "vitest";
import {
  AppServerClient,
  AppServerConnection,
  METHOD_FUZZY_FILE_SEARCH,
} from "../dist/index.js";

const params = {
  query: "app",
  roots: ["/workspace"],
  cancellationToken: "composer",
};

test("builds exact fuzzyFileSearch requests", () => {
  const client = new AppServerClient({ initialRequestId: 40 });

  assert.deepEqual(client.searchFiles(params), {
    id: 40,
    method: METHOD_FUZZY_FILE_SEARCH,
    params,
  });
});

test("connection sends and decodes fuzzyFileSearch responses", async () => {
  const sent = [];
  const connection = new AppServerConnection(
    {
      send(message) {
        sent.push(message);
      },
      async nextMessage() {
        return {
          id: 1,
          result: {
            files: [
              {
                root: "/workspace",
                path: "src/app.rs",
                match_type: "file",
                file_name: "app.rs",
                score: 84,
                indices: [4, 5, 6],
              },
            ],
          },
        };
      },
    },
    new AppServerClient(),
  );

  const result = await connection.searchFiles(params);

  assert.equal(sent[0].method, METHOD_FUZZY_FILE_SEARCH);
  assert.deepEqual(result.result.files[0], {
    root: "/workspace",
    path: "src/app.rs",
    match_type: "file",
    file_name: "app.rs",
    score: 84,
    indices: [4, 5, 6],
  });
});
