import http from "node:http";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function jsonResponse(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
  });
  response.end(`${JSON.stringify(payload)}\n`);
}

function textResponse(response, statusCode, text) {
  response.writeHead(statusCode, {
    "content-type": "text/plain; charset=utf-8",
  });
  response.end(text);
}

function readFormBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let totalBytes = 0;
    request.on("data", (chunk) => {
      totalBytes += chunk.length;
      if (totalBytes > 1_000_000) {
        reject(new Error("fixture OAuth request body exceeds 1MB"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("error", reject);
    request.on("end", () => {
      const text = Buffer.concat(chunks).toString("utf8");
      resolve(new URLSearchParams(text));
    });
  });
}

function parseUrl(request, baseUrl) {
  return new URL(request.url || "/", baseUrl);
}

function createOAuthProviderServer() {
  const state = {
    baseUrl: "",
    authorizeQueries: [],
    tokenRequests: [],
    registrationRequests: [],
    openedRequests: [],
  };

  const server = http.createServer(async (request, response) => {
    try {
      const url = parseUrl(request, state.baseUrl);

      if (
        request.method === "GET" &&
        (url.pathname === "/.well-known/oauth-authorization-server" ||
          url.pathname === "/.well-known/oauth-authorization-server/mcp" ||
          url.pathname === "/mcp/.well-known/oauth-authorization-server")
      ) {
        jsonResponse(response, 200, {
          issuer: state.baseUrl,
          authorization_endpoint: `${state.baseUrl}/authorize`,
          token_endpoint: `${state.baseUrl}/token`,
          registration_endpoint: `${state.baseUrl}/register`,
          response_types_supported: ["code"],
          scopes_supported: ["fixture.read", "fixture.write"],
        });
        return;
      }

      if (request.method === "POST" && url.pathname === "/register") {
        const body = await readFormBody(request);
        state.registrationRequests.push(Object.fromEntries(body.entries()));
        jsonResponse(response, 200, {
          client_id: "dynamic-fixture-client",
          client_name: "Lime MCP Client",
          redirect_uris: [],
        });
        return;
      }

      if (request.method === "GET" && url.pathname === "/opened") {
        state.openedRequests.push({
          userAgent: request.headers["user-agent"] || null,
        });
        textResponse(response, 200, "MCP OAuth fixture opened");
        return;
      }

      if (request.method === "GET" && url.pathname === "/authorize") {
        const query = Object.fromEntries(url.searchParams.entries());
        state.authorizeQueries.push(query);
        const redirectUri = url.searchParams.get("redirect_uri");
        const oauthState = url.searchParams.get("state");
        if (!redirectUri || !oauthState) {
          textResponse(response, 400, "missing redirect_uri or state");
          return;
        }
        const callback = new URL(redirectUri);
        callback.searchParams.set("code", "fixture-auth-code");
        callback.searchParams.set("state", oauthState);
        response.writeHead(302, { location: callback.toString() });
        response.end();
        return;
      }

      if (request.method === "POST" && url.pathname === "/token") {
        const body = await readFormBody(request);
        const tokenRequest = Object.fromEntries(body.entries());
        state.tokenRequests.push(tokenRequest);
        if (tokenRequest.grant_type !== "authorization_code") {
          jsonResponse(response, 400, { error: "unsupported_grant_type" });
          return;
        }
        if (tokenRequest.code !== "fixture-auth-code") {
          jsonResponse(response, 400, { error: "invalid_grant" });
          return;
        }
        jsonResponse(response, 200, {
          access_token: "fixture-access-token",
          token_type: "Bearer",
          refresh_token: "fixture-refresh-token",
          expires_in: 3600,
        });
        return;
      }

      textResponse(response, 404, "not found");
    } catch (error) {
      textResponse(
        response,
        500,
        error instanceof Error ? error.message : String(error),
      );
    }
  });

  return { server, state };
}

export async function startMcpOAuthFixtureProvider() {
  const { server, state } = createOAuthProviderServer();

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    await new Promise((resolve) => server.close(resolve));
    throw new Error("MCP OAuth fixture provider did not bind TCP port");
  }

  state.baseUrl = `http://127.0.0.1:${address.port}`;
  return {
    baseUrl: state.baseUrl,
    mcpUrl: `${state.baseUrl}/mcp`,
    state,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

function findServer(servers, serverName) {
  return Array.isArray(servers)
    ? servers.find((server) => server?.name === serverName)
    : null;
}

function authStatusOf(server) {
  return server?.authStatus || null;
}

async function waitForAuthorizedStatus({
  options,
  entries,
  invokeAppServerMethod,
  serverName,
}) {
  const startedAt = Date.now();
  let lastStatus = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const result = await invokeAppServerMethod(
      options,
      "mcpServerStatus/list",
      {},
      entries,
    );
    const server = findServer(result?.data, serverName);
    lastStatus = authStatusOf(server);
    if (lastStatus === "oauth") {
      return { server, authStatus: lastStatus };
    }
    await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
  }

  throw new Error(
    `MCP OAuth fixture did not reach authorized status: ${JSON.stringify(lastStatus)}`,
  );
}

function summarizeProviderState(provider) {
  const lastAuthorizeQuery = provider.state.authorizeQueries.at(-1) || null;
  const lastTokenRequest = provider.state.tokenRequests.at(-1) || null;
  return {
    authorizeRequestCount: provider.state.authorizeQueries.length,
    registrationRequestCount: provider.state.registrationRequests.length,
    tokenRequestCount: provider.state.tokenRequests.length,
    openedRequestCount: provider.state.openedRequests.length,
    lastAuthorizeQuery,
    lastTokenRequest: lastTokenRequest
      ? {
          grant_type: lastTokenRequest.grant_type,
          code: lastTokenRequest.code ? "[present]" : undefined,
          redirect_uri: lastTokenRequest.redirect_uri,
        }
      : null,
  };
}

async function deleteFixtureServer({
  options,
  entries,
  invokeAppServerMethod,
  serverId,
}) {
  await invokeAppServerMethod(
    options,
    "mcpServer/delete",
    { id: serverId },
    entries,
  ).catch((error) => {
    console.warn(
      `[smoke:mcp-current] OAuth fixture delete failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
}

export async function runMcpOAuthFixtureSmoke({
  options,
  entries,
  invokeAppServerMethod,
  invokeBridgeCommand,
}) {
  const provider = await startMcpOAuthFixtureProvider();
  const serverId = `mcp-oauth-current-${Date.now()}`;
  const serverName = serverId.replace(/[^a-zA-Z0-9_-]/g, "-");

  try {
    const createResult = await invokeAppServerMethod(
      options,
      "mcpServer/create",
      {
        server: {
          id: serverId,
          name: serverName,
          description: "Current MCP OAuth smoke fixture",
          server_config: {
            transport: "streamable_http",
            url: provider.mcpUrl,
            timeout: 3,
            scopes: ["fixture.read"],
          },
          enabled_lime: true,
          enabled_claude: false,
          enabled_codex: false,
          enabled_gemini: false,
          created_at: Date.now(),
        },
      },
      entries,
    );
    assert(
      Array.isArray(createResult?.servers),
      "mcpServer/create did not return servers",
    );

    const login = await invokeAppServerMethod(
      options,
      "mcpServer/oauth/login",
      { name: serverName, timeoutSecs: 30 },
      entries,
    );
    assert(
      typeof login?.authorizationUrl === "string" &&
        login.authorizationUrl.startsWith(`${provider.baseUrl}/authorize`),
      "mcpServer/oauth/login did not return fixture authorizationUrl",
    );

    await invokeBridgeCommand(
      options,
      "open_external_url",
      { url: `${provider.baseUrl}/opened` },
      entries,
    );

    const authResponse = await fetch(login.authorizationUrl, {
      redirect: "follow",
      signal: AbortSignal.timeout(Math.min(30_000, options.timeoutMs)),
    });
    assert(
      authResponse.ok,
      `fixture OAuth browser redirect failed: HTTP ${authResponse.status}`,
    );

    const authorized = await waitForAuthorizedStatus({
      options,
      entries,
      invokeAppServerMethod,
      serverName,
    }).catch((error) => {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${detail}; provider=${JSON.stringify(summarizeProviderState(provider))}`,
      );
    });

    const authorizeQuery = provider.state.authorizeQueries.at(-1) || {};
    assert(
      authorizeQuery.scope === "fixture.read",
      `fixture OAuth authorize scope drifted: ${authorizeQuery.scope || "<none>"}`,
    );
    assert(
      provider.state.tokenRequests.some(
        (request) =>
          request.grant_type === "authorization_code" &&
          request.code === "fixture-auth-code",
      ),
      "fixture OAuth token endpoint was not reached",
    );

    return {
      serverId,
      serverName,
      providerBaseUrl: provider.baseUrl,
      authorizationUrlHost: new URL(login.authorizationUrl).host,
      authorizeRequestCount: provider.state.authorizeQueries.length,
      registrationRequestCount: provider.state.registrationRequests.length,
      tokenRequestCount: provider.state.tokenRequests.length,
      authStatus: authorized.authStatus,
    };
  } finally {
    await deleteFixtureServer({
      options,
      entries,
      invokeAppServerMethod,
      serverId,
    });
    await provider.close();
  }
}
