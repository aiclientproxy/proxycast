import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID,
  SEEDED_SERVICE_SKILL_CATALOG_VERSION,
} from "@/lib/base-setup/seededServiceSkillPackage";
import { readStoredBaseSetupPackageSnapshot } from "@/lib/base-setup/storage";
import {
  LEGACY_SERVICE_SKILL_EXECUTION_COMPAT_NOTE,
  SERVICE_SKILL_EXECUTION_LOCATION_LABEL,
  clearServiceSkillCatalogCache,
  getSeededServiceSkillCatalog,
  getServiceSkillCatalog,
  listServiceSkills,
  refreshServiceSkillCatalogFromRemote,
  resolveServiceSkillExecutionLocationPresentation,
  saveServiceSkillCatalog,
  subscribeServiceSkillCatalogChanged,
  type ServiceSkillCatalog,
  type ServiceSkillCompatCatalog,
} from "./serviceSkills";

function buildRemoteBaseSetupPackage() {
  return {
    id: "tenant-scene-pack",
    version: "tenant-2026-04-15",
    title: "Tenant Scene Pack",
    summary: "租户下发的基础设置包",
    bundle_refs: [
      {
        id: "tenant-bundle",
        source: "remote",
        path_or_uri: "lime://bundles/tenant",
        kind: "skill_bundle",
      },
    ],
    catalog_projections: [
      {
        id: "tenant-base-setup-skill",
        target_catalog: "service_skill_catalog",
        entry_key: "tenant-base-setup-skill",
        skill_key: "tenant-base-setup-skill",
        title: "租户基础设置场景",
        summary: "通过基础设置包编译出来的目录项",
        category: "租户能力",
        output_hint: "结果包",
        bundle_ref_id: "tenant-bundle",
        slot_profile_ref: "tenant-slot-profile",
        binding_profile_ref: "tenant-binding-profile",
        artifact_profile_ref: "tenant-artifact-profile",
        scorecard_profile_ref: "tenant-scorecard-profile",
        policy_profile_ref: "tenant-policy-profile",
        aliases: ["租户场景"],
        trigger_hints: ["租户下发的新场景"],
      },
    ],
    slot_profiles: [
      {
        id: "tenant-slot-profile",
        slots: [
          {
            key: "topic",
            label: "主题",
            type: "text",
            required: true,
            placeholder: "输入主题",
          },
        ],
      },
    ],
    binding_profiles: [
      {
        id: "tenant-binding-profile",
        binding_family: "agent_turn",
        runner_type: "instant",
      },
    ],
    artifact_profiles: [
      {
        id: "tenant-artifact-profile",
        delivery_contract: "artifact_bundle",
        required_parts: ["index.md"],
        viewer_kind: "artifact_bundle",
        default_artifact_kind: "brief",
        output_destination: "workspace",
      },
    ],
    scorecard_profiles: [
      {
        id: "tenant-scorecard-profile",
        metrics: ["acceptance_rate"],
      },
    ],
    policy_profiles: [
      {
        id: "tenant-policy-profile",
        enabled: true,
        surface_scopes: ["home", "workspace"],
      },
    ],
    compatibility: {
      min_app_version: "1.11.0",
      required_kernel_capabilities: ["agent_turn", "artifact_viewer"],
      seeded_fallback: true,
      compat_catalog_projection: true,
    },
  };
}

function buildRemoteCatalog(): ServiceSkillCatalog {
  const seeded = getSeededServiceSkillCatalog();
  return {
    version: "tenant-2026-03-24",
    tenantId: "tenant-demo",
    syncedAt: "2026-03-24T12:00:00.000Z",
    items: [
      {
        ...seeded.items[0]!,
        id: "tenant-remote-skill",
        title: "租户短视频脚本",
        summary: "租户下发的远端目录项",
        version: "tenant-2026-03-24",
      },
    ],
  };
}

function buildLegacyExecutionLocationRemoteCatalog(): ServiceSkillCompatCatalog {
  const seeded = getSeededServiceSkillCatalog();
  return {
    version: "tenant-2026-04-21-legacy-compat",
    tenantId: "tenant-demo",
    syncedAt: "2026-04-21T12:00:00.000Z",
    items: [
      {
        ...seeded.items[0]!,
        id: "tenant-legacy-compat-skill",
        title: "租户旧版兼容技能",
        summary: "历史目录仍把服务技能写成 cloud_required。",
        defaultExecutorBinding: "agent_turn",
        executionLocation: "cloud_required",
        version: "tenant-2026-04-21-legacy-compat",
      },
    ],
  };
}

function buildStaleRemoteCatalog(): ServiceSkillCatalog {
  const seeded = getSeededServiceSkillCatalog();
  return {
    version: "tenant-2026-03-20",
    tenantId: "tenant-demo",
    syncedAt: "2026-03-20T08:00:00.000Z",
    items: [
      {
        ...seeded.items[0]!,
        id: "tenant-stale-skill",
        title: "过期目录项",
        summary: "旧的租户目录快照",
        version: "tenant-2026-03-20",
      },
    ],
  };
}

describe("serviceSkills API", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
    delete window.__LIME_OEM_CLOUD__;
    delete window.__LIME_SESSION_TOKEN__;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("应统一解析 current 与 compat 执行位置展示语义", () => {
    expect(
      resolveServiceSkillExecutionLocationPresentation("client_default"),
    ).toEqual({
      label: SERVICE_SKILL_EXECUTION_LOCATION_LABEL,
      legacyCompat: false,
    });
    expect(
      resolveServiceSkillExecutionLocationPresentation("cloud_required"),
    ).toEqual({
      label: SERVICE_SKILL_EXECUTION_LOCATION_LABEL,
      legacyCompat: true,
    });
    expect(LEGACY_SERVICE_SKILL_EXECUTION_COMPAT_NOTE).toBe(
      "沿用旧目录兼容标记，实际仍在客户端执行。",
    );
    expect(
      resolveServiceSkillExecutionLocationPresentation("browser_assist"),
    ).toBeNull();
  });

  it("无缓存时应回退到 seeded catalog", async () => {
    const catalog = await getServiceSkillCatalog();

    expect(catalog.tenantId).toBe(SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID);
    expect(catalog.items.length).toBeGreaterThan(0);
    expect(
      catalog.items.find((item) => item.id === "carousel-post-replication"),
    ).toEqual(
      expect.objectContaining({
        skillType: "service",
        entryHint: expect.any(String),
        aliases: expect.arrayContaining(["轮播帖"]),
        usageGuidelines: expect.arrayContaining([
          "适合先产出一版结构化草稿，再在当前工作区继续精修。",
        ]),
        surfaceScopes: expect.arrayContaining(["mention"]),
        promptTemplateKey: "replication",
        skillBundle: expect.objectContaining({
          name: "carousel-post-replication",
          standardCompliance: expect.objectContaining({
            isStandard: true,
          }),
          metadata: expect.objectContaining({
            Lime_base_setup_package_id: "lime-seeded-service-skills",
            Lime_skill_type: "service",
            Lime_prompt_template_key: "replication",
          }),
        }),
      }),
    );
    expect(
      catalog.items.find((item) => item.id === "account-performance-tracking"),
    ).toEqual(
      expect.objectContaining({
        title: "账号增长跟踪",
        aliases: expect.arrayContaining(["增长跟踪", "自动增长"]),
        triggerHints: expect.arrayContaining([
          "需要围绕目标账号持续跟踪内容节奏和提醒条件时使用。",
        ]),
      }),
    );

    const cached = window.localStorage.getItem("lime:service-skill-catalog:v1");
    expect(cached).toContain(
      `"tenantId":"${SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID}"`,
    );
  });

  it("保存远端目录后应优先返回远端 catalog", async () => {
    const remoteCatalog = buildRemoteCatalog();

    saveServiceSkillCatalog(remoteCatalog, "bootstrap_sync");

    const catalog = await getServiceSkillCatalog();
    const skills = await listServiceSkills();

    expect(catalog.tenantId).toBe("tenant-demo");
    expect(catalog.version).toBe("tenant-2026-03-24");
    expect(skills.map((item) => item.id)).toEqual(
      expect.arrayContaining(["tenant-remote-skill"]),
    );
  });

  it("旧格式远端目录缺少 skillBundle 时应自动补齐标准摘要", async () => {
    const remoteCatalog = buildRemoteCatalog();
    if (remoteCatalog.items[0]) {
      delete remoteCatalog.items[0].skillBundle;
    }

    saveServiceSkillCatalog(remoteCatalog, "bootstrap_sync");

    const catalog = await getServiceSkillCatalog();

    expect(catalog.items[0]?.skillBundle).toEqual(
      expect.objectContaining({
        name: "carousel-post-replication",
        standardCompliance: expect.objectContaining({
          isStandard: true,
        }),
        metadata: expect.objectContaining({
          Lime_skill_type: "service",
          Lime_output_destination:
            "结果会写回当前工作区中的内容草稿，方便继续改写和发布。",
        }),
      }),
    );
  });

  it("旧执行位置标记的派生 metadata 应正规化为本地执行语义", async () => {
    const remoteCatalog = buildLegacyExecutionLocationRemoteCatalog();
    if (remoteCatalog.items[0]) {
      delete remoteCatalog.items[0].skillBundle;
    }

    saveServiceSkillCatalog(remoteCatalog, "bootstrap_sync");

    const catalog = await getServiceSkillCatalog();

    expect(catalog.items[0]).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "agent_turn",
        executionLocation: "client_default",
        skillBundle: expect.objectContaining({
          metadata: expect.objectContaining({
            Lime_execution_location: "client_default",
            Lime_executor_binding: "agent_turn",
          }),
        }),
      }),
    );

    const stored = window.localStorage.getItem("lime:service-skill-catalog:v1");
    expect(stored).toContain('"Lime_execution_location":"client_default"');
    expect(stored).toContain('"Lime_executor_binding":"agent_turn"');
    expect(stored).toContain('"executionLocation":"client_default"');
    expect(stored).toContain('"defaultExecutorBinding":"agent_turn"');
    expect(stored).not.toContain('"executionLocation":"cloud_required"');
    expect(stored).not.toContain('"Lime_execution_location":"cloud_required"');
  });

  it("清空缓存后应恢复到 seeded catalog", async () => {
    saveServiceSkillCatalog(buildRemoteCatalog(), "bootstrap_sync");

    clearServiceSkillCatalogCache();
    const catalog = await getServiceSkillCatalog();

    expect(catalog.tenantId).toBe(SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID);
    expect(catalog.version).toBe(SEEDED_SERVICE_SKILL_CATALOG_VERSION);
  });

  it("当前 OEM 租户不匹配时不应读取其他租户的缓存目录", async () => {
    saveServiceSkillCatalog(buildRemoteCatalog(), "manual_override");
    window.__LIME_OEM_CLOUD__ = {
      baseUrl: "https://oem.example.com",
      tenantId: "tenant-other",
    };

    const catalog = await getServiceSkillCatalog();

    expect(catalog.tenantId).toBe(SEEDED_SERVICE_SKILL_CATALOG_TENANT_ID);
    expect(catalog.version).toBe(SEEDED_SERVICE_SKILL_CATALOG_VERSION);
  });

  it("旧的 seeded 本地缓存应自动升级到当前 seeded 目录", async () => {
    const oldSeeded = getSeededServiceSkillCatalog();
    const downgraded = {
      ...oldSeeded,
      items: oldSeeded.items.filter(
        (item) => item.id !== "daily-trend-briefing",
      ),
    };

    window.localStorage.setItem(
      "lime:service-skill-catalog:v1",
      JSON.stringify(downgraded),
    );

    const catalog = await getServiceSkillCatalog();

    expect(
      catalog.items.some((item) => item.id === "daily-trend-briefing"),
    ).toBe(true);
    const stored = window.localStorage.getItem("lime:service-skill-catalog:v1");
    expect(stored).toContain('"daily-trend-briefing"');
  });

  it("目录变更时应广播 catalog change 事件", () => {
    const callback = vi.fn();
    const unsubscribe = subscribeServiceSkillCatalogChanged(callback);

    try {
      saveServiceSkillCatalog(buildRemoteCatalog(), "bootstrap_sync");
      clearServiceSkillCatalogCache();

      expect(callback).toHaveBeenNthCalledWith(1, "bootstrap_sync");
      expect(callback).toHaveBeenNthCalledWith(2, "cache_clear");
    } finally {
      unsubscribe();
    }
  });

  it("存在 OEM 注入时应优先从远端刷新目录", async () => {
    window.__LIME_OEM_CLOUD__ = {
      baseUrl: "https://oem.example.com",
      tenantId: "tenant-demo",
    };
    window.__LIME_SESSION_TOKEN__ = "session-token-demo";

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        code: 200,
        message: "success",
        data: buildRemoteCatalog(),
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const catalog = await refreshServiceSkillCatalogFromRemote();
    const stored = await getServiceSkillCatalog();

    expect(fetchMock).toHaveBeenCalledWith(
      "https://oem.example.com/api/v1/public/tenants/tenant-demo/client/service-skills",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          Accept: "application/json",
          Authorization: "Bearer session-token-demo",
        }),
      }),
    );
    expect(catalog?.tenantId).toBe("tenant-demo");
    expect(stored.tenantId).toBe("tenant-demo");
  });

  it("远端刷新返回更旧目录时不应回退当前缓存版本", async () => {
    saveServiceSkillCatalog(buildRemoteCatalog(), "manual_override");
    window.__LIME_OEM_CLOUD__ = {
      baseUrl: "https://oem.example.com",
      tenantId: "tenant-demo",
    };
    window.__LIME_SESSION_TOKEN__ = "session-token-demo";

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        code: 200,
        message: "success",
        data: buildStaleRemoteCatalog(),
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const catalog = await refreshServiceSkillCatalogFromRemote();
    const stored = await getServiceSkillCatalog();

    expect(catalog?.version).toBe("tenant-2026-03-24");
    expect(stored.version).toBe("tenant-2026-03-24");
    expect(stored.items[0]?.id).toBe("tenant-remote-skill");
  });

  it("应支持把 Base Setup Package 编译成 compat catalog", async () => {
    saveServiceSkillCatalog(buildRemoteBaseSetupPackage(), "bootstrap_sync");

    const catalog = await getServiceSkillCatalog();
    const storedSnapshot = readStoredBaseSetupPackageSnapshot();

    expect(catalog.tenantId).toBe("base-setup");
    expect(catalog.version).toBe("tenant-2026-04-15");
    expect(storedSnapshot).toEqual(
      expect.objectContaining({
        packageId: "tenant-scene-pack",
        packageVersion: "tenant-2026-04-15",
        tenantId: "base-setup",
      }),
    );
    expect(catalog.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "tenant-base-setup-skill",
          skillKey: "tenant-base-setup-skill",
          title: "租户基础设置场景",
          defaultExecutorBinding: "agent_turn",
          outputDestination: "workspace",
          skillBundle: expect.objectContaining({
            metadata: expect.objectContaining({
              Lime_base_setup_package_id: "tenant-scene-pack",
              Lime_projection_id: "tenant-base-setup-skill",
            }),
          }),
        }),
      ]),
    );
  });

  it("切回普通 catalog 时应清理已有基础设置快照", async () => {
    saveServiceSkillCatalog(buildRemoteBaseSetupPackage(), "bootstrap_sync");
    expect(readStoredBaseSetupPackageSnapshot()?.packageId).toBe(
      "tenant-scene-pack",
    );

    saveServiceSkillCatalog(buildRemoteCatalog(), "bootstrap_sync");

    expect(readStoredBaseSetupPackageSnapshot()).toBeNull();
  });

  it("远端刷新应支持服务端直接返回 Base Setup Package", async () => {
    window.__LIME_OEM_CLOUD__ = {
      baseUrl: "https://oem.example.com",
      tenantId: "tenant-demo",
    };
    window.__LIME_SESSION_TOKEN__ = "session-token-demo";

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        code: 200,
        message: "success",
        data: buildRemoteBaseSetupPackage(),
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const catalog = await refreshServiceSkillCatalogFromRemote();
    const stored = await getServiceSkillCatalog();
    const storedSnapshot = readStoredBaseSetupPackageSnapshot();

    expect(catalog?.tenantId).toBe("tenant-demo");
    expect(stored.items[0]?.id).toBe("tenant-base-setup-skill");
    expect(storedSnapshot).toEqual(
      expect.objectContaining({
        packageId: "tenant-scene-pack",
        tenantId: "tenant-demo",
      }),
    );
  });
});
