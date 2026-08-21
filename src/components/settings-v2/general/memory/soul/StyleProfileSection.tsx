import { useCallback, useEffect, useMemo, useState } from "react";
import type { TFunction } from "i18next";
import { getLimeI18n } from "@/i18n/createI18n";
import { SUPPORTED_LOCALES, type SupportedLocale } from "@/i18n/locales";
import {
  installSoulStylePack,
  listSoulStylePacks,
  setSoulStylePackStatus,
  uninstallSoulStylePack,
  type SoulStylePackListEntry,
  type SoulStylePackMutableStatus,
} from "@/lib/api/soulStylePacks";
import {
  createSoulStyleProfileRegistry,
  normalizeSoulStyleProfileId,
  type SoulStyleProfileId,
} from "@/lib/soul/style-profiles";
import {
  STYLE_PACK_INSTALL_BUSY_ID,
  StyleProfileSelector,
  type StylePackInstallDraft,
} from "./StyleProfileSelector";

interface StyleProfileSectionProps {
  value?: string | null;
  onChange: (profileId: SoulStyleProfileId) => void;
  showMessage: (message: string) => void;
  t: TFunction<"settings">;
}

function sectionT(
  t: TFunction<"settings">,
  key: string,
  values: Record<string, string | number | boolean> = {},
): string {
  const translate = t as unknown as (
    key: string,
    values?: Record<string, string | number | boolean>,
  ) => string;
  return String(translate(key, values));
}

export function StyleProfileSection({
  value,
  onChange,
  showMessage,
  t,
}: StyleProfileSectionProps) {
  const translate = useCallback(
    (key: string, values?: Record<string, string | number | boolean>) =>
      sectionT(t, key, values),
    [t],
  );
  const [packs, setPacks] = useState<SoulStylePackListEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyPackId, setBusyPackId] = useState<string | null>(null);

  const refreshPacks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listSoulStylePacks();
      const nextPacks = response.packs ?? [];
      registerStylePackLocaleResources(nextPacks);
      setPacks(nextPacks);
    } catch (refreshError) {
      console.error("加载 Soul Style Pack 失败:", refreshError);
      setError(translate("settings.memory.soul.stylePacks.message.loadFailed"));
    } finally {
      setLoading(false);
    }
  }, [translate]);

  useEffect(() => {
    void refreshPacks();
  }, [refreshPacks]);

  const registry = useMemo(() => {
    const installedPackManifests = packs
      .filter((pack) => pack.status === "enabled")
      .map((pack) => JSON.parse(pack.manifestSource) as unknown);
    return createSoulStyleProfileRegistry({ installedPackManifests });
  }, [packs]);

  useEffect(() => {
    if (loading) {
      return;
    }
    const requestedProfileId = normalizeSoulStyleProfileId(value);
    if (!requestedProfileId || registry.findProfile(requestedProfileId)) {
      return;
    }
    const fallback = registry.getFallbackProfile();
    onChange(fallback.id);
    showMessage(
      translate("settings.memory.soul.styleProfile.missing.fallbackApplied"),
    );
  }, [loading, onChange, registry, showMessage, translate, value]);

  const handleInstallPack = async (draft: StylePackInstallDraft) => {
    setBusyPackId(STYLE_PACK_INSTALL_BUSY_ID);
    setError(null);
    try {
      await installSoulStylePack({
        manifestSource: draft.manifestSource,
        localeSources: draft.localeSources,
        enableAfterInstall: draft.enableAfterInstall,
      });
      await refreshPacks();
      showMessage(
        translate("settings.memory.soul.stylePacks.message.installed"),
      );
    } catch (installError) {
      console.error("安装 Soul Style Pack 失败:", installError);
      setError(
        translate("settings.memory.soul.stylePacks.message.installFailed"),
      );
    } finally {
      setBusyPackId(null);
    }
  };

  const handleSetPackStatus = async (
    packId: string,
    status: SoulStylePackMutableStatus,
  ) => {
    setBusyPackId(packId);
    setError(null);
    try {
      await setSoulStylePackStatus({ packId, status });
      await refreshPacks();
      showMessage(
        translate(
          status === "enabled"
            ? "settings.memory.soul.stylePacks.message.enabled"
            : "settings.memory.soul.stylePacks.message.disabled",
        ),
      );
    } catch (statusError) {
      console.error("更新 Soul Style Pack 状态失败:", statusError);
      setError(
        translate("settings.memory.soul.stylePacks.message.statusFailed"),
      );
    } finally {
      setBusyPackId(null);
    }
  };

  const handleUninstallPack = async (packId: string) => {
    setBusyPackId(packId);
    setError(null);
    try {
      await uninstallSoulStylePack({ packId });
      await refreshPacks();
      showMessage(
        translate("settings.memory.soul.stylePacks.message.uninstalled"),
      );
    } catch (uninstallError) {
      console.error("卸载 Soul Style Pack 失败:", uninstallError);
      setError(
        translate("settings.memory.soul.stylePacks.message.uninstallFailed"),
      );
    } finally {
      setBusyPackId(null);
    }
  };

  return (
    <StyleProfileSelector
      value={value}
      registry={registry}
      installedPacks={packs}
      loadingPacks={loading}
      packsError={error}
      packBusyId={busyPackId}
      onChange={onChange}
      onInstallPack={handleInstallPack}
      onSetPackStatus={handleSetPackStatus}
      onUninstallPack={handleUninstallPack}
    />
  );
}

function registerStylePackLocaleResources(
  packs: readonly SoulStylePackListEntry[],
): void {
  const i18n = getLimeI18n();
  for (const pack of packs) {
    for (const locale of SUPPORTED_LOCALES) {
      const source = pack.localeSources?.[locale];
      if (typeof source !== "string" || source.trim().length === 0) {
        continue;
      }
      const resource = parseLocaleResource(source);
      if (resource) {
        i18n.addResourceBundle(locale, "settings", resource, true, true);
      }
    }
  }
}

function parseLocaleResource(source: string): Record<string, string> | null {
  try {
    const parsed = JSON.parse(source);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] =>
          typeof entry[0] === "string" && typeof entry[1] === "string",
      ),
    );
  } catch {
    return null;
  }
}

export type SoulStylePackLocaleSources = Record<SupportedLocale, string>;
