/**
 * 设置分类 Hook
 *
 * 定义设置页面的分组和导航项
 * 参考成熟产品的分组导航设计
 */

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  Home,
  Archive,
  User,
  BarChart3,
  Palette,
  Brain,
  Image,
  Plug,
  Search,
  ShieldCheck,
  Variable,
  Monitor,
  Code,
  Info,
  LucideIcon,
} from "lucide-react";
import { SettingsGroupKey, SettingsTabs } from "@/types/settings";

/**
 * 分类项定义
 */
export interface CategoryItem {
  key: SettingsTabs;
  label: string;
  icon: LucideIcon;
  experimental?: boolean;
}

/**
 * 分类组定义
 */
export interface CategoryGroup {
  key: SettingsGroupKey;
  title: string;
  items: CategoryItem[];
}

/**
 * 设置分类 Hook
 *
 * 返回按分组组织的设置导航项
 */
export function useSettingsCategory(): CategoryGroup[] {
  const { t } = useTranslation("settings");

  return useMemo(() => {
    const groups: CategoryGroup[] = [];

    groups.push({
      key: SettingsGroupKey.Overview,
      title: t("settings.group.overview"),
      items: [
        {
          key: SettingsTabs.Home,
          label: t("settings.tab.home"),
          icon: Home,
        },
      ],
    });

    // 账号组
    groups.push({
      key: SettingsGroupKey.Account,
      title: t("settings.group.account"),
      items: [
        {
          key: SettingsTabs.Profile,
          label: t("settings.tab.profile"),
          icon: User,
        },
        {
          key: SettingsTabs.Stats,
          label: t("settings.tab.stats"),
          icon: BarChart3,
        },
      ],
    });

    // 通用组
    groups.push({
      key: SettingsGroupKey.General,
      title: t("settings.group.general"),
      items: [
        {
          key: SettingsTabs.Appearance,
          label: t("settings.tab.appearance"),
          icon: Palette,
        },
        {
          key: SettingsTabs.Memory,
          label: t("settings.tab.memory"),
          icon: Brain,
        },
        {
          key: SettingsTabs.ArchivedConversations,
          label: t("settings.tab.archivedConversations"),
          icon: Archive,
        },
      ],
    });

    // 智能体组
    groups.push({
      key: SettingsGroupKey.Agent,
      title: t("settings.group.agent"),
      items: [
        {
          key: SettingsTabs.Providers,
          label: t("settings.tab.providers"),
          icon: Brain,
        },
        {
          key: SettingsTabs.MediaServices,
          label: t("settings.tab.mediaServices"),
          icon: Image,
        },
      ],
    });

    // 系统组
    groups.push({
      key: SettingsGroupKey.System,
      title: t("settings.group.system"),
      items: [
        {
          key: SettingsTabs.McpServer,
          label: t("settings.tab.mcpServer"),
          icon: Plug,
        },
        {
          key: SettingsTabs.WebSearch,
          label: t("settings.tab.webSearch"),
          icon: Search,
        },
        {
          key: SettingsTabs.Environment,
          label: t("settings.tab.environment"),
          icon: Variable,
        },
        {
          key: SettingsTabs.ExecutionPolicy,
          label: t("settings.tab.executionPolicy"),
          icon: ShieldCheck,
        },
        {
          key: SettingsTabs.ChromeRelay,
          label: t("settings.tab.chromeRelay"),
          icon: Monitor,
        },
        {
          key: SettingsTabs.Developer,
          label: t("settings.tab.developerLab"),
          icon: Code,
          experimental: true,
        },
        {
          key: SettingsTabs.About,
          label: t("settings.tab.about"),
          icon: Info,
        },
      ],
    });

    return groups;
  }, [t]);
}
