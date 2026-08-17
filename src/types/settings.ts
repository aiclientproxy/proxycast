/**
 * 设置页面类型定义
 *
 * 定义设置分组和标签页的枚举
 */

/**
 * 设置分组 Key
 */
export enum SettingsGroupKey {
  Overview = "overview",
  Account = "account",
  General = "general",
  Agent = "agent",
  System = "system",
}

/**
 * 设置标签页
 */
export enum SettingsTabs {
  Home = "home",
  // 账号
  Profile = "profile",
  Stats = "stats",

  // 通用
  Appearance = "appearance",
  Memory = "memory",
  ArchivedConversations = "archived-conversations",

  // 智能体
  Providers = "providers",
  Skills = "skills",
  MediaServices = "media-services",

  // 系统
  McpServer = "mcp-server",
  WebSearch = "web-search",
  Environment = "environment",
  ExecutionPolicy = "execution-policy",
  ChromeRelay = "chrome-relay",
  ExecutionTracker = "execution-tracker",

  Experimental = "experimental",
  Developer = "developer",
  About = "about",
}

/**
 * 分组信息
 */
export interface SettingsGroupInfo {
  key: SettingsGroupKey;
  labelKey: string; // i18n key
}

/**
 * 标签页信息
 */
export interface SettingsTabInfo {
  key: SettingsTabs;
  labelKey: string; // i18n key
  group: SettingsGroupKey;
  experimental?: boolean;
}

/**
 * 分组到标签页的映射
 */
export const SETTINGS_GROUPS: Record<SettingsGroupKey, SettingsTabs[]> = {
  [SettingsGroupKey.Overview]: [SettingsTabs.Home],
  [SettingsGroupKey.Account]: [SettingsTabs.Profile, SettingsTabs.Stats],
  [SettingsGroupKey.General]: [
    SettingsTabs.Appearance,
    SettingsTabs.Memory,
    SettingsTabs.ArchivedConversations,
  ],
  [SettingsGroupKey.Agent]: [
    SettingsTabs.Providers,
    SettingsTabs.MediaServices,
  ],
  [SettingsGroupKey.System]: [
    SettingsTabs.McpServer,
    SettingsTabs.WebSearch,
    SettingsTabs.Environment,
    SettingsTabs.ExecutionPolicy,
    SettingsTabs.ChromeRelay,
    SettingsTabs.Developer,
    SettingsTabs.About,
  ],
};
