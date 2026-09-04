export interface TelegramBotConfig {
  enabled: boolean;
  bot_token: string;
  allowed_user_ids: string[];
  default_model?: string;
}

export interface DiscordBotConfig {
  enabled: boolean;
  bot_token: string;
  allowed_server_ids: string[];
  default_model?: string;
  default_account?: string;
  accounts?: Record<string, DiscordAccountConfig>;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, DiscordGuildConfig>;
  streaming?: string;
  reply_to_mode?: string;
  intents?: DiscordIntentsConfig;
  actions?: DiscordActionsConfig;
  thread_bindings?: DiscordThreadBindingsConfig;
  auto_presence?: DiscordAutoPresenceConfig;
  voice?: DiscordVoiceConfig;
  agent_components?: DiscordAgentComponentsConfig;
  ui?: DiscordUiConfig;
  exec_approvals?: DiscordExecApprovalsConfig;
  response_prefix?: string;
  ack_reaction?: string;
}

export interface DiscordAccountConfig {
  enabled?: boolean;
  name?: string;
  bot_token?: string;
  allowed_server_ids?: string[];
  default_model?: string;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, DiscordGuildConfig>;
  streaming?: string;
  reply_to_mode?: string;
  intents?: DiscordIntentsConfig;
  actions?: DiscordActionsConfig;
  thread_bindings?: DiscordThreadBindingsConfig;
  auto_presence?: DiscordAutoPresenceConfig;
  voice?: DiscordVoiceConfig;
  agent_components?: DiscordAgentComponentsConfig;
  ui?: DiscordUiConfig;
  exec_approvals?: DiscordExecApprovalsConfig;
  response_prefix?: string;
  ack_reaction?: string;
}

export interface DiscordGuildConfig {
  enabled?: boolean;
  require_mention?: boolean;
  group_policy?: string;
  allow_from?: string[];
  channels?: Record<string, DiscordChannelConfig>;
}

export interface DiscordChannelConfig {
  enabled?: boolean;
  require_mention?: boolean;
  group_policy?: string;
  allow_from?: string[];
}

export interface DiscordIntentsConfig {
  message_content?: boolean;
  guild_members?: boolean;
  presence?: boolean;
}

export interface DiscordActionsConfig {
  reactions?: boolean;
  messages?: boolean;
  threads?: boolean;
  moderation?: boolean;
  presence?: boolean;
}

export interface DiscordThreadBindingsConfig {
  enabled?: boolean;
  idle_hours?: number;
  max_age_hours?: number;
  spawn_subagent_sessions?: boolean;
}

export interface DiscordAutoPresenceConfig {
  enabled?: boolean;
  interval_ms?: number;
  min_update_interval_ms?: number;
  healthy_text?: string;
  degraded_text?: string;
  exhausted_text?: string;
}

export interface DiscordVoiceAutoJoinConfig {
  guild_id: string;
  channel_id: string;
}

export interface DiscordVoiceConfig {
  enabled?: boolean;
  auto_join?: DiscordVoiceAutoJoinConfig[];
  dave_encryption?: boolean;
  decryption_failure_tolerance?: number;
}

export interface DiscordAgentComponentsConfig {
  enabled?: boolean;
}

export interface DiscordUiComponentsConfig {
  accent_color?: string;
}

export interface DiscordUiConfig {
  components?: DiscordUiComponentsConfig;
}

export interface DiscordExecApprovalsConfig {
  enabled?: boolean;
  approvers?: string[];
  agent_filter?: string[];
  session_filter?: string[];
  cleanup_after_resolve?: boolean;
  target?: string;
}

export interface FeishuBotConfig {
  enabled: boolean;
  app_id: string;
  app_secret: string;
  verification_token?: string;
  encrypt_key?: string;
  default_model?: string;
  default_account?: string;
  accounts?: Record<string, FeishuAccountConfig>;
  domain?: string;
  connection_mode?: string;
  webhook_host?: string;
  webhook_port?: number;
  webhook_path?: string;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, FeishuGroupConfig>;
  streaming?: string;
  reply_to_mode?: string;
}

export interface FeishuGroupConfig {
  enabled?: boolean;
  require_mention?: boolean;
  group_policy?: string;
  allow_from?: string[];
}

export interface FeishuAccountConfig {
  enabled?: boolean;
  name?: string;
  app_id?: string;
  app_secret?: string;
  verification_token?: string;
  encrypt_key?: string;
  default_model?: string;
  domain?: string;
  connection_mode?: string;
  webhook_host?: string;
  webhook_port?: number;
  webhook_path?: string;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, FeishuGroupConfig>;
  streaming?: string;
  reply_to_mode?: string;
}

export interface CloudflareTunnelConfig {
  account_id?: string;
  tunnel_name?: string;
  tunnel_id?: string;
  run_token?: string;
  credentials_file?: string;
  dns_name?: string;
}

export interface GatewayTunnelConfig {
  enabled?: boolean;
  provider?: string;
  mode?: string;
  binary_path?: string;
  local_host?: string;
  local_port?: number;
  public_base_url?: string;
  cloudflare?: CloudflareTunnelConfig;
}

export interface GatewayConfig {
  tunnel?: GatewayTunnelConfig;
}

export interface ChannelsConfig {
  telegram: TelegramBotConfig;
  discord: DiscordBotConfig;
  feishu: FeishuBotConfig;
  wechat: WechatBotConfig;
}

export interface WechatBotConfig {
  enabled: boolean;
  bot_token: string;
  base_url: string;
  cdn_base_url: string;
  account_id?: string;
  scanner_user_id?: string;
  default_model?: string;
  default_account?: string;
  accounts?: Record<string, WechatAccountConfig>;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, WechatGroupConfig>;
  streaming?: string;
  reply_to_mode?: string;
}

export interface WechatAccountConfig {
  enabled?: boolean;
  name?: string;
  base_url?: string;
  cdn_base_url?: string;
  bot_token?: string;
  scanner_user_id?: string;
  default_model?: string;
  dm_policy?: string;
  allow_from?: string[];
  group_policy?: string;
  group_allow_from?: string[];
  groups?: Record<string, WechatGroupConfig>;
  streaming?: string;
  reply_to_mode?: string;
}

export interface WechatGroupConfig {
  enabled?: boolean;
  group_policy?: string;
  allow_from?: string[];
}

export interface TelegramGatewayAccountStatus {
  account_id: string;
  running: boolean;
  bot_username?: string | null;
  started_at?: string | null;
  last_error?: string | null;
  last_update_id?: number | null;
  last_message_at?: string | null;
}

export interface TelegramGatewayStatus {
  running_accounts: number;
  accounts: TelegramGatewayAccountStatus[];
}

export interface FeishuGatewayAccountStatus {
  account_id: string;
  running: boolean;
  connection_mode: string;
  started_at?: string | null;
  last_error?: string | null;
  last_event_at?: string | null;
  last_message_at?: string | null;
  webhook_endpoint?: string | null;
}

export interface FeishuGatewayStatus {
  running_accounts: number;
  accounts: FeishuGatewayAccountStatus[];
}

export interface DiscordGatewayAccountStatus {
  account_id: string;
  running: boolean;
  connected: boolean;
  bot_id?: string | null;
  bot_username?: string | null;
  application_id?: string | null;
  message_content_intent?: string | null;
  started_at?: string | null;
  last_error?: string | null;
  last_event_at?: string | null;
  last_message_at?: string | null;
  last_disconnect?: string | null;
  reconnect_attempts?: number | null;
}

export interface DiscordGatewayStatus {
  running_accounts: number;
  accounts: DiscordGatewayAccountStatus[];
}

export interface WechatGatewayAccountStatus {
  accountId: string;
  running: boolean;
  startedAt?: string | null;
  lastError?: string | null;
  lastUpdateAt?: string | null;
  lastMessageAt?: string | null;
  syncBufPresent: boolean;
}

export interface WechatGatewayStatus {
  running_accounts: number;
  accounts: WechatGatewayAccountStatus[];
}

export interface GatewayChannelStatusResponse {
  channel: string;
  status: unknown;
}

export interface WechatProbeResult {
  accountId: string;
  ok: boolean;
  message: string;
}

export interface WechatLoginStartResult {
  sessionKey: string;
  qrcodeUrl: string;
  message: string;
}

export interface WechatLoginWaitResult {
  connected: boolean;
  botToken?: string;
  accountId?: string;
  baseUrl?: string;
  userId?: string;
  message: string;
}

export interface WechatConfiguredAccount {
  accountId: string;
  enabled: boolean;
  name?: string;
  baseUrl?: string;
  cdnBaseUrl?: string;
  hasToken: boolean;
  scannerUserId?: string;
}

export interface TelegramProbeResult {
  account_id: string;
  ok: boolean;
  bot_id?: number | null;
  username?: string | null;
  message: string;
}

export interface FeishuProbeResult {
  account_id: string;
  ok: boolean;
  app_id?: string | null;
  message: string;
}

export interface DiscordProbeResult {
  account_id: string;
  ok: boolean;
  bot_id?: string | null;
  username?: string | null;
  application_id?: string | null;
  message_content_intent?: string | null;
  message: string;
}

export interface GatewayTunnelStatus {
  running: boolean;
  provider: string;
  mode: string;
  binary: string;
  local_url: string;
  public_base_url?: string | null;
  pid?: number | null;
  started_at?: string | null;
  last_error?: string | null;
  last_exit?: string | null;
  command_preview?: string | null;
  connector_active?: boolean | null;
  connector_message?: string | null;
}

export interface GatewayTunnelProbeResult {
  ok: boolean;
  provider: string;
  mode: string;
  binary: string;
  version?: string | null;
  config_ready: boolean;
  message: string;
}

export interface CloudflaredInstallStatus {
  installed: boolean;
  binary: string;
  version?: string | null;
  platform: string;
  package_manager?: string | null;
  install_supported: boolean;
  install_command?: string | null;
  requires_privilege: boolean;
  message: string;
}

export interface CloudflaredInstallResult {
  ok: boolean;
  attempted: boolean;
  platform: string;
  package_manager?: string | null;
  command?: string | null;
  exit_code?: number | null;
  installed: boolean;
  version?: string | null;
  stdout: string;
  stderr: string;
  message: string;
}

export interface GatewayTunnelCreateResponse {
  result: {
    ok: boolean;
    tunnel_name: string;
    tunnel_id?: string | null;
    credentials_file?: string | null;
    dns_name?: string | null;
    public_base_url?: string | null;
    message: string;
  };
  status: GatewayTunnelStatus;
}

export interface GatewayTunnelSyncWebhookResponse {
  channel: string;
  account_id?: string | null;
  webhook_path: string;
  public_base_url: string;
  webhook_url: string;
  persisted: boolean;
}
