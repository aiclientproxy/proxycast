const LOCAL_KEYLESS_PROVIDER_IDS = new Set([
  "ollama",
  "lmstudio",
  "gpustack",
  "ovms",
]);

const PROVIDER_ACCESS_URLS: Record<string, string> = {
  openai: "https://platform.openai.com/api-keys",
  anthropic: "https://console.anthropic.com/settings/keys",
  gemini: "https://aistudio.google.com/app/apikey",
  google: "https://aistudio.google.com/app/apikey",
  deepseek: "https://platform.deepseek.com/api_keys",
  moonshot: "https://platform.moonshot.cn/console/api-keys",
  groq: "https://console.groq.com/keys",
  grok: "https://console.x.ai/",
  mistral: "https://console.mistral.ai/api-keys/",
  perplexity: "https://www.perplexity.ai/settings/api",
  cohere: "https://dashboard.cohere.com/api-keys",
  zhipu: "https://open.bigmodel.cn/usercenter/apikeys",
  zhipuai: "https://open.bigmodel.cn/usercenter/apikeys",
  baichuan: "https://platform.baichuan-ai.com/console/key",
  dashscope: "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
  stepfun: "https://platform.stepfun.com/",
  doubao: "https://console.volcengine.com/ark",
  minimax:
    "https://platform.minimaxi.com/user-center/basic-information/interface-key",
  yi: "https://platform.lingyiwanwu.com/",
  hunyuan: "https://console.cloud.tencent.com/hunyuan",
  "tencent-cloud-ti": "https://console.cloud.tencent.com/ti",
  "baidu-cloud": "https://console.bce.baidu.com/qianfan/overview",
  infini: "https://cloud.infini-ai.com/",
  modelscope: "https://modelscope.cn/my/myaccesstoken",
  xirang: "https://xirang.aliyun.com/",
  mimo: "https://platform.xiaomi.com/",
  zhinao: "https://ai.360.com/",
  "azure-openai": "https://portal.azure.com/",
  vertexai: "https://console.cloud.google.com/vertex-ai",
  "aws-bedrock": "https://console.aws.amazon.com/bedrock/",
  github: "https://github.com/settings/tokens",
  copilot: "https://github.com/settings/copilot",
  silicon: "https://cloud.siliconflow.cn/account/ak",
  openrouter: "https://openrouter.ai/keys",
  aihubmix: "https://aihubmix.com/",
  "302ai": "https://302.ai/",
  together: "https://api.together.xyz/settings/api-keys",
  fireworks: "https://fireworks.ai/account/api-keys",
  nvidia: "https://build.nvidia.com/",
  hyperbolic: "https://app.hyperbolic.xyz/settings/api-keys",
  cerebras: "https://cloud.cerebras.ai/",
  ppio: "https://ppinfra.com/settings/key-management",
  qiniu: "https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key",
  tokenflux: "https://tokenflux.ai/dashboard/api-keys",
  cephalon: "https://cephalon.cloud/",
  lanyun: "https://lanyun.net/",
  ph8: "https://ph8.com/",
  sophnet: "https://sophnet.com/",
  ocoolai: "https://ocoolai.com/",
  dmxapi: "https://dmxapi.com/",
  aionly: "https://aionly.com/",
  burncloud: "https://ai.burncloud.com/api/usage/token/",
  alayanew: "https://www.alayanew.com/",
  longcat: "https://longcat.chat/",
  nearai: "https://cloud.near.ai",
  poe: "https://poe.com/api_key",
  huggingface: "https://huggingface.co/settings/tokens",
  "vercel-gateway": "https://vercel.com/account/tokens",
  "new-api": "https://docs.newapi.pro/",
  jina: "https://jina.ai/",
  voyageai: "https://dash.voyageai.com/api-keys",
  cherryin: "https://open.cherryin.ai/console",
  fal: "https://fal.ai/dashboard/keys",
};

function normalize(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

function isLikelyLocalHost(apiHost?: string | null): boolean {
  const host = normalize(apiHost);
  if (!host) {
    return false;
  }

  return (
    host.includes("://localhost") ||
    host.includes("://127.0.0.1") ||
    host.includes("://0.0.0.0") ||
    host.includes("://host.docker.internal")
  );
}

function resolveFallbackOrigin(apiHost?: string | null): string | null {
  const host = (apiHost || "").trim();
  if (!host) {
    return null;
  }

  try {
    return new URL(host).origin;
  } catch {
    return null;
  }
}

export interface ProviderAccessHelp {
  helpText: string | null;
  keylessHint: string | null;
  url: string | null;
}

export function getProviderAccessHelp(input: {
  providerId?: string | null;
  providerName?: string | null;
  apiHost?: string | null;
}): ProviderAccessHelp {
  const providerId = normalize(input.providerId);
  const providerName =
    (input.providerName || "").trim() || providerId || "当前渠道";

  if (
    LOCAL_KEYLESS_PROVIDER_IDS.has(providerId) ||
    isLikelyLocalHost(input.apiHost)
  ) {
    return {
      helpText: null,
      keylessHint:
        "当前渠道通常支持本地直连，通常无需 API Key；如果网关额外开启了鉴权，再按需填写。",
      url: null,
    };
  }

  const url =
    PROVIDER_ACCESS_URLS[providerId] ?? resolveFallbackOrigin(input.apiHost);
  const helpText = url
    ? `如何获取 API Key：前往 ${providerName} 的控制台或开发者页面`
    : input.apiHost
      ? `如何获取 API Key：请查看 ${providerName} 的控制台、文档或网关管理页`
      : `如何获取 API Key：先填写 ${providerName} 的 API Host，再前往对应控制台创建密钥`;

  return {
    helpText,
    keylessHint: null,
    url,
  };
}
