# React Hooks

## 概述

自定义 Hooks 封装业务逻辑；新代码应优先通过 `src/lib/api/*` 网关与后端通信，而不是直接在 Hook 中散落 `invoke`。

## 目录结构

```
src/hooks/
├── index.ts                # 导出入口
├── useConfiguredProviders.ts # 已配置 Provider 读取
├── useFlowEvents.ts        # 流量事件
├── useMcpServers.ts        # MCP 服务器
├── useDeepLink.ts          # Deep Link 处理
└── useSound.ts             # 音效管理
```

## 治理约束

- 新的前端能力优先落在 `src/lib/api/*`，再由 Hook 或组件消费。
- 历史桌面宿主兼容聚合层已删除，不要重新引入新的“大一统 API Hook”。
- Agent 工作台统一走 `src/components/agent/chat/hooks/index.ts` 暴露的 `useAgentChatUnified`，底层实现委托 `useAgentChat`。
- 历史 `@/hooks/useUnifiedChat` 与 `src/lib/api/unified-chat.ts` 已删除，不要重建 compat Hook / API。
- 凭证池 Hook `useProviderPool` 已删除；Provider 配置只允许走 API Key Provider / configured providers 主路径。

## 核心 Hooks

### useAgentChatUnified / useAgentChat（现役 Agent 对话）

现役 Agent / Codex 工作台事实源：

- `useAgentChatUnified -> useAgentChat -> useAgentContext / useAgentSession / useAgentTools / useAgentStream`
- 命令主链：`agentSession/turn/start -> direct v2 Thread/Turn/Item notifications -> runtime items(plan / runtime_status / artifact / tool / action) -> typed action request -> agentSession/action/respond`；`agentSession/event` 只允许显式 allowlist 内的 `message.created` durable input 及 provider/media/runtime raw side-channel，不是生命周期主链
- 适用场景：Agent 工作台、任务执行、工具审批、timeline 渲染

**相关文件**：

- 统一入口：`src/components/agent/chat/hooks/index.ts`
- Hook 实现：`src/components/agent/chat/hooks/useAgentChat.ts`
- API 封装：按职责直接使用 `src/lib/api/agentRuntime/*Client.ts`

### useConfiguredProviders

Provider 列表与默认模型读取应优先复用 `src/lib/api/modelRegistry.ts`、`src/lib/api/appConfigTypes.ts` 和现有 configured provider Hook，不要重新创建凭证池 Hook 或 OAuth Hook。

### useFlowEvents

```typescript
export function useFlowEvents() {
  const [records, setRecords] = useState<FlowRecord[]>([]);

  useEffect(() => {
    const unlisten = listen<FlowEvent>("flow-event", (event) => {
      setRecords((prev) => [event.payload.data, ...prev].slice(0, 100));
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return { records };
}
```

### useDeepLink

```typescript
export function useDeepLink() {
  useEffect(() => {
    const unlisten = listen<string>("deep-link", async (event) => {
      const url = new URL(event.payload);

      if (url.pathname === "/oem-cloud/auth/callback") {
        await handleOemCloudAuthCallback(url.searchParams);
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);
}
```

`useDeepLink` 可以处理 OEM 云登录、外部连接器回调等独立流程，但不得恢复旧 Provider OAuth / 凭证池授权回调。

## Hook 规范

### 命名约定

- 以 `use` 开头
- 描述功能: `useConfiguredProviders`, `useFlowEvents`

### 返回值

```typescript
// 返回对象，包含状态和操作
return {
  // 状态
  data,
  loading,
  error,

  // 操作
  refresh,
  add,
  remove,
};
```

## 相关文档

- [components.md](components.md) - 组件系统
- [commands.md](commands.md) - Desktop Host / App Server 命令边界
