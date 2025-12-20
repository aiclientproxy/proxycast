import { useState, useEffect } from "react";
import {
  ExternalLink,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

interface VersionInfo {
  current: string;
  latest?: string;
  hasUpdate: boolean;
  downloadUrl?: string;
  error?: string;
}

interface ToolVersion {
  name: string;
  version: string | null;
  installed: boolean;
}

export function AboutSection() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    current: "0.7.0",
    latest: undefined,
    hasUpdate: false,
    downloadUrl: undefined,
    error: undefined,
  });
  const [checking, setChecking] = useState(false);
  const [toolVersions, setToolVersions] = useState<ToolVersion[]>([]);
  const [loadingTools, setLoadingTools] = useState(true);

  // 加载本地工具版本
  useEffect(() => {
    const loadToolVersions = async () => {
      try {
        const versions = await invoke<ToolVersion[]>("get_tool_versions");
        setToolVersions(versions);
      } catch (error) {
        console.error("Failed to load tool versions:", error);
      } finally {
        setLoadingTools(false);
      }
    };
    loadToolVersions();
  }, []);

  const handleCheckUpdate = async () => {
    setChecking(true);
    try {
      const result = await invoke<VersionInfo>("check_for_updates");
      setVersionInfo(result);
    } catch (error) {
      console.error("Failed to check for updates:", error);
      setVersionInfo((prev) => ({
        ...prev,
        error: "检查更新失败",
      }));
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      {/* 应用信息 */}
      <div className="p-6 rounded-lg border text-center space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
          <span className="text-2xl font-bold text-white">PC</span>
        </div>

        <div>
          <h2 className="text-xl font-bold">ProxyCast</h2>
          <p className="text-sm text-muted-foreground">AI API 代理服务</p>
        </div>

        <div className="flex items-center justify-center gap-2">
          <span className="text-sm">版本 {versionInfo.current}</span>
          {versionInfo.hasUpdate ? (
            <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs">
              有新版本 {versionInfo.latest}
            </span>
          ) : versionInfo.error ? (
            <span className="flex items-center gap-1 text-xs text-red-500">
              <AlertCircle className="h-3 w-3" />
              {versionInfo.error}
            </span>
          ) : versionInfo.latest ? (
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <CheckCircle2 className="h-3 w-3" />
              已是最新
            </span>
          ) : null}
        </div>

        <div className="flex items-center justify-center gap-2">
          <button
            onClick={handleCheckUpdate}
            disabled={checking}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw
              className={`h-4 w-4 ${checking ? "animate-spin" : ""}`}
            />
            检查更新
          </button>

          {versionInfo.hasUpdate && versionInfo.downloadUrl && (
            <a
              href={versionInfo.downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white text-sm hover:bg-green-700"
            >
              <ExternalLink className="h-4 w-4" />
              下载新版本
            </a>
          )}
        </div>
      </div>

      {/* 链接 */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">相关链接</h3>
        <div className="space-y-2">
          <a
            href="https://github.com/aiclientproxy/proxycast"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">GitHub 仓库</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href="https://aiclientproxy.github.io/proxycast/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">文档</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href="https://github.com/aiclientproxy/proxycast/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
          >
            <span className="text-sm">问题反馈</span>
            <ExternalLink className="h-4 w-4 text-muted-foreground" />
          </a>
        </div>
      </div>

      {/* 本地工具版本 */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">本地工具版本</h3>
        <div className="p-4 rounded-lg border space-y-3">
          {loadingTools ? (
            <>
              <ToolVersionItem name="Claude Code" version="检测中..." />
              <ToolVersionItem name="Codex" version="检测中..." />
              <ToolVersionItem name="Gemini CLI" version="检测中..." />
            </>
          ) : (
            toolVersions.map((tool) => (
              <ToolVersionItem
                key={tool.name}
                name={tool.name}
                version={tool.installed ? tool.version || "已安装" : "未安装"}
              />
            ))
          )}
        </div>
      </div>

      {/* 版权信息 */}
      <div className="text-center text-xs text-muted-foreground pt-4 border-t">
        <p>Made with love for AI developers</p>
        <p className="mt-1">2025-2026 ProxyCast</p>
      </div>
    </div>
  );
}

function ToolVersionItem({ name, version }: { name: string; version: string }) {
  const isInstalled = version !== "未安装" && !version.includes("检测");

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{name}</span>
      <div className="flex items-center gap-2">
        {isInstalled ? (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        ) : (
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        )}
        <span className="text-sm text-muted-foreground font-mono">
          {version}
        </span>
      </div>
    </div>
  );
}
