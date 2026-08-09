/* global Buffer, process */
import { spawn } from "node:child_process";
import path from "node:path";

export type ProjectPathOpenTool = "vscode" | "cursor" | "terminal" | "finder";

export async function openProjectPathWithLocalTool(
  targetPath: string,
  tool: Exclude<ProjectPathOpenTool, "finder">,
): Promise<void> {
  const command = resolveProjectPathOpenCommand(targetPath, tool);
  await runProjectPathOpenCommand(command.executable, command.args, {
    cwd: command.cwd,
  });
}

function resolveProjectPathOpenCommand(
  targetPath: string,
  tool: Exclude<ProjectPathOpenTool, "finder">,
): { executable: string; args: string[]; cwd?: string } {
  if (process.platform === "darwin") {
    if (tool === "terminal") {
      return {
        executable: "open",
        args: ["-a", "Terminal", targetPath],
      };
    }
    return {
      executable: "open",
      args: [
        "-a",
        tool === "vscode" ? "Visual Studio Code" : "Cursor",
        targetPath,
      ],
    };
  }

  if (process.platform === "win32") {
    if (tool === "terminal") {
      return {
        executable: "cmd.exe",
        args: ["/c", "start", "", "cmd.exe", "/K", "cd", "/d", targetPath],
      };
    }
    return {
      executable: "cmd.exe",
      args: ["/c", tool === "vscode" ? "code" : "cursor", targetPath],
    };
  }

  if (tool === "terminal") {
    return {
      executable: "x-terminal-emulator",
      args: [],
      cwd: targetPath,
    };
  }

  return {
    executable: tool === "vscode" ? "code" : "cursor",
    args: [targetPath],
  };
}

async function runProjectPathOpenCommand(
  executable: string,
  args: string[],
  options: { cwd?: string },
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd: options.cwd,
      detached: process.platform === "win32",
      stdio: ["ignore", "ignore", "pipe"],
      windowsHide: true,
    });
    let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      reject(new Error(`打开项目工具失败: ${error.message}`));
    });
    child.on("close", (code) => {
      if (code === 0 || code === null) {
        resolve();
        return;
      }
      reject(new Error(`打开项目工具失败: ${stderr.trim() || `exit ${code}`}`));
    });
    child.unref();
  });
}
