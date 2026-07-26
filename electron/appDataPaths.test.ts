import { describe, expect, it } from "vitest";
import {
  resolveAgentRoot,
  resolveAppDataRoot,
  resolveDesktopStorageRoots,
  resolveHostSessionData,
} from "./appDataPaths";

describe("app data paths", () => {
  it("macOS 使用 Electron userData 作为机器数据根", () => {
    expect(
      resolveAppDataRoot({
        platform: "darwin",
        hostUserData: "/Users/test/Library/Application Support/lime",
        localAppData: "/tmp/should-not-be-used",
      }),
    ).toBe("/Users/test/Library/Application Support/lime");
  });

  it("Windows 使用 LOCALAPPDATA 而不是 roaming userData", () => {
    expect(
      resolveAppDataRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\Lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
      }),
    ).toBe("C:\\Users\\test\\AppData\\Local\\LimeCloud\\lime");
  });

  it("Windows 在 LOCALAPPDATA 缺失时从 home 推导 LocalAppData", () => {
    expect(
      resolveAppDataRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\Lime",
        home: "C:\\Users\\test",
      }),
    ).toBe("C:\\Users\\test\\AppData\\Local\\LimeCloud\\lime");
  });

  it("显式 app data override 优先于平台默认目录", () => {
    expect(
      resolveAppDataRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\Lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        appDataRootOverride: "C:\\Temp\\lime-e2e-user-data",
      }),
    ).toBe("C:\\Temp\\lime-e2e-user-data");
  });

  it("AgentRoot 只接受显式 override，不和默认根双写", () => {
    expect(
      resolveAgentRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\Lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        agentRootOverride: "C:\\Temp\\lime-agent",
      }),
    ).toBe("C:\\Temp\\lime-agent");
  });

  it("E2E root 压过 ambient AgentRoot override", () => {
    expect(
      resolveDesktopStorageRoots({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        e2eMode: true,
        e2eUserDataDir: "C:\\Temp\\lime-e2e-user-data",
        agentRootOverride: "C:\\Users\\test\\AppData\\Local\\real-agent-root",
      }),
    ).toEqual({
      appDataRoot: "C:\\Temp\\lime-e2e-user-data",
      agentRoot: "C:\\Temp\\lime-e2e-user-data\\app-server",
      hostSessionData: "C:\\Temp\\lime-e2e-user-data\\host-session",
    });
  });

  it("E2E 模式缺少隔离 root 时 fail closed", () => {
    expect(() =>
      resolveDesktopStorageRoots({
        platform: "darwin",
        hostUserData: "/Users/test/Library/Application Support/lime",
        e2eMode: true,
        agentRootOverride: "/Users/test/real-agent-root",
      }),
    ).toThrow("E2E 模式缺少 ELECTRON_E2E_USER_DATA_DIR");
  });

  it("非 E2E 模式继续接受显式 AgentRoot override", () => {
    expect(
      resolveDesktopStorageRoots({
        platform: "darwin",
        hostUserData: "/Users/test/Library/Application Support/lime",
        e2eMode: false,
        e2eUserDataDir: "/tmp/ignored-e2e-root",
        agentRootOverride: "/tmp/lime-agent-runtime",
      }),
    ).toEqual({
      appDataRoot: "/Users/test/Library/Application Support/lime",
      agentRoot: "/tmp/lime-agent-runtime",
      hostSessionData: "/Users/test/Library/Application Support/lime",
    });
  });

  it("macOS HostSessionData 与 host profile 同根", () => {
    expect(
      resolveHostSessionData({
        platform: "darwin",
        hostUserData: "/Users/test/Library/Application Support/lime",
      }),
    ).toBe("/Users/test/Library/Application Support/lime");
  });

  it("Windows HostSessionData 离开 roaming，落在 AppDataRoot 下", () => {
    expect(
      resolveHostSessionData({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
      }),
    ).toBe("C:\\Users\\test\\AppData\\Local\\LimeCloud\\lime\\host-session");
  });

  it("Windows HostSessionData 拒绝落入 Squirrel 安装树", () => {
    expect(() =>
      resolveHostSessionData({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        appDataRootOverride: "C:\\Users\\test\\AppData\\Local\\lime\\data",
      }),
    ).toThrow("Windows 数据根不能位于 Squirrel 安装根");
  });

  it("E2E 下 HostSessionData 不回退真实 userData", () => {
    expect(
      resolveDesktopStorageRoots({
        platform: "darwin",
        hostUserData: "/Users/test/Library/Application Support/lime",
        e2eMode: true,
        e2eUserDataDir: "/tmp/lime-e2e-user-data",
      }).hostSessionData,
    ).toBe("/tmp/lime-e2e-user-data");
  });

  it("默认 AgentRoot 位于独立机器数据根下", () => {
    expect(
      resolveAgentRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\Lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
      }),
    ).toBe("C:\\Users\\test\\AppData\\Local\\LimeCloud\\lime\\app-server");
  });

  it("Windows 无法解析任何机器数据根时 fail closed", () => {
    expect(() =>
      resolveAppDataRoot({
        platform: "win32",
        hostUserData: "",
      }),
    ).toThrow("无法解析 Windows AppDataRoot");
  });

  it("Windows AppDataRoot 拒绝落入 Squirrel 安装树", () => {
    expect(() =>
      resolveAppDataRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        appDataRootOverride: "C:\\Users\\test\\AppData\\Local\\lime\\data",
      }),
    ).toThrow("Windows 数据根不能位于 Squirrel 安装根");
  });

  it("Windows AgentRoot 使用大小写无关比较拒绝 Squirrel 安装树", () => {
    expect(() =>
      resolveAgentRoot({
        platform: "win32",
        hostUserData: "C:\\Users\\test\\AppData\\Roaming\\lime",
        localAppData: "C:\\Users\\test\\AppData\\Local",
        agentRootOverride: "c:\\users\\TEST\\appdata\\local\\LIME\\app-server",
      }),
    ).toThrow("Windows 数据根不能位于 Squirrel 安装根");
  });
});
