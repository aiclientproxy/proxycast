import { describe, expect, it, vi } from "vitest";
import {
  ComposerController,
  LARGE_PASTE_CHAR_THRESHOLD,
} from "./ComposerController";

describe("ComposerController", () => {
  it("merges persistent text history into local navigation order", () => {
    const controller = new ComposerController({ text: "current draft" });
    controller.replaceHistory([
      { text: "older" },
      { text: "newer" },
      { text: "newer" },
    ]);

    expect(controller.recallHistory("previous")?.text).toBe("newer");
    expect(controller.recallHistory("previous")?.text).toBe("older");
    expect(controller.recallHistory("next")?.text).toBe("newer");
    expect(controller.recallHistory("next")?.text).toBe("current draft");
  });

  it("does not lose a local submission when persistent history resolves late", () => {
    const controller = new ComposerController();
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("empty");

    controller.setText("本地刚提交");
    const accepted = controller.submit("start");
    expect(accepted.kind).toBe("accepted");
    if (accepted.kind === "accepted") {
      expect(controller.commit(accepted)).toBe(true);
    }

    controller.mergeHistory([{ text: "服务端旧记录" }]);

    expect(controller.getHistory().map((entry) => entry.text)).toEqual([
      "服务端旧记录",
      "本地刚提交",
    ]);
  });

  it("合并服务端快照时去掉读取边界上的重复提交", () => {
    const controller = new ComposerController();
    controller.setText("本地刚提交", { start: 0, end: 0 });
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("accepted");
    if (receipt.kind === "accepted") {
      expect(controller.commit(receipt)).toBe(true);
    }
    controller.mergeHistory([{ text: "服务端更早记录" }, { text: "本地刚提交" }]);

    expect(controller.getHistory().map((entry) => entry.text)).toEqual([
      "服务端更早记录",
      "本地刚提交",
    ]);
  });

  it("服务端同文本不会吞掉带附件的会话内历史", () => {
    const controller = new ComposerController({ text: "带图片的输入" });
    controller.setAttachments([{ data: "ZmFrZQ==", mediaType: "image/png" }]);
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("accepted");
    if (receipt.kind === "accepted") {
      expect(controller.commit(receipt)).toBe(true);
    }

    controller.mergeHistory([{ text: "带图片的输入" }]);

    expect(controller.getHistory()).toHaveLength(2);
    expect(controller.getHistory()[1]?.attachments).toHaveLength(1);
  });

  it("有界服务端快照命中中间条目时保留本地前缀", () => {
    const controller = new ComposerController();
    for (const text of ["本地较早", "本地最新"]) {
      controller.setText(text);
      const receipt = controller.submit("start");
      expect(receipt.kind).toBe("accepted");
      if (receipt.kind === "accepted") {
        expect(controller.commit(receipt)).toBe(true);
      }
    }

    controller.mergeHistory([{ text: "本地最新" }]);

    expect(controller.getHistory().map((entry) => entry.text)).toEqual([
      "本地较早",
      "本地最新",
    ]);
  });

  it("归一化换行并保留受控 textarea 的选择范围", () => {
    const controller = new ComposerController({
      text: "a\r\nb",
      selectionStart: 1,
      selectionEnd: 2,
    });

    expect(controller.getDocument()).toMatchObject({
      text: "a\nb",
      selectionStart: 1,
      selectionEnd: 2,
    });
    controller.setText("hello");
    expect(controller.getDocument()).toMatchObject({
      text: "hello",
      selectionStart: 1,
      selectionEnd: 2,
    });
  });

  it("大粘贴先使用占位符，提交时展开并在成功后清理草稿", () => {
    const controller = new ComposerController({
      text: "before",
      selectionStart: 6,
    });
    const listener = vi.fn();
    controller.subscribe(listener);
    const pasted = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 1);
    const pasteResult = controller.ingestPaste(pasted, { platform: "windows" });

    expect(pasteResult.placeholder).toMatch(/Pasted text 1/);
    expect(controller.getDocument().text).toContain(pasteResult.placeholder!);
    const receipt = controller.submit("queue");
    expect(receipt).toMatchObject({ kind: "accepted", target: "queue" });
    if (receipt.kind === "accepted") {
      expect(receipt.draft.text).toBe(`before${pasted}`);
      controller.commit(receipt);
    }
    expect(controller.getDocument().text).toBe("");
    expect(listener).toHaveBeenCalled();
  });

  it("空白草稿不会产生 start/steer 请求，但 interrupt 可接受", () => {
    const controller = new ComposerController();
    expect(controller.submit("start").kind).toBe("empty");
    expect(controller.submit("steer").kind).toBe("empty");
    expect(controller.submit("interrupt").kind).toBe("accepted");
  });

  it("异步提交期间出现新输入时不应由旧回执清空新草稿", () => {
    const controller = new ComposerController({ text: "发送内容" });
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("accepted");

    controller.setText("发送后的新草稿");
    if (receipt.kind === "accepted") {
      expect(controller.commit(receipt)).toBe(false);
    }
    expect(controller.getDocument().text).toBe("发送后的新草稿");
  });

  it("成功提交后记录会话内历史，并支持上下键回放与恢复当前草稿", () => {
    const controller = new ComposerController({ text: "第一条" });
    const firstReceipt = controller.submit("start");
    expect(firstReceipt.kind).toBe("accepted");
    if (firstReceipt.kind === "accepted") {
      expect(controller.commit(firstReceipt)).toBe(true);
    }

    controller.setText("第二条");
    const secondReceipt = controller.submit("start");
    expect(secondReceipt.kind).toBe("accepted");
    if (secondReceipt.kind === "accepted") {
      expect(controller.commit(secondReceipt)).toBe(true);
    }
    controller.setText("当前草稿");

    expect(controller.getHistory().map((entry) => entry.text)).toEqual([
      "第一条",
      "第二条",
    ]);
    expect(controller.recallHistory("previous")?.text).toBe("第二条");
    expect(controller.recallHistory("previous")?.text).toBe("第一条");
    expect(controller.recallHistory("previous")).toBeNull();
    expect(controller.recallHistory("next")?.text).toBe("第二条");
    expect(controller.recallHistory("next")?.text).toBe("当前草稿");
    expect(controller.recallHistory("next")).toBeNull();
  });

  it("未 commit 的回执不会进入会话历史", () => {
    const controller = new ComposerController({ text: "待确认" });
    const receipt = controller.submit("start");
    expect(receipt.kind).toBe("accepted");
    expect(controller.getHistory()).toHaveLength(0);
  });
});
