import { describe, expect, it } from "vitest";
import { validateMcpElicitationFormContent } from "./mcpServerElicitation";

const schema = {
  type: "object",
  properties: {
    environment: {
      type: "string",
      enum: ["staging", "production"],
    },
    retries: { type: "integer", minimum: 0, maximum: 3 },
    confirmed: { type: "boolean" },
  },
  required: ["environment", "confirmed"],
};

describe("validateMcpElicitationFormContent", () => {
  it("校验 required、enum 与 integer", () => {
    expect(
      validateMcpElicitationFormContent(schema, {
        environment: "unknown",
        retries: 1.5,
      }),
    ).toEqual([
      { code: "missing_required", field: "confirmed" },
      { code: "invalid_enum", field: "environment" },
      { code: "invalid_integer", field: "retries" },
    ]);
  });

  it("拒绝 required 引用未知 property 的非标准 schema", () => {
    expect(() =>
      validateMcpElicitationFormContent(
        {
          type: "object",
          properties: { confirmed: { type: "boolean" } },
          required: ["missing"],
        },
        {},
      ),
    ).toThrow("required field is not declared");
  });

  it("校验 email、URI、date 与 RFC3339 date-time", () => {
    const formatSchema = {
      type: "object",
      properties: {
        email: { type: "string", format: "email" },
        uri: { type: "string", format: "uri" },
        date: { type: "string", format: "date" },
        dateTime: { type: "string", format: "date-time" },
      },
      required: ["email", "uri", "date", "dateTime"],
    };
    expect(
      validateMcpElicitationFormContent(formatSchema, {
        email: "not-an-email",
        uri: "not a uri",
        date: "2026-02-30",
        dateTime: "2026-07-13T11:00:00",
      }),
    ).toEqual([
      { code: "invalid_format", field: "email" },
      { code: "invalid_format", field: "uri" },
      { code: "invalid_format", field: "date" },
      { code: "invalid_format", field: "dateTime" },
    ]);
  });
});
