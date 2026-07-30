export interface PendingMcpServerElicitation {
  key: string;
  params: {
    _meta?: unknown;
    message: string;
    mode: "form";
    requestedSchema: Record<string, unknown>;
    serverName: string;
    threadId: string;
    turnId: string | null;
  };
}

export type McpElicitationFormValue = boolean | number | string;
export type McpElicitationFormContent = Record<string, McpElicitationFormValue>;

export type McpElicitationFormIssueCode =
  | "invalid_enum"
  | "invalid_format"
  | "invalid_integer"
  | "invalid_number"
  | "invalid_type"
  | "maximum"
  | "max_length"
  | "minimum"
  | "min_length"
  | "missing_required";

export interface McpElicitationFormIssue {
  code: McpElicitationFormIssueCode;
  field: string;
}

function assertStandardFormSchema(schema: Record<string, unknown>): void {
  if (schema.type !== "object" || !isRecord(schema.properties)) {
    throw new Error("MCP server elicitation requires a standard form schema");
  }
  const properties = schema.properties;
  const required = schema.required;
  if (
    required !== undefined &&
    (!Array.isArray(required) ||
      required.some((field) => typeof field !== "string"))
  ) {
    throw new Error("MCP server elicitation required fields are invalid");
  }
  if (
    Array.isArray(required) &&
    required.some((field) => !(field in properties))
  ) {
    throw new Error(
      "MCP server elicitation required field is not declared by properties",
    );
  }
  for (const [field, candidate] of Object.entries(properties)) {
    if (!isRecord(candidate)) {
      throw new Error(`MCP server elicitation field is invalid: ${field}`);
    }
    const type = candidate.type;
    if (
      type !== "string" &&
      type !== "number" &&
      type !== "integer" &&
      type !== "boolean"
    ) {
      throw new Error(
        `MCP server elicitation field type is unsupported: ${field}`,
      );
    }
    if (
      candidate.enum !== undefined &&
      (!Array.isArray(candidate.enum) ||
        candidate.enum.some((value) => typeof value !== "string"))
    ) {
      throw new Error(`MCP server elicitation enum is invalid: ${field}`);
    }
  }
}

export function validateMcpElicitationFormContent(
  schema: Record<string, unknown>,
  content: McpElicitationFormContent,
): McpElicitationFormIssue[] {
  assertStandardFormSchema(schema);
  const properties = schema.properties as Record<
    string,
    Record<string, unknown>
  >;
  const required = new Set(
    Array.isArray(schema.required)
      ? schema.required.filter(
          (field): field is string => typeof field === "string",
        )
      : [],
  );
  const issues: McpElicitationFormIssue[] = [];

  for (const field of required) {
    if (!(field in content)) {
      issues.push({ code: "missing_required", field });
    }
  }
  for (const [field, value] of Object.entries(content)) {
    const fieldSchema = properties[field];
    if (!fieldSchema) {
      issues.push({ code: "invalid_type", field });
      continue;
    }
    const type = fieldSchema.type;
    if (type === "string") {
      if (typeof value !== "string") {
        issues.push({ code: "invalid_type", field });
        continue;
      }
      const enumValues = Array.isArray(fieldSchema.enum)
        ? fieldSchema.enum
        : undefined;
      if (enumValues && !enumValues.includes(value)) {
        issues.push({ code: "invalid_enum", field });
      }
      if (
        typeof fieldSchema.format === "string" &&
        !matchesStringFormat(fieldSchema.format, value)
      ) {
        issues.push({ code: "invalid_format", field });
      }
      if (
        typeof fieldSchema.minLength === "number" &&
        value.length < fieldSchema.minLength
      ) {
        issues.push({ code: "min_length", field });
      }
      if (
        typeof fieldSchema.maxLength === "number" &&
        value.length > fieldSchema.maxLength
      ) {
        issues.push({ code: "max_length", field });
      }
      continue;
    }
    if (type === "boolean") {
      if (typeof value !== "boolean") {
        issues.push({ code: "invalid_type", field });
      }
      continue;
    }
    if (typeof value !== "number" || !Number.isFinite(value)) {
      issues.push({ code: "invalid_number", field });
      continue;
    }
    if (type === "integer" && !Number.isInteger(value)) {
      issues.push({ code: "invalid_integer", field });
    }
    if (
      typeof fieldSchema.minimum === "number" &&
      value < fieldSchema.minimum
    ) {
      issues.push({ code: "minimum", field });
    }
    if (
      typeof fieldSchema.maximum === "number" &&
      value > fieldSchema.maximum
    ) {
      issues.push({ code: "maximum", field });
    }
  }
  return issues;
}

function matchesStringFormat(format: string, value: string): boolean {
  switch (format) {
    case "email":
      return /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/.test(
        value,
      );
    case "uri":
      try {
        new URL(value);
        return true;
      } catch {
        return false;
      }
    case "date":
      return isValidDate(value);
    case "date-time":
      return isValidRfc3339(value);
    default:
      return true;
  }
}

function isValidDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  return isValidCalendarDate(year, month, day);
}

function isValidRfc3339(value: string): boolean {
  const match =
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|([+-])(\d{2}):(\d{2}))$/.exec(
      value,
    );
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  const offsetHour = Number(match[8] ?? 0);
  const offsetMinute = Number(match[9] ?? 0);
  return (
    isValidCalendarDate(year, month, day) &&
    hour <= 23 &&
    minute <= 59 &&
    second <= 59 &&
    offsetHour <= 23 &&
    offsetMinute <= 59
  );
}

function isValidCalendarDate(
  year: number,
  month: number,
  day: number,
): boolean {
  if (month < 1 || month > 12 || day < 1) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return day <= days[month - 1];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
