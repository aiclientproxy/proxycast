export function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

export function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

export function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

export function positiveInteger(value) {
  const number = nonNegativeInteger(value);
  return number != null && number > 0 ? number : null;
}
