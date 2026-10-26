const PRIMARY_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const BACKUP_BASE = "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${PRIMARY_BASE}${path}`, options);
  } catch (err1) {
    try {
      res = await fetch(`${BACKUP_BASE}${path}`, options);
    } catch {
      throw Object.assign(new Error("Cannot reach the backend. Is `uvicorn main:app` running?"), { network: true });
    }
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = body.message;
    let reasons = body.reasons || [];
    if (!msg && typeof body.detail === "string") {
      msg = body.detail;
    } else if (!msg && typeof body.detail === "object" && body.detail !== null) {
      msg = body.detail.message || "Request failed";
      if (body.detail.reasons) reasons = body.detail.reasons;
    }
    const err = new Error(msg || `Request failed (${res.status})`);
    err.reasons = reasons;
    err.status = res.status;
    throw err;
  }
  return body;
}

export const checkHealth = () => request("/health");
export const getHistory = () => request("/history");
export const clearHistory = () => request("/history", { method: "DELETE" });
export const getModelInfo = () => request("/model-info");

/** Sends ONE image; the backend returns the prediction and the extracted features. */
export function predictImage(file) {
  const form = new FormData();
  form.append("file", file, file.name || "capture.jpg");
  return request("/predict", { method: "POST", body: form });
}
