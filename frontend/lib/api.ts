const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function get(path: string) {
  const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path: string, body?: any) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get("/health"),
  metrics: (n = 20) => get(`/metrics?n=${n}`),
  buildML: () => post("/build-ml-data"),
  monitorDrift: () => post("/monitor_drift"),
  featureColumns: () => get("/feature-columns"),
};

export type MetricLog = Record<string, { R2?: number; MAE?: number; RMSE?: number; ACC?: number; F1?: number; AUC?: number }>;
