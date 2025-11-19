"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function DataPipelines() {
  const [resp, setResp] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try { setResp(await api.buildML()); } catch(e){ console.error(e); }
    setLoading(false);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Data Pipelines</h1>
      <button onClick={run} disabled={loading}
        className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white disabled:opacity-50">
        {loading ? "Building..." : "Build ML Dataset"}
      </button>
      {resp && <pre className="card p-4 overflow-auto text-xs">{JSON.stringify(resp, null, 2)}</pre>}
    </div>
  );
}
