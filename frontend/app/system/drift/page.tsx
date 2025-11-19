"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function DriftPage() {
  const [path, setPath] = useState<string|null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const res = await api.monitorDrift();
      setPath(res.report_path);
    } catch {}
    setLoading(false);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Drift Reports</h1>
      <button onClick={run} disabled={loading}
        className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white disabled:opacity-50">
        {loading ? "Building..." : "Build Drift Report"}
      </button>
      {path && (
        <div className="card p-2">
          <iframe src={path} className="w-full h-[70vh] rounded-xl" />
        </div>
      )}
    </div>
  );
}
