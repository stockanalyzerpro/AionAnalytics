"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function HealthPage() {
  const [data, setData] = useState<any>(null);
  useEffect(()=>{ api.health().then(setData).catch(()=>{}); },[]);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">System Health</h1>
      <pre className="card p-4 overflow-auto text-xs">{data ? JSON.stringify(data, null, 2) : "Loading..."}</pre>
    </div>
  );
}
