"use client";
import { useEffect, useState } from "react";

export default function SystemBar() {
  const [status, setStatus] = useState({
    drift: "⚪ Checking...",
    retraining: "⚪ Idle",
    lastUpdate: "—",
    retrainCycles: "—",
    newsCount: "—",
    tickersTracked: "—",
    version: "SAP v1.4.2",
  });

  async function fetchStatus() {
    try {
      const res = await fetch("http://127.0.0.1:8080/system/status");
      if (!res.ok) throw new Error("Backend not reachable");
      const data = await res.json();

      setStatus({
        drift: data.drift || "✅ Stable",
        retraining: data.retraining || "🧠 Ready",
        lastUpdate:
          data.last_update ||
          new Date().toLocaleString(undefined, { hour12: false }),
        retrainCycles: data.retrain_cycles || "—",
        newsCount: data.news_articles || "—",
        tickersTracked: data.tickers_tracked || "—",
        version: data.version || "SAP v1.4.2",
      });
    } catch {
      // fallback if backend is offline
      setStatus((prev) => ({
        ...prev,
        drift: "⚠️ Offline",
        retraining: "—",
      }));
    }
  }

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 60000); // refresh every 60s
    return () => clearInterval(id);
  }, []);

  const items = [
    `Drift: ${status.drift}`,
    `Retraining: ${status.retraining}`,
    `Last Update: ${status.lastUpdate}`,
    `Retrain Cycles: ${status.retrainCycles}`,
    `News Articles: ${status.newsCount}`,
    `Tickers Tracked: ${status.tickersTracked}`,
    `${status.version}`,
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-slate-950/80 backdrop-blur-xs border-t border-slate-800">
      <div className="mx-auto max-w-7xl px-4 py-2 text-xs text-slate-400 flex flex-wrap items-center justify-center gap-4">
        {items.map((t, i) => (
          <span key={i} className="opacity-80">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
