"use client";
import { useEffect, useState } from "react";
import { Activity } from "lucide-react";

export default function AccuracyCard() {
  const [acc, setAcc] = useState<number | null>(null);
  const [wow, setWow] = useState<number | null>(null);
  const [summary, setSummary] = useState<string>("");

  useEffect(() => {
    async function fetchAccuracy() {
      try {
        const res = await fetch("http://127.0.0.1:8000/dashboard/metrics");
        const data = await res.json();
        if (data && typeof data.accuracy_30d === "number") {
          setAcc(data.accuracy_30d * 100);
          setSummary(data.summary || "");
          // You could later calculate WoW here if you start tracking weekly snapshots
          setWow(null);
        }
      } catch (err) {
        console.error("Failed to fetch dashboard metrics:", err);
      }
    }
    fetchAccuracy();
  }, []);

  const pct = acc != null ? acc.toFixed(1) : null;

  return (
    <div className="card card-hover p-6 flex-1 min-h-[180px] flex flex-col justify-between">
      <div className="flex items-center gap-2 text-slate-300">
        <Activity size={18} /> <span className="text-sm">Accuracy</span>
      </div>
      <div className="text-center">
        <div className="text-5xl font-extrabold text-brand-400">
          {pct != null ? `${pct}%` : "—"}
        </div>
        {wow != null ? (
          <div className="mt-2 text-sm text-slate-400">
            {`${wow >= 0 ? "+" : ""}${wow.toFixed(1)}% WoW`}
          </div>
        ) : (
          <div className="mt-2 text-sm text-slate-400">{summary}</div>
        )}
      </div>
    </div>
  );
}
