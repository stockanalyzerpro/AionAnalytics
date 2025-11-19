"use client";
import React, { useEffect, useMemo, useState } from "react";

// ✅ FIX: use relative import, not absolute Windows path
// (Next.js uses the @ alias for src/, not absolute system paths)
import { api } from "@/lib/api";

type Filters = {
  sector: string;
  priceMax: number | null;
  confMin: number | null;
  volMin: number | null;
};

type Row = {
  rank: number;
  ticker: string;
  currentPrice: number;
  predictedPrice: number;
  expectedReturnPct: number;
  confidence: number;
  score: number;
  sector?: string;
  marketCapBucket?: string;
  volume?: number;
  trend?: string;
};

const InsightsPage: React.FC = () => {
  const [rows, setRows] = useState<Row[]>([]);
  const [horizon, setHorizon] = useState("1w");
  const [limit, setLimit] = useState(50);
  const [sectors, setSectors] = useState<string[]>([]);
  const [filters, setFilters] = useState<Filters>({
    sector: "All",
    priceMax: null,
    confMin: null,
    volMin: null,
  });

  // ✅ Base URL from environment
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  // -----------------------------------------------------------
  // Load available sectors (filter options)
  // -----------------------------------------------------------
  useEffect(() => {
    fetch(`${API_URL}/insights/filters?horizon=${horizon}`)
      .then((r) => r.json())
      .then((d) => {
        const arr = Array.isArray(d?.sectors) ? (d.sectors as string[]) : [];
        setSectors(["All", ...arr]);
      })
      .catch((err) => console.error("Failed to fetch filters:", err));
  }, [horizon]);

  // -----------------------------------------------------------
  // Load top picks (with filters)
  // -----------------------------------------------------------
  const loadTopPicks = () => {
    const params = new URLSearchParams({
      horizon,
      limit: limit.toString(),
    });

    if (filters.sector && filters.sector !== "All") {
      params.append("sector", filters.sector);
    }
    if (filters.priceMax !== null && filters.priceMax !== undefined) {
      params.append("price_max", String(filters.priceMax));
    }
    if (filters.volMin !== null && filters.volMin !== undefined) {
      params.append("vol_min", String(filters.volMin));
    }
    if (filters.confMin !== null && filters.confMin !== undefined) {
      params.append("conf_min", String(filters.confMin));
    }

    const url = `${API_URL}/insights/top-picks?${params.toString()}`;

    fetch(url)
      .then((r) => r.json())
      .then((d) => {
        const picks = Array.isArray(d?.picks) ? d.picks : [];
        const mapped: Row[] = picks.map((p: any, idx: number) => ({
          rank: idx + 1,
          ticker: p.ticker,
          currentPrice: p.currentPrice ?? 0,
          predictedPrice: p.predictedPrice ?? 0,
          expectedReturnPct: p.expectedReturnPct ?? 0,
          confidence: p.confidence ?? 0,
          score: p.rankingScore ?? 0,
          sector: p.sector,
        }));
        setRows(mapped);
      })
      .catch((err) => console.error("Failed to fetch top picks:", err));
  };

  // Reload when filters or horizon change
  useEffect(() => {
    loadTopPicks();
  }, [filters, horizon, limit]);

  // Handle filter change
  const handleFilterChange = (key: keyof Filters, value: any) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const filteredRows = useMemo(() => rows, [rows]);

  // -----------------------------------------------------------
  // Render
  // -----------------------------------------------------------
  return (
    <div className="p-6 text-gray-100">
      <h1 className="text-2xl font-semibold mb-4">
        Top Picks — {horizon.toUpperCase()}
      </h1>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div>
          <label className="block text-sm text-gray-400">Sector</label>
          <select
            value={filters.sector}
            onChange={(e) => handleFilterChange("sector", e.target.value)}
            className="bg-gray-800 border border-gray-700 p-2 rounded"
          >
            {sectors.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400">Price ≤</label>
          <input
            type="number"
            value={filters.priceMax ?? ""}
            onChange={(e) =>
              handleFilterChange(
                "priceMax",
                e.target.value ? Number(e.target.value) : null
              )
            }
            className="bg-gray-800 border border-gray-700 p-2 rounded w-28"
            placeholder="Max"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400">Volume ≥</label>
          <input
            type="number"
            value={filters.volMin ?? ""}
            onChange={(e) =>
              handleFilterChange(
                "volMin",
                e.target.value ? Number(e.target.value) : null
              )
            }
            className="bg-gray-800 border border-gray-700 p-2 rounded w-28"
            placeholder="Min"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400">Confidence ≥</label>
          <input
            type="number"
            step="0.01"
            value={filters.confMin ?? ""}
            onChange={(e) =>
              handleFilterChange(
                "confMin",
                e.target.value ? Number(e.target.value) : null
              )
            }
            className="bg-gray-800 border border-gray-700 p-2 rounded w-28"
            placeholder="Min"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400">Horizon</label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(e.target.value)}
            className="bg-gray-800 border border-gray-700 p-2 rounded"
          >
            <option value="1w">1W</option>
            <option value="1m">1M</option>
            <option value="1y">1Y</option>
          </select>
        </div>
      </div>

      {/* Results Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-800 text-gray-400">
            <tr>
              <th className="p-2 text-left">Rank</th>
              <th className="p-2 text-left">Ticker</th>
              <th className="p-2 text-right">Price</th>
              <th className="p-2 text-right">Predicted</th>
              <th className="p-2 text-right">Return %</th>
              <th className="p-2 text-right">Confidence</th>
              <th className="p-2 text-right">Score</th>
              <th className="p-2 text-left">Sector</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.length > 0 ? (
              filteredRows.map((r) => (
                <tr
                  key={r.rank}
                  className="border-b border-gray-800 hover:bg-gray-800/40"
                >
                  <td className="p-2">{r.rank}</td>
                  <td className="p-2 font-medium">{r.ticker}</td>
                  <td className="p-2 text-right">
                    ${r.currentPrice?.toFixed(2) ?? "-"}
                  </td>
                  <td className="p-2 text-right">
                    ${r.predictedPrice?.toFixed(2) ?? "-"}
                  </td>
                  <td
                    className={`p-2 text-right ${
                      r.expectedReturnPct > 0
                        ? "text-blue-400"
                        : "text-red-400"
                    }`}
                  >
                    {r.expectedReturnPct.toFixed(2)}%
                  </td>
                  <td className="p-2 text-right">
                    {(r.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="p-2 text-right">{r.score.toFixed(3)}</td>
                  <td className="p-2 text-left">{r.sector || "—"}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td
                  colSpan={8}
                  className="text-center text-gray-500 py-6 italic"
                >
                  No predictions available for this horizon.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default InsightsPage;
