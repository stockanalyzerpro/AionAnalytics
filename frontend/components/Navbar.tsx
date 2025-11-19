"use client";
import Link from "next/link";
import { useState } from "react";
import { ChevronDown } from "lucide-react";

const tabs = [
  { href: "/", label: "Dashboard" },
  { href: "/predict", label: "Predict" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/optimizer", label: "Optimizer" },
  { href: "/reports", label: "Reports" },
  { href: "/insights", label: "Insights" },
];

function ToolsMenu({ onClose }: { onClose?: () => void }) {
  const items = [
    { href: "/system/overrides", label: "Manual Overrides" },
    { href: "/tools/data", label: "Data Pipelines" },
    { href: "/tools/metrics", label: "Model Metrics" },
    { href: "/tools/drift", label: "Drift Reports" },
    { href: "/tools/models", label: "Model Registry" },
    { href: "/tools/health", label: "System Health" },
  ];
  return (
    <div className="border-t border-slate-800 bg-slate-950/95">
      <div className="mx-auto max-w-7xl px-4 py-3 flex gap-4">
        {items.map((it) => (
          <Link
            key={it.href}
            href={it.href}
            onClick={onClose}
            className="badge hover:bg-slate-700/60"
          >
            {it.label}
          </Link>
        ))}
      </div>
    </div>
  );
}

export default function Navbar() {
  const [open, setOpen] = useState(false);
  return (
    <nav className="w-full sticky top-0 z-50 backdrop-blur-md bg-slate-950/70 border-b border-slate-800">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        {/* --- AION Branding --- */}
        <div className="flex items-center gap-3">
          <img
            src="/assets/aion_logo_icon.png"
            alt="AION logo"
            className="h-7 w-auto"
          />
          <span className="orbitron font-semibold tracking-wide text-white">
            AION Analytics
          </span>
        </div>

        {/* --- Nav Links --- */}
        <div className="hidden md:flex items-center gap-6">
          {tabs.map((t) => (
            <Link
              key={t.href}
              href={t.href}
              className="text-sm text-slate-300 hover:text-white transition-colors"
            >
              {t.label}
            </Link>
          ))}
          <button
            onClick={() => setOpen(!open)}
            className="badge bg-slate-900 hover:bg-slate-800"
          >
            <span>Tools</span>
            <ChevronDown size={16} />
          </button>
        </div>
      </div>
      {open && <ToolsMenu onClose={() => setOpen(false)} />}
    </nav>
  );
}
