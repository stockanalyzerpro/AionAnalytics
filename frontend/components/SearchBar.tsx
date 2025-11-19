"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";

export default function SearchBar() {
  const [q, setQ] = useState("");
  const router = useRouter();

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    router.push(`/predict?ticker=${encodeURIComponent(q.trim().toUpperCase())}`);
  }

  return (
    <form onSubmit={onSubmit} className="w-full flex justify-center mt-6">
      <div className="flex w-full max-w-2xl items-center gap-3 bg-[rgba(59,130,246,0.2)] border border-blue-500/40 rounded-2xl px-4 py-3 shadow-glow focus-within:ring-2 focus-within:ring-blue-400">
        <Search size={18} className="text-blue-300" />
        <input
          value={q}
          onChange={e=>setQ(e.target.value)}
          placeholder="Search ticker... e.g., AAPL"
          className="bg-transparent outline-none w-full text-blue-100 placeholder:text-blue-300/60"
        />
        <button className="px-3 py-1 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm">Search</button>
      </div>
    </form>
  );
}
