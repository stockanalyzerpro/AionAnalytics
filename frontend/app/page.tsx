import SearchBar from "@/components/SearchBar";
import LogoHeader from "@/components/LogoHeader";
import AccuracyCard from "@/components/AccuracyCard";
import TopPredictions from "@/components/TopPredictions";

export default function Page() {
  return (
    <div className="space-y-6">
      <SearchBar />
      <LogoHeader />
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
        <AccuracyCard />
        <TopPredictions title="Top 3 — 1 Week" horizon="1w" />
        <TopPredictions title="Top 3 — 1 Month" horizon="4w" />
      </section>
    </div>
  );
}
