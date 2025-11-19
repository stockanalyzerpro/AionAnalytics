import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import SystemBar from "@/components/SystemBar";

export const metadata: Metadata = {
  title: "SAP — StockAnalyzerPro",
  description: "Predict. Learn. Evolve.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <Navbar />
        <main className="mx-auto max-w-7xl px-4 pb-16 pt-6">{children}</main>
        <SystemBar />
      </body>
    </html>
  );
}
