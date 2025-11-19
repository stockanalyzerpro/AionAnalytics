"use client";
import useSWR from "swr";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8080";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export function useDashboardData() {
  const { data: metrics, error: metricsError } = useSWR(`${API_URL}/dashboard/metrics`, fetcher, {
    refreshInterval: 300000, // refresh every 5 min
  });
  const { data: top1w, error: top1wError } = useSWR(`${API_URL}/dashboard/top/1w`, fetcher);
  const { data: top1m, error: top1mError } = useSWR(`${API_URL}/dashboard/top/1m`, fetcher);

  return {
    metrics,
    top1w,
    top1m,
    isLoading: !metrics && !top1w && !top1m,
    isError: metricsError || top1wError || top1mError,
  };
}
