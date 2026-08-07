"use client";

/* ============================================================================
   DEEP TRUTH — ANALYTICS
   ----------------------------------------------------------------------------
   Every figure here is derived client-side from the case list the backend
   already returns. There is no analytics endpoint on the server, and inventing
   trend data the API can't produce would make this screen lie — so the page
   computes only what the data actually supports and says when a series is
   too thin to read.
   ========================================================================= */

import { useMemo } from "react";
import Link from "next/link";
import { BarChart3, Gauge, PieChart as PieIcon, TrendingUp } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import AppShell from "@/app/components/AppShell";
import {
  EmptyState,
  ErrorState,
  PageHeader,
  Panel,
  PanelHeading,
  StatTile,
} from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { useCases, useStats } from "@/app/hooks/use-cases";
import { VERDICT, type Case, type CaseStatus } from "@/app/shared/schema";

const CHART_MAX = 200;

export default function AnalyticsPage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data, isLoading, isError, error, refetch } = useCases({ pageSize: CHART_MAX });

  const cases = useMemo(() => data?.cases ?? [], [data]);

  /* ── Derived series ────────────────────────────────────────────────────── */

  const settled = useMemo(
    () => cases.filter((c) => c.status !== "processing" && c.status !== "failed"),
    [cases],
  );

  const riskHistogram = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({
      band: `${i * 10}–${i * 10 + 10}`,
      count: 0,
      mid: i * 10 + 5,
    }));
    settled.forEach((c) => {
      const idx = Math.min(9, Math.max(0, Math.floor(c.riskScore / 10)));
      buckets[idx].count += 1;
    });
    return buckets;
  }, [settled]);

  const modalitySplit = useMemo(() => {
    const counts: Record<string, number> = { video: 0, image: 0, audio: 0 };
    cases.forEach((c) => (counts[c.mediaType] += 1));
    return [
      { name: "Video", value: counts.video, fill: "#6366f1" },
      { name: "Image", value: counts.image, fill: "#a855f7" },
      { name: "Audio", value: counts.audio, fill: "#38bdf8" },
    ].filter((s) => s.value > 0);
  }, [cases]);

  const timeline = useMemo(() => buildTimeline(cases), [cases]);

  const meanRisk = stats?.avgRiskScore ?? 0;
  const detectionRate =
    settled.length > 0
      ? Math.round((settled.filter((c) => c.status === "manipulated").length / settled.length) * 100)
      : 0;

  if (isError) {
    return (
      <AppShell>
        <ErrorState message={error?.message ?? "Couldn't reach the server."} onRetry={() => refetch()} />
      </AppShell>
    );
  }

  const thin = settled.length < 3;

  return (
    <AppShell width="wide">
      <PageHeader
        eyebrow="Reporting"
        title="Analytics"
        description={`Aggregate view across ${cases.length} case${cases.length === 1 ? "" : "s"} in the archive.`}
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Cases analysed"
          value={stats?.totalCases ?? 0}
          icon={BarChart3}
          tone="brand"
          loading={statsLoading}
        />
        <StatTile
          label="Mean risk score"
          value={`${meanRisk.toFixed(1)}%`}
          icon={Gauge}
          tone={meanRisk >= 65 ? "bad" : meanRisk <= 35 ? "good" : "warn"}
          loading={statsLoading}
          hint="Across every settled case"
        />
        <StatTile
          label="Detection rate"
          value={`${detectionRate}%`}
          icon={TrendingUp}
          tone={detectionRate > 50 ? "bad" : "neutral"}
          hint="Share of settled cases flagged manipulated"
        />
        <StatTile
          label="Modalities in use"
          value={modalitySplit.length}
          icon={PieIcon}
          tone="neutral"
          hint={modalitySplit.map((m) => m.name).join(" · ") || "None yet"}
        />
      </div>

      {cases.length === 0 && !isLoading ? (
        <Panel>
          <EmptyState
            icon={BarChart3}
            title="Nothing to chart yet"
            description="Analytics fill in once cases start settling. Run your first scan to seed the archive."
            action={
              <Link href="/forensics">
                <Button className="bg-indigo-600 text-white hover:bg-indigo-500">Run a scan</Button>
              </Link>
            }
          />
        </Panel>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Risk distribution */}
            <Panel className="overflow-hidden lg:col-span-2">
              <PanelHeading
                title="Risk score distribution"
                hint="Settled cases grouped into ten-point bands"
              />
              <div className="p-5">
                {thin ? (
                  <ThinNotice n={settled.length} />
                ) : (
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={riskHistogram} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                      <XAxis
                        dataKey="band"
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
                        axisLine={{ stroke: "#1e293b" }}
                        tickLine={false}
                      />
                      <YAxis
                        allowDecimals={false}
                        tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: "rgba(99,102,241,0.08)" }}
                        contentStyle={tooltipStyle}
                        labelStyle={{ color: "#94a3b8", fontSize: 11 }}
                        formatter={(v) => [`${v ?? 0} case${v === 1 ? "" : "s"}`, "Count"] as [string, string]}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {riskHistogram.map((b) => (
                          <Cell
                            key={b.band}
                            fill={b.mid >= 65 ? "#f87171" : b.mid <= 35 ? "#34d399" : "#fbbf24"}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
                <p className="mt-3 text-center font-mono text-[10px] uppercase tracking-wider text-slate-600">
                  Green ≤35 authentic · Amber uncertain · Red ≥65 manipulated
                </p>
              </div>
            </Panel>

            {/* Modality split */}
            <Panel className="overflow-hidden">
              <PanelHeading title="Modality mix" hint="What's being submitted" />
              <div className="p-5">
                {modalitySplit.length === 0 ? (
                  <ThinNotice n={0} />
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={modalitySplit}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={54}
                          outerRadius={82}
                          paddingAngle={3}
                          stroke="none"
                        />
                        <Tooltip contentStyle={tooltipStyle} />
                      </PieChart>
                    </ResponsiveContainer>
                    <ul className="mt-4 space-y-2">
                      {modalitySplit.map((m) => (
                        <li key={m.name} className="flex items-center gap-2 text-xs">
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: m.fill }}
                          />
                          <span className="text-slate-400">{m.name}</span>
                          <span className="tabular ml-auto font-semibold text-white">{m.value}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </Panel>
          </div>

          {/* Volume over time */}
          <Panel className="overflow-hidden">
            <PanelHeading
              title="Case volume"
              hint="Cases opened per day, by verdict"
            />
            <div className="p-5">
              {timeline.length < 2 ? (
                <ThinNotice n={timeline.length} unit="day" />
              ) : (
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart data={timeline} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                    <defs>
                      {(["authentic", "manipulated", "inconclusive"] as CaseStatus[]).map((k) => (
                        <linearGradient key={k} id={`fill-${k}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={VERDICT[k].hex} stopOpacity={0.35} />
                          <stop offset="100%" stopColor={VERDICT[k].hex} stopOpacity={0.02} />
                        </linearGradient>
                      ))}
                    </defs>
                    <XAxis
                      dataKey="label"
                      tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
                      axisLine={{ stroke: "#1e293b" }}
                      tickLine={false}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#94a3b8", fontSize: 11 }} />
                    {(["authentic", "inconclusive", "manipulated"] as CaseStatus[]).map((k) => (
                      <Area
                        key={k}
                        type="monotone"
                        dataKey={k}
                        stackId="1"
                        stroke={VERDICT[k].hex}
                        strokeWidth={1.5}
                        fill={`url(#fill-${k})`}
                        name={VERDICT[k].label}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </Panel>
        </>
      )}
    </AppShell>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

const tooltipStyle = {
  backgroundColor: "#0f172a",
  border: "1px solid #1e293b",
  borderRadius: 8,
  fontSize: 12,
  color: "#e2e8f0",
} as const;

function ThinNotice({ n, unit = "case" }: { n: number; unit?: string }) {
  return (
    <p className="py-16 text-center text-sm text-slate-600">
      {n === 0
        ? `No ${unit}s to chart yet.`
        : `Only ${n} ${unit}${n === 1 ? "" : "s"} so far — too few to read a pattern from.`}
    </p>
  );
}

interface TimelinePoint {
  label: string;
  authentic: number;
  manipulated: number;
  inconclusive: number;
}

function buildTimeline(cases: Case[]): TimelinePoint[] {
  const byDay = new Map<string, TimelinePoint>();

  cases.forEach((c) => {
    if (!c.createdAt) return;
    const iso = c.createdAt.endsWith("Z") || c.createdAt.includes("+") ? c.createdAt : `${c.createdAt}Z`;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return;

    const key = d.toISOString().slice(0, 10);
    if (!byDay.has(key)) {
      byDay.set(key, {
        label: d.toLocaleDateString(undefined, { day: "2-digit", month: "short" }),
        authentic: 0,
        manipulated: 0,
        inconclusive: 0,
      });
    }
    const point = byDay.get(key)!;
    if (c.status === "authentic") point.authentic += 1;
    else if (c.status === "manipulated") point.manipulated += 1;
    else if (c.status === "inconclusive") point.inconclusive += 1;
  });

  return Array.from(byDay.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, v]) => v);
}
