"use client";

import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Database,
  Gauge,
  Plus,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { CaseTable } from "@/app/components/CaseTable";
import {
  EmptyState,
  LivePill,
  PageHeader,
  Panel,
  PanelHeading,
  StatTile,
} from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { useCases, useStats } from "@/app/hooks/use-cases";
import { riskTone } from "@/app/shared/schema";
import { cn } from "@/app/lib/utils";

export default function DashboardPage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: list, isLoading: listLoading, isError, error, refetch } = useCases({ pageSize: 8 });

  const recent = list?.cases ?? [];
  const avg = stats?.avgRiskScore ?? 0;
  const avgTone = riskTone(avg);

  /* Share of settled cases that came back manipulated — the one number that
     answers "is what I'm looking at mostly real or mostly fake?" */
  const settled = (stats?.authentic ?? 0) + (stats?.manipulated ?? 0);
  const manipulatedShare = settled > 0 ? Math.round(((stats?.manipulated ?? 0) / settled) * 100) : 0;

  return (
    <AppShell width="wide">
      <PageHeader
        status={<LivePill label="All systems nominal" tone="good" />}
        title="Command Center"
        description="Live view of every case moving through the detection pipeline."
        actions={
          <>
            <Link href="/cases">
              <Button variant="outline" className="border-slate-700 text-slate-300 hover:text-white">
                Browse case history
              </Button>
            </Link>
            <Link href="/forensics">
              <Button className="glow-brand bg-indigo-600 text-white hover:bg-indigo-500">
                <Plus className="mr-2 h-4 w-4" />
                New scan
              </Button>
            </Link>
          </>
        }
      />

      {/* ── Vitals ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Active scans"
          value={stats?.processing ?? 0}
          icon={Activity}
          tone="brand"
          loading={statsLoading}
          hint="Engines currently running"
        />
        <StatTile
          label="Verified authentic"
          value={stats?.authentic ?? 0}
          icon={ShieldCheck}
          tone="good"
          loading={statsLoading}
          hint="No manipulation found"
          href="/cases?status=authentic"
        />
        <StatTile
          label="Threats detected"
          value={stats?.manipulated ?? 0}
          icon={ShieldAlert}
          tone="bad"
          loading={statsLoading}
          hint={settled > 0 ? `${manipulatedShare}% of settled cases` : "None settled yet"}
          href="/cases?status=manipulated"
        />
        <StatTile
          label="Total analysed"
          value={stats?.totalCases ?? 0}
          icon={Database}
          tone="neutral"
          loading={statsLoading}
          hint="Across all modalities"
          href="/cases"
        />
      </div>

      {/* ── Mean risk + queue split ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel className="p-5 lg:col-span-1">
          <div className="mb-4 flex items-center gap-2.5">
            <Gauge className="h-4 w-4 text-slate-500" />
            <h2 className="font-display text-sm font-semibold text-white">Mean risk score</h2>
          </div>

          <div className="flex items-baseline gap-2">
            <span className={cn("tabular text-5xl font-bold leading-none", avgTone.text)}>
              {statsLoading ? "—" : avg.toFixed(1)}
            </span>
            <span className="text-lg text-slate-600">%</span>
          </div>

          <div className="mt-5 h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-700", avgTone.bar)}
              style={{ width: `${Math.min(100, Math.max(0, avg))}%` }}
            />
          </div>

          {/* Thresholds mirror fusion_engine._verdict so the scale on screen
              matches the one the backend actually decides with. */}
          <div className="mt-3 flex justify-between font-mono text-[10px] uppercase tracking-wider text-slate-600">
            <span className="text-emerald-500/70">Authentic ≤35</span>
            <span className="text-amber-500/70">Uncertain</span>
            <span className="text-red-500/70">≥65 Manipulated</span>
          </div>
        </Panel>

        <Panel className="p-5 lg:col-span-2">
          <div className="mb-5 flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold text-white">Verdict distribution</h2>
            <Link
              href="/analytics"
              className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-slate-500 transition-colors hover:text-indigo-400"
            >
              Full analytics <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <DistributionBar
            loading={statsLoading}
            segments={[
              { label: "Authentic", value: stats?.authentic ?? 0, color: "bg-emerald-500" },
              { label: "Manipulated", value: stats?.manipulated ?? 0, color: "bg-red-500" },
              { label: "Processing", value: stats?.processing ?? 0, color: "bg-sky-500" },
              {
                label: "Other",
                value: Math.max(
                  0,
                  (stats?.totalCases ?? 0) -
                    (stats?.authentic ?? 0) -
                    (stats?.manipulated ?? 0) -
                    (stats?.processing ?? 0),
                ),
                color: "bg-slate-600",
              },
            ]}
          />
        </Panel>
      </div>

      {/* ── Recent investigations ─────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          title="Recent investigations"
          hint="Newest first · updates automatically while cases are analysing"
          action={
            <Link
              href="/cases"
              className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-slate-500 transition-colors hover:text-indigo-400"
            >
              View all <ArrowRight className="h-3 w-3" />
            </Link>
          }
        />

        <CaseTable
          cases={recent}
          isLoading={listLoading}
          isError={isError}
          errorMessage={error?.message}
          onRetry={() => refetch()}
          empty={
            <EmptyState
              icon={Database}
              title="No cases yet"
              description="Upload a video, image or audio clip and the pipeline will open the first case for you."
              action={
                <Link href="/forensics">
                  <Button className="bg-indigo-600 text-white hover:bg-indigo-500">
                    <Plus className="mr-2 h-4 w-4" />
                    Start a scan
                  </Button>
                </Link>
              }
            />
          }
        />
      </Panel>
    </AppShell>
  );
}

/* ── Stacked proportion bar ──────────────────────────────────────────────── */

function DistributionBar({
  segments,
  loading,
}: {
  segments: { label: string; value: number; color: string }[];
  loading?: boolean;
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);

  if (loading) return <div className="h-3 w-full animate-pulse rounded-full bg-slate-800" />;

  if (total === 0) {
    return (
      <p className="py-6 text-center text-sm text-slate-600">
        Nothing analysed yet — the distribution appears after your first case settles.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
        {segments
          .filter((s) => s.value > 0)
          .map((s) => (
            <div
              key={s.label}
              className={cn("h-full transition-all duration-700", s.color)}
              style={{ width: `${(s.value / total) * 100}%` }}
              title={`${s.label}: ${s.value}`}
            />
          ))}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-2">
            <span className={cn("h-2 w-2 shrink-0 rounded-full", s.color)} />
            <span className="truncate text-xs text-slate-400">{s.label}</span>
            <span className="tabular ml-auto text-xs font-semibold text-white">{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
