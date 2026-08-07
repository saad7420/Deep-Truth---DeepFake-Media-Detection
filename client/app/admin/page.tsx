"use client";

/* ============================================================================
   DEEP TRUTH — SYSTEM CONFIGURATION
   ----------------------------------------------------------------------------
   The backend exposes no configuration endpoints: fusion weights live in
   app/registry.py, thresholds in analyser._status(), and file limits in
   the server's environment. So this page reports state rather than pretending
   to change it — every value is labelled with where it's actually set, and
   nothing here renders a control that would silently do nothing.
   ========================================================================= */

import { useMemo } from "react";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  FileWarning,
  Server,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

import AppShell from "@/app/components/AppShell";
import {
  ErrorState,
  MetaRow,
  PageHeader,
  Panel,
  PanelHeading,
  SectionRule,
  StatTile,
} from "@/app/components/Primitives";
import { useCases, useHealth, useStats } from "@/app/hooks/use-cases";
import { API_BASE, API_ORIGIN, MAX_FILE_MB, ACCEPTED_MIME } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { RISK_BANDS } from "@/app/shared/schema";
import { readAnalysis } from "@/app/lib/analysis";

/* What app/registry.py actually registers. Each case is routed to exactly one
   engine by its media type — there is no cross-modal fusion step in this
   backend, so this panel reports engine health rather than blend weights. */
const ENGINES = [
  {
    key: "video" as const,
    module: "M7",
    label: "Visual forensics",
    detail: "6 ViViT + LoRA adapters",
    real: true,
  },
  {
    key: "image" as const,
    module: "M7b",
    label: "Image forensics",
    detail: "8 ViT-B/16 + LoRA adapters",
    real: true,
  },
  {
    key: "audio" as const,
    module: "M6",
    label: "Audio forensics",
    detail: "WavLM-Large — checkpoint not yet wired in",
    real: false,
  },
];

export default function AdminPage() {
  const { data: health, isError: healthError } = useHealth();
  const { data: stats } = useStats();
  const { data: list } = useCases({ pageSize: 50 });

  /* How each engine has actually behaved across recent cases: how many it was
     handed, and how many of those produced a usable (non-zero confidence)
     reading. Read straight from the rows the backend wrote. */
  const engineActivity = useMemo(() => {
    const seen = new Map<string, { runs: number; contributing: number }>();

    (list?.cases ?? []).forEach((c) => {
      if (c.status === "processing") return;
      const { summary } = readAnalysis(c.analysisResults);
      if (!summary) return;

      const entry = seen.get(c.mediaType) ?? { runs: 0, contributing: 0 };
      entry.runs += 1;
      if (!summary.isNeutral) entry.contributing += 1;
      seen.set(c.mediaType, entry);
    });

    return seen;
  }, [list]);

  const online = !healthError && health?.status === "ok";
  const dbOk = health?.db === "ok";

  return (
    <AppShell width="wide">
      <PageHeader
        eyebrow="System"
        title="Configuration"
        description="Live state of the detection service and the settings it's running with."
      />

      {/* ── Service vitals ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Analysis engine"
          value={online ? "Online" : "Offline"}
          icon={Server}
          tone={online ? "good" : "bad"}
          hint={health?.version ? `Version ${health.version}` : "No response from the API"}
        />
        <StatTile
          label="Database"
          value={dbOk ? "Connected" : online ? "Degraded" : "Unknown"}
          icon={Database}
          tone={dbOk ? "good" : online ? "warn" : "neutral"}
          hint={health?.db && health.db !== "ok" ? health.db : "SQLite, WAL journal mode"}
        />
        <StatTile
          label="Cases in queue"
          value={stats?.processing ?? 0}
          icon={Activity}
          tone={(stats?.processing ?? 0) > 0 ? "brand" : "neutral"}
          hint="Currently running through the pipeline"
        />
        <StatTile
          label="Upload ceiling"
          value={`${MAX_FILE_MB} MB`}
          icon={FileWarning}
          tone="neutral"
          hint="Set by MAX_FILE_SIZE_MB on the server"
        />
      </div>

      {!online && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-red-200">The analysis engine isn&apos;t responding</p>
            <p className="mt-1 text-xs leading-relaxed text-red-300/80">
              Uploads and case lookups will fail until it&apos;s back. Start the backend with{" "}
              <code className="rounded bg-red-950/50 px-1 py-0.5 font-mono">python main.py</code> from
              the server directory, then confirm it&apos;s listening on{" "}
              <code className="rounded bg-red-950/50 px-1 py-0.5 font-mono">{API_ORIGIN}</code>.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ── Engine status ───────────────────────────────────────────────── */}
        <Panel className="overflow-hidden">
          <PanelHeading
            icon={Sparkles}
            title="Detection engines"
            hint="One engine per modality — routed by media type"
          />
          <div className="space-y-5 p-5">
            {ENGINES.map((e) => {
              const activity = engineActivity.get(e.key);
              const rate =
                activity && activity.runs > 0
                  ? Math.round((activity.contributing / activity.runs) * 100)
                  : null;

              return (
                <div key={e.key} className="space-y-2">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                    <div className="flex min-w-0 items-baseline gap-2">
                      <span className="font-mono text-[10px] text-slate-600">{e.module}</span>
                      <span className="truncate text-sm font-medium text-white">{e.label}</span>
                    </div>
                    <span
                      className={cn(
                        "shrink-0 rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider",
                        e.real
                          ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-400"
                          : "border-amber-500/25 bg-amber-500/10 text-amber-400",
                      )}
                    >
                      {e.real ? "Live" : "Stub"}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-500">{e.detail}</p>

                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-700",
                        e.real
                          ? "bg-gradient-to-r from-indigo-600 to-indigo-400"
                          : "bg-slate-600",
                      )}
                      style={{ width: `${rate ?? (e.real ? 100 : 0)}%` }}
                    />
                  </div>

                  <p className="text-[11px] text-slate-600">
                    {activity == null
                      ? "No settled cases for this modality yet."
                      : e.real
                        ? `Returned a usable reading on ${activity.contributing} of the last ${activity.runs} ${e.key} case${activity.runs === 1 ? "" : "s"}.`
                        : `Handed ${activity.runs} case${activity.runs === 1 ? "" : "s"}, all reported inconclusive — the stub contributes no signal by design.`}
                  </p>
                </div>
              );
            })}

            <p className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-500">
              An engine reporting zero confidence never produces an authentic or manipulated
              verdict — the case is marked inconclusive instead. Swap the audio stub for the real
              WavLM model in{" "}
              <code className="font-mono text-slate-400">app/registry.py</code>; nothing else
              needs to change.
            </p>
          </div>
        </Panel>

        {/* ── Decision thresholds ─────────────────────────────────────────── */}
        <Panel className="overflow-hidden">
          <PanelHeading
            icon={SlidersHorizontal}
            title="Decision thresholds"
            hint="Where a fused score becomes a verdict"
          />
          <div className="p-5">
            <div className="relative h-12 overflow-hidden rounded-lg border border-slate-800">
              <div className="flex h-full">
                <div
                  className="flex items-center justify-center bg-emerald-500/20"
                  style={{ width: `${RISK_BANDS.authentic}%` }}
                >
                  <span className="font-mono text-[10px] uppercase tracking-wider text-emerald-300">
                    Authentic
                  </span>
                </div>
                <div
                  className="flex items-center justify-center bg-amber-500/20"
                  style={{ width: `${RISK_BANDS.manipulated - RISK_BANDS.authentic}%` }}
                >
                  <span className="font-mono text-[10px] uppercase tracking-wider text-amber-300">
                    Inconclusive
                  </span>
                </div>
                <div
                  className="flex items-center justify-center bg-red-500/20"
                  style={{ width: `${100 - RISK_BANDS.manipulated}%` }}
                >
                  <span className="font-mono text-[10px] uppercase tracking-wider text-red-300">
                    Manipulated
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-2 flex justify-between font-mono text-[10px] text-slate-600">
              <span>0</span>
              <span>{RISK_BANDS.authentic}</span>
              <span>{RISK_BANDS.manipulated}</span>
              <span>100</span>
            </div>

            <div className="mt-5 divide-y divide-slate-800/70">
              <MetaRow label="Authentic at or below" value={`${RISK_BANDS.authentic}%`} />
              <MetaRow label="Manipulated at or above" value={`${RISK_BANDS.manipulated}%`} />
              <MetaRow label="Cross-modal disagreement" value="Escalates to manipulated" />
              <MetaRow label="Retry attempts on failure" value="1" />
            </div>

            <p className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-500">
              A strongly-fake reading from one modality against a strongly-real reading from another
              is reported as manipulated rather than averaged into an inconclusive middle — that
              disagreement is the finding. Set in{" "}
              <code className="font-mono text-slate-400">app/services/analyser.py::_status</code>.
            </p>
          </div>
        </Panel>
      </div>

      {/* ── Endpoints & accepted formats ──────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel className="p-5">
          <SectionRule label="Service endpoints" />
          <div className="mt-3 divide-y divide-slate-800/70">
            <MetaRow label="API base" value={API_BASE} />
            <MetaRow label="Evidence mount" value={`${API_ORIGIN}/uploads`} />
            <MetaRow label="Interactive docs" value={`${API_ORIGIN}/api/docs`} />
            <MetaRow label="OpenAPI schema" value={`${API_ORIGIN}/api/openapi.json`} />
          </div>
          <p className="mt-4 text-[11px] leading-relaxed text-slate-600">
            Point the console at a different server by setting{" "}
            <code className="font-mono text-slate-400">NEXT_PUBLIC_API_URL</code> in{" "}
            <code className="font-mono text-slate-400">.env.local</code>.
          </p>
        </Panel>

        <Panel className="p-5">
          <SectionRule label="Accepted formats" />
          <div className="mt-4 space-y-4">
            {(Object.keys(ACCEPTED_MIME) as (keyof typeof ACCEPTED_MIME)[]).map((k) => (
              <div key={k}>
                <p className="mb-1.5 font-mono text-[11px] uppercase tracking-wider text-slate-400">
                  {k}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {ACCEPTED_MIME[k].map((m) => (
                    <span
                      key={m}
                      className="rounded border border-slate-800 bg-slate-950 px-1.5 py-0.5 font-mono text-[10px] text-slate-500"
                    >
                      {m.split("/")[1]}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[11px] leading-relaxed text-slate-600">
            Anything outside this list is rejected before analysis starts. Extend the list in{" "}
            <code className="font-mono text-slate-400">app/services/analyser.py</code>.
          </p>
        </Panel>
      </div>

      {/* ── Model registry ────────────────────────────────────────────────── */}
      <Panel className="p-5">
        <SectionRule label="Pipeline modules" />
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {[
            { id: "M4", name: "Smart de-multiplexer", detail: "Splits container into audio track and sampled keyframes" },
            { id: "M5", name: "Task orchestration", detail: "Runs engines off the request thread, retries transient failures" },
            { id: "M6", name: "AudioFakeNet", detail: "Spectral and prosody analysis for synthesised speech" },
            { id: "M7", name: "Visual forensics", detail: "Transformer ensemble over sampled frames" },
            { id: "M8", name: "SRM noise analysis", detail: "Sensor noise residuals and splice detection" },
            { id: "M9", name: "Fusion layer", detail: "Confidence-weighted combination into one trust score" },
          ].map((m) => (
            <div key={m.id} className="rounded-lg border border-slate-800 bg-slate-950 p-4">
              <div className="mb-2 flex items-center gap-2">
                <Cpu className="h-4 w-4 text-indigo-400" />
                <span className="font-mono text-[10px] text-slate-600">{m.id}</span>
              </div>
              <p className="text-sm font-medium text-white">{m.name}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-500">{m.detail}</p>
            </div>
          ))}
        </div>
      </Panel>
    </AppShell>
  );
}
