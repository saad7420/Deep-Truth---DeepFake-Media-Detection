"use client";

/* ============================================================================
   DEEP TRUTH — OVERVIEW (dashboard)
   ----------------------------------------------------------------------------
   The console's home. The first thing on the page is the thing people come
   here to do — check a file — with the system's live state beside it. Below:
   verdict totals, the distribution, the media mix and the newest cases, all
   from GET /stats and GET /cases. Nothing on this page is invented: a tile
   with no data shows a dash.
   ========================================================================= */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Database, Link2, Loader2, Plus, UploadCloud } from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { CaseTable } from "@/app/components/CaseTable";
import { EmptyState, Panel, PanelHeading } from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { useAuth } from "@/app/hooks/use-auth";
import { useCases, useCreateCase, useHealth, useJobStream, useQueueOverview, useStats } from "@/app/hooks/use-cases";
import { useToast } from "@/app/hooks/use-toast";
import { ACCEPT_ATTRIBUTE, casesApi, inferMediaType, validateFile } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import type { MediaType } from "@/app/shared/schema";

/* Checkpoints shipped per stream (checkpoints/ directory). */
const MODEL_GROUPS = [
  { label: "Video · ViViT", count: 6 },
  { label: "Image · ViT", count: 8 },
  { label: "Voice · WavLM", count: 1 },
  { label: "Noise · SRM", count: 1 },
];

export default function DashboardPage() {
  useJobStream();
  const { data: stats, isLoading: statsLoading } = useStats();
  const { data: list, isLoading: listLoading, isError, error, refetch } = useCases({ pageSize: 6 });

  const total = stats?.totalCases ?? 0;
  const pct = (n?: number) => (total > 0 && typeof n === "number" ? `${Math.round((n / total) * 100)}% of all cases` : "No cases yet");
  const v = (n?: number) => (statsLoading ? "—" : (n ?? 0).toLocaleString());
  const mix = stats?.byMediaType ?? { image: 0, video: 0, audio: 0 };

  return (
    <AppShell width="wide">
      <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="eyebrow mb-2">Workspace</p>
          <h1 className="font-display text-3xl font-bold tracking-tight text-white">Overview</h1>
          <p className="mt-2 text-sm text-slate-400">Check a file, and see everything the system has analysed so far.</p>
        </div>
        <div className="flex gap-2">
          <Link href="/cases">
            <Button variant="outline" className="border-slate-700 text-slate-300 hover:text-white">All cases</Button>
          </Link>
          <Link href="/forensics">
            <Button className="bg-indigo-600 text-white hover:bg-indigo-500">
              <Plus className="mr-2 h-4 w-4" /> New analysis
            </Button>
          </Link>
        </div>
      </header>

      {/* ── Analyse + system ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.6fr_1fr]">
        <AnalysePanel />
        <SystemPanel />
      </div>

      {/* ── Verdict totals ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Cases analysed" value={v(stats?.totalCases)} hint={`${mix.image} image · ${mix.audio} audio · ${mix.video} video`} bar="bg-cyan-400" href="/cases" />
        <Kpi label="Manipulated" value={v(stats?.manipulated)} hint={pct(stats?.manipulated)} bar="bg-red-400" tone="text-red-400" href="/cases?status=manipulated" />
        <Kpi label="Authentic" value={v(stats?.authentic)} hint={pct(stats?.authentic)} bar="bg-emerald-400" tone="text-emerald-400" href="/cases?status=authentic" />
        <Kpi label="Inconclusive" value={v(stats?.inconclusive)} hint={total ? `${pct(stats?.inconclusive)} · models disagreed` : "No cases yet"} bar="bg-amber-400" tone="text-amber-400" href="/cases?status=inconclusive" />
      </div>

      {/* ── Distribution + risk/mix ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.6fr_1fr]">
        <Panel className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold text-white">Verdict distribution</h2>
            <Link href="/analytics" className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-slate-500 hover:text-indigo-400">
              Full analytics <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
          <Distribution
            loading={statsLoading}
            segments={[
              { label: "Manipulated", value: stats?.manipulated ?? 0, color: "bg-red-400" },
              { label: "Authentic", value: stats?.authentic ?? 0, color: "bg-emerald-400" },
              { label: "Inconclusive", value: stats?.inconclusive ?? 0, color: "bg-amber-400" },
              { label: "Failed", value: stats?.failed ?? 0, color: "bg-slate-500" },
              { label: "Processing", value: stats?.processing ?? 0, color: "bg-sky-400" },
            ]}
          />
        </Panel>

        <Panel className="grid grid-cols-[120px_1fr] items-center gap-5 p-5">
          <RiskRing value={stats?.avgRiskScore} loading={statsLoading} />
          <div className="space-y-3">
            <h2 className="font-display text-sm font-semibold text-white">Media mix</h2>
            {([
              ["Image", mix.image],
              ["Audio", mix.audio],
              ["Video", mix.video],
            ] as const).map(([label, n]) => (
              <div key={label} className="text-xs text-slate-400">
                {label} · <span className="tabular text-slate-200">{n}</span>
                <div className="mt-1.5 h-1.5 rounded-full bg-slate-800">
                  <div className="h-full rounded-full bg-cyan-400" style={{ width: total ? `${(n / total) * 100}%` : 0 }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* ── Recent cases ──────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          title="Recent cases"
          hint="Newest first · updates while cases are analysing"
          action={
            <Link href="/cases" className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-slate-500 hover:text-indigo-400">
              View all <ArrowRight className="h-3 w-3" />
            </Link>
          }
        />
        <CaseTable
          cases={list?.cases ?? []}
          isLoading={listLoading}
          isError={isError}
          errorMessage={error?.message}
          onRetry={() => refetch()}
          empty={
            <EmptyState
              icon={Database}
              title="No cases yet"
              description="Drop a file above and the first case opens automatically."
            />
          }
        />
      </Panel>
    </AppShell>
  );
}

/* ── Analyse: drop a file or paste a link ────────────────────────────────── */

function AnalysePanel() {
  const router = useRouter();
  const { toast } = useToast();
  const { operator } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [url, setUrl] = useState("");
  const [urlType, setUrlType] = useState<MediaType>("image");

  const upload = useCreateCase((p) => setProgress(p));
  const fromUrl = useMutation({
    mutationFn: () =>
      casesApi.createFromUrl({
        title: url.split("?")[0].split("/").pop() || "Linked media",
        mediaType: urlType,
        mediaUrl: url.trim(),
        userId: operator?.id,
      }),
    onSuccess: (c) => router.push(`/cases/${c.caseId}`),
    onError: (e: Error) => toast({ title: "Couldn't analyse that link", description: e.message, variant: "destructive" }),
  });

  const busy = upload.isPending || fromUrl.isPending;

  function handleFile(file?: File) {
    if (!file || busy) return;
    const mediaType = inferMediaType(file);
    const problem = validateFile(file, mediaType);
    if (problem) {
      toast({ title: "Can't analyse this file", description: problem, variant: "destructive" });
      return;
    }
    setProgress(0);
    upload.mutate(
      { title: file.name, mediaType, file, userId: operator?.id },
      {
        onSuccess: (c) => router.push(`/cases/${c.caseId}`),
        onSettled: () => setProgress(null),
      },
    );
  }

  return (
    <Panel className="overflow-hidden">
      <PanelHeading title="Analyse media" hint="Image · video · voice, up to 500 MB" />
      <div className="grid gap-5 p-5 lg:grid-cols-[1fr_260px] lg:items-center">
        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
          className={cn(
            "grid-bg rounded-lg border-[1.5px] border-dashed px-6 py-8 text-center transition-colors",
            dragging ? "border-cyan-300 bg-cyan-400/5" : "border-cyan-400/45 hover:border-cyan-300",
          )}
        >
          <span className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-lg border border-cyan-400/40 text-cyan-400">
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <UploadCloud className="h-5 w-5" />}
          </span>
          <span className="block font-semibold text-white">
            {upload.isPending ? `Uploading… ${progress ?? 0}%` : "Drop a file to check it"}
          </span>
          <span className="mt-1 block text-sm text-slate-400">
            or click to browse. The case opens as soon as the upload finishes.
          </span>
          <span className="mt-4 flex flex-wrap justify-center gap-1.5">
            {["JPG PNG WEBP", "MP4 MOV WEBM", "MP3 WAV FLAC"].map((c) => (
              <span key={c} className="rounded-full border border-slate-700 px-2.5 py-0.5 font-mono text-[11px] text-slate-400">{c}</span>
            ))}
          </span>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT_ATTRIBUTE}
            className="hidden"
            onChange={(e) => {
              handleFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </button>

        <form
          className="space-y-2.5"
          onSubmit={(e) => {
            e.preventDefault();
            if (url.trim() && !busy) fromUrl.mutate();
          }}
        >
          <p className="eyebrow">Or from a link</p>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://…/media.jpg"
            className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2.5 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:border-cyan-400/60 focus:outline-none"
          />
          <div className="flex gap-2">
            <select
              value={urlType}
              onChange={(e) => setUrlType(e.target.value as MediaType)}
              aria-label="Media type"
              className="rounded-lg border border-slate-800 bg-slate-950 px-2 text-xs text-slate-300"
            >
              <option value="image">Image</option>
              <option value="video">Video</option>
              <option value="audio">Audio</option>
            </select>
            <Button type="submit" disabled={!url.trim() || busy} variant="outline" className="flex-1 border-slate-700 text-slate-200 hover:text-white">
              {fromUrl.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Link2 className="mr-2 h-4 w-4" />}
              Fetch and analyse
            </Button>
          </div>
          <p className="text-xs leading-relaxed text-slate-500">Seen this file before? The saved verdict is returned instantly.</p>
        </form>
      </div>
    </Panel>
  );
}

/* ── System: is the engine up, and is anything waiting ───────────────────── */

function SystemPanel() {
  const { data: health, isError, isLoading } = useHealth();
  const { data: queue } = useQueueOverview();

  const online = !isError && health?.status === "ok";
  const workers = health?.workers ?? 0;
  const statusText = isLoading ? "Connecting" : online ? "All services running" : isError ? "Server unreachable" : workers === 0 ? "No worker running" : "Degraded";

  return (
    <Panel className="overflow-hidden">
      <PanelHeading
        title="System"
        action={
          <span className={cn("inline-flex items-center gap-1.5 text-xs", online ? "text-emerald-400" : "text-amber-400")}>
            <span className={cn("h-1.5 w-1.5 rounded-full", online ? "bg-emerald-400" : "bg-amber-400")} />
            {statusText}
          </span>
        }
      />
      <div className="px-5 pb-2 pt-1">
        {[
          ["Analysis workers", isError ? "—" : `${workers} online`],
          ["Queue", queue ? `${queue.pendingDepth} waiting` : "—"],
          ["Database · cache", isError ? "—" : `${health?.db === "ok" ? "OK" : "Down"} · ${health?.redis === "ok" ? "OK" : "Down"}`],
        ].map(([k, val]) => (
          <div key={k} className="flex justify-between border-b border-dashed border-slate-800 py-2.5 text-sm last:border-b-0">
            <span className="text-slate-400">{k}</span>
            <span className="tabular text-xs text-slate-100">{val}</span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 px-5 pb-5">
        {MODEL_GROUPS.map((m) => (
          <div key={m.label} className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2.5">
            <p className="eyebrow text-[9px]">{m.label}</p>
            <p className="tabular mt-1.5 text-[15px] font-semibold text-white">
              {m.count} <span className="text-[11px] font-normal text-slate-500">{m.count === 1 ? "model" : "models"}</span>
            </p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/* ── Pieces ──────────────────────────────────────────────────────────────── */

function Kpi({ label, value, hint, bar, tone, href }: { label: string; value: string; hint: string; bar: string; tone?: string; href: string }) {
  return (
    <Link href={href}>
      <Panel className="panel-hover relative h-full overflow-hidden p-5">
        <span className={cn("absolute inset-y-0 left-0 w-[3px]", bar)} />
        <p className="eyebrow">{label}</p>
        <p className={cn("tabular mb-1 mt-3 text-4xl font-semibold", tone ?? "text-white")}>{value}</p>
        <p className="text-xs text-slate-400">{hint}</p>
      </Panel>
    </Link>
  );
}

function Distribution({ segments, loading }: { segments: { label: string; value: number; color: string }[]; loading?: boolean }) {
  const total = segments.reduce((s, x) => s + x.value, 0);
  if (loading) return <div className="h-3.5 w-full animate-pulse rounded-full bg-slate-800" />;
  if (total === 0) return <p className="py-6 text-center text-sm text-slate-500">The distribution appears after your first case settles.</p>;

  const shown = segments.filter((s) => s.value > 0 || s.label !== "Processing");
  return (
    <div className="space-y-4">
      <div className="flex h-3.5 w-full overflow-hidden rounded-full bg-slate-800">
        {segments
          .filter((s) => s.value > 0)
          .map((s) => (
            <div key={s.label} className={cn("h-full", s.color)} style={{ width: `${(s.value / total) * 100}%` }} title={`${s.label}: ${s.value}`} />
          ))}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {shown.map((s) => (
          <div key={s.label} className="text-xs text-slate-400">
            <span className={cn("mr-1.5 inline-block h-2 w-2 rounded-sm", s.color)} />
            {s.label}
            <p className="tabular mt-1 text-lg font-semibold text-white">{s.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function RiskRing({ value, loading }: { value?: number; loading?: boolean }) {
  const v = Math.max(0, Math.min(100, value ?? 0));
  const color = v >= 65 ? "#f87171" : v <= 35 ? "#34d399" : "#fbbf24";
  return (
    <div
      className="grid h-[120px] w-[120px] place-items-center rounded-full"
      style={{ background: `conic-gradient(${color} 0 ${v * 3.6}deg, #1d2536 ${v * 3.6}deg 360deg)` }}
    >
      <div className="grid h-24 w-24 place-items-center rounded-full bg-slate-900 text-center">
        <div>
          <p className="tabular text-2xl font-semibold text-white">{loading ? "—" : v.toFixed(1)}</p>
          <p className="font-mono text-[9px] tracking-[0.14em] text-slate-500">MEAN RISK</p>
        </div>
      </div>
    </div>
  );
}
