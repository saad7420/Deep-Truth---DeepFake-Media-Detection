"use client";

/* ============================================================================
   DEEP TRUTH — FORENSIC REPORT (single case)
   ----------------------------------------------------------------------------
   The previous version of this page read `params.caseId` while the route
   segment is `[case_id]`, so the id was always undefined and the fetch never
   ran. It now reads `case_id`, and looks the case up by its public
   CASE-XXXXXXXX id — which is what GET /cases/{case_id} matches on.
   ========================================================================= */

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Check,
  Clock,
  Copy,
  Download,
  FileText,
  Layers,
  Loader2,
  Pencil,
  Printer,
  Trash2,
} from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { ArtifactMapPanel } from "@/app/components/ArtifactMap";
import { MEDIA_ICON } from "@/app/components/CaseTable";
import { ConfirmDialog } from "@/app/components/ConfirmDialog";
import { JobBadge, JobStatusPanel } from "@/app/components/JobStatus";
import { MediaPreview } from "@/app/components/MediaPreview";
import {
  ErrorState,
  MetaRow,
  Panel,
  PanelHeading,
  SectionRule,
  Spinner,
} from "@/app/components/Primitives";
import { ResultBarChart } from "@/app/components/ResultBarChart";
import { StatusBadge } from "@/app/components/StatusBadge";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { useToast } from "@/app/hooks/use-toast";
import { useCase, useDeleteCase, useJobStream, useUpdateCase } from "@/app/hooks/use-cases";
import { ApiError, formatBytes, resolveMediaUrl } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { engineNameFor, readAnalysis } from "@/app/lib/analysis";
import { VERDICT, riskTone, type AnalysisResult } from "@/app/shared/schema";

export default function CaseReportPage() {
  const params = useParams<{ case_id: string }>();
  const router = useRouter();
  const { toast } = useToast();

  const caseId = params?.case_id;
  const { data: c, isLoading, isError, error, refetch, isFetching } = useCase(caseId);

  // Backend-pushed transitions for this case. The `useCase` poll above stays
  // as the fallback — see useJobStream's note on why both exist.
  useJobStream(caseId);

  const updateCase = useUpdateCase();
  const deleteCase = useDeleteCase();

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editingNotes, setEditingNotes] = useState(false);
  const [notes, setNotes] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (c && !editingNotes) setNotes(c.notes ?? "");
  }, [c, editingNotes]);

  /* ── Loading / error ─────────────────────────────────────────────────── */

  if (isLoading) {
    return (
      <AppShell>
        <Spinner label="Loading case" />
      </AppShell>
    );
  }

  if (isError || !c) {
    const notFound = error instanceof ApiError && error.isNotFound;
    return (
      <AppShell>
        <ErrorState
          title={notFound ? "No such case" : "Couldn't load this case"}
          message={
            notFound
              ? `There's no case with the id ${caseId}. It may have been deleted from the archive.`
              : (error?.message ?? "The analysis server didn't respond.")
          }
          onRetry={notFound ? undefined : () => refetch()}
        />
        <div className="flex justify-center">
          <Link href="/cases">
            <Button variant="outline" className="border-slate-700 text-slate-300 hover:text-white">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to case history
            </Button>
          </Link>
        </div>
      </AppShell>
    );
  }

  const meta = VERDICT[c.status];
  const tone = riskTone(c.riskScore);
  const MediaIcon = MEDIA_ICON[c.mediaType];
  const analysing = c.status === "processing";
  const downloadUrl = resolveMediaUrl(c.fileUrl);

  const { summary } = readAnalysis(c.analysisResults);

  async function copyId() {
    await navigator.clipboard.writeText(c!.caseId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }

  function saveNotes() {
    updateCase.mutate(
      { caseId: c!.caseId, notes },
      { onSuccess: () => setEditingNotes(false) },
    );
  }

  return (
    <AppShell width="wide">
      {/* ── Top bar ───────────────────────────────────────────────────────── */}
      <div className="no-print flex flex-wrap items-center justify-between gap-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.push("/cases")}
          className="-ml-2 text-slate-400 hover:text-white"
        >
          <ArrowLeft className="mr-1.5 h-4 w-4" />
          Case history
        </Button>

        <div className="flex flex-wrap gap-2">
          {downloadUrl && (
            <a href={downloadUrl} download={c.fileName ?? undefined}>
              <Button variant="outline" size="sm" className="border-slate-700 text-slate-300 hover:text-white">
                <Download className="mr-1.5 h-4 w-4" />
                Evidence file
              </Button>
            </a>
          )}
          <Link href={`/cases/${c.caseId}/report`}>
            <Button variant="outline" size="sm" className="border-slate-700 text-slate-300 hover:text-white">
              <Printer className="mr-1.5 h-4 w-4" />
              Export report
            </Button>
          </Link>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmDelete(true)}
            className="border-slate-800 text-slate-400 hover:border-red-500/40 hover:text-red-400"
          >
            <Trash2 className="mr-1.5 h-4 w-4" />
            Delete
          </Button>
        </div>
      </div>

      {/* ── Verdict banner ────────────────────────────────────────────────── */}
      <Panel className={cn("overflow-hidden border-l-4 p-6", meta.border)}>
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button
                onClick={copyId}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-800 bg-slate-950 px-2 py-1 font-mono text-[11px] text-indigo-400 transition-colors hover:border-indigo-500/40"
                aria-label="Copy case id"
              >
                {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                {c.caseId}
              </button>
              <StatusBadge status={c.status} size="sm" />
              <JobBadge job={c.job} size="sm" />
              <span className="inline-flex items-center gap-1.5 text-xs capitalize text-slate-500">
                <MediaIcon className="h-3.5 w-3.5" />
                {c.mediaType}
              </span>
              {isFetching && analysing && (
                <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-sky-400">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Checking for results
                </span>
              )}
            </div>

            <h1 className="font-display text-2xl font-bold tracking-tight text-white lg:text-3xl">
              {c.title}
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">{meta.summary}</p>

            {summary && !analysing && (
              <span
                className={cn(
                  "mt-3 inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-[11px] font-semibold",
                  summary.isNeutral
                    ? "border-amber-500/20 bg-amber-500/10 text-amber-400"
                    : "border-slate-700 bg-slate-800/60 text-slate-300",
                )}
              >
                <Layers className="h-3 w-3" />
                {summary.isNeutral
                  ? "No engine signal — verdict withheld"
                  : engineNameFor(c.mediaType)}
              </span>
            )}

            {/* Why this case is waiting, retrying, or failed — and, on a
                cache hit, why the verdict appeared instantly. */}
            <div className="mt-4 max-w-2xl">
              <JobStatusPanel job={c.job} />
            </div>
          </div>

          {/* Risk dial */}
          <div className="shrink-0">
            <RiskDial value={c.riskScore} pending={analysing} hex={tone.hex} />
          </div>
        </div>
      </Panel>

      {/* ── Evidence + analysis ───────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        {/* Left column */}
        <div className="space-y-6 xl:col-span-2">
          <Panel className="overflow-hidden">
            <PanelHeading icon={MediaIcon} title="Evidence" hint={c.fileName ?? undefined} />
            <div className="p-4">
              <MediaPreview
                fileUrl={c.fileUrl}
                mediaType={c.mediaType}
                fileName={c.fileName}
                analysing={analysing}
              />
            </div>
          </Panel>

          {/* Directly under the evidence it explains, so the map and the image
              it came from are read together. */}
          <ArtifactMapPanel
            map={summary?.evidence.artifact_map}
            originalUrl={resolveMediaUrl(c.fileUrl)}
          />

          <Panel className="p-5">
            <SectionRule label="Chain of custody" />
            <div className="mt-3 divide-y divide-slate-800/70">
              <MetaRow label="Case ID" value={c.caseId} />
              <MetaRow label="Internal ID" value={<span title={c.id}>{c.id.slice(0, 13)}…</span>} />
              <MetaRow label="Modality" value={<span className="uppercase">{c.mediaType}</span>} />
              <MetaRow label="File name" value={c.fileName ?? "—"} mono={false} />
              <MetaRow label="File size" value={formatBytes(c.fileSize)} />
              <MetaRow label="Opened" value={formatFull(c.createdAt)} />
              <MetaRow label="Last updated" value={formatFull(c.updatedAt)} />
              <MetaRow label="Submitted by" value={c.userId ?? "Unattributed"} />
              <MetaRow label="Engine rows" value={c.analysisResults.length} />
            </div>
          </Panel>

          {/* Investigator notes */}
          <Panel className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <SectionRule label="Investigator notes" />
              {!editingNotes && (
                <button
                  onClick={() => setEditingNotes(true)}
                  className="no-print ml-3 shrink-0 text-slate-500 transition-colors hover:text-indigo-400"
                  aria-label="Edit notes"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {editingNotes ? (
              <div className="space-y-3">
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={5}
                  placeholder="Record context, provenance, or why this case was opened."
                  className="resize-none border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setNotes(c.notes ?? "");
                      setEditingNotes(false);
                    }}
                    className="text-slate-400"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={saveNotes}
                    disabled={updateCase.isPending}
                    className="bg-indigo-600 text-white hover:bg-indigo-500"
                  >
                    {updateCase.isPending && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
                    Save notes
                  </Button>
                </div>
              </div>
            ) : c.notes ? (
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300">{c.notes}</p>
            ) : (
              <button
                onClick={() => setEditingNotes(true)}
                className="w-full rounded-lg border border-dashed border-slate-800 py-6 text-sm text-slate-600 transition-colors hover:border-indigo-500/40 hover:text-slate-400"
              >
                Add a note about this case
              </button>
            )}
          </Panel>
        </div>

        {/* Right column */}
        <div className="space-y-6 xl:col-span-3">
          <Panel className="overflow-hidden">
            <PanelHeading
              icon={FileText}
              title="Analysis report"
              hint="Fused verdict, per-modality contributions and checkpoint detail"
              action={
                <button
                  onClick={() => toast({ title: "Report ready", description: "Opening the export view." })}
                  className="hidden"
                  aria-hidden
                />
              }
            />

            <div className="p-5">
              {analysing ? (
                <AnalysingSkeleton />
              ) : c.status === "failed" ? (
                <ErrorState
                  title="Analysis didn't finish"
                  message="The pipeline stopped before producing a verdict. Re-upload the file to run it again."
                />
              ) : (
                <ResultBarChart
                  riskScore={c.riskScore}
                  syntheticLikelihood={c.syntheticLikelihood}
                  analysisResults={c.analysisResults}
                />
              )}
            </div>
          </Panel>

          {c.analysisResults.length > 0 && <EngineLog rows={c.analysisResults} />}
        </div>
      </div>

      <ConfirmDialog
        open={confirmDelete}
        title="Delete this case?"
        description={`${c.caseId} and its stored evidence file will be removed from the server. This can't be undone.`}
        confirmLabel="Delete case"
        destructive
        loading={deleteCase.isPending}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => {
          deleteCase.mutate(c.caseId, { onSuccess: () => router.push("/cases") });
          setConfirmDelete(false);
        }}
      />
    </AppShell>
  );
}

/* ── Risk dial ───────────────────────────────────────────────────────────────
   A ring rather than a bar: on the report the score is the headline, and the
   ring reads at a glance from across a desk.
   ------------------------------------------------------------------------- */

function RiskDial({ value, pending, hex }: { value: number; pending: boolean; hex: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped / 100);

  return (
    <div className="relative flex h-36 w-36 items-center justify-center">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={radius} fill="none" stroke="#1e293b" strokeWidth="9" />
        {!pending && (
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke={hex}
            strokeWidth="9"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 900ms cubic-bezier(0.16,1,0.3,1)" }}
          />
        )}
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {pending ? (
          <Loader2 className="h-7 w-7 animate-spin text-sky-400" />
        ) : (
          <>
            <span className="tabular text-3xl font-bold leading-none" style={{ color: hex }}>
              {clamped.toFixed(0)}
              <span className="text-base text-slate-600">%</span>
            </span>
            <span className="eyebrow mt-1.5">Risk</span>
          </>
        )}
      </div>
    </div>
  );
}

/* ── Engine log ──────────────────────────────────────────────────────────── */

function EngineLog({ rows }: { rows: AnalysisResult[] }) {
  return (
    <Panel className="overflow-hidden">
      <PanelHeading
        icon={Clock}
        title="Engine log"
        hint={`${rows.length} rows written by the pipeline, in execution order`}
      />
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
            <tr className="border-b border-slate-800 font-mono uppercase tracking-wider text-slate-500">
              <th scope="col" className="px-5 py-2.5 font-medium">Model</th>
              <th scope="col" className="px-5 py-2.5 font-medium">Label</th>
              <th scope="col" className="px-5 py-2.5 text-right font-medium">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {rows.map((r) => (
              <tr key={r.id} className="transition-colors hover:bg-slate-800/30">
                <td className="max-w-[280px] truncate px-5 py-2.5 font-mono text-slate-300">
                  {r.model_name}
                </td>
                <td className="px-5 py-2.5">
                  <span
                    className={cn(
                      "rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold",
                      r.label === "SYNTHETIC"
                        ? "bg-red-500/10 text-red-400"
                        : "bg-emerald-500/10 text-emerald-400",
                    )}
                  >
                    {r.label}
                  </span>
                </td>
                <td className="tabular px-5 py-2.5 text-right font-semibold text-white">
                  {r.confidence.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* ── Analysing placeholder ───────────────────────────────────────────────── */

function AnalysingSkeleton() {
  const stages = [
    "De-multiplexing streams",
    "Sampling informative frames",
    "Running visual forensics",
    "Running audio analysis",
    "Fusing modality scores",
  ];

  return (
    <div className="space-y-5">
      {stages.map((stage, i) => (
        <div key={stage} className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] text-slate-500">{stage}</span>
            <Loader2 className="h-3 w-3 animate-spin text-slate-700" />
          </div>
          <div className="shimmer h-1.5 overflow-hidden rounded-full bg-slate-800" style={{ animationDelay: `${i * 120}ms` }} />
        </div>
      ))}
      <p className="pt-2 text-center font-mono text-[11px] text-slate-600">
        Results appear here automatically — no need to refresh.
      </p>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function formatFull(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
