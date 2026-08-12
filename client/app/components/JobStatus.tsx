"use client";

/* ============================================================================
   DEEP TRUTH — ORCHESTRATION STATE
   ----------------------------------------------------------------------------
   Renders `case.job` — what the analysis job is doing, as opposed to what it
   concluded (that is StatusBadge, driven by `case.status`).

   The distinction matters most while a case sits at "processing". That badge
   alone cannot say whether the case is third in a queue, executing on a
   worker, or on its second attempt after a transient failure, and an operator
   staring at a spinner for four minutes deserves to know which.
   ========================================================================= */

import {
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Loader2,
  RotateCw,
  XCircle,
  type LucideIcon,
} from "lucide-react";

import { cn } from "@/app/lib/utils";
import { JOB_STATE_PRESENTATION, type JobState, type JobStateName } from "@/app/shared/schema";

const ICON: Record<JobStateName, LucideIcon> = {
  queued: Clock,
  running: Loader2,
  retrying: RotateCw,
  succeeded: CheckCircle2,
  cached: Database,
  failed: XCircle,
};

const TONE: Record<string, string> = {
  waiting: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  active: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  bad: "border-red-500/30 bg-red-500/10 text-red-300",
};

/* ── Badge ───────────────────────────────────────────────────────────────── */

export function JobBadge({
  job,
  className,
  size = "sm",
}: {
  job?: JobState | null;
  className?: string;
  size?: "sm" | "md";
}) {
  if (!job) return null;

  const meta = JOB_STATE_PRESENTATION[job.state];
  const Icon = ICON[job.state];

  // Queue position is the single most useful thing to show on a waiting job,
  // so it is promoted into the badge itself rather than buried in the detail
  // panel below.
  const suffix =
    job.state === "queued" && job.position ? ` · #${job.position}` : "";

  return (
    <span
      title={meta.blurb}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-mono font-medium uppercase tracking-wider",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-xs",
        TONE[meta.tone],
        className,
      )}
    >
      <Icon
        className={cn(
          size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5",
          job.state === "running" && "animate-spin",
        )}
      />
      {meta.label}
      {suffix}
    </span>
  );
}

/* ── Detail panel ────────────────────────────────────────────────────────── */

function elapsed(from?: number | null, to?: number | null): string | null {
  if (!from) return null;
  const end = to ?? Date.now() / 1000;
  const secs = Math.max(0, Math.round(end - from));
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  return `${mins}m ${secs % 60}s`;
}

/**
 * The explanatory block shown beneath a case's headline while it is being
 * processed, and after it fails.
 *
 * Returns null for a plain successful run: once a verdict is on screen, how
 * many attempts it took is noise. A cache hit is the exception — that one is
 * worth stating, because it explains why a result appeared instantly and
 * tells the operator the verdict was not recomputed for this upload.
 */
export function JobStatusPanel({ job }: { job?: JobState | null }) {
  if (!job) return null;
  if (job.state === "succeeded") return null;

  const meta = JOB_STATE_PRESENTATION[job.state];
  const runtime = elapsed(job.startedAt, job.finishedAt);
  const waited = elapsed(job.queuedAt, job.startedAt);

  return (
    <div
      className={cn(
        "rounded-lg border px-4 py-3 text-xs",
        job.state === "failed"
          ? "border-red-500/30 bg-red-500/5"
          : "border-slate-800 bg-slate-900/40",
      )}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <JobBadge job={job} />
        <span className="text-slate-400">{meta.blurb}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3">
        {job.state === "queued" && job.position != null && (
          <Field label="Position" value={`#${job.position} in line`} />
        )}
        {job.worker && <Field label="Worker" value={job.worker} icon={Cpu} />}
        {job.maxAttempts > 0 && job.attempt > 0 && (
          <Field label="Attempt" value={`${job.attempt} of ${job.maxAttempts}`} />
        )}
        {waited && job.state !== "queued" && <Field label="Waited" value={waited} />}
        {runtime && <Field label="Runtime" value={runtime} />}
        {job.cacheHit && <Field label="Source" value="Cached result" icon={Database} />}
      </dl>

      {job.state === "retrying" && (
        <p className="mt-3 text-[11px] leading-relaxed text-amber-300/80">
          The previous attempt failed with a transient error. It has been put
          back in the queue automatically — no action needed.
        </p>
      )}

      {job.error && (
        <p className="mt-3 break-words font-mono text-[11px] leading-relaxed text-red-300/90">
          {job.error}
        </p>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon?: LucideIcon;
}) {
  return (
    <div className="min-w-0">
      <dt className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="mt-0.5 flex items-center gap-1.5 truncate font-mono text-xs text-slate-300">
        {Icon && <Icon className="h-3 w-3 shrink-0 text-slate-500" />}
        <span className="truncate">{value}</span>
      </dd>
    </div>
  );
}
