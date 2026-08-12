/* ============================================================================
   DEEP TRUTH — SHARED SCHEMAS
   ----------------------------------------------------------------------------
   Mirrors server/app/models.py one-for-one. The server serialises case fields
   with camelCase aliases (CaseResponse) but leaves the nested AnalysisResult
   rows in snake_case — both are reproduced faithfully here rather than
   normalised, so a shape drift on either side surfaces as a parse error
   instead of a silent `undefined` in the UI.
   ========================================================================= */

import { z } from "zod";

/* ── Enums ───────────────────────────────────────────────────────────────── */

export const MediaTypeSchema = z.enum(["image", "video", "audio"]);

export const CaseStatusSchema = z.enum([
  "processing",
  "authentic",
  "manipulated",
  "inconclusive",
  "failed",
]);

/* ── Analysis result row ─────────────────────────────────────────────────── */

/**
 * The analyser emits two tiers of rows per case, distinguished by
 * `details.tier` (see app/lib/analysis.ts, which is the only place that
 * should interpret them):
 *   tier "summary"     -> the fused verdict for the modality that ran,
 *                         e.g. "Video Ensemble (fused)"
 *   tier "checkpoint"  -> one row per LoRA adapter that voted, named by
 *                         the corpus it was fine-tuned on ("celebdf_v2")
 */
export const AnalysisResultSchema = z.object({
  id: z.string(),
  case_id: z.string(),
  model_name: z.string(),
  confidence: z.number(),
  label: z.string(),
  details: z.record(z.string(), z.any()).nullable().optional(),
  created_at: z.string(),
});

/** Shape of `details` on a tier-"summary" row, as written by
    analyser._engine_result_row. Every field is optional because which keys
    are present depends on which engine ran. */
export const SummaryDetailsSchema = z.object({
  tier: z.literal("summary").optional(),
  modality: z.string().optional(),
  fake_prob: z.number().optional(),
  real_prob: z.number().optional(),
  confidence: z.number().optional(),
  model_version: z.string().optional(),
  rationale: z.string().optional(),
  error: z.string().optional(),
  note: z.string().optional(),
});

/* ── Orchestration state ─────────────────────────────────────────────────────
   Mirrors server/app/models.py JobState. Distinct from `CaseStatus` on
   purpose: `status` is what the analysis *concluded*, `job.state` is what the
   job is *doing*. A case can sit at status "processing" while its job is
   "retrying" on attempt 2 of 3, and an operator watching an upload needs to
   see that rather than an unexplained wait.

   `job` is null for cases older than the queue's state retention. Treat that
   as "no queue information", never as an error — those cases still have a
   perfectly good verdict on them.
   ------------------------------------------------------------------------- */

export const JobStateSchema = z.object({
  state: z.enum(["queued", "running", "retrying", "succeeded", "failed", "cached"]),
  caseId: z.string().nullable().optional(),
  mediaType: z.string().nullable().optional(),
  /** 1-based place in line; null once the job is no longer waiting. */
  position: z.number().nullable().optional(),
  attempt: z.number().default(0),
  maxAttempts: z.number().default(0),
  worker: z.string().nullable().optional(),
  queuedAt: z.number().nullable().optional(),
  startedAt: z.number().nullable().optional(),
  finishedAt: z.number().nullable().optional(),
  retryAt: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
  /** True when the verdict was replayed from cache instead of computed. */
  cacheHit: z.boolean().default(false),
  contentHash: z.string().nullable().optional(),
  sourceUrl: z.string().nullable().optional(),
});

export const QueueWorkersSchema = z.object({
  online: z.number(),
  names: z.array(z.string()).default([]),
  active: z.number(),
  reserved: z.number(),
  concurrency: z.number(),
  reachable: z.boolean().default(false),
});

export const QueueCacheStatsSchema = z.object({
  hits: z.number(),
  misses: z.number(),
  entries: z.number(),
  hitRate: z.number(),
  version: z.string(),
});

export const QueueOverviewSchema = z.object({
  redisOnline: z.boolean(),
  pendingDepth: z.number(),
  workers: QueueWorkersSchema,
  activeJobs: z
    .array(
      z.object({
        worker: z.string(),
        taskId: z.string().nullable().optional(),
        caseDbId: z.string().nullable().optional(),
        mediaType: z.string().nullable().optional(),
        startedAt: z.number().nullable().optional(),
      }),
    )
    .default([]),
  cache: QueueCacheStatsSchema,
  /** Operator-facing hint, e.g. "no workers online". Null when healthy. */
  message: z.string().nullable().optional(),
});

/* ── Case ────────────────────────────────────────────────────────────────── */

export const CaseSchema = z.object({
  id: z.string(),
  caseId: z.string(),
  title: z.string(),
  mediaType: MediaTypeSchema,
  status: CaseStatusSchema,
  riskScore: z.number(),
  syntheticLikelihood: z.number(),
  fileName: z.string().nullable().optional(),
  fileUrl: z.string().nullable().optional(),
  fileSize: z.number().nullable().optional(),
  userId: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
  analysisResults: z.array(AnalysisResultSchema).default([]),
  job: JobStateSchema.nullable().optional(),
});

/** GET /cases returns a pagination envelope, not a bare array. */
export const CaseListResponseSchema = z.object({
  cases: z.array(CaseSchema),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
});

/* ── Stats & health ──────────────────────────────────────────────────────── */

export const DashboardStatsSchema = z.object({
  totalCases: z.number(),
  authentic: z.number(),
  manipulated: z.number(),
  processing: z.number(),
  avgRiskScore: z.number(),
});

export const HealthResponseSchema = z.object({
  /** "ok" | "idle" (no workers) | "degraded" (db or redis unreachable) */
  status: z.string(),
  version: z.string(),
  db: z.string(),
  redis: z.string().default("unknown"),
  workers: z.number().default(0),
});

/* ── Types ───────────────────────────────────────────────────────────────── */

export type MediaType = z.infer<typeof MediaTypeSchema>;
export type CaseStatus = z.infer<typeof CaseStatusSchema>;
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;
export type SummaryDetails = z.infer<typeof SummaryDetailsSchema>;
export type Case = z.infer<typeof CaseSchema>;
export type CaseListResponse = z.infer<typeof CaseListResponseSchema>;
export type DashboardStats = z.infer<typeof DashboardStatsSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type JobState = z.infer<typeof JobStateSchema>;
export type JobStateName = JobState["state"];
export type QueueOverview = z.infer<typeof QueueOverviewSchema>;

/* ── Job-state presentation ──────────────────────────────────────────────────
   One map so a job reads the same in the sidebar, the case table and the
   report header — the same reason VERDICT_PRESENTATION exists below.
   ------------------------------------------------------------------------- */

export const JOB_STATE_PRESENTATION: Record<
  JobStateName,
  { label: string; blurb: string; tone: "waiting" | "active" | "good" | "bad" }
> = {
  queued: {
    label: "Queued",
    blurb: "Waiting for a free worker.",
    tone: "waiting",
  },
  running: {
    label: "Analysing",
    blurb: "A worker is running the models now.",
    tone: "active",
  },
  retrying: {
    label: "Retrying",
    blurb: "The last attempt hit a transient error; it is queued again.",
    tone: "waiting",
  },
  succeeded: {
    label: "Complete",
    blurb: "Analysis finished.",
    tone: "good",
  },
  cached: {
    label: "From cache",
    blurb: "This exact file was analysed before; the stored verdict was reused.",
    tone: "good",
  },
  failed: {
    label: "Failed",
    blurb: "Every attempt failed. See the error for details.",
    tone: "bad",
  },
};

/* ── Verdict presentation ────────────────────────────────────────────────────
   Every surface that shows a status - badges, tables, the report header -
   reads from this one map, so a case never reads "AUTHENTIC" in one place
   and "Verified" in another.
   ------------------------------------------------------------------------- */

export interface VerdictMeta {
  label: string;
  /** Plain-language line shown under the verdict on the report page. */
  summary: string;
  text: string;
  bg: string;
  border: string;
  ring: string;
  /** Hex, for charts and inline SVG where Tailwind classes don't reach. */
  hex: string;
}

export const VERDICT: Record<CaseStatus, VerdictMeta> = {
  processing: {
    label: "Analysing",
    summary: "Engines are still running. Results appear here automatically.",
    text: "text-sky-400",
    bg: "bg-sky-500/10",
    border: "border-sky-500/20",
    ring: "ring-sky-500/30",
    hex: "#38bdf8",
  },
  authentic: {
    label: "Authentic",
    summary: "No synthetic manipulation detected across the engines that ran.",
    text: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
    ring: "ring-emerald-500/30",
    hex: "#34d399",
  },
  manipulated: {
    label: "Manipulated",
    summary: "Synthetic manipulation detected. Review the modality breakdown below.",
    text: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    ring: "ring-red-500/30",
    hex: "#f87171",
  },
  inconclusive: {
    label: "Inconclusive",
    summary: "Signals disagree or fall in the uncertain band. Manual review advised.",
    text: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
    ring: "ring-amber-500/30",
    hex: "#fbbf24",
  },
  failed: {
    label: "Failed",
    summary: "The pipeline could not finish. Re-upload the file to try again.",
    text: "text-slate-400",
    bg: "bg-slate-500/10",
    border: "border-slate-500/20",
    ring: "ring-slate-500/30",
    hex: "#94a3b8",
  },
};

/** Verdict bands, mirrored from server/app/services/analyser.py::_status. */
export const RISK_BANDS = {
  manipulated: 65,
  authentic: 35,
} as const;

export function riskTone(risk: number): { text: string; bar: string; hex: string } {
  if (risk >= RISK_BANDS.manipulated)
    return { text: "text-red-400", bar: "from-red-600 to-red-400", hex: "#f87171" };
  if (risk <= RISK_BANDS.authentic)
    return { text: "text-emerald-400", bar: "from-emerald-600 to-emerald-400", hex: "#34d399" };
  return { text: "text-amber-400", bar: "from-amber-600 to-amber-400", hex: "#fbbf24" };
}
