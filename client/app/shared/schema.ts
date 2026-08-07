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
  status: z.string(),
  version: z.string(),
  db: z.string(),
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
