import { z } from "zod";

export const MediaTypeSchema = z.enum(["image", "video", "audio"]);

export const CaseStatusSchema = z.enum([
  "processing",
  "authentic",
  "manipulated",
  "inconclusive",
  "failed",
]);

export const AnalysisResultSchema = z.object({
  id: z.string(),
  case_id: z.string(),
  model_name: z.string(),
  confidence: z.number(),
  label: z.string(),
  details: z.record(z.string(), z.any()).nullable().optional(),
  created_at: z.string(),
});

export const ArtifactMapSchema = z.object({
  url: z.string(),
  method: z.string().optional(),
  grid: z.number().optional(),
  concentration: z.number().optional(),
  localised: z.boolean().optional(),
  contributors: z.array(z.string()).default([]),
  regions: z
    .array(
      z.object({
        x: z.number(),
        y: z.number(),
        w: z.number(),
        h: z.number(),
        relevance: z.number(),
      }),
    )
    .default([]),

  segments: z.number().optional(),
  temporal_profile: z.array(z.number()).optional(),
  peak_segment: z.number().optional(),
  temporally_localised: z.boolean().optional(),
  branch: z.string().optional(),
});

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
  artifact_map: ArtifactMapSchema.nullable().optional(),
});

export const JobStateSchema = z.object({
  state: z.enum(["queued", "running", "retrying", "succeeded", "failed", "cached"]),
  caseId: z.string().nullable().optional(),
  mediaType: z.string().nullable().optional(),
  position: z.number().nullable().optional(),
  attempt: z.number().default(0),
  maxAttempts: z.number().default(0),
  worker: z.string().nullable().optional(),
  queuedAt: z.number().nullable().optional(),
  startedAt: z.number().nullable().optional(),
  finishedAt: z.number().nullable().optional(),
  retryAt: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
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
  message: z.string().nullable().optional(),
});

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

export const CaseListResponseSchema = z.object({
  cases: z.array(CaseSchema),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
});

export const DashboardStatsSchema = z.object({
  totalCases: z.number(),
  authentic: z.number(),
  manipulated: z.number(),
  processing: z.number(),
  inconclusive: z.number().default(0),
  failed: z.number().default(0),
  byMediaType: z
    .object({ image: z.number(), video: z.number(), audio: z.number() })
    .default({ image: 0, video: 0, audio: 0 }),
  avgRiskScore: z.number(),
});

export const HealthResponseSchema = z.object({
  status: z.string(),
  version: z.string(),
  db: z.string(),
  redis: z.string().default("unknown"),
  workers: z.number().default(0),
});

export type MediaType = z.infer<typeof MediaTypeSchema>;
export type CaseStatus = z.infer<typeof CaseStatusSchema>;
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;
export type SummaryDetails = z.infer<typeof SummaryDetailsSchema>;
export type Case = z.infer<typeof CaseSchema>;
export type CaseListResponse = z.infer<typeof CaseListResponseSchema>;
export type DashboardStats = z.infer<typeof DashboardStatsSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type ArtifactMap = z.infer<typeof ArtifactMapSchema>;
export type JobState = z.infer<typeof JobStateSchema>;
export type JobStateName = JobState["state"];
export type QueueOverview = z.infer<typeof QueueOverviewSchema>;

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

export interface VerdictMeta {
  label: string;
  summary: string;
  text: string;
  bg: string;
  border: string;
  ring: string;
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
    hex: "#818cf8",
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
    hex: "#8190a8",
  },
};

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
