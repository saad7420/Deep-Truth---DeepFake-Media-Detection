/* ============================================================================
   DEEP TRUTH — ANALYSIS ROW INTERPRETATION
   ----------------------------------------------------------------------------
   Every surface that renders `analysisResults` reads it through this file.

   Why it exists: the UI used to classify rows by matching `model_name`
   against "Fusion (M9)", "Visual Forensics Engine", "AudioFakeNet" and
   "SRM Noise Analysis". The backend emits none of those names. What
   app/services/analyser.py actually writes is:

     per-checkpoint rows   model_name = the adapter slug ("celebdf_v2",
                           "commforensics", "ffpp_facecrop", ...)
     one summary row       model_name = "Video Ensemble (fused)"
                                      | "Image Ensemble (fused)"
                                      | "Audio Ensemble (fused)"
                                      | "AudioFakeNet (stub)"

   The consequence of the mismatch was not a crash: the summary row fell
   through into the "checkpoint" bucket and the modality section rendered
   empty, so the fused verdict — the single most important row — was shown
   as if it were one more anonymous adapter.

   The backend now stamps `details.tier` on every row, so classification is
   explicit. The name-based path below is kept only for rows written before
   that field existed, which are still sitting in forensics.db.
   ========================================================================= */

import type { AnalysisResult, CaseStatus, MediaType } from "@/app/shared/schema";

/* ── Checkpoint display names ────────────────────────────────────────────────
   Each adapter is named after the corpus it was fine-tuned on. The slug is
   what the pipeline emits; these are what an operator should read. The
   backend sends `details.label_text` too — this map is the fallback for
   rows written before that, and keeps the UI readable if the API is ever
   consumed without it. */
const CHECKPOINT_LABELS: Record<string, string> = {
  // video — ViViT + LoRA
  celebdf_v2: "Celeb-DF v2",
  deeperforensics: "DeeperForensics",
  dfdc: "DFDC",
  ffpp: "FaceForensics++",
  genvideo: "GenVideo",
  wilddeepfake: "WildDeepfake",
  // image — ViT-B/16 + LoRA
  genimage: "GenImage",
  mscocoai: "MS-COCO AI",
  wildrf: "WildRF",
  commforensics: "CommunityForensics",
  ntire: "NTIRE",
  dff: "DiffusionFace",
  ffpp_facecrop: "FaceForensics++ (face crop)",
  // audio — WavLM-Large + 3-layer head
  wavlm_large: "WavLM-Large",
};

/** One line on what each adapter is good at, shown on hover in the breakdown. */
const CHECKPOINT_BLURBS: Record<string, string> = {
  celebdf_v2: "High-quality celebrity face swaps",
  deeperforensics: "Face swaps under varied lighting and compression",
  dfdc: "Facebook Deepfake Detection Challenge corpus",
  ffpp: "Four classic face-manipulation families",
  genvideo: "Fully generated video, no real source footage",
  wilddeepfake: "Deepfakes scraped from the open internet",
  genimage: "Diffusion and GAN generated stills",
  mscocoai: "AI re-renders of natural photographs",
  wildrf: "Real-vs-fake images collected in the wild",
  commforensics: "Broad community-sourced generator coverage",
  ntire: "Restoration and super-resolution artefacts",
  dff: "Diffusion-generated faces specifically",
  ffpp_facecrop: "FaceForensics++ scored on the cropped face only",
  wavlm_large: "Synthesised speech and voice cloning (ASVspoof)",
};

/** Summary-row names the backend can emit, mapped to the modality they cover. */
const SUMMARY_ROWS: Record<string, MediaType> = {
  "Video Ensemble (fused)": "video",
  "Image Ensemble (fused)": "image",
  "Audio Ensemble (fused)": "audio",
  "AudioFakeNet (stub)": "audio",
};

/* ── Evidence shapes ─────────────────────────────────────────────────────── */

/** Keys the visual (ViViT) engine puts in `evidence`. */
export interface VideoEvidence {
  rationale?: string;
  face_avg?: number;
  genvideo_score?: number;
  n_face_detected?: number;
  heatmap_path?: string | null;
}

/** Keys the image (ViT-B/16) engine puts in `evidence`. */
export interface ImageEvidence {
  rationale?: string;
  policy?: string;
  generalist_avg?: number;
  face_avg?: number;
  face_detected?: boolean;
  face_trusted?: boolean;
  n_generalist?: number;
  n_face?: number;
  skipped?: string[];
  heatmap_path?: string | null;
}

/** Keys the audio (WavLM-Large) engine puts in `evidence`. */
export interface AudioEvidence {
  rationale?: string;
  /** Decision threshold actually used. Heavy class weighting during training
      pushes this far from 0.5 — it is read from the checkpoint's
      metadata.json rather than assumed. */
  threshold?: number;
  sample_rate?: number;
  max_audio_sec?: number;
  /** True while the checkpoint has not been validated on diverse real audio. */
  experimental?: boolean;
  /** The model's untouched output, before threshold-anchored rescaling. */
  raw_fake_prob?: number;
  /** True when the displayed risk score was rescaled around the model's
      own decision threshold rather than being the raw network output. */
  calibrated?: boolean;
}

export interface SummaryEvidence extends VideoEvidence, ImageEvidence, AudioEvidence {
  tier?: string;
  modality?: string;
  fake_prob?: number;
  real_prob?: number;
  confidence?: number;
  model_version?: string;
  /** Set when the engine failed but returned a neutral result instead of raising. */
  error?: string;
  /** Set by `neutral_result()` — explains why the engine contributed nothing. */
  note?: string;
}

/* ── Derived row types ───────────────────────────────────────────────────── */

export interface CheckpointRow {
  /** Adapter slug, e.g. "celebdf_v2". */
  slug: string;
  /** Human-readable corpus name. */
  label: string;
  /** What this adapter specialises in, or undefined if unknown. */
  blurb?: string;
  /** 0–100. */
  score: number;
  /** "generalist" | "face" on the image branch; absent on video. */
  role?: string;
  isSynthetic: boolean;
}

export interface SummaryRow {
  modelName: string;
  /** 0–100 — the fused fake probability. */
  score: number;
  /** 0–1 — the engine's own certainty. 0 means "ignore me". */
  confidence: number;
  modality?: string;
  modelVersion?: string;
  evidence: SummaryEvidence;
  /** True when the engine declined to contribute (stub, or a caught failure). */
  isNeutral: boolean;
  /** True when the checkpoint behind this result is not yet validated for
      production use. Set by the audio engine while
      DEEPTRUTH_AUDIO_EXPERIMENTAL is on. */
  isExperimental: boolean;
  label: string;
}

export interface AnalysisView {
  summary?: SummaryRow;
  checkpoints: CheckpointRow[];
  /** Checkpoints split by role — only populated on the image branch. */
  generalists: CheckpointRow[];
  faceCheckpoints: CheckpointRow[];
  /** True when there is nothing to show yet. */
  isEmpty: boolean;
}

/* ── Classification ──────────────────────────────────────────────────────── */

type Row = Pick<AnalysisResult, "model_name" | "confidence" | "label" | "details">;

function isSummary(row: Row): boolean {
  const tier = row.details?.tier;
  if (typeof tier === "string") return tier === "summary";
  // Legacy rows (written before `tier` existed) — fall back to the name.
  return row.model_name in SUMMARY_ROWS || /\(fused\)|\(stub\)$/.test(row.model_name);
}

function toCheckpoint(row: Row): CheckpointRow {
  const slug = row.model_name;
  const labelText = row.details?.label_text;
  return {
    slug,
    label: typeof labelText === "string" ? labelText : (CHECKPOINT_LABELS[slug] ?? slug),
    blurb: CHECKPOINT_BLURBS[slug],
    score: row.confidence,
    role: typeof row.details?.role === "string" ? row.details.role : undefined,
    isSynthetic: row.label === "SYNTHETIC",
  };
}

function toSummary(row: Row): SummaryRow {
  const evidence = (row.details ?? {}) as SummaryEvidence;
  const confidence = typeof evidence.confidence === "number" ? evidence.confidence : 0;
  return {
    modelName: row.model_name,
    score: row.confidence,
    confidence,
    modality: evidence.modality ?? SUMMARY_ROWS[row.model_name],
    modelVersion: evidence.model_version,
    evidence,
    // The engine contract: confidence 0 means the result carries no
    // information and must never drive a verdict.
    isNeutral: confidence <= 0,
    isExperimental: evidence.experimental === true,
    label: row.label,
  };
}

/**
 * Split a case's analysis rows into the fused summary and the individual
 * checkpoints that fed it. Safe on an empty or still-processing case.
 */
export function readAnalysis(rows: readonly Row[] | undefined | null): AnalysisView {
  const all = rows ?? [];
  const summaryRow = all.find(isSummary);
  const checkpoints = all.filter((r) => !isSummary(r)).map(toCheckpoint);

  // Loudest signal first — an operator scanning the breakdown wants the
  // adapter that flagged hardest at the top, not alphabetical order.
  checkpoints.sort((a, b) => b.score - a.score);

  return {
    summary: summaryRow ? toSummary(summaryRow) : undefined,
    checkpoints,
    generalists: checkpoints.filter((c) => c.role === "generalist"),
    faceCheckpoints: checkpoints.filter((c) => c.role === "face"),
    isEmpty: all.length === 0,
  };
}

/* ── Plain-language explanation ──────────────────────────────────────────────
   The engines emit a `rationale` string plus a handful of numeric evidence
   keys. Rather than dumping the raw keys into the report, turn them into
   sentences an investigator can put in front of someone who has never heard
   of a LoRA adapter. */

export function explainEvidence(summary: SummaryRow | undefined): string[] {
  if (!summary) return [];
  const e = summary.evidence;
  const lines: string[] = [];

  if (summary.isNeutral) {
    lines.push(
      e.note ??
        e.error ??
        "This engine returned no usable signal, so it did not affect the verdict.",
    );
    return lines;
  }

  if (e.rationale) lines.push(e.rationale);

  // Video branch
  if (typeof e.n_face_detected === "number") {
    lines.push(
      e.n_face_detected > 0
        ? `A face was found in ${e.n_face_detected} sampled frame${e.n_face_detected === 1 ? "" : "s"}, so the face-specialised adapters were included.`
        : "No face was found in the sampled frames, so scoring fell back to the whole-frame adapters.",
    );
  }
  if (typeof e.face_avg === "number" && typeof e.genvideo_score === "number") {
    lines.push(
      `Face-manipulation adapters averaged ${(e.face_avg * 100).toFixed(0)}%; the fully-generated-video adapter read ${(e.genvideo_score * 100).toFixed(0)}%. The higher of the two is taken, because a clip only has to be fake one way to be fake.`,
    );
  }

  // Image branch
  if (typeof e.n_generalist === "number" && e.n_generalist > 0) {
    lines.push(
      `${e.n_generalist} generalist adapter${e.n_generalist === 1 ? "" : "s"} scored the full image${
        typeof e.generalist_avg === "number"
          ? `, averaging ${(e.generalist_avg * 100).toFixed(0)}%`
          : ""
      }.`,
    );
  }
  if (e.face_detected === false) {
    lines.push("No face was detected, so the face-crop adapters were left out of the ensemble.");
  } else if (e.face_trusted === false) {
    lines.push(
      "A face was detected but the crop was not clean enough to trust, so the face adapters were down-weighted.",
    );
  }
  // Audio branch
  if (typeof e.max_audio_sec === "number") {
    lines.push(
      `A single WavLM-Large model scored the first ${e.max_audio_sec} seconds of audio at ${e.sample_rate ? `${e.sample_rate / 1000} kHz` : "16 kHz"} mono. Unlike the video and image channels there is no ensemble here — one model, one score.`,
    );
  }
  // The audio threshold is routinely nowhere near 0.5. ASVspoof training data
  // is roughly 1:9 bonafide:spoof, and the class weighting used to correct
  // that squeezes the model's raw scores toward zero. The equal-error point
  // measured on the dev set is what the verdict actually uses, so showing the
  // raw percentage without it invites the reader to judge "0.4%" against a
  // mental 50% baseline and conclude the opposite of what the model found.
  if (
    e.calibrated === true &&
    typeof e.threshold === "number" &&
    typeof e.raw_fake_prob === "number"
  ) {
    const thr =
      e.threshold < 0.01 ? e.threshold.toExponential(2) : e.threshold.toFixed(4);
    const raw =
      e.raw_fake_prob < 0.01
        ? e.raw_fake_prob.toExponential(2)
        : `${(e.raw_fake_prob * 100).toFixed(2)}%`;
    lines.push(
      `The model's raw output was ${raw}, judged against its own decision threshold of ${thr} rather than 50% — heavy class balancing during training pushes its scores toward zero. The score shown above has been rescaled so that threshold sits at the 50% mark, which is what makes it comparable to the video and image channels. The rescaling preserves ranking exactly; it moves the midpoint, it does not add separation.`,
    );
  } else if (typeof e.threshold === "number" && (e.threshold < 0.4 || e.threshold > 0.6)) {
    lines.push(
      `Scores are judged against a threshold of ${e.threshold < 0.01 ? e.threshold.toExponential(2) : e.threshold.toFixed(4)}, not 50%.`,
    );
  }

  if (Array.isArray(e.skipped) && e.skipped.length > 0) {
    lines.push(
      `Skipped: ${e.skipped.join(", ")} — these adapters could not produce a usable score for this file.`,
    );
  }

  return lines;
}

/* ── Verdict helpers ─────────────────────────────────────────────────────────
   Thresholds live in one place, mirroring analyser._status on the server:
   >= 65 manipulated, <= 35 authentic, anything between is inconclusive.
   The chart previously used 70 for its headline while the rest of the app
   used 65, so a case at 67% read "manipulated" in the table and
   "inconclusive" on its own chart. */

export const THRESHOLDS = { manipulated: 65, authentic: 35 } as const;

export function verdictFromRisk(risk: number): CaseStatus {
  if (risk >= THRESHOLDS.manipulated) return "manipulated";
  if (risk <= THRESHOLDS.authentic) return "authentic";
  return "inconclusive";
}

/** Which engine ran, in words, for the modality label on the report. */
export function engineNameFor(mediaType: MediaType): string {
  return {
    video: "Visual forensics — ViViT ensemble",
    image: "Image forensics — ViT-B/16 ensemble",
    audio: "Audio forensics — WavLM-Large",
  }[mediaType];
}
