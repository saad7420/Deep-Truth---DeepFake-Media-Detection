import type {
  AnalysisResult,
  ArtifactMap,
  CaseStatus,
  MediaType,
} from "@/app/shared/schema";

const CHECKPOINT_LABELS: Record<string, string> = {
  celebdf_v2: "Celeb-DF v2",
  deeperforensics: "DeeperForensics",
  dfdc: "DFDC",
  ffpp: "FaceForensics++",
  genvideo: "GenVideo",
  wilddeepfake: "WildDeepfake",
  genimage: "GenImage",
  mscocoai: "MS-COCO AI",
  wildrf: "WildRF",
  commforensics: "CommunityForensics",
  ntire: "NTIRE",
  dff: "DiffusionFace",
  ffpp_facecrop: "FaceForensics++ (face crop)",
  wavlm_large: "WavLM-Large",
};

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

const SUMMARY_ROWS: Record<string, MediaType> = {
  "Video Ensemble (fused)": "video",
  "Image Ensemble (fused)": "image",
  "Audio Ensemble (fused)": "audio",
  "AudioFakeNet (stub)": "audio",
};

export interface VideoEvidence {
  rationale?: string;
  face_avg?: number;
  genvideo_score?: number;
  n_face_detected?: number;
  heatmap_path?: string | null;
}

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

export interface AudioEvidence {
  rationale?: string;
  threshold?: number;
  sample_rate?: number;
  max_audio_sec?: number;
  experimental?: boolean;
  raw_fake_prob?: number;
  calibrated?: boolean;
}

export interface SummaryEvidence extends VideoEvidence, ImageEvidence, AudioEvidence {
  tier?: string;
  modality?: string;
  fake_prob?: number;
  real_prob?: number;
  confidence?: number;
  model_version?: string;
  error?: string;
  note?: string;
  artifact_map?: ArtifactMap | null;
}

export interface CheckpointRow {
  slug: string;
  label: string;
  blurb?: string;
  score: number;
  role?: string;
  isSynthetic: boolean;
}

export interface SummaryRow {
  modelName: string;
  score: number;
  confidence: number;
  modality?: string;
  modelVersion?: string;
  evidence: SummaryEvidence;
  isNeutral: boolean;
  isExperimental: boolean;
  label: string;
}

export interface AnalysisView {
  summary?: SummaryRow;
  checkpoints: CheckpointRow[];
  generalists: CheckpointRow[];
  faceCheckpoints: CheckpointRow[];
  secondarySignals: SecondarySignal[];
  isEmpty: boolean;
}

export interface SecondarySignal {
  signal: string;
  label: string;
  hasVerdict: boolean;
  score?: number;
  rationale?: string;
  note?: string;
  evidence: Record<string, unknown>;
}

type Row = Pick<AnalysisResult, "model_name" | "confidence" | "label" | "details">;

// the verdict row is the one tagged tier: summary
function isSummary(row: Row): boolean {
  const tier = row.details?.tier;
  if (typeof tier === "string") return tier === "summary";
  return row.model_name in SUMMARY_ROWS || /\(fused\)|\(stub\)$/.test(row.model_name);
}

function isSecondary(row: Row): boolean {
  return row.details?.tier === "secondary";
}

function toSecondary(row: Row): SecondarySignal {
  const evidence = (row.details ?? {}) as Record<string, unknown>;
  const confidence = typeof evidence.confidence === "number" ? evidence.confidence : 0;
  return {
    signal: typeof evidence.signal === "string" ? evidence.signal : "unknown",
    label: row.model_name,
    hasVerdict: confidence > 0,
    score: confidence > 0 ? row.confidence : undefined,
    rationale: typeof evidence.rationale === "string" ? evidence.rationale : undefined,
    note: typeof evidence.note === "string" ? evidence.note : undefined,
    evidence,
  };
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
    isNeutral: confidence <= 0,
    isExperimental: evidence.experimental === true,
    label: row.label,
  };
}

// split rows into the verdict, the checkpoints and the secondary signals
export function readAnalysis(rows: readonly Row[] | undefined | null): AnalysisView {
  const all = rows ?? [];
  const summaryRow = all.find(isSummary);
  const secondaryRows = all.filter(isSecondary);
  const checkpoints = all
    .filter((r) => !isSummary(r) && !isSecondary(r))
    .map(toCheckpoint);

  checkpoints.sort((a, b) => b.score - a.score);

  return {
    summary: summaryRow ? toSummary(summaryRow) : undefined,
    checkpoints,
    generalists: checkpoints.filter((c) => c.role === "generalist"),
    faceCheckpoints: checkpoints.filter((c) => c.role === "face"),
    secondarySignals: secondaryRows.map(toSecondary),
    isEmpty: all.length === 0,
  };
}

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
  if (typeof e.max_audio_sec === "number") {
    lines.push(
      `A single WavLM-Large model scored the first ${e.max_audio_sec} seconds of audio at ${e.sample_rate ? `${e.sample_rate / 1000} kHz` : "16 kHz"} mono. Unlike the video and image channels there is no ensemble here — one model, one score.`,
    );
  }
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

export const THRESHOLDS = { manipulated: 65, authentic: 35 } as const;

export function verdictFromRisk(risk: number): CaseStatus {
  if (risk >= THRESHOLDS.manipulated) return "manipulated";
  if (risk <= THRESHOLDS.authentic) return "authentic";
  return "inconclusive";
}

export function engineNameFor(mediaType: MediaType): string {
  return {
    video: "Visual forensics — ViViT ensemble",
    image: "Image forensics — ViT-B/16 ensemble",
    audio: "Audio forensics — WavLM-Large",
  }[mediaType];
}
