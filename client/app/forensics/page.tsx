"use client";

/* ============================================================================
   DEEP TRUTH — FORENSICS HUB
   ----------------------------------------------------------------------------
   Upload evidence, watch the pipeline run, read the verdict. The server queues
   analysis itself on POST /cases, so there's no separate "start" step — the
   case comes back `processing` and useCase polls until it settles.
   ========================================================================= */

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  CheckCircle2,
  FileAudio,
  FileVideo,
  Image as ImageIcon,
  Loader2,
  RotateCcw,
  UploadCloud,
  X,
} from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { CaseTable } from "@/app/components/CaseTable";
import { MediaPreview } from "@/app/components/MediaPreview";
import {
  EmptyState,
  LivePill,
  PageHeader,
  Panel,
  PanelHeading,
} from "@/app/components/Primitives";
import { ResultBarChart } from "@/app/components/ResultBarChart";
import { StatusBadge } from "@/app/components/StatusBadge";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { useAuth } from "@/app/hooks/use-auth";
import { useCase, useCases, useCreateCase } from "@/app/hooks/use-cases";
import {
  ACCEPT_ATTRIBUTE,
  MAX_FILE_MB,
  formatBytes,
  inferMediaType,
  validateFile,
} from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { VERDICT, type MediaType } from "@/app/shared/schema";

const MODALITIES: { id: MediaType; icon: typeof ImageIcon; label: string; blurb: string }[] = [
  { id: "image", icon: ImageIcon, label: "Image", blurb: "Noise-residual and GAN artefact analysis" },
  { id: "video", icon: FileVideo, label: "Video", blurb: "Keyframe sampling with temporal consistency checks" },
  { id: "audio", icon: FileAudio, label: "Audio", blurb: "Spectral and prosody analysis for cloned speech" },
];

export default function ForensicsHubPage() {
  const router = useRouter();
  const { operator } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [mediaType, setMediaType] = useState<MediaType>("video");
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  const createCase = useCreateCase(setProgress);
  const { data: activeCase } = useCase(activeCaseId ?? undefined);
  const { data: list, isLoading: listLoading } = useCases({ pageSize: 6 });

  /* ── File selection ────────────────────────────────────────────────────── */

  const acceptFile = useCallback((picked: File) => {
    const detected = inferMediaType(picked);
    const problem = validateFile(picked, detected);

    setMediaType(detected);
    setFile(picked);
    setFileError(problem);
    setTitle((current) => current || picked.name.replace(/\.[^.]+$/, ""));

    setObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return problem ? null : URL.createObjectURL(picked);
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const picked = e.dataTransfer.files?.[0];
      if (picked) acceptFile(picked);
    },
    [acceptFile],
  );

  function clearSelection() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(null);
    setFile(null);
    setTitle("");
    setFileError(null);
    setProgress(0);
    if (inputRef.current) inputRef.current.value = "";
  }

  function startOver() {
    clearSelection();
    setActiveCaseId(null);
  }

  function submit() {
    if (!file || fileError || !title.trim()) return;
    createCase.mutate(
      { title: title.trim(), mediaType, file, userId: operator?.id },
      {
        onSuccess: (created) => {
          setActiveCaseId(created.caseId);
          clearSelection();
        },
      },
    );
  }

  /* ── Derived view state ────────────────────────────────────────────────── */

  const uploading = createCase.isPending;
  const analysing = activeCase?.status === "processing";
  const settled = activeCase && !analysing;
  const canSubmit = Boolean(file) && !fileError && title.trim().length > 0 && !uploading;

  return (
    <AppShell width="wide">
      <PageHeader
        status={<LivePill label="Detection pipeline ready" tone="good" />}
        title="Forensics Hub"
        description="Upload a video, image or audio clip. The pipeline splits the streams, runs each engine, and fuses the results into one verdict."
        actions={
          activeCaseId ? (
            <Button variant="outline" onClick={startOver} className="border-slate-700 text-slate-300 hover:text-white">
              <RotateCcw className="mr-2 h-4 w-4" />
              Analyse another
            </Button>
          ) : undefined
        }
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        {/* ── Upload ──────────────────────────────────────────────────────── */}
        <div className="space-y-6 xl:col-span-2">
          <Panel className="overflow-hidden">
            <PanelHeading
              icon={UploadCloud}
              title="Submit evidence"
              hint={`Up to ${MAX_FILE_MB} MB · analysed in memory, not retained after the case is deleted`}
            />

            <div className="space-y-5 p-5">
              {/* Drop zone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                className={cn(
                  "relative rounded-xl border-2 border-dashed transition-colors",
                  dragging
                    ? "border-indigo-500 bg-indigo-500/5"
                    : fileError
                      ? "border-red-500/50 bg-red-500/5"
                      : file
                        ? "border-slate-700 bg-slate-950"
                        : "border-slate-800 bg-slate-950 hover:border-slate-700",
                )}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept={ACCEPT_ATTRIBUTE}
                  onChange={(e) => {
                    const picked = e.target.files?.[0];
                    if (picked) acceptFile(picked);
                  }}
                  disabled={uploading}
                  className="absolute inset-0 cursor-pointer opacity-0 disabled:cursor-not-allowed"
                  aria-label="Choose an evidence file"
                />

                {!file ? (
                  <div className="pointer-events-none flex flex-col items-center gap-3 px-6 py-12 text-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-slate-800 bg-slate-900">
                      <UploadCloud className="h-6 w-6 text-slate-500" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">Drop a file here</p>
                      <p className="mt-1 text-xs text-slate-500">or click to browse</p>
                    </div>
                    <p className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
                      MP4 · MOV · WEBM · JPG · PNG · WAV · MP3 · FLAC
                    </p>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 p-4">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-slate-800 bg-slate-900">
                      {(() => {
                        const Icon = MODALITIES.find((m) => m.id === mediaType)!.icon;
                        return <Icon className="h-5 w-5 text-indigo-400" />;
                      })()}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-white">{file.name}</p>
                      <p className="font-mono text-[11px] text-slate-500">
                        {formatBytes(file.size)} · {mediaType}
                      </p>
                    </div>
                    {!uploading && (
                      <button
                        onClick={clearSelection}
                        aria-label="Remove selected file"
                        className="relative z-10 rounded-md p-1.5 text-slate-500 hover:bg-slate-800 hover:text-white"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                )}
              </div>

              {fileError && (
                <p className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs leading-relaxed text-red-300">
                  {fileError}
                </p>
              )}

              {/* Local preview before upload */}
              {objectUrl && file && !uploading && (
                <MediaPreview fileUrl={objectUrl} mediaType={mediaType} fileName={file.name} />
              )}

              {/* Case title */}
              <div className="space-y-1.5">
                <label htmlFor="case-title" className="eyebrow block">
                  Case title
                </label>
                <Input
                  id="case-title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Senator interview clip"
                  disabled={uploading}
                  className="h-10 border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
                />
                <p className="text-[11px] text-slate-600">
                  How this case will read in the archive.
                </p>
              </div>

              {/* Upload progress */}
              {uploading && (
                <div className="space-y-2">
                  <div className="flex justify-between font-mono text-[11px] text-slate-400">
                    <span>Uploading</span>
                    <span className="tabular">{progress}%</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-600 to-indigo-400 transition-all duration-200"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              <Button
                onClick={submit}
                disabled={!canSubmit}
                className="glow-brand h-11 w-full bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:shadow-none"
              >
                {uploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Uploading evidence
                  </>
                ) : (
                  <>
                    Start analysis
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          </Panel>

          {/* What each modality runs */}
          <Panel className="p-5">
            <p className="eyebrow mb-3">What runs per modality</p>
            <ul className="space-y-3">
              {MODALITIES.map((m) => (
                <li key={m.id} className="flex gap-3">
                  <m.icon
                    className={cn(
                      "mt-0.5 h-4 w-4 shrink-0",
                      mediaType === m.id ? "text-indigo-400" : "text-slate-600",
                    )}
                  />
                  <div className="min-w-0">
                    <p
                      className={cn(
                        "text-sm font-medium",
                        mediaType === m.id ? "text-white" : "text-slate-400",
                      )}
                    >
                      {m.label}
                    </p>
                    <p className="text-xs leading-relaxed text-slate-500">{m.blurb}</p>
                  </div>
                </li>
              ))}
            </ul>
          </Panel>
        </div>

        {/* ── Results ─────────────────────────────────────────────────────── */}
        <div className="xl:col-span-3">
          <Panel className="overflow-hidden">
            <PanelHeading
              title={activeCase ? `Result · ${activeCase.caseId}` : "Result"}
              hint={
                activeCase
                  ? VERDICT[activeCase.status].summary
                  : "Submit a file and the verdict appears here"
              }
              action={activeCase ? <StatusBadge status={activeCase.status} size="sm" /> : undefined}
            />

            <div className="p-5">
              {!activeCase ? (
                <EmptyState
                  icon={UploadCloud}
                  title="Nothing under analysis"
                  description="Pick a file on the left to open a case. Results stream in here as each engine reports."
                />
              ) : analysing ? (
                <PipelineProgress mediaType={activeCase.mediaType} />
              ) : (
                <div className="space-y-6">
                  {settled && (
                    <div className="flex items-start gap-3 rounded-lg border border-slate-800 bg-slate-950 p-4">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-white">Analysis complete</p>
                        <p className="mt-0.5 text-xs text-slate-500">
                          {activeCase.analysisResults.length} engine rows written for{" "}
                          {activeCase.title}.
                        </p>
                      </div>
                      <Link href={`/cases/${activeCase.caseId}`}>
                        <Button size="sm" variant="outline" className="border-slate-700 text-slate-300 hover:text-white">
                          Full report
                        </Button>
                      </Link>
                    </div>
                  )}

                  <ResultBarChart
                    riskScore={activeCase.riskScore}
                    syntheticLikelihood={activeCase.syntheticLikelihood}
                    analysisResults={activeCase.analysisResults}
                  />
                </div>
              )}
            </div>
          </Panel>
        </div>
      </div>

      {/* ── Recent cases ──────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          title="Recent scans"
          hint="Your last six investigations"
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
          cases={list?.cases ?? []}
          isLoading={listLoading}
          empty={
            <EmptyState
              icon={UploadCloud}
              title="No scans yet"
              description="Your first case will appear here as soon as you upload evidence."
            />
          }
        />
      </Panel>
    </AppShell>
  );
}

/* ── Pipeline progress ───────────────────────────────────────────────────────
   Stages mirror the real order in server/app/services/analyser.py, so what the
   operator watches matches what the backend is doing.
   ------------------------------------------------------------------------- */

function PipelineProgress({ mediaType }: { mediaType: MediaType }) {
  const stages =
    mediaType === "video"
      ? ["De-multiplexing audio and video", "Sampling informative keyframes", "Visual forensics engine", "Audio engine", "Noise-residual analysis", "Fusing scores"]
      : mediaType === "audio"
        ? ["Normalising sample rate", "Building spectrogram", "Audio engine", "Fusing scores"]
        : ["Decoding image", "Noise-residual analysis", "Fusing scores"];

  return (
    <div className="space-y-5 py-2">
      <div className="flex items-center gap-2.5">
        <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
        <p className="text-sm font-medium text-white">Running the pipeline</p>
      </div>

      <ol className="space-y-3">
        {stages.map((stage, i) => (
          <li key={stage} className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] text-slate-400">
                <span className="mr-2 text-slate-600">{String(i + 1).padStart(2, "0")}</span>
                {stage}
              </span>
            </div>
            <div
              className="shimmer h-1 overflow-hidden rounded-full bg-slate-800"
              style={{ animationDelay: `${i * 140}ms` }}
            />
          </li>
        ))}
      </ol>

      <p className="pt-1 text-center font-mono text-[11px] text-slate-600">
        This usually takes 5–15 seconds. Results appear automatically.
      </p>
    </div>
  );
}
