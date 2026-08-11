"use client";

/* ============================================================================
   DEEP TRUTH — RESULT BREAKDOWN
   ----------------------------------------------------------------------------
   Renders one case's analysis: the fused verdict, the evidence behind it in
   plain language, and the individual checkpoint votes underneath.

   Row interpretation lives in app/lib/analysis.ts. This file only decides how
   the result looks — it does no name-matching of its own, which is what the
   previous version got wrong.
   ========================================================================= */

import { useEffect, useState } from "react";
import { ChevronDown, FlaskConical, Info, ShieldQuestion } from "lucide-react";

import { cn } from "@/app/lib/utils";
import {
  explainEvidence,
  readAnalysis,
  verdictFromRisk,
  type CheckpointRow,
  type SecondarySignal,
} from "@/app/lib/analysis";
import { VERDICT, type AnalysisResult } from "@/app/shared/schema";

/* ── Single animated bar ─────────────────────────────────────────────────── */

interface BarProps {
  label: string;
  value: number;
  barClass?: string;
  delay?: number;
  showPercent?: boolean;
  size?: "sm" | "md";
  title?: string;
}

export function ConfidenceBar({
  label,
  value,
  barClass = "bg-indigo-500",
  delay = 0,
  showPercent = true,
  size = "md",
  title,
}: BarProps) {
  const [width, setWidth] = useState(0);
  const clamped = Math.max(0, Math.min(100, value));

  useEffect(() => {
    // Respect a reduced-motion preference: land on the final width straight
    // away rather than sweeping the bar across the panel.
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (reduced) {
      setWidth(clamped);
      return;
    }
    const t = setTimeout(() => setWidth(clamped), delay + 60);
    return () => clearTimeout(t);
  }, [clamped, delay]);

  return (
    <div className="space-y-1.5" title={title}>
      <div className="flex items-center justify-between gap-3">
        <span
          className={cn(
            "min-w-0 truncate font-mono text-slate-400",
            size === "sm" ? "text-[10px]" : "text-xs",
          )}
        >
          {label}
        </span>
        {showPercent && (
          <span
            className={cn(
              "shrink-0 font-mono font-bold tabular-nums text-white",
              size === "sm" ? "text-[10px]" : "text-xs",
            )}
          >
            {clamped.toFixed(1)}%
          </span>
        )}
      </div>
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-slate-800/80",
          size === "sm" ? "h-1" : "h-1.5",
        )}
      >
        <div
          className={cn("h-full rounded-full transition-all duration-700 ease-out", barClass)}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

/* ── Main breakdown ──────────────────────────────────────────────────────── */

interface ResultBarChartProps {
  riskScore: number;
  syntheticLikelihood: number;
  analysisResults: Pick<AnalysisResult, "model_name" | "confidence" | "label" | "details">[];
}

export function ResultBarChart({
  riskScore,
  syntheticLikelihood,
  analysisResults,
}: ResultBarChartProps) {
  const [showDetail, setShowDetail] = useState(false);

  const { summary, checkpoints, generalists, faceCheckpoints, secondarySignals } =
    readAnalysis(analysisResults);
  const clamped = Math.max(0, Math.min(100, riskScore));

  /* A neutral summary means the engine declined to contribute. Reporting a
     percentage in that case would be inventing a reading the pipeline never
     took, so the headline says so outright. */
  const neutral = summary?.isNeutral ?? false;
  const status = neutral ? "inconclusive" : verdictFromRisk(clamped);
  const meta = VERDICT[status];
  const explanation = explainEvidence(summary);

  const mainBarClass =
    status === "manipulated"
      ? "bg-gradient-to-r from-red-600 to-red-400"
      : status === "authentic"
        ? "bg-gradient-to-r from-emerald-600 to-emerald-400"
        : "bg-gradient-to-r from-amber-600 to-amber-400";

  /* The image branch tags each adapter generalist/face; the video branch
     doesn't, so it falls back to one flat list. */
  const grouped = generalists.length > 0 || faceCheckpoints.length > 0;

  return (
    <div className="space-y-6">
      {/* ── Verdict header ───────────────────────────────────────────────── */}
      <div className="space-y-3 border-b border-slate-800 pb-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div className="min-w-0">
            <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-slate-500">
              Verdict
            </p>
            <p className={cn("font-display text-xl font-black tracking-tight", meta.text)}>
              {meta.label}
            </p>
          </div>
          <div className="text-right">
            <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-slate-500">
              {neutral ? "No reading" : "Risk score"}
            </p>
            <p className={cn("font-mono text-3xl font-black tabular-nums", meta.text)}>
              {neutral ? "—" : <>{clamped.toFixed(0)}<span className="text-lg">%</span></>}
            </p>
          </div>
        </div>

        {summary?.modelVersion && !neutral && (
          <p className="font-mono text-[10px] tracking-wide text-slate-600">
            {summary.modelName} · {summary.modelVersion}
            {summary.confidence > 0 && (
              <> · engine confidence {(summary.confidence * 100).toFixed(0)}%</>
            )}
          </p>
        )}
      </div>

      {/* ── Unvalidated checkpoint ────────────────────────────────────────────
          Sits above the score deliberately. A model that has not been checked
          against real-world data can be confidently wrong, and a reader who
          scrolls past the number without seeing this has been misled by the
          interface rather than by the model. */}
      {summary?.isExperimental && !neutral && (
        <div className="flex gap-3 rounded-lg border border-orange-500/30 bg-orange-500/10 px-3.5 py-3">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-orange-400" />
          <div className="min-w-0 space-y-1">
            <p className="text-xs font-semibold text-orange-300">
              Unvalidated model — treat this verdict as provisional
            </p>
            <p className="text-[11px] leading-relaxed text-slate-400">
              This checkpoint has not been confirmed against diverse real-world
              recordings. It may report genuine media as fake. Use it to exercise the
              pipeline, not to support a conclusion about this file.
            </p>
          </div>
        </div>
      )}

      {/* ── Neutral notice ───────────────────────────────────────────────── */}
      {neutral && (
        <div className="flex gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3.5 py-3">
          <ShieldQuestion className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <div className="min-w-0 space-y-1">
            <p className="text-xs font-semibold text-amber-300">
              This modality produced no usable signal
            </p>
            {explanation.map((line, i) => (
              <p key={i} className="text-[11px] leading-relaxed text-slate-400">
                {line}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* ── Primary metrics ──────────────────────────────────────────────── */}
      {!neutral && (
        <div className="space-y-3">
          <ConfidenceBar
            label="Fused risk score"
            value={riskScore}
            barClass={mainBarClass}
            delay={0}
          />
          <ConfidenceBar
            label="Synthetic likelihood"
            value={syntheticLikelihood}
            barClass="bg-gradient-to-r from-purple-600 to-purple-400"
            delay={120}
          />
        </div>
      )}

      {/* ── How the engine got there ─────────────────────────────────────── */}
      {!neutral && explanation.length > 0 && (
        <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900/60 px-3.5 py-3">
          <div className="flex items-center gap-2">
            <Info className="h-3.5 w-3.5 shrink-0 text-slate-500" />
            <p className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
              How this was decided
            </p>
          </div>
          {explanation.map((line, i) => (
            <p key={i} className="text-[11px] leading-relaxed text-slate-400">
              {line}
            </p>
          ))}
        </div>
      )}

      {/* ── Checkpoint votes ─────────────────────────────────────────────── */}
      {checkpoints.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setShowDetail((s) => !s)}
            aria-expanded={showDetail}
            className="group flex w-full items-center gap-2 rounded text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/60"
          >
            <p className="font-mono text-[10px] uppercase tracking-widest text-slate-600 transition-colors group-hover:text-slate-400">
              Individual model votes ({checkpoints.length})
            </p>
            <div className="h-px flex-1 bg-slate-800" />
            <ChevronDown
              className={cn(
                "h-3.5 w-3.5 shrink-0 text-slate-600 transition-transform",
                showDetail && "rotate-180",
              )}
            />
          </button>

          {showDetail &&
            (grouped ? (
              <div className="space-y-4 pt-1">
                <CheckpointGroup
                  title="Whole-image models"
                  hint="Scored the full frame"
                  rows={generalists}
                />
                <CheckpointGroup
                  title="Face-crop models"
                  hint="Scored the detected face only"
                  rows={faceCheckpoints}
                />
              </div>
            ) : (
              <div className="space-y-2.5 pt-1">
                {checkpoints.map((c, i) => (
                  <CheckpointBar key={c.slug} row={c} delay={i * 60} />
                ))}
              </div>
            ))}
        </div>
      )}

      {/* ── Secondary signals (SRM, etc.) ────────────────────────────────────
          Deliberately styled apart from everything above: no risk bar, no
          colour tied to the primary verdict palette, and its own heading —
          nothing here should read as "another vote" the way a checkpoint
          does. This is supplementary evidence the primary engine's fused
          score never saw. */}
      {secondarySignals.length > 0 && (
        <div className="space-y-2 border-t border-slate-800 pt-4">
          <p className="font-mono text-[10px] uppercase tracking-widest text-slate-600">
            Supplementary signals
          </p>
          {secondarySignals.map((s, i) => (
            <SecondarySignalCard key={`${s.signal}-${i}`} signal={s} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Secondary signal card ──────────────────────────────────────────────── */

function SecondarySignalCard({ signal }: { signal: SecondarySignal }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-3.5 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium text-slate-300">{signal.label}</p>
        {signal.hasVerdict && typeof signal.score === "number" ? (
          <span className="font-mono text-xs font-semibold text-slate-300">
            {signal.score.toFixed(1)}%
          </span>
        ) : (
          <span className="rounded-full border border-slate-700 bg-slate-800/60 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-slate-500">
            Not yet trained
          </span>
        )}
      </div>
      {(signal.rationale || signal.note) && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
          {signal.rationale ?? signal.note}
        </p>
      )}
    </div>
  );
}

/* ── Checkpoint pieces ───────────────────────────────────────────────────── */

function CheckpointBar({ row, delay }: { row: CheckpointRow; delay: number }) {
  return (
    <ConfidenceBar
      label={row.label}
      value={row.score}
      title={row.blurb}
      barClass={
        row.isSynthetic
          ? "bg-gradient-to-r from-red-700/70 to-red-500/70"
          : "bg-gradient-to-r from-emerald-700/70 to-emerald-500/70"
      }
      delay={delay}
      size="sm"
    />
  );
}

function CheckpointGroup({
  title,
  hint,
  rows,
}: {
  title: string;
  hint: string;
  rows: CheckpointRow[];
}) {
  if (rows.length === 0) return null;
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <p className="text-[11px] font-semibold text-slate-300">{title}</p>
        <p className="font-mono text-[10px] text-slate-600">{hint}</p>
      </div>
      {rows.map((c, i) => (
        <CheckpointBar key={c.slug} row={c} delay={i * 60} />
      ))}
    </div>
  );
}