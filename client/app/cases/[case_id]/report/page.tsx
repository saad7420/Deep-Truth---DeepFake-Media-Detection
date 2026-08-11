"use client";

/* ============================================================================
   DEEP TRUTH — EXPORTABLE FORENSIC REPORT
   ----------------------------------------------------------------------------
   A print-first document rather than a console screen. "Export report" hands
   off to the browser's own print-to-PDF, which keeps selectable text, working
   links and correct pagination — all of which a client-side canvas capture
   would lose, and none of which needs a server-side PDF service the backend
   doesn't have.

   The integrity line is a plain content digest over the fields shown, not a
   cryptographic signature: it detects an altered copy of this document, and
   the report says exactly that rather than implying more.
   ========================================================================= */

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, Printer, Shield } from "lucide-react";

import { ErrorState, Spinner } from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { useCase } from "@/app/hooks/use-cases";
import { formatBytes } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { VERDICT, riskTone } from "@/app/shared/schema";
import { engineNameFor, explainEvidence, readAnalysis } from "@/app/lib/analysis";

export default function ReportPage() {
  const params = useParams<{ case_id: string }>();
  const caseId = params?.case_id;
  const { data: c, isLoading, isError, error } = useCase(caseId);
  const [digest, setDigest] = useState<string>("");

  const fingerprintSource = useMemo(() => {
    if (!c) return "";
    return [c.caseId, c.title, c.mediaType, c.status, c.riskScore, c.syntheticLikelihood, c.createdAt]
      .join("|");
  }, [c]);

  useEffect(() => {
    if (!fingerprintSource || typeof window === "undefined" || !window.crypto?.subtle) return;
    const bytes = new TextEncoder().encode(fingerprintSource);
    window.crypto.subtle.digest("SHA-256", bytes).then((buf) => {
      const hex = Array.from(new Uint8Array(buf))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      setDigest(hex.toUpperCase());
    });
  }, [fingerprintSource]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-950">
        <Spinner label="Preparing report" />
      </div>
    );
  }

  if (isError || !c) {
    return (
      <div className="min-h-screen bg-slate-950 pt-20">
        <ErrorState title="Couldn't build the report" message={error?.message ?? "Case not found."} />
      </div>
    );
  }

  const meta = VERDICT[c.status];
  const tone = riskTone(c.riskScore);
  const { summary, checkpoints, secondarySignals } = readAnalysis(c.analysisResults);
  const explanation = explainEvidence(summary);
  const rationale = summary?.evidence.rationale;

  return (
    <div className="min-h-screen bg-slate-950 print:bg-white">
      {/* Screen-only toolbar */}
      <div className="no-print sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-3">
          <Link href={`/cases/${c.caseId}`}>
            <Button variant="ghost" size="sm" className="-ml-2 text-slate-400 hover:text-white">
              <ArrowLeft className="mr-1.5 h-4 w-4" />
              Back to case
            </Button>
          </Link>
          <Button
            onClick={() => window.print()}
            className="glow-brand bg-indigo-600 text-white hover:bg-indigo-500"
          >
            <Printer className="mr-2 h-4 w-4" />
            Print or save as PDF
          </Button>
        </div>
      </div>

      <article className="mx-auto max-w-4xl px-6 py-10 print:max-w-none print:px-0 print:py-0">
        {/* ── Masthead ────────────────────────────────────────────────────── */}
        <header className="mb-8 border-b border-slate-800 pb-6 print:border-slate-300">
          <div className="flex items-start justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-indigo-500/40 bg-indigo-600/20 print:border-slate-300 print:bg-transparent">
                <Shield className="h-6 w-6 text-indigo-400 print:text-slate-700" />
              </div>
              <div>
                <p className="font-display text-lg font-bold tracking-wide text-white print:text-slate-900">
                  DEEP TRUTH
                </p>
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500">
                  Forensic analysis report
                </p>
              </div>
            </div>

            <div className="text-right">
              <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">Case</p>
              <p className="font-mono text-sm font-semibold text-indigo-400 print:text-slate-900">
                {c.caseId}
              </p>
              <p className="mt-1 font-mono text-[10px] text-slate-500">
                Generated {new Date().toLocaleString()}
              </p>
            </div>
          </div>
        </header>

        {/* ── Verdict ─────────────────────────────────────────────────────── */}
        <Section title="1 · Verdict">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <h1 className="font-display text-2xl font-bold text-white print:text-slate-900">
                {c.title}
              </h1>
              <p className={cn("mt-1 font-display text-lg font-bold", meta.text)}>{meta.label}</p>
              <p className="mt-1.5 max-w-md text-sm text-slate-400 print:text-slate-600">
                {meta.summary}
              </p>
              <p className="mt-2 font-mono text-xs text-slate-500">
                {engineNameFor(c.mediaType)}
                {summary?.modelVersion ? ` · ${summary.modelVersion}` : ""}
              </p>
            </div>

            <div className="shrink-0 text-right">
              <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                Fused risk score
              </p>
              <p className={cn("tabular text-5xl font-bold leading-none", tone.text)}>
                {c.riskScore.toFixed(1)}
                <span className="text-2xl text-slate-600">%</span>
              </p>
            </div>
          </div>

          {rationale && (
            <p className="mt-5 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 text-xs leading-relaxed text-slate-400 print-plain">
              {rationale}
            </p>
          )}
        </Section>

        {/* ── Evidence ────────────────────────────────────────────────────── */}
        <Section title="2 · Evidence">
          <dl className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-3">
            <Field label="File name" value={c.fileName ?? "—"} />
            <Field label="Modality" value={c.mediaType.toUpperCase()} />
            <Field label="File size" value={formatBytes(c.fileSize)} />
            <Field label="Case opened" value={fmt(c.createdAt)} />
            <Field label="Analysis completed" value={fmt(c.updatedAt)} />
            <Field label="Submitted by" value={c.userId ?? "Unattributed"} />
          </dl>
        </Section>

        {/* ── Engine result ───────────────────────────────────────────────── */}
        <Section title="3 · Engine result">
          {!summary ? (
            <p className="text-sm text-slate-500">
              No engine has reported for this case yet.
            </p>
          ) : (
            <>
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-800 font-mono text-[10px] uppercase tracking-wider text-slate-500 print:border-slate-300">
                    <th scope="col" className="py-2 font-medium">Engine</th>
                    <th scope="col" className="py-2 font-medium">Finding</th>
                    <th scope="col" className="py-2 text-right font-medium">Fake probability</th>
                    <th scope="col" className="py-2 text-right font-medium">Engine confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 print:divide-slate-200">
                  <tr>
                    <td className="py-2.5 text-slate-300 print:text-slate-800">
                      {summary.modelName}
                    </td>
                    <td className="py-2.5">
                      <span
                        className={cn(
                          "font-mono text-xs font-semibold",
                          summary.label === "SYNTHETIC"
                            ? "text-red-400"
                            : summary.label === "AUTHENTIC"
                              ? "text-emerald-400"
                              : "text-amber-400",
                        )}
                      >
                        {summary.label}
                      </span>
                    </td>
                    <td className="tabular py-2.5 text-right text-slate-300 print:text-slate-800">
                      {summary.isNeutral ? "—" : `${summary.score.toFixed(2)}%`}
                    </td>
                    <td className="tabular py-2.5 text-right text-slate-400 print:text-slate-600">
                      {summary.confidence > 0
                        ? `${(summary.confidence * 100).toFixed(0)}%`
                        : "not contributing"}
                    </td>
                  </tr>
                </tbody>
              </table>

              {explanation.length > 0 && (
                <div className="mt-4 space-y-1.5">
                  <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                    How this was decided
                  </p>
                  {explanation.map((line, i) => (
                    <p
                      key={i}
                      className="text-xs leading-relaxed text-slate-400 print:text-slate-600"
                    >
                      {line}
                    </p>
                  ))}
                </div>
              )}
            </>
          )}
        </Section>

        {/* ── Checkpoint detail ───────────────────────────────────────────── */}
        {checkpoints.length > 0 && (
          <Section title="4 · Checkpoint detail">
            <p className="mb-3 text-xs text-slate-500">
              Each adapter is fine-tuned on a different corpus of manipulated media and votes
              independently. The fused score above is not a plain average of these — see the
              reasoning in section 3.
            </p>
            <ul className="grid grid-cols-1 gap-x-8 gap-y-1.5 sm:grid-cols-2">
              {checkpoints.map((r) => (
                <li key={r.slug} className="flex items-baseline justify-between gap-3 text-xs">
                  <span className="truncate text-slate-400 print:text-slate-600">
                    {r.label}
                    {r.role && (
                      <span className="ml-1.5 font-mono text-[10px] text-slate-600">
                        {r.role}
                      </span>
                    )}
                  </span>
                  <span
                    className={cn(
                      "tabular shrink-0 font-semibold",
                      r.isSynthetic ? "text-red-400" : "text-emerald-400",
                    )}
                  >
                    {r.score.toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {/* ── Supplementary signals ─────────────────────────────────────────
            Deliberately its own section, not folded into "Checkpoint
            detail" above. These are additional evidence passes that do not
            feed the primary verdict (currently: SRM noise analysis) — the
            report previously had no separate section for these, so a
            secondary-tier row like SRM sorted straight into the checkpoint
            list by score and printed above every real adapter vote,
            reading as if it were one of them. */}
        {secondarySignals.length > 0 && (
          <Section title={`${checkpoints.length > 0 ? "5" : "4"} · Supplementary signals`}>
            <p className="mb-3 text-xs text-slate-500">
              Additional evidence gathered alongside the primary engine. These do not
              contribute to the verdict or fused risk score above.
            </p>
            <ul className="space-y-2">
              {secondarySignals.map((s, i) => (
                <li key={`${s.signal}-${i}`} className="text-xs">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-slate-400 print:text-slate-600">{s.label}</span>
                    {s.hasVerdict && typeof s.score === "number" ? (
                      <span className="tabular shrink-0 font-semibold text-slate-300">
                        {s.score.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="shrink-0 font-mono text-[10px] uppercase tracking-wider text-slate-600">
                        Not yet trained
                      </span>
                    )}
                  </div>
                  {(s.rationale || s.note) && (
                    <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500 print:text-slate-600">
                      {s.rationale ?? s.note}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {/* ── Notes ───────────────────────────────────────────────────────── */}
        {c.notes && (
          <Section
            title={`${
              (checkpoints.length > 0 ? 1 : 0) + (secondarySignals.length > 0 ? 1 : 0) + 4
            } · Investigator notes`}
          >
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300 print:text-slate-800">
              {c.notes}
            </p>
          </Section>
        )}

        {/* ── Integrity ───────────────────────────────────────────────────── */}
        <footer className="mt-10 border-t border-slate-800 pt-5 print:border-slate-300">
          <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            Content digest (SHA-256)
          </p>
          <p className="mt-1 break-all font-mono text-[10px] text-slate-400 print:text-slate-700">
            {digest || "computing…"}
          </p>
          <p className="mt-3 max-w-2xl text-[10px] leading-relaxed text-slate-600">
            This digest covers the case id, title, modality, verdict and scores printed above. It
            detects alteration of this document; it is not a signature and does not attest to the
            provenance of the underlying media. Scores are produced by statistical models and
            should be read as evidence supporting a conclusion, not as proof on their own.
          </p>
          <p className="mt-3 font-mono text-[10px] text-slate-600">
            Deep Truth · Active Deepfake Defense System · COMSATS University Islamabad
          </p>
        </footer>
      </article>
    </div>
  );
}

/* ── Report primitives ───────────────────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-8 break-inside-avoid">
      <h2 className="mb-4 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-400 print:text-slate-500">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{label}</dt>
      <dd className="mt-0.5 truncate text-sm text-slate-300 print:text-slate-800">{value}</dd>
    </div>
  );
}

function fmt(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}