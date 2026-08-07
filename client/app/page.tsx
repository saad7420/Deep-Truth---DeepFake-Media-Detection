"use client";

/* ============================================================================
   DEEP TRUTH — LANDING
   ----------------------------------------------------------------------------
   The public face of the console. The hero leads with the thing that actually
   distinguishes this system — that it watches and listens at once, and reports
   when the two disagree — rather than a generic capability grid.

   Figures in the "live" strip come from GET /stats. When the backend isn't
   running they fall back to em-dashes instead of invented numbers.
   ========================================================================= */

import Link from "next/link";
import {
  ArrowRight,
  AudioLines,
  Eye,
  FileAudio,
  FileVideo,
  Fingerprint,
  Image as ImageIcon,
  Layers,
  ShieldCheck,
  Waves,
} from "lucide-react";

import { PublicHeader } from "@/app/components/Navigation";
import { Button } from "@/app/components/ui/button";
import { useStats } from "@/app/hooks/use-cases";
import { cn } from "@/app/lib/utils";
import { RISK_BANDS } from "@/app/shared/schema";

const ENGINES = [
  {
    icon: Eye,
    module: "M7",
    name: "Visual forensics",
    body: "A transformer ensemble reads sampled keyframes for generation artefacts — warped texture, inconsistent lighting, blending seams at the edge of a swapped face.",
    accent: "text-indigo-400",
    ring: "border-indigo-500/30",
  },
  {
    icon: AudioLines,
    module: "M6",
    name: "Audio analysis",
    body: "Speech is converted to a spectral representation where the rhythm and timing irregularities of cloned voices become visible, in a way they aren't in the raw waveform.",
    accent: "text-purple-400",
    ring: "border-purple-500/30",
  },
  {
    icon: Waves,
    module: "M8",
    name: "Noise residuals",
    body: "Every camera sensor leaves a consistent noise fingerprint. Regions spliced in from elsewhere — or generated outright — break that consistency.",
    accent: "text-sky-400",
    ring: "border-sky-500/30",
  },
];

const MODALITIES = [
  { icon: FileVideo, label: "Video", detail: "Keyframe sampling, not frame-by-frame" },
  { icon: ImageIcon, label: "Image", detail: "Pixel and sensor-noise analysis" },
  { icon: FileAudio, label: "Audio", detail: "Spectral and prosody analysis" },
];

export default function LandingPage() {
  const { data: stats } = useStats();

  const fmt = (n?: number) => (typeof n === "number" ? n.toLocaleString() : "—");

  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-indigo-500/30">
      <PublicHeader />

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden pb-24 pt-36 lg:pb-32 lg:pt-48">
        <div className="grid-bg pointer-events-none absolute inset-0 opacity-40" />
        <div className="aurora pointer-events-none absolute inset-0" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-slate-950 to-transparent" />

        <div className="relative mx-auto max-w-5xl px-4 text-center sm:px-6 lg:px-8">
          <span className="animate-fade-up mb-8 inline-flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-indigo-300">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-indigo-500" />
            </span>
            Detection pipeline online
          </span>

          <h1
            className="animate-fade-up font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-6xl lg:text-7xl"
            style={{ animationDelay: "80ms" }}
          >
            Most detectors only
            <br />
            <span className="text-slate-500">watch,</span> or only{" "}
            <span className="text-slate-500">listen.</span>
            <br />
            <span className="neon-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
              Deep Truth does both.
            </span>
          </h1>

          <p
            className="animate-fade-up mx-auto mt-8 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg"
            style={{ animationDelay: "160ms" }}
          >
            A real video dubbed with a cloned voice passes a video-only detector, because the frames
            genuinely are real. This system runs visual, audio and noise-residual analysis together,
            then reports when they disagree — because that disagreement is the finding.
          </p>

          <div
            className="animate-fade-up mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row"
            style={{ animationDelay: "240ms" }}
          >
            <Link href="/forensics">
              <Button
                size="lg"
                className="glow-brand-strong h-13 bg-indigo-600 px-8 text-base text-white transition-transform hover:scale-[1.02] hover:bg-indigo-500"
              >
                Analyse a file
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button
                size="lg"
                variant="outline"
                className="h-13 border-slate-700 px-8 text-base text-slate-300 hover:bg-slate-900 hover:text-white"
              >
                How it works
              </Button>
            </Link>
          </div>

          {/* Live counters */}
          <div
            className="animate-fade-up mx-auto mt-16 grid max-w-2xl grid-cols-3 divide-x divide-slate-800 rounded-xl border border-slate-800 bg-slate-900/40 backdrop-blur"
            style={{ animationDelay: "320ms" }}
          >
            {[
              { label: "Cases analysed", value: fmt(stats?.totalCases) },
              { label: "Verified authentic", value: fmt(stats?.authentic) },
              { label: "Manipulation found", value: fmt(stats?.manipulated) },
            ].map((s) => (
              <div key={s.label} className="px-4 py-5">
                <p className="tabular text-2xl font-bold text-white sm:text-3xl">{s.value}</p>
                <p className="mt-1 text-[11px] uppercase tracking-wider text-slate-500">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── The signature: cross-modal disagreement ───────────────────────── */}
      <section className="border-y border-slate-800 bg-slate-900/30 py-20">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <p className="eyebrow mb-3">The case that breaks other tools</p>
          <h2 className="font-display max-w-2xl text-2xl font-bold tracking-tight sm:text-3xl">
            Real footage. Synthetic voice.
          </h2>

          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
            <SignalCard
              label="Visual engine"
              reading="4% fake"
              verdict="Reads authentic"
              tone="emerald"
              note="The frames are genuine, so a video-only detector stops here and clears it."
            />
            <SignalCard
              label="Audio engine"
              reading="91% fake"
              verdict="Reads synthetic"
              tone="red"
              note="The voice was cloned. An audio-only tool catches this but sees no video context."
            />
            <SignalCard
              label="Fusion layer"
              reading="Disagreement"
              verdict="Voice clone"
              tone="indigo"
              note="Two confident engines pointing opposite ways is escalated, not averaged into an inconclusive middle."
            />
          </div>

          <p className="mt-8 max-w-3xl text-sm leading-relaxed text-slate-400">
            A plain weighted average would let those two readings cancel out into an unhelpful{" "}
            <span className="text-amber-400">inconclusive</span>. Instead the fusion layer treats the
            conflict itself as the signal and names the manipulation type, so the report tells you{" "}
            <em className="not-italic text-white">what</em> was faked, not just that something was.
          </p>
        </div>
      </section>

      {/* ── Engines ───────────────────────────────────────────────────────── */}
      <section id="capabilities" className="py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          <p className="eyebrow mb-3">Three independent readings</p>
          <h2 className="font-display max-w-2xl text-2xl font-bold tracking-tight sm:text-3xl">
            Every engine votes with its own confidence
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            An engine that isn&apos;t sure contributes proportionally less, and one that can&apos;t
            run at all drops out entirely rather than dragging the result toward the middle.
          </p>

          <div className="mt-10 grid grid-cols-1 gap-5 md:grid-cols-3">
            {ENGINES.map((e) => (
              <div
                key={e.name}
                className={cn(
                  "group rounded-2xl border border-slate-800 bg-slate-900/40 p-6 transition-colors",
                  "hover:bg-slate-900/70",
                  `hover:${e.ring}`,
                )}
              >
                <div className="mb-5 flex items-center justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-slate-800 bg-slate-950">
                    <e.icon className={cn("h-5 w-5", e.accent)} />
                  </div>
                  <span className="font-mono text-[10px] tracking-widest text-slate-700">
                    {e.module}
                  </span>
                </div>
                <h3 className="font-display text-lg font-semibold text-white">{e.name}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{e.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline ──────────────────────────────────────────────────────── */}
      <section id="pipeline" className="border-y border-slate-800 bg-slate-900/30 py-20">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <p className="eyebrow mb-3">Upload to verdict</p>
          <h2 className="font-display max-w-2xl text-2xl font-bold tracking-tight sm:text-3xl">
            Usually 5–15 seconds
          </h2>

          <ol className="mt-10 space-y-0">
            {[
              { n: "01", t: "Validate and store", d: "Type and size are checked before anything touches disk." },
              { n: "02", t: "Split the streams", d: "Video is separated into its audio track and a sampled set of informative keyframes — not every frame, which is what keeps this fast." },
              { n: "03", t: "Run the engines", d: "Visual, audio and noise-residual analysis run off the request thread, in parallel where the modality allows." },
              { n: "04", t: "Fuse and report", d: `Scores combine into one trust score: authentic at or below ${RISK_BANDS.authentic}%, manipulated at or above ${RISK_BANDS.manipulated}%.` },
            ].map((s, i, arr) => (
              <li key={s.n} className="flex gap-6">
                <div className="flex flex-col items-center">
                  <span className="tabular flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-indigo-500/30 bg-indigo-500/10 text-xs font-bold text-indigo-400">
                    {s.n}
                  </span>
                  {i < arr.length - 1 && <span className="w-px flex-1 bg-slate-800" />}
                </div>
                <div className={cn("min-w-0", i < arr.length - 1 && "pb-8")}>
                  <p className="font-display text-base font-semibold text-white">{s.t}</p>
                  <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">{s.d}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── Modalities + CTA ──────────────────────────────────────────────── */}
      <section className="py-20">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {MODALITIES.map((m) => (
              <div
                key={m.label}
                className="flex items-center gap-4 rounded-xl border border-slate-800 bg-slate-900/40 p-5"
              >
                <m.icon className="h-5 w-5 shrink-0 text-slate-500" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-white">{m.label}</p>
                  <p className="truncate text-xs text-slate-500">{m.detail}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-12 overflow-hidden rounded-2xl border border-indigo-500/20 bg-gradient-to-br from-indigo-950/40 to-slate-900/40 p-8 text-center sm:p-12">
            <Fingerprint className="mx-auto mb-5 h-9 w-9 text-indigo-400" />
            <h2 className="font-display text-2xl font-bold tracking-tight sm:text-3xl">
              Check something now
            </h2>
            <p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-slate-400">
              No account, no key. Drop a file into the console and the pipeline opens a case, runs
              every applicable engine, and writes a report you can export.
            </p>
            <Link href="/forensics">
              <Button
                size="lg"
                className="glow-brand mt-7 bg-indigo-600 px-8 text-white hover:bg-indigo-500"
              >
                Open the Forensics Hub
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Honest limits — kept on the marketing page on purpose ──────────── */}
      <section className="border-t border-slate-800 py-14">
        <div className="mx-auto max-w-3xl px-4 text-center sm:px-6">
          <Layers className="mx-auto mb-4 h-5 w-5 text-slate-600" />
          <p className="text-sm leading-relaxed text-slate-500">
            Scores are produced by statistical models. A high score means the media resembles
            synthetic content the models were trained on — it is evidence supporting a conclusion,
            not proof on its own. Heavy compression can degrade accuracy, and an engine that
            returns no confidence tells you nothing about that modality.{" "}
            <Link href="/docs" className="text-indigo-400 underline-offset-4 hover:underline">
              Read the limits in full
            </Link>
            .
          </p>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800 bg-slate-950 py-12">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-10 md:grid-cols-3">
            <div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-indigo-500" />
                <span className="font-display font-bold text-white">DEEP TRUTH</span>
              </div>
              <p className="mt-3 max-w-xs text-sm leading-relaxed text-slate-500">
                Active deepfake defense system. Multi-modal forensic verification for audio, video
                and images.
              </p>
            </div>

            <div>
              <p className="eyebrow mb-3">Console</p>
              <ul className="space-y-2 text-sm text-slate-500">
                {[
                  { href: "/dashboard", label: "Command Center" },
                  { href: "/forensics", label: "Forensics Hub" },
                  { href: "/cases", label: "Case History" },
                  { href: "/docs", label: "Documentation" },
                ].map((l) => (
                  <li key={l.href}>
                    <Link href={l.href} className="transition-colors hover:text-white">
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <p className="eyebrow mb-3">Project</p>
              <p className="font-mono text-sm leading-relaxed text-slate-500">
                Saad Mehmood
                <br />
                Ramish Naseer
                <br />
                <span className="text-slate-600">COMSATS University Islamabad</span>
              </p>
            </div>
          </div>

          <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-slate-900 pt-6 sm:flex-row">
            <p className="font-mono text-[11px] uppercase tracking-wider text-slate-600">
              Deep Truth · Final Year Project 2023–2027
            </p>
            <span className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
                Pipeline active
              </span>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

/* ── Signal card used in the disagreement walkthrough ────────────────────── */

function SignalCard({
  label,
  reading,
  verdict,
  note,
  tone,
}: {
  label: string;
  reading: string;
  verdict: string;
  note: string;
  tone: "emerald" | "red" | "indigo";
}) {
  const tones = {
    emerald: { border: "border-emerald-500/30", text: "text-emerald-400", bar: "bg-emerald-500" },
    red: { border: "border-red-500/30", text: "text-red-400", bar: "bg-red-500" },
    indigo: { border: "border-indigo-500/40", text: "text-indigo-400", bar: "bg-indigo-500" },
  }[tone];

  return (
    <div className={cn("rounded-xl border bg-slate-950/60 p-5", tones.border)}>
      <p className="eyebrow mb-3">{label}</p>
      <p className={cn("tabular text-2xl font-bold", tones.text)}>{reading}</p>
      <p className={cn("mt-1 text-sm font-medium", tones.text)}>{verdict}</p>
      <div className="my-4 h-px bg-slate-800" />
      <p className="text-xs leading-relaxed text-slate-500">{note}</p>
    </div>
  );
}
