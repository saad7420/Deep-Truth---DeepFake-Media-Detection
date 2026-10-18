"use client";

/* ============================================================================
   DEEP TRUTH — LANDING
   ----------------------------------------------------------------------------
   A dark hero over light, editorial sections. Every claim on this page is
   something the system actually does; model counts and datasets match
   config.py and the checkpoints directory. The hero card is labelled as an
   example so it is never mistaken for a real case.
   ========================================================================= */

import Link from "next/link";
import { ArrowRight, Plus } from "lucide-react";

import { PublicHeader } from "@/app/components/Navigation";
import { Button } from "@/app/components/ui/button";
import { cn } from "@/app/lib/utils";

/* ── Content ─────────────────────────────────────────────────────────────── */

const STATS = [
  { value: "16", label: "trained AI models" },
  { value: "3", label: "media types: image, video, audio" },
  { value: "1-click", label: "checks from any web page" },
  { value: "PDF", label: "forensic report for every case" },
];

const DETECTORS = [
  {
    key: "video",
    eyebrow: "Video detector · M7",
    title: "Catch face swaps and generated clips.",
    body: "Samples frames across the clip, finds faces, and scores them with six video transformers, including one that reads the whole frame for fully generated video.",
    points: ["6 ViViT + LoRA checkpoints", "Face-aware and whole-frame models", "Celeb-DF, FF++, DFDC, GenVideo"],
    cta: "Check a video",
  },
  {
    key: "image",
    eyebrow: "Image detector · M7",
    title: "Spot AI images and edited faces.",
    body: "Eight image transformers look for the fingerprints generators leave behind, and a heat map shows the regions that drove the verdict.",
    points: ["8 ViT-B/16 + LoRA checkpoints", "Grad-CAM artifact map", "GenImage, WildRF, DiffusionFace"],
    cta: "Check an image",
  },
  {
    key: "audio",
    eyebrow: "Voice detector · M6",
    title: "Hear the clone the ear misses.",
    body: "A WavLM speech model listens for the timing and spectral traces of text-to-speech and voice cloning, in standalone audio or a video's soundtrack.",
    points: ["WavLM-Large, fine-tuned", "Calibrated decision threshold", "ASVspoof 2019 LA"],
    cta: "Check audio",
  },
] as const;

const ARCHITECTURES = [
  { eyebrow: "Video stream", name: "ViViT", rows: [["Input", "16 frames, 224 px"], ["Checkpoints", "6"], ["Face finder", "MTCNN"]] },
  { eyebrow: "Image stream", name: "ViT-B/16", rows: [["Input", "224 × 224"], ["Checkpoints", "8"], ["Explains", "Grad-CAM map"]] },
  { eyebrow: "Voice stream", name: "WavLM-Large", rows: [["Input", "16 kHz speech"], ["Checkpoints", "1"], ["Output", "Calibrated score"]] },
  { eyebrow: "Noise stream", name: "SRM residuals", rows: [["Input", "5 noise filters"], ["Features", "45 statistics"], ["Role", "Supporting signal"]] },
];

const CHECKPOINTS = [
  ["celebdf_v2", "Video", "Celeb-DF v2", "Faces"],
  ["ffpp", "Video", "FaceForensics++", "Faces"],
  ["dfdc", "Video", "DFDC", "Faces"],
  ["deeperforensics", "Video", "DeeperForensics-1.0", "Faces"],
  ["wilddeepfake", "Video", "WildDeepfake", "Faces"],
  ["genvideo", "Video", "GenVideo-100K", "Whole frame"],
  ["genimage · mscocoai · wildrf · commforensics · ntire", "Image", "GenImage, MS-COCO AI, WildRF, Community Forensics, NTIRE", "Whole image"],
  ["dff · ffpp · ffpp_facecrop", "Image", "DiffusionFace, FaceForensics++", "Faces"],
  ["wavlm_large", "Voice", "ASVspoof 2019 LA", "Speech"],
  ["srm_classifier", "Noise", "CASIA v2.0", "Sensor noise"],
];

const STEPS = [
  { n: "01", title: "Capture", body: "Click Verify on any image or video while browsing, or drop a file into the console." },
  { n: "02", title: "Analyse", body: "The file is checked, queued and examined by every detector for its media type, plus the SRM noise stream." },
  { n: "03", title: "Verdict and report", body: "A colour-coded verdict, the score from each model, an artifact map and a one-click PDF report." },
];

const SOLUTIONS = [
  { title: "Journalists and fact-checkers", body: "Verify a viral clip or photo before it goes into a story, and keep the PDF report on file." },
  { title: "Everyday users", body: "Check a forwarded video or a “family member's” voice note before you believe it or send money." },
  { title: "Education and research", body: "See every model's score and the regions it focused on: a teaching tool for media literacy and forensics." },
  { title: "Legal and investigation teams", body: "Keep each case with its file hash, timeline and PDF report, ready to attach to a review or complaint." },
];

const EVIDENCE = [
  { title: "Every model's vote", body: "See how each detector scored the file and how the ensemble combined them." },
  { title: "Artifact maps", body: "A heat map over the image shows which regions drove a “synthetic” call." },
  { title: "Chain of custody", body: "Each report records the file's SHA-256 hash, the time and the analysis details." },
  { title: "Honest uncertainty", body: "When the models disagree, the verdict is “inconclusive”, never a guess dressed up as a fact." },
];

const WHY = [
  { eyebrow: "Multi-model", title: "Many models, not one", body: "Sixteen models trained on different datasets cover each other's blind spots." },
  { eyebrow: "Fast repeats", title: "Seen before? Instant.", body: "Files analysed before are answered from the cache without running the models again." },
  { eyebrow: "Explainable", title: "Show your working", body: "Scores, heat maps and a PDF report a reviewer can check." },
  { eyebrow: "Case history", title: "Nothing gets lost", body: "Every check is saved with its verdict, evidence and report, ready to revisit." },
];

const FAQ = [
  {
    q: "Is a “manipulated” verdict proof that a file is fake?",
    a: "No. It means the models found strong signs of manipulation. “Authentic” means no evidence was found, not that none exists. Treat the result as expert evidence to weigh, not a final ruling.",
  },
  {
    q: "Which formats can I check?",
    a: "Images (JPG, PNG, WEBP, BMP, TIFF), video (MP4, MOV, AVI, WEBM, MKV) and audio (MP3, WAV, FLAC, OGG, M4A), up to 500 MB per file.",
  },
  {
    q: "How long does a check take?",
    a: "It depends on the hardware the server runs on. A file that was analysed before is answered immediately from the cache.",
  },
  {
    q: "What happens to the files I upload?",
    a: "The file is kept with its case so the report can show the evidence, and it is removed when you delete the case.",
  },
  {
    q: "Does heavy compression affect the result?",
    a: "Yes. Media re-compressed by messaging apps loses detail the detectors rely on, so results on such files are less reliable.",
  },
];

const BADGES = [
  { label: "Verify", cls: "bg-cyan-100 text-cyan-800" },
  // `sky` is remapped to violet in globals.css: the processing colour.
  { label: "Analysing", cls: "bg-sky-100 text-sky-700" },
  { label: "Authentic", cls: "bg-emerald-100 text-emerald-700" },
  { label: "Inconclusive", cls: "bg-amber-100 text-amber-700" },
  { label: "Manipulated", cls: "bg-red-100 text-red-700" },
];

/* ── Small building blocks ───────────────────────────────────────────────── */

function Eyebrow({ children, dark }: { children: React.ReactNode; dark?: boolean }) {
  return (
    <p className={cn("font-mono text-[11px] uppercase tracking-[0.16em]", dark ? "text-cyan-400" : "text-paper-accent")}>
      {children}
    </p>
  );
}

function SectionHead({ eyebrow, title, accent, body }: { eyebrow: string; title: string; accent: string; body: string }) {
  return (
    <div className="mb-12 grid items-end gap-6 lg:grid-cols-2 lg:gap-12">
      <div>
        <Eyebrow>{eyebrow}</Eyebrow>
        <h2 className="mt-3 text-3xl font-bold leading-tight tracking-tight text-paper-ink sm:text-4xl">
          {title} <span className="text-paper-accent">{accent}</span>
        </h2>
      </div>
      <p className="text-base leading-relaxed text-paper-mute">{body}</p>
    </div>
  );
}

function Pill({ label, cls }: { label: string; cls: string }) {
  return (
    <span className={cn("inline-block rounded-full px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.1em]", cls)}>
      {label}
    </span>
  );
}

/* Small dark illustrations at the top of each detector card. */
function DetectorVisual({ kind }: { kind: "video" | "image" | "audio" }) {
  if (kind === "video") {
    return (
      <div className="relative flex h-full w-full items-center justify-center">
        <div className="flex h-24 w-36 items-center justify-center border border-dashed border-cyan-400/60 text-slate-300">▶</div>
        <div className="absolute bottom-4 flex gap-1">
          {[0, 0, 1, 1, 0, 0].map((hot, i) => (
            <span key={i} className={cn("h-3.5 w-5 border", hot ? "border-red-500 bg-red-500/25" : "border-slate-700 bg-slate-800")} />
          ))}
        </div>
      </div>
    );
  }
  if (kind === "image") {
    return (
      <div className="reticle relative h-28 w-28 bg-gradient-to-br from-[#1f3346] to-[#123f52]">
        <span className="absolute left-[30%] top-[28%] h-[46%] w-[46%] rounded-full bg-[radial-gradient(circle,rgba(239,68,68,.6),transparent_70%)]" />
      </div>
    );
  }
  const bars = [14, 28, 20, 44, 56, 38, 50, 24, 32, 18, 42, 30, 16, 26, 12];
  return (
    <div className="flex h-16 items-center gap-[3px]">
      {bars.map((h, i) => (
        <span key={i} className={cn("w-[3px] rounded-sm", i >= 3 && i <= 6 ? "bg-cyan-400" : i === 10 || i === 11 ? "bg-cyan-400" : "bg-slate-600")} style={{ height: h }} />
      ))}
    </div>
  );
}

/* ── Page ────────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-paper text-paper-ink">
      <PublicHeader />

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <header className="relative overflow-hidden bg-slate-950 text-slate-100">
        <div className="lp-hero-grid pointer-events-none absolute inset-0" />
        <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-4 py-20 sm:px-6 lg:grid-cols-[1.05fr_.95fr] lg:px-8 lg:py-24">
          <div>
            <Eyebrow dark>● Active deepfake defense · v1.0</Eyebrow>
            <h1 className="mt-5 text-5xl font-bold leading-[1.04] tracking-tight text-white sm:text-6xl">
              Know what&apos;s real
              <br />
              before you <span className="text-cyan-400">share it.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-300">
              Deep Truth checks images, video and voice for AI generation and face manipulation, right from your
              browser. Sixteen trained models across video, image, voice and sensor noise examine every file, and you
              get a clear verdict with the evidence behind it.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/forensics">
                <Button size="lg" className="bg-indigo-600 text-white hover:bg-indigo-500">
                  Analyse a file <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <a href="#extension">
                <Button size="lg" variant="outline" className="border-slate-700 bg-transparent text-slate-100 hover:bg-slate-900 hover:text-white">
                  Add the browser extension
                </Button>
              </a>
            </div>
            <p className="mt-7 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-slate-500">
              <span className="text-slate-300">JPG · PNG · WEBP</span>
              <span className="text-slate-300">MP4 · MOV · WEBM</span>
              <span className="text-slate-300">MP3 · WAV · FLAC</span>
            </p>
          </div>

          {/* Example verdict card — illustrative, labelled as such */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-[0_30px_80px_rgba(0,0,0,.45)]">
            <div className="mb-3 flex justify-between font-mono text-[11px] text-slate-500">
              <span>Example result · image</span>
              <span>8 detectors</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="lp-frame lp-real">
                <div className="lp-shoulders" />
                <div className="lp-face" />
                <div className="lp-box"><span>REAL · 8%</span></div>
              </div>
              <div className="lp-frame lp-fake">
                <div className="lp-shoulders" />
                <div className="lp-face" />
                <div className="lp-heat" />
                <div className="lp-box"><span>FAKE · 96%</span></div>
              </div>
            </div>
            <div className="mt-4 space-y-2">
              {[
                ["ntire", 0.99, true],
                ["commforensics", 0.94, true],
                ["ffpp_facecrop", 0.32, false],
              ].map(([name, v, hot]) => (
                <div key={name as string} className="grid grid-cols-[120px_1fr_40px] items-center gap-3 font-mono text-xs text-slate-300">
                  {name as string}
                  <div className="h-1.5 overflow-hidden rounded bg-slate-800">
                    <div className={cn("h-full", hot ? "bg-red-500" : "bg-cyan-400")} style={{ width: `${(v as number) * 100}%` }} />
                  </div>
                  <span className="text-right">{(v as number).toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center justify-between rounded-lg border border-red-600/35 bg-red-600/10 px-4 py-3">
              <span className="font-mono text-xs tracking-[0.1em] text-red-300">VERDICT · MANIPULATED</span>
              <span className="font-mono text-2xl font-semibold text-white">96%</span>
            </div>
          </div>
        </div>

        <div className="relative border-t border-slate-800">
          <div className="mx-auto grid max-w-7xl grid-cols-2 px-4 sm:px-6 lg:grid-cols-4 lg:px-8">
            {STATS.map((s) => (
              <div key={s.label} className="border-slate-800 py-6 pr-4 lg:border-r lg:pl-6 lg:first:pl-0 lg:last:border-r-0">
                <p className="font-mono text-3xl font-semibold text-white">{s.value}</p>
                <p className="mt-1 text-sm text-slate-400">{s.label}</p>
              </div>
            ))}
          </div>
        </div>
      </header>

      <main>
        {/* ── Detectors ───────────────────────────────────────────────────── */}
        <section id="detectors" className="scroll-mt-16 border-b border-paper-line py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHead
              eyebrow="Detectors"
              title="One tool for every kind of"
              accent="synthetic media."
              body="Each media type has its own specialised detector group, trained on public research datasets. They run independently, so a convincing face can't hide a cloned voice."
            />
            <div className="grid gap-5 md:grid-cols-3">
              {DETECTORS.map((d) => (
                <div key={d.key} className="overflow-hidden rounded-xl border border-paper-line bg-white">
                  <div className="flex h-44 items-center justify-center bg-slate-950">
                    <DetectorVisual kind={d.key} />
                  </div>
                  <div className="p-6">
                    <Eyebrow>{d.eyebrow}</Eyebrow>
                    <h3 className="mt-3 text-lg font-semibold tracking-tight">{d.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-paper-mute">{d.body}</p>
                    <ul className="mt-4 space-y-1 font-mono text-xs text-gray-700">
                      {d.points.map((p) => (
                        <li key={p}><span className="text-paper-accent">› </span>{p}</li>
                      ))}
                    </ul>
                    <Link href="/forensics" className="mt-5 inline-flex items-center gap-1 text-sm font-semibold text-paper-accent hover:underline">
                      {d.cta} <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── AI models ───────────────────────────────────────────────────── */}
        <section id="models" className="scroll-mt-16 border-b border-paper-line bg-white py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHead
              eyebrow="AI models"
              title="Sixteen models, four streams,"
              accent="one verdict."
              body="Every model is a pretrained transformer fine-tuned with LoRA adapters on public deepfake datasets. Each media type has its own group, and their scores are fused into a single verdict."
            />
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {ARCHITECTURES.map((a) => (
                <div key={a.name} className="rounded-xl border border-paper-line bg-paper p-5">
                  <Eyebrow>{a.eyebrow}</Eyebrow>
                  <h3 className="mb-4 mt-2.5 font-mono text-xl font-semibold">{a.name}</h3>
                  <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
                    {a.rows.map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="text-paper-mute">{k}</dt>
                        <dd className="text-right font-mono">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
            </div>

            <div className="my-5 grid gap-4 rounded-xl bg-slate-950 p-5 text-sm leading-relaxed text-slate-300 sm:grid-cols-[120px_1fr]">
              <Eyebrow dark>Fusion</Eyebrow>
              <p>
                <b className="text-white">Any specialist alarm counts.</b> When a face is clearly visible, the face models
                and the whole-frame model each get a say and the strongest warning wins. With no reliable face, the
                whole-frame models lead. When the detectors disagree strongly, the verdict is{" "}
                <b className="text-white">inconclusive</b> instead of a guess.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-sm">
                <thead>
                  <tr>
                    {["Checkpoint", "Stream", "Trained on", "Looks at"].map((h) => (
                      <th key={h} className="border-b border-paper-line px-3 py-2.5 text-left font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-paper-mute">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CHECKPOINTS.map((row) => (
                    <tr key={row[0]}>
                      {row.map((cell, i) => (
                        <td key={i} className={cn("border-b border-paper-line px-3 py-2.5", i === 0 && "font-mono text-cyan-900")}>
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── How it works ────────────────────────────────────────────────── */}
        <section id="how" className="scroll-mt-16 border-b border-paper-line py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHead
              eyebrow="How it works"
              title="Three steps from"
              accent="doubt to evidence."
              body="No account set-up, no settings to tune. Send a file and read the verdict. The heavy work runs in the background, so the page never freezes."
            />
            <div className="grid overflow-hidden rounded-xl border border-paper-line bg-white md:grid-cols-4">
              {STEPS.map((s) => (
                <div key={s.n} className="border-b border-paper-line p-6 md:border-b-0 md:border-r">
                  <Eyebrow>Step {s.n}</Eyebrow>
                  <h3 className="mb-2 mt-3 font-semibold">{s.title}</h3>
                  <p className="text-sm leading-relaxed text-paper-mute">{s.body}</p>
                </div>
              ))}
              <div className="bg-slate-950 p-6 text-slate-100">
                <Eyebrow dark>Try it now</Eyebrow>
                <h3 className="mb-2 mt-3 font-semibold">See it on your own file.</h3>
                <p className="text-sm leading-relaxed text-slate-300">Open the console and drop in any image, video or voice note.</p>
                <div className="mt-4 flex flex-col gap-2">
                  <Link href="/forensics">
                    <Button className="w-full bg-white text-slate-950 hover:bg-slate-200">Analyse a file →</Button>
                  </Link>
                  <a href="#extension">
                    <Button variant="outline" className="w-full border-slate-700 bg-transparent text-slate-100 hover:bg-slate-900 hover:text-white">
                      Get the extension
                    </Button>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Solutions ───────────────────────────────────────────────────── */}
        <section id="solutions" className="scroll-mt-16 border-b border-paper-line py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHead
              eyebrow="Solutions"
              title="Built for people who"
              accent="can't afford to be fooled."
              body="Deepfakes are cheap to make and fast to spread. Deep Truth puts a forensic check one click away for the people who see them first."
            />
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {SOLUTIONS.map((u, i) => (
                <div key={u.title} className="rounded-xl border border-paper-line bg-white p-6">
                  <span className="grid h-9 w-9 place-items-center rounded-lg bg-cyan-50 font-mono text-sm font-semibold text-paper-accent">
                    0{i + 1}
                  </span>
                  <h3 className="mb-1.5 mt-4 font-semibold">{u.title}</h3>
                  <p className="text-sm leading-relaxed text-paper-mute">{u.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Evidence ────────────────────────────────────────────────────── */}
        <section className="border-b border-paper-line bg-white py-24">
          <div className="mx-auto grid max-w-7xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-2 lg:px-8">
            <div>
              <Eyebrow>Explainable by design</Eyebrow>
              <h2 className="mt-3 text-3xl font-bold leading-tight tracking-tight sm:text-4xl">
                Evidence, <span className="text-paper-accent">not just a score.</span>
              </h2>
              <ul className="mt-8 space-y-5">
                {EVIDENCE.map((e, i) => (
                  <li key={e.title} className="grid grid-cols-[30px_1fr] gap-3">
                    <span className="grid h-7 place-items-center rounded-md border border-cyan-200 font-mono text-xs font-semibold text-paper-accent">
                      0{i + 1}
                    </span>
                    <div>
                      <p className="font-semibold">{e.title}</p>
                      <p className="mt-0.5 text-sm leading-relaxed text-paper-mute">{e.body}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-xl border border-paper-line bg-white p-6 shadow-[0_20px_50px_rgba(17,24,39,.08)]">
              <div className="flex justify-between border-b border-paper-line pb-3 font-mono text-[11px] text-paper-mute">
                <span>FORENSIC REPORT · EXAMPLE</span>
                <span>PDF ↓</span>
              </div>
              <p className="mb-1 mt-4 text-sm text-paper-mute">Verdict</p>
              <div className="flex items-center justify-between">
                <Pill label="Manipulated" cls="bg-red-100 text-red-700" />
                <span className="font-mono text-2xl font-semibold">96.3%</span>
              </div>
              <table className="mt-3 w-full font-mono text-xs">
                <tbody>
                  {[
                    ["genimage", "0.91"],
                    ["ntire", "0.99"],
                    ["commforensics", "0.94"],
                    ["ffpp_facecrop", "0.32"],
                    ["SRM noise (secondary)", "info"],
                    ["SHA-256", "9f2c…a71e"],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <td className="border-b border-dashed border-paper-line py-2">{k}</td>
                      <td className="border-b border-dashed border-paper-line py-2 text-right">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── Extension ───────────────────────────────────────────────────── */}
        <section id="extension" className="scroll-mt-16 bg-slate-950 py-24 text-slate-100">
          <div className="mx-auto grid max-w-7xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-2 lg:px-8">
            <div>
              <Eyebrow dark>Browser extension</Eyebrow>
              <h2 className="mb-4 mt-3 text-3xl font-bold leading-tight tracking-tight text-white sm:text-4xl">
                Verify media without leaving the page.
              </h2>
              <p className="leading-relaxed text-slate-300">
                The Deep Truth extension adds a Verify badge to images and videos on the sites you already use. One click
                sends the media for analysis, and the badge changes colour with the verdict.
              </p>
              <div className="mt-6 flex flex-wrap gap-2">
                {BADGES.map((b) => (
                  <Pill key={b.label} {...b} />
                ))}
              </div>
              <ol className="mt-6 space-y-2 font-mono text-[13px] text-slate-300">
                {[
                  "Open chrome://extensions and turn on Developer mode",
                  "Load unpacked → choose the extension folder",
                  "Set the server to your Deep Truth address",
                ].map((t, i) => (
                  <li key={t}>
                    <span className="text-cyan-400">0{i + 1}</span> {t}
                  </li>
                ))}
              </ol>
            </div>
            <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
              <div className="flex h-9 items-center gap-1.5 border-b border-slate-800 bg-[#0d1422] px-3">
                {[0, 1, 2].map((i) => (
                  <span key={i} className="h-2.5 w-2.5 rounded-full bg-slate-700" />
                ))}
                <span className="ml-3 rounded bg-slate-900 px-2.5 py-0.5 font-mono text-[11px] text-slate-500">social-feed.example/post/…</span>
              </div>
              <div className="space-y-3.5 p-4">
                <div className="relative h-36 rounded-lg bg-gradient-to-br from-[#1f3346] to-[#0e1a28]">
                  <span className="absolute right-2.5 top-2.5"><Pill label="Manipulated · 96%" cls="bg-red-100 text-red-700" /></span>
                </div>
                <div className="relative h-24 rounded-lg bg-gradient-to-br from-[#1f3346] to-[#0e1a28]">
                  <span className="absolute right-2.5 top-2.5"><Pill label="Verify" cls="bg-cyan-100 text-cyan-800" /></span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Why + FAQ ───────────────────────────────────────────────────── */}
        <section id="faq" className="scroll-mt-16 py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="grid gap-6 border-t border-paper-line sm:grid-cols-2 lg:grid-cols-4">
              {WHY.map((w) => (
                <div key={w.title} className="pt-6">
                  <Eyebrow>{w.eyebrow}</Eyebrow>
                  <p className="mb-1.5 mt-2.5 font-semibold">{w.title}</p>
                  <p className="text-sm leading-relaxed text-paper-mute">{w.body}</p>
                </div>
              ))}
            </div>

            <div className="mt-24 grid gap-10 lg:grid-cols-[1fr_1.4fr] lg:gap-14">
              <div>
                <Eyebrow>FAQ</Eyebrow>
                <h2 className="mt-3 text-3xl font-bold leading-tight tracking-tight sm:text-4xl">
                  Questions people <span className="text-paper-accent">usually ask.</span>
                </h2>
              </div>
              <div>
                {FAQ.map((f, i) => (
                  <details key={f.q} open={i === 0} className="group border-b border-paper-line py-5">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-semibold">
                      {f.q}
                      <Plus className="h-4 w-4 shrink-0 text-paper-accent transition-transform group-open:rotate-45" />
                    </summary>
                    <p className="mt-2.5 text-sm leading-relaxed text-paper-mute">{f.a}</p>
                  </details>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── Final call to action ────────────────────────────────────────── */}
        <section className="bg-slate-950 py-24 text-center text-slate-100">
          <div className="mx-auto max-w-3xl px-4">
            <Eyebrow dark>Don&apos;t share it until you&apos;ve checked it</Eyebrow>
            <h2 className="mb-3 mt-4 text-4xl font-bold tracking-tight text-white sm:text-5xl">Check a file in one click.</h2>
            <p className="mb-8 text-slate-300">Upload an image, video or voice note and get a verdict with the evidence behind it.</p>
            <div className="flex flex-wrap justify-center gap-3">
              <Link href="/forensics">
                <Button size="lg" className="bg-indigo-600 text-white hover:bg-indigo-500">
                  Analyse a file <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/dashboard">
                <Button size="lg" variant="outline" className="border-slate-700 bg-transparent text-slate-100 hover:bg-slate-900 hover:text-white">
                  Open console
                </Button>
              </Link>
            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="bg-[#060b14] py-12 text-sm text-slate-400">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-[2fr_1fr_1fr_1fr]">
            <div>
              <p className="font-semibold tracking-wide text-white">DEEP TRUTH</p>
              <p className="mt-3 max-w-xs leading-relaxed">
                Active deepfake defense system. Final Year Project, BS Artificial Intelligence, COMSATS University
                Islamabad.
              </p>
            </div>
            <FooterCol
              title="Product"
              links={[
                ["Analyse a file", "/forensics"],
                ["Dashboard", "/dashboard"],
                ["Case history", "/cases"],
                ["Analytics", "/analytics"],
              ]}
            />
            <FooterCol
              title="Resources"
              links={[
                ["AI models", "#models"],
                ["Browser extension", "#extension"],
                ["Documentation", "/docs"],
                ["System status", "/admin"],
              ]}
            />
            <div>
              <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-200">Project</p>
              <ul className="space-y-2">
                <li>Ramish Naseer</li>
                <li>Saad Mehmood</li>
                <li>Supervisor: Dr. Muhammad Imran</li>
              </ul>
            </div>
          </div>
          <div className="mt-10 flex flex-wrap justify-between gap-2 border-t border-slate-800 pt-5 font-mono text-[11px] text-slate-500">
            <span>© 2026 Deep Truth · COMSATS University Islamabad</span>
            <span>Results are probabilistic. Not legal proof.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FooterCol({ title, links }: { title: string; links: [string, string][] }) {
  return (
    <div>
      <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-200">{title}</p>
      <ul className="space-y-2">
        {links.map(([label, href]) => (
          <li key={label}>
            {href.startsWith("#") ? (
              <a href={href} className="hover:text-white">{label}</a>
            ) : (
              <Link href={href} className="hover:text-white">{label}</Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
