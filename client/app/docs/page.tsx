"use client";

/* ============================================================================
   DEEP TRUTH — DOCUMENTATION
   ----------------------------------------------------------------------------
   Written for the person operating the console, not for whoever wrote the
   backend: what a verdict means, what the limits are, and what the API does if
   you want to drive it directly.
   ========================================================================= */

import Link from "next/link";
import { ArrowUpRight, BookOpen, GitBranch, Terminal, TriangleAlert } from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { PageHeader, Panel, PanelHeading, SectionRule } from "@/app/components/Primitives";
import { API_BASE, API_ORIGIN, MAX_FILE_MB } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { RISK_BANDS, VERDICT, type CaseStatus } from "@/app/shared/schema";

const ENDPOINTS = [
  { method: "GET", path: "/cases", note: "List cases. Supports page, page_size, media_type, status, q." },
  { method: "POST", path: "/cases", note: "Open a case. multipart/form-data: title, media_type, file." },
  { method: "GET", path: "/cases/{case_id}", note: "One case with all analysis rows." },
  { method: "PATCH", path: "/cases/{case_id}", note: "Update title, notes, status or scores." },
  { method: "DELETE", path: "/cases/{case_id}", note: "Delete the case and its evidence file." },
  { method: "GET", path: "/stats", note: "Aggregate counts and mean risk score." },
  { method: "GET", path: "/health", note: "Service and database reachability." },
];

const PIPELINE = [
  { id: "01", title: "Validate", body: "MIME type and size are checked before anything is written to disk." },
  { id: "02", title: "Store", body: "The file is saved under a generated name and the case row is created as processing." },
  { id: "03", title: "De-multiplex", body: "Video is split into its audio track and a sampled set of keyframes rather than every frame." },
  { id: "04", title: "Analyse", body: "Each applicable engine runs off the request thread: visual, audio, noise-residual." },
  { id: "05", title: "Fuse", body: "Scores are combined by confidence-weighted average into one trust score." },
  { id: "06", title: "Settle", body: "The case status flips to its verdict and the console stops polling." },
];

export default function DocsPage() {
  return (
    <AppShell width="default">
      <PageHeader
        eyebrow="Reference"
        title="Documentation"
        description="How the detection pipeline reaches a verdict, and what that verdict does and doesn't tell you."
        actions={
          <a href={`${API_ORIGIN}/api/docs`} target="_blank" rel="noreferrer">
            <button className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-sm font-medium text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white">
              Interactive API docs
              <ArrowUpRight className="h-3.5 w-3.5" />
            </button>
          </a>
        }
      />

      {/* ── Reading a verdict ─────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading icon={BookOpen} title="Reading a verdict" hint="What each status means in practice" />
        <div className="divide-y divide-slate-800/70">
          {(Object.keys(VERDICT) as CaseStatus[]).map((k) => (
            <div key={k} className="flex gap-4 px-5 py-4">
              <span
                className={cn(
                  "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                  VERDICT[k].bg.replace("/10", ""),
                )}
                style={{ backgroundColor: VERDICT[k].hex }}
              />
              <div className="min-w-0">
                <p className={cn("text-sm font-semibold", VERDICT[k].text)}>{VERDICT[k].label}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-slate-400">{VERDICT[k].summary}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="border-t border-slate-800 bg-slate-950/60 px-5 py-4">
          <p className="text-xs leading-relaxed text-slate-500">
            A case is <span className="text-emerald-400">authentic</span> at or below{" "}
            <span className="tabular text-white">{RISK_BANDS.authentic}%</span> risk and{" "}
            <span className="text-red-400">manipulated</span> at or above{" "}
            <span className="tabular text-white">{RISK_BANDS.manipulated}%</span>. Between those it is
            reported as inconclusive rather than pushed to the nearer edge.
          </p>
        </div>
      </Panel>

      {/* ── Limits — stated plainly, because a forensic tool that oversells
             itself is the failure mode that matters most here ────────────── */}
      <Panel className="border-amber-500/20 p-5">
        <div className="flex gap-4">
          <TriangleAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
          <div className="min-w-0">
            <h2 className="font-display text-sm font-semibold text-white">What this tool can&apos;t tell you</h2>
            <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-400">
              <li>
                <span className="text-slate-300">A score is evidence, not proof.</span> The models
                are statistical. A high score means the media resembles synthetic content the models
                were trained on — it is not a determination that the media is fake.
              </li>
              <li>
                <span className="text-slate-300">Heavy compression degrades accuracy.</span> Platforms
                that re-encode aggressively can blur the high-frequency artefacts detection relies on.
              </li>
              <li>
                <span className="text-slate-300">An engine that returns no confidence is skipped.</span>{" "}
                If the audio engine contributed nothing, the verdict says nothing about the audio —
                check the modality breakdown before concluding a track is clean.
              </li>
              <li>
                <span className="text-slate-300">Files above {MAX_FILE_MB} MB are rejected outright</span>{" "}
                before any analysis begins.
              </li>
            </ul>
          </div>
        </div>
      </Panel>

      {/* ── Pipeline ──────────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          icon={GitBranch}
          title="Pipeline"
          hint="What happens between upload and verdict, in order"
        />
        <ol className="divide-y divide-slate-800/70">
          {PIPELINE.map((step) => (
            <li key={step.id} className="flex gap-4 px-5 py-4">
              <span className="tabular mt-0.5 shrink-0 text-xs font-semibold text-indigo-400">
                {step.id}
              </span>
              <div className="min-w-0">
                <p className="text-sm font-medium text-white">{step.title}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-slate-500">{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </Panel>

      {/* ── API ───────────────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading icon={Terminal} title="API" hint={API_BASE} />
        <div className="divide-y divide-slate-800/70">
          {ENDPOINTS.map((e) => (
            <div key={`${e.method}${e.path}`} className="flex flex-col gap-1 px-5 py-3.5 sm:flex-row sm:items-baseline sm:gap-4">
              <div className="flex shrink-0 items-baseline gap-2.5">
                <span
                  className={cn(
                    "w-14 shrink-0 rounded px-1.5 py-0.5 text-center font-mono text-[10px] font-bold",
                    e.method === "GET" && "bg-sky-500/10 text-sky-400",
                    e.method === "POST" && "bg-emerald-500/10 text-emerald-400",
                    e.method === "PATCH" && "bg-amber-500/10 text-amber-400",
                    e.method === "DELETE" && "bg-red-500/10 text-red-400",
                  )}
                >
                  {e.method}
                </span>
                <code className="font-mono text-xs text-slate-300">{e.path}</code>
              </div>
              <p className="text-xs leading-relaxed text-slate-500 sm:ml-auto sm:text-right">{e.note}</p>
            </div>
          ))}
        </div>
      </Panel>

      {/* ── Example ───────────────────────────────────────────────────────── */}
      <Panel className="p-5">
        <SectionRule label="Open a case from the command line" />
        <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-4 font-mono text-[11px] leading-relaxed text-slate-300">
{`curl -X POST ${API_BASE}/cases \\
  -F "title=Senator interview clip" \\
  -F "media_type=video" \\
  -F "file=@evidence.mp4"

# Returns a case with status "processing".
# Poll until it settles:
curl ${API_BASE}/cases/CASE-XXXXXXXX`}
        </pre>
        <p className="mt-3 text-[11px] leading-relaxed text-slate-600">
          <code className="font-mono text-slate-400">media_type</code> must be one of image, video or
          audio, and must match the file you attach.
        </p>
      </Panel>

      <div className="flex justify-center pt-2">
        <Link href="/forensics">
          <button className="glow-brand rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500">
            Run your first scan
          </button>
        </Link>
      </div>
    </AppShell>
  );
}
