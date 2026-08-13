"use client";

/* ============================================================================
   DEEP TRUTH — ARTIFACT MAP (M7 FE-3)
   ----------------------------------------------------------------------------
   Shows where the image ensemble found its evidence, as a heat overlay on the
   224x224 tensor the models actually saw.

   The hard part here is not the picture, it is not overclaiming with it.
   Grad-CAM always returns a map; whether that map is a statement about a
   *region* depends on how concentrated it is. For a spliced face the relevance
   piles into a few cells and "here is the manipulated area" is fair. For a
   fully synthesised image the giveaway is texture everywhere, the map comes
   back near-uniform, and the same sentence would be a fabrication.

   So `localised` drives the wording, and the caveat is always visible rather
   than tucked behind a tooltip. A forensic tool that lets a diffuse map read
   as a located finding is worse than one that shows no map at all.
   ========================================================================= */

import { useState } from "react";
import { Layers, ScanSearch } from "lucide-react";

import { Panel, PanelHeading } from "@/app/components/Primitives";
import { cn } from "@/app/lib/utils";
import type { ArtifactMap as ArtifactMapData } from "@/app/shared/schema";

export function ArtifactMapPanel({
  map,
  originalUrl,
  className,
}: {
  map?: ArtifactMapData | null;
  originalUrl?: string;
  className?: string;
}) {
  const [showOverlay, setShowOverlay] = useState(true);

  if (!map?.url) return null;

  const localised = map.localised === true;
  const pct =
    typeof map.concentration === "number" ? Math.round(map.concentration * 100) : null;

  const profile = map.temporal_profile ?? [];
  const isVideo = profile.length > 0;
  const peak = map.peak_segment ?? 0;
  const timeLocalised = map.temporally_localised === true;

  return (
    <Panel className={cn("overflow-hidden", className)}>
      <PanelHeading
        icon={ScanSearch}
        title="Artifact map"
        hint={
          isVideo
            ? `Per-segment heat over the ${map.branch === "face" ? "face crops" : "sampled frames"}`
            : "Regions that drove the verdict"
        }
        action={
          // A clip's map is a contact sheet of many frames, so there is no
          // single "original" to toggle back to.
          originalUrl && !isVideo ? (
            <button
              type="button"
              onClick={() => setShowOverlay((v) => !v)}
              className="rounded-md border border-slate-700 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white"
            >
              {showOverlay ? "Show original" : "Show map"}
            </button>
          ) : undefined
        }
      />

      <div className="p-4">
        <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={showOverlay || !originalUrl ? map.url : originalUrl}
            alt={
              showOverlay
                ? "Heat map of the regions that contributed to the verdict"
                : "The analysed image without the overlay"
            }
            className="w-full"
          />
        </div>

        {/* ── When, for a clip ────────────────────────────────────────── */}
        {isVideo && (
          <div className="mt-3">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                Relevance over time
              </span>
              <span className="font-mono text-[10px] text-slate-500">
                {profile.length} segments
              </span>
            </div>

            {/* Bars share one scale, so a flat clip looks flat. Normalising
                each bar to its own height would make every clip look like it
                had a decisive moment. */}
            <div className="flex h-12 items-end gap-0.5">
              {profile.map((share, i) => (
                <div
                  key={i}
                  title={`Segment ${i + 1}: ${Math.round(share * 100)}% of total relevance`}
                  className={cn(
                    "flex-1 rounded-t-sm transition-colors",
                    i === peak && timeLocalised
                      ? "bg-amber-400"
                      : i === peak
                        ? "bg-slate-500"
                        : "bg-slate-700",
                  )}
                  style={{
                    height: `${Math.max(3, (share / Math.max(...profile, 0.0001)) * 100)}%`,
                  }}
                />
              ))}
            </div>

            <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
              {timeLocalised ? (
                <>
                  <span className="font-semibold text-amber-300">
                    Segment {peak + 1} of {profile.length}
                  </span>{" "}
                  carries {Math.round((profile[peak] ?? 0) * 100)}% of the total
                  relevance — the evidence is concentrated in one part of the
                  clip rather than spread through it.
                </>
              ) : (
                <>
                  Relevance is spread fairly evenly across the clip (strongest
                  segment {Math.round((profile[peak] ?? 0) * 100)}%, an even
                  share would be {Math.round((1 / profile.length) * 100)}%), so
                  there is no single moment to point at. Note that temporal
                  discrimination is the weakest part of this method and has not
                  yet been validated against a clip with a known localised
                  edit — read the spatial heat, not the timing.
                </>
              )}
            </p>
          </div>
        )}

        {/* ── What this map does and does not say ─────────────────────── */}
        <div
          className={cn(
            "mt-3 rounded-lg border px-3 py-2.5 text-[11px] leading-relaxed",
            localised
              ? "border-amber-500/25 bg-amber-500/5 text-amber-200/90"
              : "border-slate-800 bg-slate-900/50 text-slate-400",
          )}
        >
          {localised ? (
            <>
              <span className="font-semibold text-amber-300">
                Evidence is concentrated.
              </span>{" "}
              The highlighted regions carry most of what drove this verdict
              {pct !== null && <> — the top tenth of the frame holds {pct}% of the
              total relevance</>}
              . Warmer colour means a stronger contribution.
            </>
          ) : (
            <>
              <span className="font-semibold text-slate-300">
                Evidence is distributed, not localised.
              </span>{" "}
              Relevance is spread across the frame
              {pct !== null && <> (the strongest tenth holds only {pct}%)</>}, which
              is what wholly synthetic {isVideo ? "footage" : "imagery"} looks
              like: the signal is texture and frequency across the whole
              {isVideo ? " picture" : " picture"} rather than one edited region.
              Read this as attention, not as a marked-up area.
            </>
          )}
        </div>

        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[11px]">
          <Field label="Method" value={map.method ?? "grad-cam"} />
          <Field label="Resolution" value={`${map.grid ?? 14}×${map.grid ?? 14} cells`} />
          <Field
            label="Contributing models"
            value={String(map.contributors?.length ?? 0)}
          />
          <Field
            label="Concentration"
            value={pct !== null ? `${pct}%` : "—"}
          />
        </dl>

        {map.contributors?.length > 0 && (
          <p className="mt-2 flex items-start gap-1.5 text-[10px] leading-relaxed text-slate-600">
            <Layers className="mt-0.5 h-3 w-3 shrink-0" />
            <span>
              Fused from {map.contributors.join(", ")} — each weighted by how
              strongly it called the image synthetic. Checkpoints that found
              nothing contribute nothing.
            </span>
          </p>
        )}
      </div>
    </Panel>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="mt-0.5 truncate font-mono text-slate-300">{value}</dd>
    </div>
  );
}
