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

  return (
    <Panel className={cn("overflow-hidden", className)}>
      <PanelHeading
        icon={ScanSearch}
        title="Artifact map"
        hint="Regions that drove the verdict"
        action={
          originalUrl ? (
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
              is what a wholly synthetic image looks like: the signal is texture
              and frequency across the whole picture rather than one edited
              region. Read this as attention, not as a marked-up area.
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
