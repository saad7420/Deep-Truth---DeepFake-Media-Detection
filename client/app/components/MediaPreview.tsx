"use client";

/* ============================================================================
   DEEP TRUTH — EVIDENCE PREVIEW
   ----------------------------------------------------------------------------
   Renders the stored evidence file straight from the backend's /uploads mount.
   While the pipeline is still running, a scanline sweeps the frame — the one
   piece of motion in the console, reserved for the moment analysis is actually
   happening so it reads as machine state rather than decoration.
   ========================================================================= */

import { useState } from "react";
import { FileAudio, FileVideo, ImageOff, Image as ImageIcon } from "lucide-react";

import { resolveMediaUrl } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import type { MediaType } from "@/app/shared/schema";

export function MediaPreview({
  fileUrl,
  mediaType,
  fileName,
  analysing = false,
  className,
}: {
  fileUrl?: string | null;
  mediaType: MediaType;
  fileName?: string | null;
  analysing?: boolean;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const src = resolveMediaUrl(fileUrl);

  const Fallback = { image: ImageIcon, video: FileVideo, audio: FileAudio }[mediaType];

  if (!src || failed) {
    return (
      <div
        className={cn(
          "flex aspect-video flex-col items-center justify-center gap-2 rounded-xl border border-slate-800 bg-slate-950",
          className,
        )}
      >
        {failed ? (
          <ImageOff className="h-8 w-8 text-slate-700" />
        ) : (
          <Fallback className="h-8 w-8 text-slate-700" />
        )}
        <p className="px-4 text-center text-xs text-slate-600">
          {failed
            ? "The evidence file couldn't be loaded from the server."
            : "No evidence file is attached to this case."}
        </p>
      </div>
    );
  }

  return (
    <figure
      className={cn(
        "relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950",
        analysing && "scanline",
        className,
      )}
    >
      {mediaType === "video" && (
        <video
          src={src}
          controls
          preload="metadata"
          playsInline
          onError={() => setFailed(true)}
          className="aspect-video w-full bg-black object-contain"
        />
      )}

      {mediaType === "image" && (
        // Evidence is served from an arbitrary backend origin, so next/image
        // optimisation is deliberately bypassed here.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={fileName ? `Evidence: ${fileName}` : "Evidence under analysis"}
          onError={() => setFailed(true)}
          className="aspect-video w-full bg-black object-contain"
        />
      )}

      {mediaType === "audio" && (
        <div className="flex aspect-video flex-col items-center justify-center gap-5 bg-gradient-to-b from-slate-900 to-slate-950 px-6">
          <WaveformMark active={analysing} />
          <audio
            src={src}
            controls
            preload="metadata"
            onError={() => setFailed(true)}
            className="w-full max-w-md"
          />
        </div>
      )}

      {analysing && (
        <figcaption className="absolute left-3 top-3 inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-slate-950/80 px-2.5 py-1 backdrop-blur-sm">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-sky-300">
            Analysing
          </span>
        </figcaption>
      )}
    </figure>
  );
}

/* Static bar mark standing in for the waveform — the backend returns no
   per-sample amplitude data, so this signals "audio" without implying it is
   a real rendering of this clip. */
function WaveformMark({ active }: { active: boolean }) {
  const bars = [30, 55, 80, 45, 95, 60, 35, 70, 50, 85, 40, 65, 30, 75, 45];

  return (
    <div className="flex h-20 items-center gap-[3px]" aria-hidden>
      {bars.map((h, i) => (
        <span
          key={i}
          className={cn(
            "w-1.5 rounded-full transition-all duration-500",
            active ? "bg-sky-500/50" : "bg-slate-700",
          )}
          style={{
            height: `${h}%`,
            animation: active ? `pulse 1.4s ease-in-out ${i * 80}ms infinite` : undefined,
          }}
        />
      ))}
    </div>
  );
}
