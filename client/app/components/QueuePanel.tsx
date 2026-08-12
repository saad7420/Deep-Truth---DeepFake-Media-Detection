"use client";

/* ============================================================================
   DEEP TRUTH — QUEUE HEALTH
   ----------------------------------------------------------------------------
   System-wide view of the orchestration layer: how deep the backlog is, how
   many workers are alive, and how much work the cache is saving.

   The failure this panel is really for is the quiet one. Without workers, the
   API still accepts uploads and every case sits at "processing" indefinitely,
   which is indistinguishable from slow analysis unless something says so.
   ========================================================================= */

import { AlertTriangle, Cpu, Database, Layers, Zap } from "lucide-react";

import { Panel, PanelHeading } from "@/app/components/Primitives";
import { useQueueOverview } from "@/app/hooks/use-cases";
import { cn } from "@/app/lib/utils";

export function QueuePanel({ className }: { className?: string }) {
  const { data, isError, isPending } = useQueueOverview();

  // Three distinct states, and conflating them is actively misleading: before
  // the first response resolves we know nothing yet, which is not the same as
  // knowing the API is down. Reporting "unreachable" during normal startup
  // would send an operator to debug a backend that is fine.
  if (isPending) {
    return (
      <Panel className={cn("overflow-hidden", className)}>
        <PanelHeading icon={Layers} title="Analysis queue" />
        <div className="px-5 py-4 text-xs text-slate-500">Reading queue state…</div>
      </Panel>
    );
  }

  if (isError || !data) {
    return (
      <Panel className={cn("overflow-hidden", className)}>
        <PanelHeading icon={Layers} title="Analysis queue" />
        <div className="px-5 py-4 text-xs text-slate-500">
          Queue status unavailable — the API is not reachable.
        </div>
      </Panel>
    );
  }

  const { workers, cache } = data;
  const capacity = workers.concurrency || 0;
  const busy = Math.min(workers.active, capacity || workers.active);

  return (
    <Panel className={cn("overflow-hidden", className)}>
      <PanelHeading
        icon={Layers}
        title="Analysis queue"
        hint={
          data.redisOnline
            ? `${workers.online} worker${workers.online === 1 ? "" : "s"} · ${capacity} parallel slot${capacity === 1 ? "" : "s"}`
            : "Broker offline"
        }
      />

      {/* Operator-facing warnings come from the server so the threshold logic
          lives in one place rather than being re-derived in the client. */}
      {(!data.redisOnline || data.message) && (
        <div
          className={cn(
            "flex items-start gap-2 border-b px-5 py-3 text-xs",
            data.redisOnline
              ? "border-slate-800 bg-amber-500/5 text-amber-300"
              : "border-slate-800 bg-red-500/5 text-red-300",
          )}
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span className="leading-relaxed">
            {data.message ??
              "Redis is unreachable. New uploads are rejected until it returns."}
          </span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-px bg-slate-800 sm:grid-cols-4">
        <Stat
          icon={Layers}
          label="Waiting"
          value={String(data.pendingDepth)}
          tone={data.pendingDepth > 0 ? "warn" : "idle"}
        />
        <Stat
          icon={Cpu}
          label="Running"
          value={capacity ? `${busy}/${capacity}` : String(workers.active)}
          tone={workers.active > 0 ? "active" : "idle"}
        />
        <Stat
          icon={Database}
          label="Cached"
          value={String(cache.entries)}
          hint={`${cache.hits} hit${cache.hits === 1 ? "" : "s"}`}
        />
        <Stat
          icon={Zap}
          label="Hit rate"
          value={`${Math.round(cache.hitRate * 100)}%`}
          tone={cache.hitRate > 0 ? "good" : "idle"}
          hint="analyses skipped"
        />
      </div>

      {data.activeJobs.length > 0 && (
        <div className="border-t border-slate-800 px-5 py-3">
          <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
            In flight
          </p>
          <ul className="mt-2 space-y-1">
            {data.activeJobs.map((job) => (
              <li
                key={job.taskId ?? `${job.worker}-${job.caseDbId}`}
                className="flex items-center justify-between gap-3 font-mono text-[11px] text-slate-400"
              >
                <span className="truncate">
                  {job.caseDbId ? job.caseDbId.slice(0, 8) : "unknown"}
                  <span className="ml-2 uppercase text-slate-600">{job.mediaType}</span>
                </span>
                <span className="shrink-0 truncate text-slate-600">{job.worker}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  hint,
  tone = "idle",
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
  hint?: string;
  tone?: "idle" | "active" | "warn" | "good";
}) {
  const toneClass = {
    idle: "text-slate-400",
    active: "text-sky-300",
    warn: "text-amber-300",
    good: "text-emerald-300",
  }[tone];

  return (
    <div className="bg-slate-900/60 px-4 py-3.5">
      <div className="flex items-center gap-1.5">
        <Icon className="h-3 w-3 text-slate-600" />
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
          {label}
        </span>
      </div>
      <p className={cn("mt-1 font-mono text-lg font-bold tabular-nums", toneClass)}>{value}</p>
      {hint && <p className="mt-0.5 truncate text-[10px] text-slate-600">{hint}</p>}
    </div>
  );
}
