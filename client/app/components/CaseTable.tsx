"use client";

/* ============================================================================
   DEEP TRUTH — CASE TABLE
   ----------------------------------------------------------------------------
   Shared by the Command Center and Case History so a case row looks and links
   identically in both. Rows link by the public `caseId` (CASE-XXXXXXXX), which
   is what GET /cases/{case_id} looks up — the internal UUID `id` is never a
   valid URL segment.

   Two renderings of the same data: stacked cards below `md`, the full table
   from `md` up. The table previously carried `min-w-[720px]` inside a
   horizontal scroller, which on a phone meant every column past "Subject"
   was off-screen and reachable only by sideways-scrolling a nested region.
   Delete is a always-visible control here rather than a hover-reveal, since
   `opacity-0 group-hover:opacity-100` never resolves on a touch device.
   ========================================================================= */

import Link from "next/link";
import { FileAudio, FileVideo, Image as ImageIcon, Trash2, type LucideIcon } from "lucide-react";

import { ErrorState, Spinner } from "@/app/components/Primitives";
import { StatusBadge } from "@/app/components/StatusBadge";
import { formatBytes } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";
import { riskTone, type Case, type MediaType } from "@/app/shared/schema";

export const MEDIA_ICON: Record<MediaType, LucideIcon> = {
  image: ImageIcon,
  video: FileVideo,
  audio: FileAudio,
};

function formatDate(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}

export function CaseTable({
  cases,
  isLoading,
  isError,
  errorMessage,
  onRetry,
  empty,
  onDelete,
  showSize = false,
}: {
  cases: Case[];
  isLoading?: boolean;
  isError?: boolean;
  errorMessage?: string;
  onRetry?: () => void;
  empty?: React.ReactNode;
  onDelete?: (caseId: string) => void;
  showSize?: boolean;
}) {
  if (isError) {
    return (
      <ErrorState
        title="Couldn't load cases"
        message={errorMessage ?? "The analysis server didn't respond."}
        onRetry={onRetry}
      />
    );
  }

  if (isLoading && cases.length === 0) return <SkeletonRows showSize={showSize} />;

  if (!isLoading && cases.length === 0) return <>{empty}</>;

  return (
    <>
      {/* ── Mobile: stacked cards ─────────────────────────────────────────── */}
      <ul className="divide-y divide-slate-800/70 md:hidden">
        {cases.map((c) => {
          const Icon = MEDIA_ICON[c.mediaType];
          const tone = riskTone(c.riskScore);

          return (
            <li key={c.id} className="px-4 py-3.5">
              <div className="flex items-start justify-between gap-3">
                <Link href={`/cases/${c.caseId}`} className="min-w-0 flex-1">
                  <p className="truncate font-medium text-white">{c.title}</p>
                  <p className="mt-0.5 truncate font-mono text-[10px] text-indigo-400">
                    {c.caseId}
                  </p>
                </Link>

                <div className="flex shrink-0 items-center gap-2">
                  {c.status === "processing" ? (
                    <span className="font-mono text-xs text-slate-600">pending</span>
                  ) : (
                    <span className={cn("tabular text-base font-bold", tone.text)}>
                      {c.riskScore.toFixed(0)}%
                    </span>
                  )}
                  {onDelete && (
                    <button
                      onClick={() => onDelete(c.caseId)}
                      aria-label={`Delete ${c.caseId}`}
                      className="rounded-md p-2 text-slate-500 transition-colors hover:bg-red-950/30 hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5">
                <StatusBadge status={c.status} size="sm" />
                <span className="inline-flex items-center gap-1.5 text-[11px] capitalize text-slate-400">
                  <Icon className="h-3 w-3 text-slate-500" />
                  {c.mediaType}
                </span>
                {showSize && (
                  <span className="tabular font-mono text-[10px] text-slate-600">
                    {formatBytes(c.fileSize)}
                  </span>
                )}
                <span className="tabular ml-auto font-mono text-[10px] text-slate-600">
                  {formatDate(c.createdAt)}
                </span>
              </div>
            </li>
          );
        })}
      </ul>

      {/* ── Desktop: full table ───────────────────────────────────────────── */}
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-800 font-mono text-[10px] uppercase tracking-[0.15em] text-slate-500">
              <th scope="col" className="px-5 py-3 font-medium">Case</th>
              <th scope="col" className="px-5 py-3 font-medium">Subject</th>
              <th scope="col" className="px-5 py-3 font-medium">Modality</th>
              {showSize && <th scope="col" className="px-5 py-3 font-medium">Size</th>}
              <th scope="col" className="px-5 py-3 font-medium">Opened</th>
              <th scope="col" className="px-5 py-3 font-medium">Verdict</th>
              <th scope="col" className="px-5 py-3 text-right font-medium">Risk</th>
              {onDelete && <th scope="col" className="px-5 py-3 text-right font-medium">Actions</th>}
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/70">
            {cases.map((c) => {
              const Icon = MEDIA_ICON[c.mediaType];
              const tone = riskTone(c.riskScore);

              return (
                <tr key={c.id} className="group transition-colors hover:bg-slate-800/30">
                  <td className="whitespace-nowrap px-5 py-3.5">
                    <Link
                      href={`/cases/${c.caseId}`}
                      className="font-mono text-xs font-medium text-indigo-400 hover:text-indigo-300 hover:underline"
                    >
                      {c.caseId}
                    </Link>
                  </td>

                  <td className="max-w-[260px] px-5 py-3.5">
                    <Link
                      href={`/cases/${c.caseId}`}
                      className="block truncate font-medium text-white hover:text-indigo-200"
                    >
                      {c.title}
                    </Link>
                    {c.fileName && (
                      <p className="truncate font-mono text-[10px] text-slate-600">{c.fileName}</p>
                    )}
                  </td>

                  <td className="whitespace-nowrap px-5 py-3.5">
                    <span className="inline-flex items-center gap-1.5 text-xs capitalize text-slate-400">
                      <Icon className="h-3.5 w-3.5 text-slate-500" />
                      {c.mediaType}
                    </span>
                  </td>

                  {showSize && (
                    <td className="tabular whitespace-nowrap px-5 py-3.5 text-xs text-slate-400">
                      {formatBytes(c.fileSize)}
                    </td>
                  )}

                  <td className="tabular whitespace-nowrap px-5 py-3.5 text-xs text-slate-400">
                    {formatDate(c.createdAt)}
                  </td>

                  <td className="whitespace-nowrap px-5 py-3.5">
                    <StatusBadge status={c.status} size="sm" />
                  </td>

                  <td className="whitespace-nowrap px-5 py-3.5 text-right">
                    {c.status === "processing" ? (
                      <span className="font-mono text-xs text-slate-600">pending</span>
                    ) : (
                      <span className={cn("tabular text-sm font-bold", tone.text)}>
                        {c.riskScore.toFixed(0)}%
                      </span>
                    )}
                  </td>

                  {onDelete && (
                    <td className="whitespace-nowrap px-5 py-3.5 text-right">
                      <button
                        onClick={() => onDelete(c.caseId)}
                        aria-label={`Delete ${c.caseId}`}
                        className="rounded-md p-1.5 text-slate-600 transition-all hover:bg-red-950/30 hover:text-red-400 focus-visible:opacity-100 md:opacity-0 md:group-hover:opacity-100"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {isLoading && cases.length > 0 && <Spinner label="Refreshing" className="py-4" />}
    </>
  );
}

/* ── Loading placeholder ─────────────────────────────────────────────────── */

function SkeletonRows({ showSize }: { showSize?: boolean }) {
  const widths = ["w-24", "w-40", "w-16", "w-14", "w-20", "w-24", "w-10"];
  const cols = showSize ? 7 : 6;

  return (
    <div className="divide-y divide-slate-800/70">
      {Array.from({ length: 5 }).map((_, row) => (
        <div key={row} className="flex items-center gap-6 px-5 py-4">
          {Array.from({ length: cols }).map((__, col) => (
            <div
              key={col}
              className={cn("h-3.5 animate-pulse rounded bg-slate-800", widths[col] ?? "w-16")}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
