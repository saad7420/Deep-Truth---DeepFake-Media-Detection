"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, FolderSearch, Plus, Search, X } from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { CaseTable } from "@/app/components/CaseTable";
import { ConfirmDialog } from "@/app/components/ConfirmDialog";
import { EmptyState, PageHeader, Panel, PanelHeading } from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { useCases, useDeleteCase } from "@/app/hooks/use-cases";
import { cn } from "@/app/lib/utils";
import {
  CaseStatusSchema,
  MediaTypeSchema,
  type CaseStatus,
  type MediaType,
} from "@/app/shared/schema";

const PAGE_SIZE = 15;

const STATUS_FILTERS: { value: CaseStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "authentic", label: "Authentic" },
  { value: "manipulated", label: "Manipulated" },
  { value: "inconclusive", label: "Inconclusive" },
  { value: "processing", label: "Analysing" },
  { value: "failed", label: "Failed" },
];

const MEDIA_FILTERS: { value: MediaType | "all"; label: string }[] = [
  { value: "all", label: "All media" },
  { value: "video", label: "Video" },
  { value: "image", label: "Image" },
  { value: "audio", label: "Audio" },
];

function CaseHistory() {
  const router = useRouter();
  const params = useSearchParams();

  /* Filters live in the URL so a filtered view can be linked and shared —
     the Command Center's stat tiles deep-link straight into them. */
  const statusParam = params.get("status");
  const mediaParam = params.get("media");
  const status = CaseStatusSchema.safeParse(statusParam).success ? (statusParam as CaseStatus) : undefined;
  const mediaType = MediaTypeSchema.safeParse(mediaParam).success ? (mediaParam as MediaType) : undefined;

  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState(params.get("q") ?? "");
  const [query, setQuery] = useState(params.get("q") ?? "");
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  const deleteCase = useDeleteCase();

  // Debounce typing so a search doesn't fire a request per keystroke.
  useEffect(() => {
    const t = setTimeout(() => {
      setQuery(searchInput);
      setPage(1);
    }, 350);
    return () => clearTimeout(t);
  }, [searchInput]);

  useEffect(() => setPage(1), [status, mediaType]);

  const { data, isLoading, isFetching, isError, error, refetch } = useCases({
    page,
    pageSize: PAGE_SIZE,
    status,
    mediaType,
    q: query || undefined,
  });

  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const cases = useMemo(() => data?.cases ?? [], [data]);
  const hasFilters = Boolean(status || mediaType || query);

  function setFilter(key: "status" | "media", value: string) {
    const next = new URLSearchParams(params.toString());
    if (value === "all") next.delete(key);
    else next.set(key, value);
    router.replace(`/cases${next.toString() ? `?${next}` : ""}`, { scroll: false });
  }

  function clearFilters() {
    setSearchInput("");
    router.replace("/cases", { scroll: false });
  }

  const rangeStart = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const rangeEnd = Math.min(page * PAGE_SIZE, total);

  return (
    <AppShell width="wide">
      <PageHeader
        eyebrow="Archive"
        title="Case History"
        description="Every investigation opened against the detection pipeline, with its evidence file and final verdict."
        actions={
          <Link href="/forensics">
            <Button className="glow-brand bg-indigo-600 text-white hover:bg-indigo-500">
              <Plus className="mr-2 h-4 w-4" />
              New scan
            </Button>
          </Link>
        }
      />

      {/* ── Controls ──────────────────────────────────────────────────────── */}
      <Panel className="p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <Input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search case titles"
              aria-label="Search case titles"
              className="h-10 border-slate-800 bg-slate-950 pl-9 pr-9 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
            />
            {searchInput && (
              <button
                onClick={() => setSearchInput("")}
                aria-label="Clear search"
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <FilterGroup
            label="Media"
            options={MEDIA_FILTERS}
            active={mediaType ?? "all"}
            onSelect={(v) => setFilter("media", v)}
          />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-slate-800 pt-4">
          <FilterGroup
            label="Verdict"
            options={STATUS_FILTERS}
            active={status ?? "all"}
            onSelect={(v) => setFilter("status", v)}
          />
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="ml-auto inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-slate-500 transition-colors hover:text-white"
            >
              <X className="h-3 w-3" />
              Clear filters
            </button>
          )}
        </div>
      </Panel>

      {/* ── Results ───────────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          title={total === 1 ? "1 case" : `${total} cases`}
          hint={total > 0 ? `Showing ${rangeStart}–${rangeEnd}` : undefined}
          action={
            isFetching && !isLoading ? (
              <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
                Refreshing
              </span>
            ) : undefined
          }
        />

        <CaseTable
          cases={cases}
          isLoading={isLoading}
          isError={isError}
          errorMessage={error?.message}
          onRetry={() => refetch()}
          onDelete={setPendingDelete}
          showSize
          empty={
            hasFilters ? (
              <EmptyState
                icon={Search}
                title="No cases match these filters"
                description="Try a different verdict or modality, or clear the filters to see the full archive."
                action={
                  <Button
                    variant="outline"
                    onClick={clearFilters}
                    className="border-slate-700 text-slate-300 hover:text-white"
                  >
                    Clear filters
                  </Button>
                }
              />
            ) : (
              <EmptyState
                icon={FolderSearch}
                title="The archive is empty"
                description="Cases appear here the moment you upload evidence to the Forensics Hub."
                action={
                  <Link href="/forensics">
                    <Button className="bg-indigo-600 text-white hover:bg-indigo-500">
                      <Plus className="mr-2 h-4 w-4" />
                      Upload evidence
                    </Button>
                  </Link>
                }
              />
            )
          }
        />

        {pageCount > 1 && (
          <div className="flex items-center justify-between border-t border-slate-800 px-5 py-3">
            <p className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
              Page {page} of {pageCount}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="border-slate-700 text-slate-300 disabled:opacity-40"
              >
                <ChevronLeft className="mr-1 h-4 w-4" />
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= pageCount}
                onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
                className="border-slate-700 text-slate-300 disabled:opacity-40"
              >
                Next
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </Panel>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete this case?"
        description={`${pendingDelete} and its stored evidence file will be removed from the server. This can't be undone.`}
        confirmLabel="Delete case"
        destructive
        loading={deleteCase.isPending}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) deleteCase.mutate(pendingDelete);
          setPendingDelete(null);
        }}
      />
    </AppShell>
  );
}

/* ── Filter pills ────────────────────────────────────────────────────────── */

function FilterGroup<T extends string>({
  label,
  options,
  active,
  onSelect,
}: {
  label: string;
  options: { value: T; label: string }[];
  active: string;
  onSelect: (value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="eyebrow mr-1">{label}</span>
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onSelect(o.value)}
          aria-pressed={active === o.value}
          className={cn(
            "rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors",
            active === o.value
              ? "border-indigo-500/40 bg-indigo-600/20 text-indigo-200"
              : "border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700 hover:text-white",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* useSearchParams needs a Suspense boundary during static prerender. */
export default function CasesPage() {
  return (
    <Suspense
      fallback={
        <AppShell width="wide">
          <div className="h-40 animate-pulse rounded-xl bg-slate-900/60" />
        </AppShell>
      }
    >
      <CaseHistory />
    </Suspense>
  );
}
