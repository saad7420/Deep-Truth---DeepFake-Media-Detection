"use client";

/* ============================================================================
   DEEP TRUTH — SECURE VAULT
   ----------------------------------------------------------------------------
   Every evidence file the backend is currently holding on disk, with what it
   belongs to and how to remove it. The previous version of this page showed a
   hard-coded key-rotation log and an "immunize" button that called nothing —
   both removed, because a security screen that displays invented audit entries
   is worse than one that shows less.
   ========================================================================= */

import { useMemo, useState } from "react";
import Link from "next/link";
import { Database, Download, HardDrive, Lock, ShieldCheck, Trash2, Upload } from "lucide-react";

import AppShell from "@/app/components/AppShell";
import { MEDIA_ICON } from "@/app/components/CaseTable";
import { ConfirmDialog } from "@/app/components/ConfirmDialog";
import {
  EmptyState,
  ErrorState,
  PageHeader,
  Panel,
  PanelHeading,
  SectionRule,
  Spinner,
  StatTile,
} from "@/app/components/Primitives";
import { StatusBadge } from "@/app/components/StatusBadge";
import { Button } from "@/app/components/ui/button";
import { useCases, useDeleteCase } from "@/app/hooks/use-cases";
import { formatBytes, resolveMediaUrl } from "@/app/lib/api-client";

export default function VaultPage() {
  const { data, isLoading, isError, error, refetch } = useCases({ pageSize: 100 });
  const deleteCase = useDeleteCase();
  const [pending, setPending] = useState<string | null>(null);

  const stored = useMemo(
    () => (data?.cases ?? []).filter((c) => Boolean(c.fileUrl)),
    [data],
  );

  const totalBytes = stored.reduce((sum, c) => sum + (c.fileSize ?? 0), 0);
  const largest = stored.reduce((max, c) => Math.max(max, c.fileSize ?? 0), 0);

  return (
    <AppShell width="wide">
      <PageHeader
        eyebrow="Evidence"
        title="Secure Vault"
        description="Files held by the analysis server, one per open case. Deleting a case removes its file from disk immediately."
        actions={
          <Link href="/forensics">
            <Button className="glow-brand bg-indigo-600 text-white hover:bg-indigo-500">
              <Upload className="mr-2 h-4 w-4" />
              Add evidence
            </Button>
          </Link>
        }
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatTile
          label="Files held"
          value={stored.length}
          icon={Database}
          tone="brand"
          loading={isLoading}
          hint="One per case with an attached file"
        />
        <StatTile
          label="Storage used"
          value={formatBytes(totalBytes)}
          icon={HardDrive}
          tone="neutral"
          loading={isLoading}
          hint={largest ? `Largest single file ${formatBytes(largest)}` : undefined}
        />
        <StatTile
          label="Retention"
          value="Manual"
          icon={Lock}
          tone="warn"
          hint="Files persist until their case is deleted"
        />
      </div>

      {/* ── Handling policy — states what the server actually does ─────────── */}
      <Panel className="p-5">
        <SectionRule label="How evidence is handled" />
        <ul className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
          {[
            {
              icon: ShieldCheck,
              title: "Written once, never rewritten",
              body: "Each upload is stored under a generated name, so two files with the same name can't collide or overwrite each other.",
            },
            {
              icon: Lock,
              title: "Served only to this console",
              body: "Files are exposed on the server's /uploads mount, restricted to the origins listed in its CORS allow-list.",
            },
            {
              icon: Trash2,
              title: "Deleted with the case",
              body: "Removing a case unlinks its file in the same request. There is no separate retention window or archive copy.",
            },
          ].map((item) => (
            <li key={item.title} className="rounded-lg border border-slate-800 bg-slate-950 p-4">
              <item.icon className="mb-3 h-5 w-5 text-indigo-400" />
              <p className="text-sm font-medium text-white">{item.title}</p>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{item.body}</p>
            </li>
          ))}
        </ul>
      </Panel>

      {/* ── Stored files ──────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          icon={Database}
          title="Stored evidence"
          hint={stored.length > 0 ? `${stored.length} file${stored.length === 1 ? "" : "s"} on disk` : undefined}
        />

        {isError ? (
          <ErrorState message={error?.message ?? "Couldn't reach the server."} onRetry={() => refetch()} />
        ) : isLoading ? (
          <Spinner label="Reading vault" />
        ) : stored.length === 0 ? (
          <EmptyState
            icon={Database}
            title="The vault is empty"
            description="Evidence files appear here as soon as a case is opened against them."
            action={
              <Link href="/forensics">
                <Button className="bg-indigo-600 text-white hover:bg-indigo-500">
                  <Upload className="mr-2 h-4 w-4" />
                  Add evidence
                </Button>
              </Link>
            }
          />
        ) : (
          <ul className="divide-y divide-slate-800/70">
            {stored.map((c) => {
              const Icon = MEDIA_ICON[c.mediaType];
              const url = resolveMediaUrl(c.fileUrl);

              return (
                <li key={c.id} className="group flex items-center gap-4 px-5 py-3.5 transition-colors hover:bg-slate-800/30">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-800 bg-slate-950">
                    <Icon className="h-4 w-4 text-slate-500" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <Link
                      href={`/cases/${c.caseId}`}
                      className="block truncate text-sm font-medium text-white hover:text-indigo-200"
                    >
                      {c.fileName ?? c.title}
                    </Link>
                    <p className="truncate font-mono text-[11px] text-slate-600">
                      {c.caseId} · {formatBytes(c.fileSize)} · {c.mediaType}
                    </p>
                  </div>

                  <StatusBadge status={c.status} size="sm" className="hidden sm:inline-flex" />

                  <div className="flex shrink-0 gap-1">
                    {url && (
                      <a
                        href={url}
                        download={c.fileName ?? undefined}
                        aria-label={`Download ${c.fileName ?? c.title}`}
                        className="rounded-md p-1.5 text-slate-600 transition-colors hover:bg-slate-800 hover:text-white"
                      >
                        <Download className="h-4 w-4" />
                      </a>
                    )}
                    <button
                      onClick={() => setPending(c.caseId)}
                      aria-label={`Delete ${c.caseId}`}
                      className="rounded-md p-1.5 text-slate-600 transition-colors hover:bg-red-950/30 hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </Panel>

      <ConfirmDialog
        open={Boolean(pending)}
        title="Delete this evidence?"
        description={`${pending} and its stored file will be removed from the server. This can't be undone.`}
        confirmLabel="Delete evidence"
        destructive
        loading={deleteCase.isPending}
        onCancel={() => setPending(null)}
        onConfirm={() => {
          if (pending) deleteCase.mutate(pending);
          setPending(null);
        }}
      />
    </AppShell>
  );
}
