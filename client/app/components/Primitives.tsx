"use client";

/* ============================================================================
   DEEP TRUTH — SHARED UI PRIMITIVES
   Small, unopinionated pieces reused across every console page so headers,
   stat tiles, empty states and error states stay identical everywhere.
   ========================================================================= */

import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { cn } from "@/app/lib/utils";

/* ── Eyebrow ─────────────────────────────────────────────────────────────── */

export function Eyebrow({ children, className }: { children: React.ReactNode; className?: string }) {
  return <p className={cn("eyebrow", className)}>{children}</p>;
}

/* ── Page header ─────────────────────────────────────────────────────────── */

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  status,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: React.ReactNode;
  /** Optional live pill, e.g. the system-online indicator. */
  status?: React.ReactNode;
}) {
  return (
    <header className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {status}
        {eyebrow && <Eyebrow className="mb-2">{eyebrow}</Eyebrow>}
        <h1 className="font-display text-3xl font-bold tracking-tight text-white lg:text-4xl">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

/* ── Live status pill ────────────────────────────────────────────────────── */

export function LivePill({
  label,
  tone = "brand",
}: {
  label: string;
  tone?: "brand" | "good" | "bad";
}) {
  const tones = {
    brand: "bg-indigo-500/10 border-indigo-500/20 text-indigo-300",
    good: "bg-emerald-500/10 border-emerald-500/20 text-emerald-300",
    bad: "bg-red-500/10 border-red-500/20 text-red-300",
  } as const;

  const dot = {
    brand: "bg-indigo-400",
    good: "bg-emerald-400",
    bad: "bg-red-400",
  } as const;

  return (
    <span
      className={cn(
        "mb-3 inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[10px] font-medium uppercase tracking-[0.18em]",
        tones[tone],
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", dot[tone], tone !== "bad" && "animate-pulse")} />
      {label}
    </span>
  );
}

/* ── Panel ───────────────────────────────────────────────────────────────── */

export function Panel({
  children,
  className,
  hover = false,
  as: Tag = "div",
}: {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  as?: React.ElementType;
}) {
  return (
    <Tag
      className={cn(
        "rounded-xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm print-plain",
        hover && "panel-hover",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function PanelHeading({
  icon: Icon,
  title,
  hint,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-800 px-5 py-4">
      <div className="flex min-w-0 items-center gap-2.5">
        {Icon && <Icon className="h-4 w-4 shrink-0 text-slate-500" />}
        <div className="min-w-0">
          <h2 className="font-display text-sm font-semibold tracking-wide text-white">{title}</h2>
          {hint && <p className="mt-0.5 truncate text-xs text-slate-500">{hint}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

/* ── Stat tile ───────────────────────────────────────────────────────────── */

export function StatTile({
  label,
  value,
  icon: Icon,
  tone = "brand",
  hint,
  loading,
  href,
}: {
  label: string;
  value: string | number;
  icon: LucideIcon;
  tone?: "brand" | "good" | "bad" | "warn" | "neutral";
  hint?: string;
  loading?: boolean;
  href?: string;
}) {
  const tones = {
    brand: { text: "text-indigo-400", bg: "bg-indigo-500/10", border: "border-indigo-500/20" },
    good: { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
    bad: { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" },
    warn: { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
    neutral: { text: "text-slate-300", bg: "bg-slate-500/10", border: "border-slate-500/20" },
  } as const;

  const t = tones[tone];

  const body = (
    <Panel hover={Boolean(href)} className="group relative overflow-hidden p-5">
      <Icon
        className={cn(
          "pointer-events-none absolute -right-4 -top-4 h-24 w-24 opacity-[0.06] transition-opacity group-hover:opacity-[0.12]",
          t.text,
        )}
      />
      <div className="relative">
        <div
          className={cn(
            "mb-4 flex h-10 w-10 items-center justify-center rounded-lg border",
            t.bg,
            t.border,
          )}
        >
          <Icon className={cn("h-5 w-5", t.text)} />
        </div>
        {loading ? (
          <div className="mb-1 h-8 w-16 animate-pulse rounded bg-slate-800" />
        ) : (
          <p className="tabular mb-1 text-3xl font-bold leading-none text-white">{value}</p>
        )}
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
        {hint && <p className="mt-1.5 text-[11px] text-slate-600">{hint}</p>}
      </div>
    </Panel>
  );

  return href ? <Link href={href}>{body}</Link> : body;
}

/* ── Empty state ─────────────────────────────────────────────────────────────
   An empty screen is an invitation to act, so every one of these takes an
   action rather than just explaining that nothing is here.
   ------------------------------------------------------------------------- */

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900">
        <Icon className="h-7 w-7 text-slate-600" />
      </div>
      <h3 className="font-display text-lg font-semibold text-white">{title}</h3>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-slate-500">{description}</p>
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

/* ── Error state ─────────────────────────────────────────────────────────────
   Errors say what happened and what to do about it. No apologies, no
   "something went wrong".
   ------------------------------------------------------------------------- */

export function ErrorState({
  title = "Couldn't load this",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-red-500/20 bg-red-500/10">
        <AlertTriangle className="h-7 w-7 text-red-400" />
      </div>
      <h3 className="font-display text-lg font-semibold text-white">{title}</h3>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-400">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-6 inline-flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white"
        >
          <RefreshCw className="h-4 w-4" />
          Try again
        </button>
      )}
    </div>
  );
}

/* ── Inline spinner ──────────────────────────────────────────────────────── */

export function Spinner({ label, className }: { label?: string; className?: string }) {
  return (
    <div className={cn("flex items-center justify-center gap-2 py-12 text-slate-500", className)}>
      <Loader2 className="h-4 w-4 animate-spin" />
      {label && <span className="font-mono text-xs uppercase tracking-widest">{label}</span>}
    </div>
  );
}

/* ── Key/value row, used across every metadata panel ─────────────────────── */

export function MetaRow({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="shrink-0 font-mono text-[11px] uppercase tracking-wider text-slate-500">
        {label}
      </span>
      <span
        className={cn(
          "min-w-0 truncate text-right text-xs text-slate-300",
          mono && "font-mono tabular-nums",
        )}
      >
        {value}
      </span>
    </div>
  );
}

/* ── Section divider with label ──────────────────────────────────────────── */

export function SectionRule({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3">
      <Eyebrow>{label}</Eyebrow>
      <div className="h-px flex-1 bg-slate-800" />
    </div>
  );
}
