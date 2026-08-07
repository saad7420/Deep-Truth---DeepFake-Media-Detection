"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertOctagon, Home, RefreshCw } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[Deep Truth] Unhandled error:", error);
  }, [error]);

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 px-4">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-40" />

      <div className="animate-fade-up relative max-w-md text-center">
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl border border-red-500/20 bg-red-500/10">
          <AlertOctagon className="h-8 w-8 text-red-400" />
        </div>

        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-slate-600">
          Console fault
        </p>
        <h1 className="font-display mt-2 text-3xl font-bold tracking-tight text-white">
          This screen stopped working
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          Reloading usually clears it. If it keeps happening, check that the analysis server is
          running and reachable.
        </p>

        {error.digest && (
          <p className="mt-4 break-all rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 font-mono text-[10px] text-slate-500">
            Reference: {error.digest}
          </p>
        )}

        <div className="mt-8 flex flex-col justify-center gap-2 sm:flex-row">
          <button
            onClick={reset}
            className="glow-brand inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-indigo-600 px-5 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
          >
            <RefreshCw className="h-4 w-4" />
            Reload this screen
          </button>
          <Link href="/dashboard">
            <button className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-5 text-sm font-medium text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white sm:w-auto">
              <Home className="h-4 w-4" />
              Command Center
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
