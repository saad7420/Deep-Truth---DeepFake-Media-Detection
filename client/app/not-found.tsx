import Link from "next/link";
import { FileQuestion, Home, Search } from "lucide-react";

export default function NotFound() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 px-4">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-40" />
      <div className="aurora pointer-events-none absolute inset-0" />

      <div className="animate-fade-up relative max-w-md text-center">
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900">
          <FileQuestion className="h-8 w-8 text-slate-600" />
        </div>

        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-slate-600">Error 404</p>
        <h1 className="font-display mt-2 text-3xl font-bold tracking-tight text-white">
          Nothing at this address
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          The page you followed doesn&apos;t exist. If you were opening a case, check the case ID —
          it should look like <code className="font-mono text-slate-300">CASE-4A2F91C8</code>.
        </p>

        <div className="mt-8 flex flex-col justify-center gap-2 sm:flex-row">
          <Link href="/dashboard">
            <button className="glow-brand inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 sm:w-auto">
              <Home className="h-4 w-4" />
              Command Center
            </button>
          </Link>
          <Link href="/cases">
            <button className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-5 text-sm font-medium text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white sm:w-auto">
              <Search className="h-4 w-4" />
              Search cases
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
