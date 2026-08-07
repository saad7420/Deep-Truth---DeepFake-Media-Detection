import { Shield } from "lucide-react";

export default function Loading() {
  return (
    <div className="relative flex min-h-screen items-center justify-center bg-slate-950">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-30" />

      <div className="relative flex flex-col items-center gap-4">
        <div className="animate-pulse-ring flex h-12 w-12 items-center justify-center rounded-xl border border-indigo-500/40 bg-indigo-600/20">
          <Shield className="h-6 w-6 text-indigo-400" />
        </div>
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-slate-600">
          Loading console
        </p>
      </div>
    </div>
  );
}
