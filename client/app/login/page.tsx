"use client";

/* ============================================================================
   DEEP TRUTH — SIGN IN
   ----------------------------------------------------------------------------
   The analysis service has no auth layer, and this page says so plainly rather
   than staging a password field that validates nothing. What it collects is
   attribution: a name and email stamped onto the cases you open, so an
   investigation can be traced back to a person.
   ========================================================================= */

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, Info, Shield } from "lucide-react";

import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { useAuth } from "@/app/hooks/use-auth";
import { cn } from "@/app/lib/utils";
import type { Operator } from "@/app/hooks/use-auth";

const ROLES: { id: Operator["role"]; label: string; blurb: string; clearance: string }[] = [
  { id: "analyst", label: "Analyst", blurb: "Run scans and read reports", clearance: "CLASS-B" },
  { id: "investigator", label: "Investigator", blurb: "Full case archive and exports", clearance: "CLASS-A" },
  { id: "admin", label: "Administrator", blurb: "Plus system configuration", clearance: "CLASS-S" },
];

export default function LoginPage() {
  const router = useRouter();
  const { signIn } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Operator["role"]>("investigator");
  const [error, setError] = useState<string | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();

    if (!name.trim()) return setError("Enter the name that should appear on your cases.");
    if (!email.trim() || !email.includes("@")) return setError("Enter a valid email address.");

    signIn({
      name: name.trim(),
      email: email.trim(),
      role,
      clearance: ROLES.find((r) => r.id === role)!.clearance,
    });
    router.push("/dashboard");
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 px-4 py-12">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-40" />
      <div className="aurora pointer-events-none absolute inset-0" />

      <div className="animate-fade-up relative w-full max-w-md">
        <Link
          href="/"
          className="mb-8 inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-slate-500 transition-colors hover:text-white"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to overview
        </Link>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-7 backdrop-blur-xl">
          <div className="mb-7 flex items-center gap-3">
            <div className="glow-brand flex h-11 w-11 items-center justify-center rounded-xl border border-indigo-500/40 bg-indigo-600/20">
              <Shield className="h-6 w-6 text-indigo-400" />
            </div>
            <div>
              <h1 className="font-display text-lg font-bold tracking-wide text-white">
                Access the console
              </h1>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500">
                Deep Truth defense system
              </p>
            </div>
          </div>

          <form onSubmit={submit} className="space-y-5">
            <div className="space-y-1.5">
              <label htmlFor="name" className="eyebrow block">
                Your name
              </label>
              <Input
                id="name"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setError(null);
                }}
                placeholder="Ramish Naseer"
                autoComplete="name"
                className="h-11 border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="email" className="eyebrow block">
                Email
              </label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError(null);
                }}
                placeholder="analyst@example.com"
                autoComplete="email"
                className="h-11 border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
              />
            </div>

            <fieldset className="space-y-2">
              <legend className="eyebrow mb-2">Role</legend>
              <div className="space-y-2">
                {ROLES.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => setRole(r.id)}
                    aria-pressed={role === r.id}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-lg border px-3.5 py-2.5 text-left transition-colors",
                      role === r.id
                        ? "border-indigo-500/40 bg-indigo-600/15"
                        : "border-slate-800 bg-slate-950 hover:border-slate-700",
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2 transition-colors",
                        role === r.id ? "border-indigo-500" : "border-slate-700",
                      )}
                    >
                      {role === r.id && <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-medium text-white">{r.label}</span>
                      <span className="block text-xs text-slate-500">{r.blurb}</span>
                    </span>
                    <span className="shrink-0 font-mono text-[10px] tracking-wider text-slate-600">
                      {r.clearance}
                    </span>
                  </button>
                ))}
              </div>
            </fieldset>

            {error && (
              <p className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                {error}
              </p>
            )}

            <Button
              type="submit"
              className="glow-brand h-11 w-full bg-indigo-600 text-white hover:bg-indigo-500"
            >
              Enter console
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </form>

          <div className="mt-6 flex gap-2.5 rounded-lg border border-slate-800 bg-slate-950 p-3">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-500" />
            <p className="text-[11px] leading-relaxed text-slate-500">
              There&apos;s no password because the analysis service doesn&apos;t authenticate. These
              details are kept in this browser and attached to the cases you open, so a report can
              be traced back to whoever ran it.
            </p>
          </div>
        </div>

        <p className="mt-5 text-center text-xs text-slate-600">
          Just want to try it?{" "}
          <Link href="/forensics" className="text-indigo-400 underline-offset-4 hover:underline">
            Skip and analyse a file
          </Link>
        </p>
      </div>
    </div>
  );
}
