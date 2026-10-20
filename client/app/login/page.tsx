"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, Info, Shield } from "lucide-react";

import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { useAuth } from "@/app/hooks/use-auth";

export default function LoginPage() {
  const router = useRouter();
  const { signIn } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();

    if (!name.trim()) return setError("Enter the name that should appear on your cases.");
    if (!email.trim() || !email.includes("@")) return setError("Enter a valid email address.");

    signIn({
      name: name.trim(),
      email: email.trim(),
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
