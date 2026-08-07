"use client";

/* ============================================================================
   DEEP TRUTH — SETTINGS
   ----------------------------------------------------------------------------
   Only settings the console can actually honour appear here. Anything that
   would need a backend endpoint the server doesn't expose (model selection,
   retention windows, API keys) lives on Configuration as read-only state
   instead of as a control that does nothing.
   ========================================================================= */

import { useEffect, useState } from "react";
import { Check, Loader2, Plug, RotateCcw, Save, User } from "lucide-react";

import AppShell from "@/app/components/AppShell";
import {
  MetaRow,
  PageHeader,
  Panel,
  PanelHeading,
  SectionRule,
} from "@/app/components/Primitives";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { useAuth } from "@/app/hooks/use-auth";
import { useHealth } from "@/app/hooks/use-cases";
import { useToast } from "@/app/hooks/use-toast";
import { API_BASE, API_ORIGIN, MAX_FILE_MB } from "@/app/lib/api-client";
import { cn } from "@/app/lib/utils";

const PREFS_KEY = "deeptruth.prefs";

interface Prefs {
  defaultModality: "video" | "image" | "audio";
  confirmBeforeDelete: boolean;
  reduceMotion: boolean;
}

const DEFAULT_PREFS: Prefs = {
  defaultModality: "video",
  confirmBeforeDelete: true,
  reduceMotion: false,
};

export default function SettingsPage() {
  const { operator, signIn } = useAuth();
  const { toast } = useToast();
  const { data: health, isError, refetch, isFetching } = useHealth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (operator) {
      setName(operator.name);
      setEmail(operator.email);
    }
    try {
      const raw = localStorage.getItem(PREFS_KEY);
      if (raw) setPrefs({ ...DEFAULT_PREFS, ...(JSON.parse(raw) as Prefs) });
    } catch {
      /* fall back to defaults */
    }
  }, [operator]);

  function saveProfile() {
    if (!name.trim() || !email.trim()) {
      toast({
        title: "Name and email are required",
        description: "Both are stamped on the cases you open.",
        variant: "destructive",
      });
      return;
    }
    signIn({ ...operator, name: name.trim(), email: email.trim() });
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    toast({ title: "Settings saved", description: "Your profile and preferences are up to date." });
  }

  function updatePref<K extends keyof Prefs>(key: K, value: Prefs[K]) {
    const next = { ...prefs, [key]: value };
    setPrefs(next);
    localStorage.setItem(PREFS_KEY, JSON.stringify(next));
  }

  function resetPrefs() {
    setPrefs(DEFAULT_PREFS);
    localStorage.setItem(PREFS_KEY, JSON.stringify(DEFAULT_PREFS));
    toast({ title: "Preferences reset", description: "Back to the console defaults." });
  }

  const online = !isError && health?.status === "ok";

  return (
    <AppShell width="default">
      <PageHeader
        eyebrow="Account"
        title="Settings"
        description="Your operator profile and how this console behaves."
      />

      {/* ── Profile ───────────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading
          icon={User}
          title="Operator profile"
          hint="Stamped on every case you open, so an investigation can be traced back to a person"
        />
        <div className="space-y-5 p-5">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field label="Display name" htmlFor="op-name">
              <Input
                id="op-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Ramish Naseer"
                className="h-10 border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
              />
            </Field>

            <Field label="Email" htmlFor="op-email">
              <Input
                id="op-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="analyst@example.com"
                className="h-10 border-slate-800 bg-slate-950 text-sm text-white placeholder:text-slate-600 focus-visible:ring-indigo-500"
              />
            </Field>
          </div>

          <div className="divide-y divide-slate-800/70 rounded-lg border border-slate-800 bg-slate-950 px-4 py-2">
            <MetaRow label="Operator ID" value={operator?.id ?? "—"} />
            <MetaRow label="Role" value={<span className="capitalize">{operator?.role ?? "—"}</span>} />
            <MetaRow label="Clearance" value={operator?.clearance ?? "—"} />
          </div>

          <p className="text-[11px] leading-relaxed text-slate-600">
            The analysis service has no accounts of its own, so this profile is stored in your
            browser and sent along with each upload as an attribution field.
          </p>

          <div className="flex justify-end">
            <Button
              onClick={saveProfile}
              className="glow-brand bg-indigo-600 text-white hover:bg-indigo-500"
            >
              {saved ? <Check className="mr-2 h-4 w-4" /> : <Save className="mr-2 h-4 w-4" />}
              {saved ? "Saved" : "Save changes"}
            </Button>
          </div>
        </div>
      </Panel>

      {/* ── Preferences ───────────────────────────────────────────────────── */}
      <Panel className="overflow-hidden">
        <PanelHeading title="Console preferences" hint="Applied on this device only" />
        <div className="divide-y divide-slate-800/70">
          <Row
            title="Default modality"
            description="Which analysis type the Forensics Hub starts on before you pick a file."
          >
            <div className="flex gap-1.5">
              {(["video", "image", "audio"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => updatePref("defaultModality", m)}
                  aria-pressed={prefs.defaultModality === m}
                  className={cn(
                    "rounded-lg border px-3 py-1.5 text-xs font-medium capitalize transition-colors",
                    prefs.defaultModality === m
                      ? "border-indigo-500/40 bg-indigo-600/20 text-indigo-200"
                      : "border-slate-800 bg-slate-950 text-slate-400 hover:text-white",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
          </Row>

          <Row
            title="Confirm before deleting"
            description="Ask for confirmation before a case and its evidence file are removed."
          >
            <Toggle
              checked={prefs.confirmBeforeDelete}
              onChange={(v) => updatePref("confirmBeforeDelete", v)}
              label="Confirm before deleting"
            />
          </Row>

          <Row
            title="Reduce motion"
            description="Turn off the scanline sweep and progress shimmer during analysis."
          >
            <Toggle
              checked={prefs.reduceMotion}
              onChange={(v) => updatePref("reduceMotion", v)}
              label="Reduce motion"
            />
          </Row>
        </div>

        <div className="flex justify-end border-t border-slate-800 px-5 py-3">
          <Button variant="ghost" size="sm" onClick={resetPrefs} className="text-slate-400 hover:text-white">
            <RotateCcw className="mr-2 h-3.5 w-3.5" />
            Reset to defaults
          </Button>
        </div>
      </Panel>

      {/* ── Connection ────────────────────────────────────────────────────── */}
      <Panel className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <SectionRule label="Analysis server" />
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
            className="ml-3 shrink-0 border-slate-700 text-slate-300 hover:text-white"
          >
            {isFetching ? (
              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Plug className="mr-2 h-3.5 w-3.5" />
            )}
            Test connection
          </Button>
        </div>

        <div className="divide-y divide-slate-800/70">
          <MetaRow
            label="Status"
            value={
              <span className={online ? "text-emerald-400" : "text-red-400"}>
                {online ? "Reachable" : "Not responding"}
              </span>
            }
          />
          <MetaRow label="Endpoint" value={API_BASE} />
          <MetaRow label="Evidence mount" value={`${API_ORIGIN}/uploads`} />
          <MetaRow label="Version" value={health?.version ?? "—"} />
          <MetaRow label="Upload ceiling" value={`${MAX_FILE_MB} MB`} />
        </div>
      </Panel>
    </AppShell>
  );
}

/* ── Local pieces ────────────────────────────────────────────────────────── */

function Field({
  label,
  htmlFor,
  children,
}: {
  label: string;
  htmlFor: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="eyebrow block">
        {label}
      </label>
      {children}
    </div>
  );
}

function Row({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 sm:pr-8">
        <p className="text-sm font-medium text-white">{title}</p>
        <p className="mt-0.5 text-xs leading-relaxed text-slate-500">{description}</p>
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full border transition-colors",
        checked ? "border-indigo-500/40 bg-indigo-600" : "border-slate-700 bg-slate-800",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 h-4.5 w-4.5 rounded-full bg-white transition-transform",
          checked ? "translate-x-[22px]" : "translate-x-0.5",
        )}
        style={{ height: 18, width: 18 }}
      />
    </button>
  );
}
