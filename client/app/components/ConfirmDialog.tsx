"use client";

import { AlertTriangle, Loader2 } from "lucide-react";
import { useEffect } from "react";

import { Button } from "@/app/components/ui/button";
import { cn } from "@/app/lib/utils";

/**
 * Confirmation for irreversible actions. The confirm button repeats the verb
 * from the title ("Delete case", not "OK") so the operator reads what will
 * happen at the moment they commit to it.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  loading = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
    >
      <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={onCancel} />

      <div className="animate-fade-up relative w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
        <div className="flex gap-4">
          <div
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border",
              destructive
                ? "border-red-500/20 bg-red-500/10"
                : "border-indigo-500/20 bg-indigo-500/10",
            )}
          >
            <AlertTriangle
              className={cn("h-5 w-5", destructive ? "text-red-400" : "text-indigo-400")}
            />
          </div>

          <div className="min-w-0 flex-1">
            <h2 id="confirm-title" className="font-display text-base font-semibold text-white">
              {title}
            </h2>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{description}</p>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <Button
            variant="ghost"
            onClick={onCancel}
            disabled={loading}
            className="text-slate-400 hover:text-white"
          >
            {cancelLabel}
          </Button>
          <Button
            autoFocus
            onClick={onConfirm}
            disabled={loading}
            className={cn(
              "text-white",
              destructive ? "bg-red-600 hover:bg-red-500" : "bg-indigo-600 hover:bg-indigo-500",
            )}
          >
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
