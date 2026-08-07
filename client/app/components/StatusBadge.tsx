"use client";

import { cn } from "@/app/lib/utils";
import { VERDICT, type CaseStatus } from "@/app/shared/schema";
import { AlertTriangle, CheckCircle2, HelpCircle, Loader2, XCircle } from "lucide-react";

const ICON: Record<CaseStatus, typeof CheckCircle2> = {
  processing: Loader2,
  authentic: CheckCircle2,
  manipulated: AlertTriangle,
  inconclusive: HelpCircle,
  failed: XCircle,
};

export function StatusBadge({
  status,
  className,
  size = "md",
}: {
  status: string;
  className?: string;
  size?: "sm" | "md";
}) {
  const key = (status in VERDICT ? status : "processing") as CaseStatus;
  const meta = VERDICT[key];
  const Icon = ICON[key];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-mono font-medium uppercase tracking-wider",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-xs",
        meta.bg,
        meta.text,
        meta.border,
        className,
      )}
    >
      <Icon
        className={cn(size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5", key === "processing" && "animate-spin")}
      />
      {meta.label}
    </span>
  );
}
