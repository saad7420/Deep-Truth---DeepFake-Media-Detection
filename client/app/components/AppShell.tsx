"use client";

/* ============================================================================
   DEEP TRUTH — CONSOLE SHELL
   ----------------------------------------------------------------------------
   The single layout wrapper for every signed-in page. Previously each page
   rendered its own <Sidebar /> *and* was wrapped in AppShell, which mounted the
   rail twice and shifted the content 64px off. Pages now render content only.
   ========================================================================= */

import { cn } from "@/app/lib/utils";
import { MobileNav, Sidebar } from "@/app/components/Navigation";

export default function AppShell({
  children,
  /** Turn off the grid substrate on pages that carry their own artwork. */
  grid = true,
  /** Wider container for dense tables and dashboards. */
  width = "default",
}: {
  children: React.ReactNode;
  grid?: boolean;
  width?: "default" | "wide" | "narrow";
}) {
  const maxWidth = {
    narrow: "max-w-3xl",
    default: "max-w-7xl",
    wide: "max-w-[110rem]",
  }[width];

  return (
    <div className={cn("min-h-screen bg-slate-950", grid && "grid-bg")}>
      <Sidebar />
      <MobileNav />

      <div className="aurora lg:pl-64">
        <main
          className={cn(
            "mx-auto w-full px-4 pb-20 pt-24 sm:px-6 lg:px-10 lg:pt-12",
            maxWidth,
          )}
        >
          <div className="animate-fade-up space-y-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
