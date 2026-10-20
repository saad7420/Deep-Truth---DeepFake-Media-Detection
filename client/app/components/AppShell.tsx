"use client";

import { cn } from "@/app/lib/utils";
import { MobileNav, Sidebar } from "@/app/components/Navigation";

export default function AppShell({
  children,
  grid = true,
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
