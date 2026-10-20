"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Cpu,
  FileSearch,
  FolderSearch,
  LayoutDashboard,
  LogOut,
  Menu,
  Settings,
  Shield,
  X,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/app/components/ui/button";
import { useAuth } from "@/app/hooks/use-auth";
import { useHealth } from "@/app/hooks/use-cases";
import { cn } from "@/app/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  prefix?: boolean;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: "Workspace",
    items: [
      { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
      { href: "/forensics", label: "New analysis", icon: FileSearch },
      { href: "/cases", label: "Cases", icon: FolderSearch, prefix: true },
      { href: "/analytics", label: "Analytics", icon: BarChart3 },
    ],
  },
  {
    label: "System",
    items: [
      { href: "/#models", label: "AI models", icon: Cpu },
      { href: "/admin", label: "System status", icon: Activity },
      { href: "/docs", label: "Documentation", icon: BookOpen },
      { href: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

function isActive(pathname: string, item: NavItem) {
  return item.prefix ? pathname.startsWith(item.href) : pathname === item.href;
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-3">
      <div className="glow-brand flex h-9 w-9 items-center justify-center rounded-lg border border-indigo-500/40 bg-indigo-600/20">
        <Shield className="h-5 w-5 text-indigo-400" />
      </div>
      <div className="leading-tight">
        <p className="font-display text-base font-bold tracking-wide text-white">DEEP TRUTH</p>
        <p className="font-mono text-[10px] tracking-[0.18em] text-slate-500">DEFENSE SYSTEM</p>
      </div>
    </Link>
  );
}

// online only when the api reports at least one worker as well
function EngineStatus() {
  const { data, isError, isLoading } = useHealth();

  const online = !isError && data?.status === "ok";
  const dbOk = data?.db === "ok";

  const label = isLoading
    ? "Connecting"
    : !online
      ? "Engine offline"
      : dbOk
        ? "Engine online"
        : "Database degraded";

  const tone = isLoading
    ? "text-slate-500"
    : !online
      ? "text-red-400"
      : dbOk
        ? "text-emerald-400"
        : "text-amber-400";

  const dot = isLoading
    ? "bg-slate-600"
    : !online
      ? "bg-red-500"
      : dbOk
        ? "bg-emerald-500"
        : "bg-amber-500";

  return (
    <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
      <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dot, online && "animate-pulse")} />
      <span className={cn("font-mono text-[10px] uppercase tracking-wider", tone)}>{label}</span>
      {data?.version && (
        <span className="ml-auto font-mono text-[10px] text-slate-600">v{data.version}</span>
      )}
    </div>
  );
}

function NavList({ pathname, onNavigate }: { pathname: string; onNavigate?: () => void }) {
  return (
    <nav className="space-y-6">
      {NAV_GROUPS.map((group) => (
        <div key={group.label} className="space-y-1">
          <p className="eyebrow px-3 pb-1">{group.label}</p>
          {group.items.map((item) => {
            const active = isActive(pathname, item);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onNavigate}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-150",
                  active
                    ? "border border-indigo-500/20 bg-indigo-950/40 text-indigo-200 shadow-lg shadow-indigo-950/40"
                    : "border border-transparent text-slate-400 hover:bg-slate-900/70 hover:text-white",
                )}
              >
                <item.icon
                  className={cn(
                    "h-[18px] w-[18px] shrink-0 transition-colors",
                    active ? "text-indigo-400" : "text-slate-500 group-hover:text-slate-300",
                  )}
                />
                <span className="truncate">{item.label}</span>
                {active && (
                  <span className="ml-auto h-1.5 w-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(34,211,238,0.9)]" />
                )}
              </Link>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

function OperatorCard({ onNavigate }: { onNavigate?: () => void }) {
  const { operator, initials, isAuthenticated, signOut } = useAuth();

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-700 bg-slate-800 font-mono text-xs font-semibold text-slate-300">
            {initials}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-white">{operator?.name ?? "—"}</p>
            <p className="truncate text-[11px] text-slate-500">
              {isAuthenticated ? operator?.email : "Not signed in"}
            </p>
          </div>
        </div>
      </div>

      {isAuthenticated ? (
        <Button
          variant="ghost"
          onClick={() => {
            signOut();
            onNavigate?.();
          }}
          className="w-full justify-start text-slate-400 hover:bg-red-950/20 hover:text-red-400"
        >
          <LogOut className="mr-2 h-4 w-4" />
          End session
        </Button>
      ) : (
        <Link href="/login" onClick={onNavigate}>
          <Button className="w-full bg-indigo-600 text-white hover:bg-indigo-500">Sign in</Button>
        </Link>
      )}
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="no-print fixed left-0 top-0 z-40 hidden h-screen w-64 flex-col border-r border-slate-800 bg-slate-950/95 backdrop-blur-xl lg:flex">
      <div className="border-b border-slate-800 p-5">
        <Brand />
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-6">
        <NavList pathname={pathname} />
      </div>

      <div className="space-y-3 border-t border-slate-800 p-3">
        <EngineStatus />
        <OperatorCard />
      </div>
    </aside>
  );
}

export function MobileNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => setOpen(false), [pathname]);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      <div className="no-print fixed inset-x-0 top-0 z-50 flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950/90 px-4 backdrop-blur-md lg:hidden">
        <Brand />
        <Button
          variant="ghost"
          size="icon"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>
      </div>

      {open && (
        <>
          <div
            className="fixed inset-0 top-16 z-40 bg-slate-950/70 backdrop-blur-sm lg:hidden"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <div className="fixed inset-x-0 top-16 z-40 max-h-[calc(100vh-4rem)] overflow-y-auto border-b border-slate-800 bg-slate-950 p-4 lg:hidden">
            <NavList pathname={pathname} onNavigate={() => setOpen(false)} />
            <div className="mt-6 space-y-3 border-t border-slate-800 pt-4">
              <EngineStatus />
              <OperatorCard onNavigate={() => setOpen(false)} />
            </div>
          </div>
        </>
      )}
    </>
  );
}

const PUBLIC_LINKS = [
  { href: "#detectors", label: "Detectors" },
  { href: "#models", label: "AI Models" },
  { href: "#solutions", label: "Solutions" },
  { href: "#how", label: "How it works" },
  { href: "#extension", label: "Extension" },
  { href: "#faq", label: "FAQ" },
];

export function PublicHeader() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-8 px-4 sm:px-6 lg:px-8">
        <Brand />
        <div className="hidden items-center gap-6 text-sm text-slate-300 lg:flex">
          {PUBLIC_LINKS.map((l) => (
            <a key={l.href} href={l.href} className="transition-colors hover:text-white">
              {l.label}
            </a>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Link href="/dashboard" className="hidden sm:block">
            <Button variant="outline" className="border-slate-700 bg-transparent text-slate-200 hover:bg-slate-900 hover:text-white">
              Open console
            </Button>
          </Link>
          <Link href="/forensics">
            <Button className="bg-indigo-600 text-white hover:bg-indigo-500">
              <Activity className="mr-2 h-4 w-4" />
              Analyse a file
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>
      {open && (
        <div className="border-t border-slate-800 px-4 py-3 lg:hidden">
          {PUBLIC_LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              className="block rounded-md px-2 py-2.5 text-sm text-slate-300 hover:bg-slate-900 hover:text-white"
            >
              {l.label}
            </a>
          ))}
          <Link href="/dashboard" className="mt-1 block rounded-md px-2 py-2.5 text-sm text-slate-300 hover:bg-slate-900 hover:text-white">
            Open console
          </Link>
        </div>
      )}
    </nav>
  );
}
