"use client";

/* ============================================================================
   DEEP TRUTH — OPERATOR SESSION
   ----------------------------------------------------------------------------
   The FastAPI service has no auth layer (see server/app/core.py — only the
   cases and health routers are mounted), so there is no token to exchange and
   nothing to verify against. This hook keeps a local operator profile instead,
   which is what the console actually needs today: a name and an id to stamp on
   uploads via the `user_id` form field, and a session to end.

   When the backend grows real auth, swap the body of `signIn`/`signOut` for
   calls to it and leave every consumer of this hook untouched.
   ========================================================================= */

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "deeptruth.operator";

export interface Operator {
  id: string;
  name: string;
  email: string;
  role: "analyst" | "investigator" | "admin";
  clearance: string;
}

const GUEST: Operator = {
  id: "OP-GUEST",
  name: "Guest Operator",
  email: "guest@deeptruth.local",
  role: "analyst",
  clearance: "CLASS-C",
};

function read(): Operator | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Operator) : null;
  } catch {
    return null;
  }
}

export function useAuth() {
  const [operator, setOperator] = useState<Operator | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setOperator(read() ?? GUEST);
    setIsLoading(false);

    // Keep tabs in sync when the operator signs in or out elsewhere.
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setOperator(read() ?? GUEST);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const signIn = useCallback((profile: Partial<Operator> & { name: string; email: string }) => {
    const next: Operator = {
      id: profile.id ?? `OP-${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
      name: profile.name,
      email: profile.email,
      role: profile.role ?? "investigator",
      clearance: profile.clearance ?? "CLASS-A",
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setOperator(next);
    return next;
  }, []);

  const signOut = useCallback(() => {
    window.localStorage.removeItem(STORAGE_KEY);
    setOperator(GUEST);
  }, []);

  const initials = (operator?.name ?? "")
    .split(" ")
    .map((p) => p[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return {
    operator,
    initials: initials || "OP",
    isLoading,
    isAuthenticated: Boolean(operator && operator.id !== GUEST.id),
    signIn,
    signOut,
  };
}
