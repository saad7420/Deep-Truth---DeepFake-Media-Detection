"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "deeptruth.operator";

export interface Operator {
  id: string;
  name: string;
  email: string;
}

const GUEST: Operator = {
  id: "OP-GUEST",
  name: "Guest Operator",
  email: "guest@deeptruth.local",
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

// local operator profile only, the api has no login yet
export function useAuth() {
  const [operator, setOperator] = useState<Operator | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setOperator(read() ?? GUEST);
    setIsLoading(false);

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
