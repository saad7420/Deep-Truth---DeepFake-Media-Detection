"use client";

/* ============================================================================
   DEEP TRUTH — CASE DATA HOOKS
   ----------------------------------------------------------------------------
   Every read/write against the forensic backend goes through here. Replaces
   the previous hook set, which pointed at the Next.js mock routes under
   app/api/cases (numeric ids, JSON upload body, a /process endpoint) rather
   than the FastAPI service.
   ========================================================================= */

import {
  useMutation,
  useQuery,
  useQueryClient,
  keepPreviousData,
} from "@tanstack/react-query";

import {
  ApiError,
  casesApi,
  healthApi,
  statsApi,
  type CaseListParams,
  type CreateCasePayload,
  type UpdateCasePayload,
} from "@/app/lib/api-client";
import type { Case } from "@/app/shared/schema";
import { useToast } from "@/app/hooks/use-toast";

/* ── Query keys ──────────────────────────────────────────────────────────── */

export const caseKeys = {
  all: ["cases"] as const,
  list: (params?: CaseListParams) => ["cases", "list", params ?? {}] as const,
  detail: (caseId: string) => ["cases", "detail", caseId] as const,
  stats: ["stats"] as const,
  health: ["health"] as const,
};

/** How often a case still in `processing` is re-checked. */
const POLL_MS = 2000;

/* ── Reads ───────────────────────────────────────────────────────────────── */

/**
 * Paginated case list. Keeps the previous page on screen while the next one
 * loads so filters and pagination don't flash an empty table.
 *
 * Auto-refreshes every few seconds while any case in the current page is
 * still processing, then stops — no polling on a settled list.
 */
export function useCases(params?: CaseListParams) {
  return useQuery({
    queryKey: caseKeys.list(params),
    queryFn: () => casesApi.list(params),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      return data.cases.some((c) => c.status === "processing") ? POLL_MS : false;
    },
  });
}

/** Single case, polled until the pipeline settles. */
export function useCase(caseId?: string) {
  return useQuery({
    queryKey: caseKeys.detail(caseId ?? ""),
    queryFn: () => casesApi.get(caseId as string),
    enabled: Boolean(caseId),
    retry: (count, error) => !(error instanceof ApiError && error.isNotFound) && count < 2,
    refetchInterval: (query) =>
      query.state.data?.status === "processing" ? POLL_MS : false,
  });
}

/** Aggregate counts for the Command Center. */
export function useStats() {
  return useQuery({
    queryKey: caseKeys.stats,
    queryFn: () => statsApi.get(),
    staleTime: 15_000,
    refetchInterval: 30_000,
  });
}

/** Backend + database reachability, shown in the sidebar status strip. */
export function useHealth() {
  return useQuery({
    queryKey: caseKeys.health,
    queryFn: () => healthApi.get(),
    retry: false,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

/* ── Writes ──────────────────────────────────────────────────────────────── */

function invalidateLists(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: caseKeys.all });
  qc.invalidateQueries({ queryKey: caseKeys.stats });
}

/**
 * Upload evidence and open a case. The server queues analysis itself, so the
 * returned case arrives as `processing`; callers navigate to the detail page
 * and `useCase` takes over polling.
 */
export function useCreateCase(onProgress?: (percent: number) => void) {
  const qc = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: (payload: CreateCasePayload) => casesApi.create(payload, onProgress),
    onSuccess: (created) => {
      qc.setQueryData(caseKeys.detail(created.caseId), created);
      invalidateLists(qc);
      toast({
        title: "Analysis started",
        description: `${created.caseId} is queued. Results appear as engines report in.`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Upload failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });
}

/** Edit a case's title, notes, or override its verdict. */
export function useUpdateCase() {
  const qc = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ caseId, ...payload }: UpdateCasePayload & { caseId: string }) =>
      casesApi.update(caseId, payload),
    onSuccess: (updated) => {
      qc.setQueryData(caseKeys.detail(updated.caseId), updated);
      invalidateLists(qc);
      toast({ title: "Case updated", description: `${updated.caseId} saved.` });
    },
    onError: (error: Error) => {
      toast({ title: "Couldn't save", description: error.message, variant: "destructive" });
    },
  });
}

/**
 * Delete a case and its stored evidence file. Optimistically drops the row
 * from every cached list so the table responds immediately, and restores it
 * if the server rejects the delete.
 */
export function useDeleteCase() {
  const qc = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: (caseId: string) => casesApi.remove(caseId),

    onMutate: async (caseId) => {
      await qc.cancelQueries({ queryKey: caseKeys.all });
      const snapshot = qc.getQueriesData({ queryKey: caseKeys.all });

      qc.setQueriesData<{ cases: Case[]; total: number }>(
        { queryKey: caseKeys.all },
        (old) =>
          old && Array.isArray(old.cases)
            ? {
                ...old,
                cases: old.cases.filter((c) => c.caseId !== caseId),
                total: Math.max(0, old.total - 1),
              }
            : old,
      );

      return { snapshot };
    },

    onError: (error: Error, _caseId, context) => {
      context?.snapshot.forEach(([key, value]) => qc.setQueryData(key, value));
      toast({ title: "Couldn't delete", description: error.message, variant: "destructive" });
    },

    onSuccess: (_data, caseId) => {
      qc.removeQueries({ queryKey: caseKeys.detail(caseId) });
      toast({ title: "Case deleted", description: `${caseId} and its evidence file are gone.` });
    },

    onSettled: () => invalidateLists(qc),
  });
}
