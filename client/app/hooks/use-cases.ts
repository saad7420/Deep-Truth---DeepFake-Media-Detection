"use client";

import { useEffect } from "react";
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
  queueApi,
  statsApi,
  subscribeToJobs,
  type CaseListParams,
  type CreateCasePayload,
  type UpdateCasePayload,
} from "@/app/lib/api-client";
import type { Case } from "@/app/shared/schema";
import { useToast } from "@/app/hooks/use-toast";

export const caseKeys = {
  all: ["cases"] as const,
  list: (params?: CaseListParams) => ["cases", "list", params ?? {}] as const,
  detail: (caseId: string) => ["cases", "detail", caseId] as const,
  stats: ["stats"] as const,
  health: ["health"] as const,
  queue: ["queue"] as const,
};

// refresh while a case is still processing
const POLL_MS = 5000;

// do not hammer the api after a 429, that only keeps the bucket empty
const retryUnlessLimited = (count: number, error: Error) =>
  !(error instanceof ApiError && (error.isRateLimited || error.isNotFound)) && count < 2;

export function useCases(params?: CaseListParams) {
  return useQuery({
    queryKey: caseKeys.list(params),
    queryFn: () => casesApi.list(params),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    retry: retryUnlessLimited,
    refetchInterval: (query) => {
      if (query.state.error instanceof ApiError && query.state.error.isRateLimited) {
        return 30_000;
      }
      const data = query.state.data;
      if (!data) return false;
      return data.cases.some((c) => c.status === "processing") ? POLL_MS : false;
    },
  });
}

export function useCase(caseId?: string) {
  return useQuery({
    queryKey: caseKeys.detail(caseId ?? ""),
    queryFn: () => casesApi.get(caseId as string),
    enabled: Boolean(caseId),
    retry: retryUnlessLimited,
    refetchInterval: (query) => {
      if (query.state.error instanceof ApiError && query.state.error.isRateLimited) {
        return 30_000;
      }
      return query.state.data?.status === "processing" ? POLL_MS : false;
    },
  });
}

export function useStats() {
  return useQuery({
    queryKey: caseKeys.stats,
    queryFn: () => statsApi.get(),
    staleTime: 15_000,
    refetchInterval: 30_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: caseKeys.health,
    queryFn: () => healthApi.get(),
    retry: false,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

export function useQueueOverview(enabled = true) {
  return useQuery({
    queryKey: caseKeys.queue,
    queryFn: () => queueApi.overview(),
    enabled,
    retry: false,
    refetchInterval: (query) =>
      query.state.error instanceof ApiError && query.state.error.isRateLimited
        ? 30_000
        : 5_000,
    staleTime: 2_000,
  });
}

// live job updates over server-sent events, no polling needed
export function useJobStream(caseId?: string) {
  const qc = useQueryClient();

  useEffect(() => {
    const unsubscribe = subscribeToJobs({
      caseId,
      onJob: (event) => {
        const settled =
          event.state === "succeeded" ||
          event.state === "failed" ||
          event.state === "cached";

        if (caseId) {
          qc.invalidateQueries({ queryKey: caseKeys.detail(caseId) });
        } else {
          qc.invalidateQueries({ queryKey: caseKeys.all });
        }

        qc.invalidateQueries({ queryKey: caseKeys.queue });
        if (settled) qc.invalidateQueries({ queryKey: caseKeys.stats });
      },
    });

    return unsubscribe;
  }, [caseId, qc]);
}

function invalidateLists(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: caseKeys.all });
  qc.invalidateQueries({ queryKey: caseKeys.stats });
}

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
