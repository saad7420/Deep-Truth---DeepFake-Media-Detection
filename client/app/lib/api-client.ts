import {
  CaseSchema,
  CaseListResponseSchema,
  DashboardStatsSchema,
  HealthResponseSchema,
  JobStateSchema,
  QueueOverviewSchema,
  type Case,
  type CaseListResponse,
  type CaseStatus,
  type DashboardStats,
  type HealthResponse,
  type JobState,
  type MediaType,
  type QueueOverview,
} from "@/app/shared/schema";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000/api";

export const API_ORIGIN = API_BASE.replace(/\/api$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  get isValidation() {
    return this.status === 422 || this.status === 400;
  }

  get isNotFound() {
    return this.status === 404;
  }

  get isRateLimited() {
    return this.status === 429;
  }
}

function readDetail(body: unknown, fallback: string): string {
  if (typeof body === "string" && body) return body;
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      const first = d[0] as { msg?: string } | undefined;
      if (first?.msg) return first.msg;
    }
  }
  return fallback;
}

// every call goes through here so errors look the same everywhere
async function request<T>(
  path: string,
  init: RequestInit | undefined,
  parse: (raw: unknown) => T,
): Promise<T> {
  let res: Response;

  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      0,
      `Can't reach the analysis server at ${API_ORIGIN}. Start the backend, then try again.`,
    );
  }

  if (res.status === 204) return parse(undefined);

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    throw new ApiError(res.status, readDetail(body, `Request failed (${res.status})`), body);
  }

  return parse(body);
}

export interface CaseListParams {
  page?: number;
  pageSize?: number;
  mediaType?: MediaType;
  status?: CaseStatus;
  q?: string;
}

function toQuery(params: CaseListParams = {}): string {
  const sp = new URLSearchParams();
  if (params.page) sp.set("page", String(params.page));
  if (params.pageSize) sp.set("page_size", String(params.pageSize));
  if (params.mediaType) sp.set("media_type", params.mediaType);
  if (params.status) sp.set("status", params.status);
  if (params.q?.trim()) sp.set("q", params.q.trim());
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export interface CreateCasePayload {
  title: string;
  mediaType: MediaType;
  file: File;
  userId?: string;
  notes?: string;
  sourceUrl?: string;
}

export interface UpdateCasePayload {
  title?: string;
  status?: CaseStatus;
  riskScore?: number;
  syntheticLikelihood?: number;
  notes?: string;
}

export const casesApi = {
  list(params?: CaseListParams): Promise<CaseListResponse> {
    return request(`/cases${toQuery(params)}`, { method: "GET" }, (raw) =>
      CaseListResponseSchema.parse(raw),
    );
  },

  get(caseId: string): Promise<Case> {
    return request(`/cases/${encodeURIComponent(caseId)}`, { method: "GET" }, (raw) =>
      CaseSchema.parse(raw),
    );
  },

  createFromUrl(payload: { title: string; mediaType: MediaType; mediaUrl: string; userId?: string }): Promise<Case> {
    const form = new FormData();
    form.append("title", payload.title);
    form.append("media_type", payload.mediaType);
    form.append("media_url", payload.mediaUrl);
    if (payload.userId) form.append("user_id", payload.userId);
    return request(`/cases`, { method: "POST", body: form }, (raw) => CaseSchema.parse(raw));
  },

  create(payload: CreateCasePayload, onProgress?: (percent: number) => void): Promise<Case> {
    const form = new FormData();
    form.append("title", payload.title);
    form.append("media_type", payload.mediaType);
    form.append("file", payload.file);
    if (payload.userId) form.append("user_id", payload.userId);
    if (payload.notes) form.append("notes", payload.notes);
    if (payload.sourceUrl) form.append("source_url", payload.sourceUrl);

    if (!onProgress) {
      return request(`/cases`, { method: "POST", body: form }, (raw) => CaseSchema.parse(raw));
    }

    return new Promise<Case>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/cases`);
      xhr.responseType = "json";

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };

      xhr.onload = () => {
        const body = xhr.response;
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(CaseSchema.parse(body));
          } catch (err) {
            reject(new ApiError(xhr.status, "Server returned an unexpected case shape.", err));
          }
        } else {
          reject(new ApiError(xhr.status, readDetail(body, `Upload failed (${xhr.status})`), body));
        }
      };

      xhr.onerror = () =>
        reject(new ApiError(0, `Can't reach the analysis server at ${API_ORIGIN}.`));
      xhr.onabort = () => reject(new ApiError(0, "Upload cancelled."));

      xhr.send(form);
    });
  },

  update(caseId: string, payload: UpdateCasePayload): Promise<Case> {
    const body: Record<string, unknown> = {};
    if (payload.title !== undefined) body.title = payload.title;
    if (payload.status !== undefined) body.status = payload.status;
    if (payload.riskScore !== undefined) body.risk_score = payload.riskScore;
    if (payload.syntheticLikelihood !== undefined)
      body.synthetic_likelihood = payload.syntheticLikelihood;
    if (payload.notes !== undefined) body.notes = payload.notes;

    return request(
      `/cases/${encodeURIComponent(caseId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      (raw) => CaseSchema.parse(raw),
    );
  },

  remove(caseId: string): Promise<void> {
    return request(`/cases/${encodeURIComponent(caseId)}`, { method: "DELETE" }, () => undefined);
  },
};

export const statsApi = {
  get(): Promise<DashboardStats> {
    return request(`/stats`, { method: "GET" }, (raw) => DashboardStatsSchema.parse(raw));
  },
};

export const healthApi = {
  get(): Promise<HealthResponse> {
    return request(`/health`, { method: "GET" }, (raw) => HealthResponseSchema.parse(raw));
  },
};

export function reportPdfUrl(caseId: string): string {
  return `${API_BASE}/cases/${encodeURIComponent(caseId)}/report.pdf`;
}

export async function downloadCaseReport(caseId: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(reportPdfUrl(caseId), { headers: { Accept: "application/pdf" } });
  } catch {
    throw new ApiError(0, `Can't reach the analysis server at ${API_ORIGIN}.`);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, readDetail(body, `Report failed (${res.status})`), body);
  }

  const blob = await res.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = `deeptruth-${caseId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export const queueApi = {
  overview(): Promise<QueueOverview> {
    return request(`/queue`, { method: "GET" }, (raw) => QueueOverviewSchema.parse(raw));
  },

  forCase(caseId: string): Promise<JobState> {
    return request(
      `/queue/cases/${encodeURIComponent(caseId)}`,
      { method: "GET" },
      (raw) => JobStateSchema.parse(raw),
    );
  },
};

export interface JobEvent extends Partial<JobState> {
  caseDbId?: string | null;
}

export interface SubscribeOptions {
  caseId?: string;
  onJob?: (event: JobEvent) => void;
  onSnapshot?: (event: JobEvent & { pendingDepth?: number }) => void;
  onError?: (error: Event) => void;
}

// server-sent events, returns a function that closes the stream
export function subscribeToJobs(options: SubscribeOptions = {}): () => void {
  if (typeof window === "undefined" || typeof EventSource === "undefined") {
    return () => {};
  }

  const qs = options.caseId ? `?case_id=${encodeURIComponent(options.caseId)}` : "";
  const source = new EventSource(`${API_BASE}/queue/stream${qs}`);

  const parse = (raw: string): JobEvent | null => {
    try {
      return JSON.parse(raw) as JobEvent;
    } catch {
      return null;
    }
  };

  source.addEventListener("snapshot", (e) => {
    const payload = parse((e as MessageEvent).data);
    if (payload) options.onSnapshot?.(payload);
  });

  source.addEventListener("job", (e) => {
    const payload = parse((e as MessageEvent).data);
    if (payload) options.onJob?.(payload);
  });

  source.onerror = (e) => options.onError?.(e);

  return () => source.close();
}

export function resolveMediaUrl(fileUrl?: string | null): string | undefined {
  if (!fileUrl) return undefined;
  if (/^https?:\/\//i.test(fileUrl)) return fileUrl;
  return `${API_ORIGIN}${fileUrl.startsWith("/") ? "" : "/"}${fileUrl}`;
}

export function inferMediaType(file: File): MediaType {
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return "image";
}

export const ACCEPTED_MIME: Record<MediaType, string[]> = {
  image: ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
  video: ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"],
  audio: ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "audio/mp4"],
};

export const ACCEPT_ATTRIBUTE = Object.values(ACCEPTED_MIME).flat().join(",");

export const MAX_FILE_MB = Number(process.env.NEXT_PUBLIC_MAX_FILE_MB ?? 500);

// same type and size rules as the server, checked before uploading
export function validateFile(file: File, mediaType: MediaType): string | null {
  if (!ACCEPTED_MIME[mediaType].includes(file.type)) {
    return `${file.type || "This file type"} isn't supported for ${mediaType}. Accepted: ${ACCEPTED_MIME[
      mediaType
    ]
      .map((m) => m.split("/")[1])
      .join(", ")}.`;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    return `File is ${(file.size / 1e6).toFixed(1)} MB. The limit is ${MAX_FILE_MB} MB.`;
  }
  return null;
}

export function formatBytes(bytes?: number | null): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
