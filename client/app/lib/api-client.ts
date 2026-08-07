/* ============================================================================
   DEEP TRUTH — API CLIENT
   ----------------------------------------------------------------------------
   Single source of truth for talking to the FastAPI backend
   (server/app/routers/cases.py + health.py).

   This replaces the older `app/lib/api.ts` and `app/shared/routes.ts`
   contracts, which described a different backend: numeric case ids, a bare
   array from GET /cases, a JSON POST body, and a POST /cases/{id}/process
   endpoint. None of those exist on the real server — ids are strings, the
   list endpoint returns a pagination envelope, uploads are multipart, and
   analysis starts automatically as a BackgroundTask on create.
   ========================================================================= */

import {
  CaseSchema,
  CaseListResponseSchema,
  DashboardStatsSchema,
  HealthResponseSchema,
  type Case,
  type CaseListResponse,
  type CaseStatus,
  type DashboardStats,
  type HealthResponse,
  type MediaType,
} from "@/app/shared/schema";

/* ── Configuration ───────────────────────────────────────────────────────── */

/** Base URL of the FastAPI service, including the `/api` prefix. */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000/api";

/** Origin only — used to resolve `/uploads/...` media URLs. */
export const API_ORIGIN = API_BASE.replace(/\/api$/, "");

/* ── Error type ──────────────────────────────────────────────────────────── */

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** True when the server rejected the file itself (type or size). */
  get isValidation() {
    return this.status === 422 || this.status === 400;
  }

  get isNotFound() {
    return this.status === 404;
  }
}

/** FastAPI returns errors as `{ detail: string }` or `{ detail: [{msg, loc}] }`. */
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

/* ── Core fetch ──────────────────────────────────────────────────────────── */

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
    // Network-level failure: server down, wrong port, or CORS refused.
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

/* ── Query string helper ─────────────────────────────────────────────────── */

export interface CaseListParams {
  page?: number;
  pageSize?: number;
  mediaType?: MediaType;
  status?: CaseStatus;
  /** Free-text search across case titles. */
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

/* ── Upload payload ──────────────────────────────────────────────────────── */

export interface CreateCasePayload {
  title: string;
  mediaType: MediaType;
  file: File;
  userId?: string;
  notes?: string;
}

export interface UpdateCasePayload {
  title?: string;
  status?: CaseStatus;
  riskScore?: number;
  syntheticLikelihood?: number;
  notes?: string;
}

/* ── Endpoints ───────────────────────────────────────────────────────────── */

export const casesApi = {
  /** GET /cases — paginated, filterable. */
  list(params?: CaseListParams): Promise<CaseListResponse> {
    return request(`/cases${toQuery(params)}`, { method: "GET" }, (raw) =>
      CaseListResponseSchema.parse(raw),
    );
  },

  /** GET /cases/{case_id} — looked up by the public `CASE-XXXXXXXX` id. */
  get(caseId: string): Promise<Case> {
    return request(`/cases/${encodeURIComponent(caseId)}`, { method: "GET" }, (raw) =>
      CaseSchema.parse(raw),
    );
  },

  /**
   * POST /cases — multipart upload. Analysis is queued server-side on the
   * BackgroundTask; the case comes back with status `processing` and the
   * caller should poll `get()` until it settles.
   *
   * `onProgress` reports upload progress only (0–100), which is why this one
   * endpoint uses XHR rather than fetch — fetch has no upload progress event.
   */
  create(payload: CreateCasePayload, onProgress?: (percent: number) => void): Promise<Case> {
    const form = new FormData();
    form.append("title", payload.title);
    form.append("media_type", payload.mediaType);
    form.append("file", payload.file);
    if (payload.userId) form.append("user_id", payload.userId);
    if (payload.notes) form.append("notes", payload.notes);

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

  /** PATCH /cases/{case_id} — edit title, notes, or override the verdict. */
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

  /** DELETE /cases/{case_id} — removes the row and the stored evidence file. */
  remove(caseId: string): Promise<void> {
    return request(`/cases/${encodeURIComponent(caseId)}`, { method: "DELETE" }, () => undefined);
  },
};

export const statsApi = {
  /** GET /stats — aggregate counts for the Command Center. */
  get(): Promise<DashboardStats> {
    return request(`/stats`, { method: "GET" }, (raw) => DashboardStatsSchema.parse(raw));
  },
};

export const healthApi = {
  /** GET /health — service + database reachability. */
  get(): Promise<HealthResponse> {
    return request(`/health`, { method: "GET" }, (raw) => HealthResponseSchema.parse(raw));
  },
};

/* ── Helpers shared by the UI ────────────────────────────────────────────── */

/** Resolve a stored evidence file to a URL the browser can load. */
export function resolveMediaUrl(fileUrl?: string | null): string | undefined {
  if (!fileUrl) return undefined;
  if (/^https?:\/\//i.test(fileUrl)) return fileUrl;
  return `${API_ORIGIN}${fileUrl.startsWith("/") ? "" : "/"}${fileUrl}`;
}

/** Infer the modality the server expects from the browser's MIME type. */
export function inferMediaType(file: File): MediaType {
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return "image";
}

/** MIME types the server accepts, mirrored from analyser.ALLOWED_TYPES. */
export const ACCEPTED_MIME: Record<MediaType, string[]> = {
  image: ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
  video: ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"],
  audio: ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "audio/mp4"],
};

export const ACCEPT_ATTRIBUTE = Object.values(ACCEPTED_MIME).flat().join(",");

/** Server rejects anything above MAX_FILE_SIZE_MB (default 500). */
export const MAX_FILE_MB = Number(process.env.NEXT_PUBLIC_MAX_FILE_MB ?? 500);

/**
 * Client-side pre-check so an oversized or wrong-typed file fails instantly
 * instead of after a long upload. The server re-validates regardless.
 */
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
