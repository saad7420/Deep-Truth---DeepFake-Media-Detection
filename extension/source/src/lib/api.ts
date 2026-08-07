/* ============================================================================
   DEEP TRUTH EXTENSION — API CLIENT
   ----------------------------------------------------------------------------
   The one place the extension talks to the FastAPI backend
   (server/app/routers/cases.py).

   The flow the server actually supports is upload-then-poll:

     1. fetch the media bytes off the page
     2. POST them to /api/cases as multipart/form-data
     3. the server stores the file, opens a case, and queues analysis on a
        BackgroundTask — the response comes straight back as "processing"
     4. GET /api/cases/{caseId} until status leaves "processing"

   There is no /api/v1/analyze, no task id, and no server-side URL fetching:
   the backend never sees the page, only the bytes we hand it.
   ========================================================================= */

import type { CaseResponse, MediaType } from '../types';

/** Mirrors ALLOWED_TYPES in server/app/services/analyser.py exactly. The
    server re-validates, so this is only here to fail fast with a message
    that names the actual problem instead of surfacing a raw 422. */
export const ACCEPTED_MIME: Record<MediaType, string[]> = {
  image: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'],
  video: ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'],
  audio: ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/mp4'],
};

/** Mirrors MAX_FILE_SIZE_MB (server default 500). */
export const MAX_FILE_MB = 500;

/** File extension to give the upload, so the server's `Path(...).suffix`
    lands on something sensible when the page URL has no extension of its own
    (blob: URLs, CDN paths ending in a hash, and so on). */
const EXT_FOR_MIME: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/bmp': '.bmp',
  'image/tiff': '.tiff',
  'video/mp4': '.mp4',
  'video/quicktime': '.mov',
  'video/x-msvideo': '.avi',
  'video/webm': '.webm',
  'video/x-matroska': '.mkv',
  'audio/mpeg': '.mp3',
  'audio/wav': '.wav',
  'audio/x-wav': '.wav',
  'audio/flac': '.flac',
  'audio/ogg': '.ogg',
  'audio/mp4': '.m4a',
};

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Normalise whatever the user typed in settings into a clean origin. */
export function normaliseApiUrl(raw: string): string {
  const trimmed = (raw || '').trim().replace(/\/+$/, '');
  // Tolerate someone pasting the URL with /api already on the end — the
  // console's own .env.local wants it that way, so it's an easy mix-up.
  return trimmed.replace(/\/api$/, '');
}

function apiBase(apiUrl: string): string {
  return `${normaliseApiUrl(apiUrl)}/api`;
}

/** FastAPI errors are `{detail: string}` or `{detail: [{msg}]}`. */
function readDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d) && d[0] && typeof d[0] === 'object') {
      const msg = (d[0] as { msg?: string }).msg;
      if (msg) return msg;
    }
  }
  return fallback;
}

/* ── Health ──────────────────────────────────────────────────────────────── */

export async function checkHealth(apiUrl: string): Promise<{ ok: boolean; detail: string }> {
  try {
    const res = await fetch(`${apiBase(apiUrl)}/health`, { method: 'GET' });
    if (!res.ok) return { ok: false, detail: `Server replied ${res.status}` };
    const body = (await res.json()) as { status?: string; db?: string };
    if (body.db && body.db !== 'ok') {
      return { ok: false, detail: `Database unavailable (${body.db})` };
    }
    return { ok: true, detail: 'Connected' };
  } catch {
    return {
      ok: false,
      detail: `No response from ${normaliseApiUrl(apiUrl)} — is the backend running?`,
    };
  }
}

/* ── Media retrieval ─────────────────────────────────────────────────────── */

/**
 * Pull the media bytes down so they can be uploaded.
 *
 * This runs in the service worker rather than the content script because the
 * worker has the extension's host permissions and is not subject to the
 * page's own CSP — a content script fetching a CDN asset is frequently
 * blocked by connect-src on sites like Instagram and X.
 */
export async function fetchMedia(
  mediaUrl: string,
  expected: MediaType,
): Promise<{ blob: Blob; contentType: string }> {
  let res: Response;
  try {
    res = await fetch(mediaUrl);
  } catch {
    throw new ApiError(0, 'Could not download this media from the page.');
  }
  if (!res.ok) {
    throw new ApiError(res.status, `The page returned ${res.status} for this media.`);
  }

  const blob = await res.blob();

  if (blob.size === 0) {
    throw new ApiError(0, 'This media element is empty — nothing to analyse.');
  }
  if (blob.size > MAX_FILE_MB * 1024 * 1024) {
    throw new ApiError(
      0,
      `File is ${(blob.size / 1e6).toFixed(0)} MB; the server accepts up to ${MAX_FILE_MB} MB.`,
    );
  }

  // A stream served without a usable Content-Type can't be validated by the
  // server, and guessing wrong wastes a long upload. Say so plainly.
  const contentType = (blob.type || '').split(';')[0].trim();
  if (!contentType) {
    throw new ApiError(0, 'The server did not report a media type for this file.');
  }
  if (!ACCEPTED_MIME[expected].includes(contentType)) {
    throw new ApiError(
      0,
      `${contentType} isn't a supported ${expected} format. ` +
        `Accepted: ${ACCEPTED_MIME[expected].map((m) => m.split('/')[1]).join(', ')}.`,
    );
  }

  return { blob, contentType };
}

/* ── Case creation ───────────────────────────────────────────────────────── */

/** Build a readable case title from the page and media URL. */
export function buildTitle(pageUrl: string, mediaUrl: string): string {
  let host = 'page';
  try {
    host = new URL(pageUrl).hostname.replace(/^www\./, '');
  } catch {
    /* keep the fallback */
  }

  let name = '';
  try {
    const path = new URL(mediaUrl, pageUrl).pathname;
    name = decodeURIComponent(path.split('/').filter(Boolean).pop() ?? '');
  } catch {
    /* blob: and data: URLs have no useful path */
  }

  const base = name ? `${host} — ${name}` : `${host} — in-page media`;
  // The server caps title at 200 characters (CaseBase).
  return base.length > 200 ? `${base.slice(0, 197)}...` : base;
}

/**
 * POST the media to /api/cases. Returns the freshly opened case, which the
 * server hands back as `processing` with analysis already queued.
 */
export async function createCase(
  apiUrl: string,
  opts: { blob: Blob; contentType: string; mediaType: MediaType; title: string; notes?: string },
): Promise<CaseResponse> {
  const ext = EXT_FOR_MIME[opts.contentType] ?? '.bin';
  const file = new File([opts.blob], `capture${ext}`, { type: opts.contentType });

  const form = new FormData();
  form.append('title', opts.title);
  form.append('media_type', opts.mediaType);
  form.append('file', file);
  if (opts.notes) form.append('notes', opts.notes);

  let res: Response;
  try {
    res = await fetch(`${apiBase(apiUrl)}/cases`, { method: 'POST', body: form });
  } catch {
    throw new ApiError(
      0,
      `Can't reach the analysis server at ${normaliseApiUrl(apiUrl)}. Start the backend and try again.`,
    );
  }

  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(res.status, readDetail(body, `Upload failed (${res.status})`));
  }
  return body as CaseResponse;
}

/* ── Polling ─────────────────────────────────────────────────────────────── */

export async function getCase(apiUrl: string, caseId: string): Promise<CaseResponse> {
  let res: Response;
  try {
    res = await fetch(`${apiBase(apiUrl)}/cases/${encodeURIComponent(caseId)}`);
  } catch {
    throw new ApiError(0, 'Lost contact with the analysis server.');
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(res.status, readDetail(body, `Could not read case (${res.status})`));
  }
  return body as CaseResponse;
}

/* ── Result interpretation ───────────────────────────────────────────────── */

/**
 * Pull the engine's own one-line reasoning out of the summary row, so the
 * badge can say why rather than only how much.
 *
 * Row classification matches the console's app/lib/analysis.ts: the summary
 * row carries `details.tier === "summary"`, with a name fallback for rows
 * written before that field existed.
 */
export function explain(c: CaseResponse): string | undefined {
  const summary = c.analysisResults.find((r) => {
    const tier = (r.details as { tier?: unknown } | null | undefined)?.tier;
    if (typeof tier === 'string') return tier === 'summary';
    return /\(fused\)|\(stub\)$/.test(r.model_name);
  });
  if (!summary?.details) return undefined;

  const d = summary.details as { rationale?: unknown; note?: unknown; error?: unknown };
  for (const v of [d.rationale, d.note, d.error]) {
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return undefined;
}

/** True once the pipeline has finished, whatever the outcome. */
export function isSettled(status: CaseResponse['status']): boolean {
  return status !== 'processing';
}
