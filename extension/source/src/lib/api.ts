import type { CacheLookup, CaseResponse, MediaType } from '../types';

export const ACCEPTED_MIME: Record<MediaType, string[]> = {
  image: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'],
  video: ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'],
  audio: ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/mp4'],
};

export const MAX_FILE_MB = 500;

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

export function normaliseApiUrl(raw: string): string {
  const trimmed = (raw || '').trim().replace(/\/+$/, '');
  return trimmed.replace(/\/api$/, '');
}

function apiBase(apiUrl: string): string {
  return `${normaliseApiUrl(apiUrl)}/api`;
}

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

// one cheap request tells us if this media was already analysed
export async function lookupCachedUrl(
  apiUrl: string,
  mediaUrl: string,
  mediaType: MediaType,
): Promise<CacheLookup | null> {
  if (!/^https?:/i.test(mediaUrl)) return null;

  const qs = new URLSearchParams({ media_type: mediaType, url: mediaUrl });
  try {
    const res = await fetch(`${apiBase(apiUrl)}/cache/lookup?${qs.toString()}`);
    if (!res.ok) return null;
    const body = (await res.json()) as CacheLookup;
    return body.hit ? body : null;
  } catch {
    return null;
  }
}

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

export function buildTitle(pageUrl: string, mediaUrl: string): string {
  let host = 'page';
  try {
    host = new URL(pageUrl).hostname.replace(/^www\./, '');
  } catch {
  }

  let name = '';
  try {
    const path = new URL(mediaUrl, pageUrl).pathname;
    name = decodeURIComponent(path.split('/').filter(Boolean).pop() ?? '');
  } catch {
  }

  const base = name ? `${host} — ${name}` : `${host} — in-page media`;
  return base.length > 200 ? `${base.slice(0, 197)}...` : base;
}

// let the server download it, so the user does not pay for the transfer
export async function createCaseFromUrl(
  apiUrl: string,
  opts: { mediaUrl: string; mediaType: MediaType; title: string; notes?: string },
): Promise<CaseResponse> {
  const form = new FormData();
  form.append('title', opts.title);
  form.append('media_type', opts.mediaType);
  form.append('media_url', opts.mediaUrl);
  form.append('source_url', opts.mediaUrl);
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
    throw new ApiError(res.status, readDetail(body, `Server-side fetch failed (${res.status})`));
  }
  return body as CaseResponse;
}

// blob: and data: urls only exist inside the browser
export function isBrowserOnlyUrl(mediaUrl: string): boolean {
  return !/^https?:/i.test(mediaUrl);
}

export async function createCase(
  apiUrl: string,
  opts: {
    blob: Blob;
    contentType: string;
    mediaType: MediaType;
    title: string;
    notes?: string;
    sourceUrl?: string;
  },
): Promise<CaseResponse> {
  const ext = EXT_FOR_MIME[opts.contentType] ?? '.bin';
  const file = new File([opts.blob], `capture${ext}`, { type: opts.contentType });

  const form = new FormData();
  form.append('title', opts.title);
  form.append('media_type', opts.mediaType);
  form.append('file', file);
  if (opts.notes) form.append('notes', opts.notes);
  if (opts.sourceUrl) form.append('source_url', opts.sourceUrl);

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

function summaryRow(c: CaseResponse) {
  return c.analysisResults.find((r) => {
    const tier = (r.details as { tier?: unknown } | null | undefined)?.tier;
    if (typeof tier === 'string') return tier === 'summary';
    return /\(fused\)|\(stub\)$/.test(r.model_name);
  });
}

export function explain(c: CaseResponse): string | undefined {
  const summary = summaryRow(c);
  if (!summary?.details) return undefined;

  const d = summary.details as { rationale?: unknown; note?: unknown; error?: unknown };
  for (const v of [d.rationale, d.note, d.error]) {
    if (typeof v === 'string' && v.trim()) {
      const text = v.trim();
      const marker = text.indexOf(' | ');
      if (text.startsWith('EXPERIMENTAL CHECKPOINT:') && marker > 0) {
        return text.slice(marker + 3);
      }
      return text;
    }
  }
  return undefined;
}

export function isExperimental(c: CaseResponse): boolean {
  const details = summaryRow(c)?.details as { experimental?: unknown } | undefined;
  return details?.experimental === true;
}

export function isSettled(status: CaseResponse['status']): boolean {
  return status !== 'processing';
}
