/* ============================================================================
   DEEP TRUTH EXTENSION — SERVICE WORKER
   ----------------------------------------------------------------------------
   Owns every network call. The content script never talks to the backend
   directly: it hands over a media URL and gets progress back by message.

   What changed from the previous version, and why it matters:

   - The simulation engine is gone. It was on by default and produced verdicts
     from Math.random() with hand-written explanations ("eye blink coordination
     shows neural mismatch"). A detection tool that invents confident findings
     when it can't reach its backend is worse than one that plainly fails, so
     an unreachable server is now reported as an error and nothing is written
     to history.

   - The endpoints are the ones the server actually serves. /api/v1/analyze and
     /api/v1/task/{id}/status never existed.

   MV3 service workers are killed when idle, which would abandon an in-flight
   poll. Each poll therefore re-reads its case from storage and reschedules
   with chrome.alarms rather than setInterval, so a scan survives the worker
   being torn down and restarted.
   ========================================================================= */

import {
  ApiError,
  buildTitle,
  createCase,
  createCaseFromUrl,
  explain,
  fetchMedia,
  getCase,
  isBrowserOnlyUrl,
  isExperimental,
  isSettled,
  lookupCachedUrl,
} from '../lib/api';
import {
  DEFAULT_SETTINGS,
  type CaseResponse,
  type ExtensionSettings,
  type ScanRecord,
  type StartAnalysisPayload,
  type StartAnalysisResponse,
} from '../types';

const HISTORY_KEY = 'scanHistory';
const PENDING_KEY = 'pendingScans';
const HISTORY_LIMIT = 50;
const POLL_ALARM = 'deep-truth-poll';
/** Alarms cannot fire faster than once a minute in MV3, so the poll runs on a
    self-rescheduling timer while the worker is alive and uses the alarm only
    as a safety net if it gets suspended mid-scan. */
const POLL_MS = 2000;
/** A case that never settles shouldn't be polled forever. */
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

/* ── Settings ────────────────────────────────────────────────────────────── */

async function getSettings(): Promise<ExtensionSettings> {
  const res = await chrome.storage.local.get(['isEnabled', 'apiUrl']);
  return {
    isEnabled: res.isEnabled !== undefined ? Boolean(res.isEnabled) : DEFAULT_SETTINGS.isEnabled,
    apiUrl: typeof res.apiUrl === 'string' && res.apiUrl ? res.apiUrl : DEFAULT_SETTINGS.apiUrl,
  };
}

chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(['isEnabled', 'apiUrl', HISTORY_KEY]);
  await chrome.storage.local.set({
    isEnabled: current.isEnabled !== undefined ? current.isEnabled : DEFAULT_SETTINGS.isEnabled,
    // An earlier build shipped a default of https://api.deep-truth.ai, which
    // does not resolve. Migrate anyone carrying it to the local default.
    apiUrl:
      typeof current.apiUrl === 'string' && !/api\.deep-truth\.ai/.test(current.apiUrl)
        ? current.apiUrl
        : DEFAULT_SETTINGS.apiUrl,
    [HISTORY_KEY]: current[HISTORY_KEY] ?? [],
    // isSimulatedMode is deliberately not carried forward.
  });
  await chrome.storage.local.remove('isSimulatedMode');
});

/* ── History ─────────────────────────────────────────────────────────────── */

async function upsertRecord(record: ScanRecord): Promise<void> {
  const res = await chrome.storage.local.get([HISTORY_KEY]);
  const history: ScanRecord[] = res[HISTORY_KEY] ?? [];

  const idx = history.findIndex((r) => r.caseId === record.caseId);
  if (idx >= 0) history[idx] = record;
  else history.unshift(record);

  await chrome.storage.local.set({ [HISTORY_KEY]: history.slice(0, HISTORY_LIMIT) });
  broadcast(record);
}

/** Push an update to the popup and to every tab showing this media. */
function broadcast(record: ScanRecord): void {
  // The popup may be closed; a rejected sendMessage here is expected.
  chrome.runtime.sendMessage({ type: 'SCAN_UPDATED', payload: record }).catch(() => {});
  chrome.tabs.query({}, (tabs) => {
    tabs.forEach((tab) => {
      if (tab.id !== undefined) {
        chrome.tabs.sendMessage(tab.id, { type: 'SCAN_UPDATED', payload: record }).catch(() => {});
      }
    });
  });
}

function recordFromCase(c: CaseResponse, base: ScanRecord): ScanRecord {
  return {
    ...base,
    caseId: c.caseId,
    status: c.status,
    riskScore: c.status === 'processing' ? undefined : c.riskScore,
    explanation: explain(c),
    experimental: isExperimental(c),
    timestamp: Date.now(),
  };
}

/* ── Polling ─────────────────────────────────────────────────────────────── */

interface PendingScan extends ScanRecord {
  startedAt: number;
}

async function readPending(): Promise<PendingScan[]> {
  const res = await chrome.storage.local.get([PENDING_KEY]);
  return res[PENDING_KEY] ?? [];
}

async function writePending(list: PendingScan[]): Promise<void> {
  await chrome.storage.local.set({ [PENDING_KEY]: list });
  if (list.length > 0) {
    // Safety net for a suspended worker — the minimum period is 1 minute.
    chrome.alarms.create(POLL_ALARM, { periodInMinutes: 1 });
  } else {
    chrome.alarms.clear(POLL_ALARM);
  }
}

async function trackScan(scan: PendingScan): Promise<void> {
  const pending = await readPending();
  await writePending([...pending.filter((p) => p.caseId !== scan.caseId), scan]);
  scheduleTick();
}

let tickHandle: ReturnType<typeof setTimeout> | null = null;

function scheduleTick(): void {
  if (tickHandle !== null) return;
  tickHandle = setTimeout(() => {
    tickHandle = null;
    void pollPending();
  }, POLL_MS);
}

/** One pass over every unsettled case. */
async function pollPending(): Promise<void> {
  const pending = await readPending();
  if (pending.length === 0) {
    await writePending([]);
    return;
  }

  const { apiUrl } = await getSettings();
  const stillPending: PendingScan[] = [];

  for (const scan of pending) {
    if (Date.now() - scan.startedAt > POLL_TIMEOUT_MS) {
      await upsertRecord({
        ...scan,
        status: 'failed',
        error: 'The analysis did not finish in time. Check the server logs.',
        timestamp: Date.now(),
      });
      continue;
    }

    try {
      const c = await getCase(apiUrl, scan.caseId);
      const updated = recordFromCase(c, scan);
      await upsertRecord(updated);

      if (!isSettled(c.status)) {
        stillPending.push({ ...scan, ...updated, startedAt: scan.startedAt });
      }
    } catch (err) {
      // A transient network blip shouldn't kill the scan — keep it queued and
      // let the timeout above end it if the server really is gone.
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.warn('[Deep-Truth] poll failed for', scan.caseId, message);
      stillPending.push(scan);
    }
  }

  await writePending(stillPending);
  if (stillPending.length > 0) scheduleTick();
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === POLL_ALARM) void pollPending();
});

/* ── Analysis entry point ────────────────────────────────────────────────── */

async function startAnalysis(payload: StartAnalysisPayload): Promise<StartAnalysisResponse> {
  const { apiUrl } = await getSettings();
  const { mediaUrl, pageUrl, mediaType } = payload;

  try {
    // ── Cache pre-check ──────────────────────────────────────────────────
    // Before downloading anything, ask whether this URL already has a
    // verdict. On a hit the answer is immediate and the whole
    // download → upload → poll cycle is skipped. The record links back to
    // the case that originally produced the verdict, so the console still
    // has one authoritative case rather than a duplicate per sighting.
    //
    // Skipped when the server is too old to know the case's public id:
    // without it the record could not link anywhere, and a scan the operator
    // cannot open is worse than one that took a few seconds longer.
    const cached = await lookupCachedUrl(apiUrl, mediaUrl, mediaType);
    if (cached?.sourceCaseId && cached.status) {
      const record: ScanRecord = {
        caseId: cached.sourceCaseId,
        mediaUrl,
        pageUrl,
        mediaType,
        status: cached.status,
        riskScore: cached.riskScore,
        fromCache: true,
        timestamp: Date.now(),
      };
      await upsertRecord(record);
      return { ok: true, caseId: cached.sourceCaseId };
    }

    const title = buildTitle(pageUrl, mediaUrl);
    const notes = `Captured by the Deep Truth extension from ${pageUrl}`;

    // ── Preferred: let the server fetch it (M4 FE-1) ─────────────────────
    // Saves the user downloading the media only to upload it again. Skipped
    // outright for blob:/data: URLs, which name nothing outside this page.
    let c: CaseResponse | null = null;
    if (!isBrowserOnlyUrl(mediaUrl)) {
      try {
        c = await createCaseFromUrl(apiUrl, { mediaUrl, mediaType, title, notes });
      } catch (err) {
        // Fall through to uploading the bytes ourselves. The server refusing
        // a URL says nothing about whether *we* can read it — media behind a
        // login is the common case, and the browser holds the session the
        // server does not.
        const why = err instanceof Error ? err.message : 'unknown error';
        console.info('[Deep-Truth] server-side fetch declined, uploading instead:', why);
      }
    }

    // ── Fallback: download here and upload the bytes ─────────────────────
    if (!c) {
      const { blob, contentType } = await fetchMedia(mediaUrl, mediaType);
      c = await createCase(apiUrl, {
        blob,
        contentType,
        mediaType,
        title,
        notes,
        // Teaches the server which URL serves these bytes, so the pre-check
        // above can short-circuit the next encounter.
        sourceUrl: mediaUrl,
      });
    }

    const base: ScanRecord = {
      caseId: c.caseId,
      mediaUrl,
      pageUrl,
      mediaType,
      status: c.status,
      timestamp: Date.now(),
    };

    await upsertRecord(base);
    await trackScan({ ...base, startedAt: Date.now() });

    return { ok: true, caseId: c.caseId };
  } catch (err) {
    const message =
      err instanceof ApiError || err instanceof Error
        ? err.message
        : 'Analysis could not be started.';
    console.error('[Deep-Truth] startAnalysis failed:', message);
    return { ok: false, error: message };
  }
}

/* ── Message routing ─────────────────────────────────────────────────────── */

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === 'START_ANALYSIS') {
    startAnalysis(message.payload as StartAnalysisPayload).then(sendResponse);
    return true; // keep the channel open for the async reply
  }

  if (message?.type === 'RESUME_POLLING') {
    void pollPending();
    sendResponse({ ok: true });
    return true;
  }

  return false;
});

// Pick up any scan left unfinished when the worker was last suspended.
void pollPending();

export {};
