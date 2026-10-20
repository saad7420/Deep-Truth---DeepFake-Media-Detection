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
// check the case every 2 seconds, an alarm covers a suspended worker
const POLL_MS = 2000;
// cpu videos can take around 8 minutes, so wait much longer than that
const POLL_TIMEOUT_MS = 35 * 60 * 1000;

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
    apiUrl:
      typeof current.apiUrl === 'string' && !/api\.deep-truth\.ai/.test(current.apiUrl)
        ? current.apiUrl
        : DEFAULT_SETTINGS.apiUrl,
    [HISTORY_KEY]: current[HISTORY_KEY] ?? [],
  });
  await chrome.storage.local.remove('isSimulatedMode');
});

async function upsertRecord(record: ScanRecord): Promise<void> {
  const res = await chrome.storage.local.get([HISTORY_KEY]);
  const history: ScanRecord[] = res[HISTORY_KEY] ?? [];

  const idx = history.findIndex((r) => r.caseId === record.caseId);
  if (idx >= 0) history[idx] = record;
  else history.unshift(record);

  await chrome.storage.local.set({ [HISTORY_KEY]: history.slice(0, HISTORY_LIMIT) });
  broadcast(record);
}

function broadcast(record: ScanRecord): void {
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

// fetch every pending case, fail it after the timeout, keep the unfinished ones
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

// cache hit first, then server-side fetch, then upload from the browser as fallback
async function startAnalysis(payload: StartAnalysisPayload): Promise<StartAnalysisResponse> {
  const { apiUrl } = await getSettings();
  const { mediaUrl, pageUrl, mediaType } = payload;

  try {
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

    let c: CaseResponse | null = null;
    if (!isBrowserOnlyUrl(mediaUrl)) {
      try {
        c = await createCaseFromUrl(apiUrl, { mediaUrl, mediaType, title, notes });
      } catch (err) {
        const why = err instanceof Error ? err.message : 'unknown error';
        console.info('[Deep-Truth] server-side fetch declined, uploading instead:', why);
      }
    }

    if (!c) {
      const { blob, contentType } = await fetchMedia(mediaUrl, mediaType);
      c = await createCase(apiUrl, {
        blob,
        contentType,
        mediaType,
        title,
        notes,
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

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === 'START_ANALYSIS') {
    startAnalysis(message.payload as StartAnalysisPayload).then(sendResponse);
    return true;
  }

  if (message?.type === 'RESUME_POLLING') {
    void pollPending();
    sendResponse({ ok: true });
    return true;
  }

  return false;
});

void pollPending();

export {};
