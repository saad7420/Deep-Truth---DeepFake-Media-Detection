/* ============================================================================
   DEEP TRUTH EXTENSION — CONTENT SCRIPT
   ----------------------------------------------------------------------------
   Finds media on the page, offers to verify it, and shows the verdict in
   place.

   Positioning approach: every badge lives in one fixed-position layer
   appended to <body>, and is placed from the element's viewport rect on a
   rAF-throttled scroll/resize pass. The previous version injected an
   absolutely-positioned wrapper into each element's parent, which inherits
   the parent's transforms, clipping and stacking context — that is why
   badges drifted or vanished inside the scroll containers on X and
   Instagram. One layer, viewport coordinates, no inherited context.
   ========================================================================= */

import type { CaseStatus, MediaType, ScanRecord, StartAnalysisResponse } from '../types';

const LAYER_ID = 'deep-truth-overlay-layer';
/** Anything smaller than this is an avatar, icon or tracking pixel. */
const MIN_EDGE_PX = 120;

type MediaEl = HTMLVideoElement | HTMLImageElement | HTMLAudioElement;

interface Tracked {
  el: MediaEl;
  badge: HTMLButtonElement;
  mediaType: MediaType;
  /** Set once a scan is opened, so pushed updates can be matched back. */
  caseId?: string;
  resolvedUrl?: string;
}

const tracked = new Map<MediaEl, Tracked>();
let layer: HTMLDivElement | null = null;
let observer: MutationObserver | null = null;
let scanning = true;
let rafPending = false;

const contextValid = (): boolean =>
  typeof chrome !== 'undefined' && !!chrome.runtime && !!chrome.runtime.id;

/* ── Verdict presentation ────────────────────────────────────────────────────
   Colours and thresholds mirror the console's VERDICT map so the extension
   and the web app never disagree about what a score means. */

const PALETTE: Record<string, { fg: string; bg: string; border: string; text: string }> = {
  idle: { fg: '#c7d2fe', bg: 'rgba(67,56,202,0.92)', border: '#6366f1', text: 'Verify' },
  processing: { fg: '#fff', bg: 'rgba(2,132,199,0.94)', border: '#38bdf8', text: 'Analysing' },
  authentic: { fg: '#fff', bg: 'rgba(4,120,87,0.94)', border: '#34d399', text: 'Authentic' },
  manipulated: { fg: '#fff', bg: 'rgba(185,28,28,0.94)', border: '#f87171', text: 'Manipulated' },
  inconclusive: { fg: '#fff', bg: 'rgba(180,83,9,0.94)', border: '#fbbf24', text: 'Inconclusive' },
  failed: { fg: '#e2e8f0', bg: 'rgba(51,65,85,0.94)', border: '#94a3b8', text: 'Failed' },
};

function paint(badge: HTMLButtonElement, key: string, label: string, title?: string): void {
  const p = PALETTE[key] ?? PALETTE.idle;
  badge.textContent = label;
  badge.style.color = p.fg;
  badge.style.background = p.bg;
  badge.style.borderColor = p.border;
  badge.title = title ?? label;
}

/* ── Media URL resolution ────────────────────────────────────────────────── */

function mediaTypeOf(el: MediaEl): MediaType {
  if (el.tagName === 'VIDEO') return 'video';
  if (el.tagName === 'AUDIO') return 'audio';
  return 'image';
}

/**
 * Work out a URL the service worker can actually download.
 *
 * Returns null with a reason for media the extension genuinely cannot fetch,
 * rather than starting a scan that is bound to fail. MSE-backed players
 * (YouTube, most streaming sites) expose only a `blob:` URL that is valid
 * inside the page and meaningless anywhere else, so those are declined up
 * front with an explanation.
 */
function resolveUrl(el: MediaEl): { url: string } | { error: string } {
  let src = '';

  if (el.tagName === 'IMG') {
    const img = el as HTMLImageElement;
    // currentSrc reflects what the browser picked from srcset.
    src = img.currentSrc || img.src || '';
  } else {
    const media = el as HTMLVideoElement | HTMLAudioElement;
    src = media.currentSrc || media.src || '';
    if (!src) {
      const source = media.querySelector('source');
      src = source?.src ?? '';
    }
  }

  if (!src) return { error: 'This element has no downloadable source.' };
  if (src.startsWith('blob:')) {
    return {
      error:
        'This is a streamed player, not a file. The extension can only verify media served as a plain file — download the clip and upload it in the console instead.',
    };
  }
  if (src.startsWith('data:')) {
    return { error: 'Inline data URLs are not supported. Upload the file in the console instead.' };
  }
  return { url: src };
}

/* ── Overlay layer ───────────────────────────────────────────────────────── */

function ensureLayer(): HTMLDivElement {
  if (layer && layer.isConnected) return layer;
  const el = document.createElement('div');
  el.id = LAYER_ID;
  Object.assign(el.style, {
    position: 'fixed',
    inset: '0',
    pointerEvents: 'none',
    // Above almost everything, below native browser UI.
    zIndex: '2147483000',
  } satisfies Partial<CSSStyleDeclaration>);
  document.body.appendChild(el);
  layer = el;
  return el;
}

function makeBadge(): HTMLButtonElement {
  const badge = document.createElement('button');
  badge.type = 'button';
  badge.className = 'deep-truth-badge';
  Object.assign(badge.style, {
    position: 'fixed',
    pointerEvents: 'auto',
    cursor: 'pointer',
    font: '600 11px/1.4 ui-sans-serif, system-ui, -apple-system, sans-serif',
    letterSpacing: '0.02em',
    padding: '4px 9px',
    borderRadius: '6px',
    borderWidth: '1px',
    borderStyle: 'solid',
    boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
    backdropFilter: 'blur(6px)',
    transition: 'background-color .2s, border-color .2s, transform .12s',
    maxWidth: '220px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    // Never inherit the host page's text styling.
    textTransform: 'none',
    margin: '0',
  } satisfies Partial<CSSStyleDeclaration>);

  badge.addEventListener('mouseenter', () => (badge.style.transform = 'translateY(-1px)'));
  badge.addEventListener('mouseleave', () => (badge.style.transform = 'none'));
  return badge;
}

/** Place every badge from its element's current viewport rect. */
function reposition(): void {
  tracked.forEach((t) => {
    if (!t.el.isConnected) {
      detach(t.el);
      return;
    }
    const r = t.el.getBoundingClientRect();

    // Off-screen or collapsed: hide rather than pile badges at the origin.
    const offscreen =
      r.bottom < 0 || r.top > window.innerHeight || r.right < 0 || r.left > window.innerWidth;
    if (offscreen || r.width === 0 || r.height === 0) {
      t.badge.style.visibility = 'hidden';
      return;
    }

    t.badge.style.visibility = 'visible';
    t.badge.style.left = `${Math.round(r.left + 8)}px`;
    t.badge.style.top = `${Math.round(r.top + 8)}px`;
  });
}

function requestReposition(): void {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    reposition();
  });
}

/* ── Attach / detach ─────────────────────────────────────────────────────── */

function eligible(el: MediaEl): boolean {
  if (tracked.has(el)) return false;
  if (el.closest(`#${LAYER_ID}`)) return false;

  const r = el.getBoundingClientRect();
  // Audio elements are controls-sized by nature — exempt them from the
  // dimension filter that screens out image thumbnails and icons.
  if (el.tagName === 'AUDIO') return r.width > 0;
  return r.width >= MIN_EDGE_PX && r.height >= MIN_EDGE_PX;
}

function attach(el: MediaEl): void {
  if (!eligible(el)) return;

  const mediaType = mediaTypeOf(el);
  const badge = makeBadge();
  const entry: Tracked = { el, badge, mediaType };

  paint(badge, 'idle', '⚡ Verify with Deep Truth');

  badge.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    void beginScan(entry);
  });

  ensureLayer().appendChild(badge);
  tracked.set(el, entry);
  requestReposition();
}

function detach(el: MediaEl): void {
  const t = tracked.get(el);
  if (!t) return;
  t.badge.remove();
  tracked.delete(el);
}

function clearAll(): void {
  tracked.forEach((t) => t.badge.remove());
  tracked.clear();
  layer?.remove();
  layer = null;
}

/* ── Scanning ────────────────────────────────────────────────────────────── */

async function beginScan(entry: Tracked): Promise<void> {
  if (!contextValid()) return;

  const resolved = resolveUrl(entry.el);
  if ('error' in resolved) {
    paint(entry.badge, 'failed', '⚠ Cannot verify', resolved.error);
    return;
  }

  entry.resolvedUrl = resolved.url;
  paint(entry.badge, 'processing', '◌ Uploading…', 'Sending this media to the analysis server');
  entry.badge.disabled = true;

  let response: StartAnalysisResponse | undefined;
  try {
    response = await chrome.runtime.sendMessage({
      type: 'START_ANALYSIS',
      payload: {
        mediaUrl: resolved.url,
        pageUrl: window.location.href,
        mediaType: entry.mediaType,
      },
    });
  } catch {
    response = { ok: false, error: 'The extension was reloaded. Refresh the page and try again.' };
  }

  entry.badge.disabled = false;

  if (!response?.ok) {
    paint(entry.badge, 'failed', '⚠ Failed', response?.error ?? 'Analysis could not be started.');
    return;
  }

  entry.caseId = response.caseId;
  paint(entry.badge, 'processing', '◌ Analysing…', `Case ${response.caseId} is running`);
}

/** Render a pushed update from the service worker. */
function applyUpdate(record: ScanRecord): void {
  tracked.forEach((t) => {
    const matches =
      (t.caseId && t.caseId === record.caseId) ||
      (!t.caseId && t.resolvedUrl && t.resolvedUrl === record.mediaUrl);
    if (!matches) return;

    t.caseId = record.caseId;

    if (record.status === 'processing') {
      paint(t.badge, 'processing', '◌ Analysing…', `Case ${record.caseId}`);
      return;
    }

    const status: CaseStatus = record.status;
    const p = PALETTE[status] ?? PALETTE.failed;
    const score = typeof record.riskScore === 'number' ? ` · ${Math.round(record.riskScore)}%` : '';
    const icon =
      status === 'manipulated'
        ? '⛔'
        : status === 'authentic'
          ? '✓'
          : status === 'inconclusive'
            ? '?'
            : '⚠';

    // An unvalidated engine gets a visible marker, not just a longer
    // tooltip — someone glancing at a red badge reading "Manipulated 99%"
    // should be able to see that the number is provisional without hovering.
    const flask = record.experimental ? ' \u{1F9EA}' : '';
    const detail = [
      record.experimental
        ? 'UNVALIDATED MODEL — provisional result, may report real media as fake.'
        : null,
      record.error ?? record.explanation ?? `Case ${record.caseId}`,
    ]
      .filter(Boolean)
      .join('\n\n');

    paint(t.badge, status, `${icon} ${p.text}${score}${flask}`, detail);
  });
}

/* ── DOM discovery ───────────────────────────────────────────────────────── */

function scan(): void {
  if (!scanning) return;
  document.querySelectorAll<MediaEl>('video, img, audio').forEach(attach);
  requestReposition();
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

let scanTimer: ReturnType<typeof setInterval> | null = null;

function startScanning(): void {
  scanning = true;
  scan();

  observer?.disconnect();
  observer = new MutationObserver(() => {
    if (contextValid()) scan();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // Catch elements that only gain their dimensions after media metadata
  // loads, which no mutation event reports.
  if (scanTimer === null) {
    scanTimer = setInterval(() => {
      if (contextValid()) scan();
    }, 3000);
  }

  window.addEventListener('scroll', requestReposition, { passive: true, capture: true });
  window.addEventListener('resize', requestReposition, { passive: true });
}

function stopScanning(): void {
  scanning = false;
  observer?.disconnect();
  observer = null;
  if (scanTimer !== null) {
    clearInterval(scanTimer);
    scanTimer = null;
  }
  window.removeEventListener('scroll', requestReposition, { capture: true });
  window.removeEventListener('resize', requestReposition);
  clearAll();
}

function init(): void {
  if (!contextValid()) return;

  chrome.storage.local.get(['isEnabled'], (res) => {
    if (res.isEnabled === false) {
      scanning = false;
      return;
    }
    startScanning();
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (!contextValid()) return;
    if (msg?.type === 'TOGGLE_SCANNING') {
      if (msg.enabled) startScanning();
      else stopScanning();
    } else if (msg?.type === 'SCAN_UPDATED') {
      applyUpdate(msg.payload as ScanRecord);
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export {};
