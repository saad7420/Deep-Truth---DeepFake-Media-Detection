/* ============================================================================
   DEEP TRUTH EXTENSION — POPUP
   ----------------------------------------------------------------------------
   Three jobs: show whether the backend is reachable, let the operator point
   the extension at it, and list recent scans with links into the console.

   The "simulated mode" toggle is gone. It defaulted to on and filled the
   history with randomly generated verdicts, which meant the honest failure
   state — "the server isn't running" — was never visible. That state is now
   the first thing this panel reports.
   ========================================================================= */

import React, { useCallback, useEffect, useState } from 'react';
import './Popup.css';

import { checkHealth, normaliseApiUrl } from '../lib/api';
import { DEFAULT_SETTINGS, type CaseStatus, type ScanRecord } from '../types';

type Health = { state: 'checking' | 'ok' | 'down'; detail: string };

const VERDICT: Record<CaseStatus, { label: string; cls: string }> = {
  processing: { label: 'Analysing', cls: 'v-processing' },
  authentic: { label: 'Authentic', cls: 'v-authentic' },
  manipulated: { label: 'Manipulated', cls: 'v-manipulated' },
  inconclusive: { label: 'Inconclusive', cls: 'v-inconclusive' },
  failed: { label: 'Failed', cls: 'v-failed' },
};

function timeAgo(ts: number): string {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url.slice(0, 40);
  }
}

const Popup: React.FC = () => {
  const [isEnabled, setIsEnabled] = useState(DEFAULT_SETTINGS.isEnabled);
  const [apiUrl, setApiUrl] = useState(DEFAULT_SETTINGS.apiUrl);
  const [draftUrl, setDraftUrl] = useState(DEFAULT_SETTINGS.apiUrl);
  const [history, setHistory] = useState<ScanRecord[]>([]);
  const [health, setHealth] = useState<Health>({ state: 'checking', detail: 'Checking…' });
  const [showConfig, setShowConfig] = useState(false);
  const [notice, setNotice] = useState('');

  const runHealthCheck = useCallback(async (url: string) => {
    setHealth({ state: 'checking', detail: 'Checking…' });
    const res = await checkHealth(url);
    setHealth({ state: res.ok ? 'ok' : 'down', detail: res.detail });
  }, []);

  useEffect(() => {
    chrome.storage.local.get(['isEnabled', 'apiUrl', 'scanHistory'], (res) => {
      const url =
        typeof res.apiUrl === 'string' && res.apiUrl ? res.apiUrl : DEFAULT_SETTINGS.apiUrl;
      setIsEnabled(res.isEnabled !== undefined ? Boolean(res.isEnabled) : true);
      setApiUrl(url);
      setDraftUrl(url);
      setHistory(Array.isArray(res.scanHistory) ? res.scanHistory : []);
      void runHealthCheck(url);
    });

    const onMessage = (msg: { type?: string }) => {
      if (msg?.type === 'SCAN_UPDATED') {
        chrome.storage.local.get(['scanHistory'], (res) => {
          setHistory(Array.isArray(res.scanHistory) ? res.scanHistory : []);
        });
      }
    };
    chrome.runtime.onMessage.addListener(onMessage);
    return () => chrome.runtime.onMessage.removeListener(onMessage);
  }, [runHealthCheck]);

  const togglePower = () => {
    const next = !isEnabled;
    setIsEnabled(next);
    chrome.storage.local.set({ isEnabled: next }, () => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const id = tabs[0]?.id;
        if (id !== undefined) {
          chrome.tabs.sendMessage(id, { type: 'TOGGLE_SCANNING', enabled: next }).catch(() => {});
        }
      });
    });
  };

  const saveUrl = () => {
    const clean = normaliseApiUrl(draftUrl);
    if (!/^https?:\/\/.+/.test(clean)) {
      setNotice('Enter a full URL, including http:// or https://');
      setTimeout(() => setNotice(''), 4000);
      return;
    }
    setApiUrl(clean);
    setDraftUrl(clean);
    chrome.storage.local.set({ apiUrl: clean }, () => {
      setNotice('Server address saved');
      setTimeout(() => setNotice(''), 3000);
      void runHealthCheck(clean);
    });
  };

  const clearHistory = () => {
    chrome.storage.local.set({ scanHistory: [] }, () => setHistory([]));
  };

  const openCase = (caseId: string) => {
    // The console and the API share a host in the default local setup; the
    // console serves the case page, so swap the API port for the Next.js one
    // only when the operator has not moved things around.
    const consoleBase = apiUrl.replace(/:8000$/, ':3000');
    chrome.tabs.create({ url: `${consoleBase}/cases/${caseId}` });
  };

  return (
    <div className="dt-popup">
      <header className="dt-header">
        <div className="dt-brand">
          <span className="dt-mark" aria-hidden="true" />
          <div>
            <p className="dt-title">DEEP TRUTH</p>
            <p className="dt-subtitle">Active deepfake defense</p>
          </div>
        </div>

        <button
          type="button"
          className={`dt-power ${isEnabled ? 'on' : 'off'}`}
          onClick={togglePower}
          aria-pressed={isEnabled}
          title={isEnabled ? 'Scanning is on' : 'Scanning is off'}
        >
          <span className="dt-power-dot" />
          {isEnabled ? 'On' : 'Off'}
        </button>
      </header>

      {/* ── Connection ────────────────────────────────────────────────────── */}
      <section className={`dt-status dt-status-${health.state}`}>
        <span className="dt-status-dot" aria-hidden="true" />
        <div className="dt-status-body">
          <p className="dt-status-label">
            {health.state === 'ok'
              ? 'Analysis server connected'
              : health.state === 'checking'
                ? 'Contacting analysis server'
                : 'Analysis server unreachable'}
          </p>
          <p className="dt-status-detail">{health.detail}</p>
        </div>
        <button type="button" className="dt-link" onClick={() => void runHealthCheck(apiUrl)}>
          Retry
        </button>
      </section>

      {health.state === 'down' && (
        <p className="dt-hint">
          Start the backend with <code>python main.py</code> in <code>server/</code>, then press
          Retry. Nothing is analysed while the server is down — no results are estimated.
        </p>
      )}

      {/* ── Settings ──────────────────────────────────────────────────────── */}
      <section className="dt-section">
        <button
          type="button"
          className="dt-section-toggle"
          onClick={() => setShowConfig((s) => !s)}
          aria-expanded={showConfig}
        >
          <span>Server address</span>
          <span className="dt-chevron">{showConfig ? '▾' : '▸'}</span>
        </button>

        {showConfig && (
          <div className="dt-config">
            <label className="dt-label" htmlFor="dt-api-url">
              FastAPI origin
            </label>
            <input
              id="dt-api-url"
              className="dt-input"
              value={draftUrl}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setDraftUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && saveUrl()}
              placeholder="http://localhost:8000"
            />
            <p className="dt-help">
              Without the <code>/api</code> suffix. This is the address{' '}
              <code>server/main.py</code> prints on start-up.
            </p>
            <button type="button" className="dt-button" onClick={saveUrl}>
              Save address
            </button>
          </div>
        )}
        {notice && <p className="dt-notice">{notice}</p>}
      </section>

      {/* ── History ───────────────────────────────────────────────────────── */}
      <section className="dt-section dt-history">
        <div className="dt-history-head">
          <span>Recent scans</span>
          {history.length > 0 && (
            <button type="button" className="dt-link" onClick={clearHistory}>
              Clear
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="dt-empty">
            No scans yet. Open a page with images or video and click the Verify badge on any of
            them.
          </p>
        ) : (
          <ul className="dt-list">
            {history.map((r) => {
              const v = VERDICT[r.status] ?? VERDICT.failed;
              return (
                <li key={r.caseId} className="dt-item">
                  <div className="dt-item-top">
                    <span className={`dt-verdict ${v.cls}`}>{v.label}</span>
                    {r.experimental && (
                      <span
                        className="dt-experimental"
                        title="This engine is not yet validated — the result may be wrong."
                      >
                        unvalidated
                      </span>
                    )}
                    {typeof r.riskScore === 'number' && (
                      <span className="dt-score">{Math.round(r.riskScore)}%</span>
                    )}
                    <span className="dt-time">{timeAgo(r.timestamp)}</span>
                  </div>

                  <p className="dt-item-source">
                    <span className="dt-badge-type">{r.mediaType}</span>
                    {hostOf(r.pageUrl)}
                  </p>

                  {(r.error || r.explanation) && (
                    <p className="dt-item-detail">{r.error ?? r.explanation}</p>
                  )}

                  <button type="button" className="dt-link" onClick={() => openCase(r.caseId)}>
                    {r.caseId} — open in console
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
};

export default Popup;
