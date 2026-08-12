/* ============================================================================
   DEEP TRUTH EXTENSION — TYPES
   ----------------------------------------------------------------------------
   These mirror server/app/models.py. The previous version of this file
   described a different service entirely — taskId/trustScore/verdict against
   `https://api.deep-truth.ai/api/v1/...` — none of which the backend in this
   repo implements. The real API is:

     POST /api/cases          multipart: title, media_type, file -> CaseResponse
     GET  /api/cases/{caseId} poll until status leaves "processing"

   so a case id, a status and a 0-100 risk score are what actually travel.
   ========================================================================= */

/** Matches MediaType in server/app/models.py. */
export type MediaType = 'image' | 'video' | 'audio';

/** Matches CaseStatus in server/app/models.py. */
export type CaseStatus =
  | 'processing'
  | 'authentic'
  | 'manipulated'
  | 'inconclusive'
  | 'failed';

/** One row from CaseResponse.analysisResults (snake_case on the wire). */
export interface AnalysisRow {
  id: string;
  case_id: string;
  model_name: string;
  confidence: number;
  label: string;
  details?: Record<string, unknown> | null;
  created_at: string;
}

/** Matches JobState in server/app/models.py — what the *job* is doing, as
    opposed to what the analysis concluded. Null on cases older than the
    queue's state retention. */
export interface JobState {
  state: 'queued' | 'running' | 'retrying' | 'succeeded' | 'failed' | 'cached';
  position?: number | null;
  attempt: number;
  maxAttempts: number;
  worker?: string | null;
  error?: string | null;
  cacheHit: boolean;
}

/** Reply from GET /api/cache/lookup. A miss is a normal 200 with hit=false. */
export interface CacheLookup {
  hit: boolean;
  mediaType: MediaType;
  contentHash?: string | null;
  normalisedUrl?: string | null;
  status?: CaseStatus;
  riskScore?: number;
  syntheticLikelihood?: number;
  computedAt?: number | null;
  sourceCaseDbId?: string | null;
  /** Public CASE-XXXXXXXX id of the case that produced this verdict. Absent
      for entries cached before the server started recording it. */
  sourceCaseId?: string | null;
  analysisResults?: Omit<AnalysisRow, 'id' | 'case_id' | 'created_at'>[];
}

/** The server's CaseResponse, camelCase as serialised. */
export interface CaseResponse {
  id: string;
  caseId: string;
  title: string;
  mediaType: MediaType;
  status: CaseStatus;
  riskScore: number;
  syntheticLikelihood: number;
  fileName?: string | null;
  fileUrl?: string | null;
  fileSize?: number | null;
  notes?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
  analysisResults: AnalysisRow[];
  /** Live queue state. Absent on older servers; treat as "no information". */
  job?: JobState | null;
}

/** What the extension keeps in chrome.storage.local per scanned element. */
export interface ScanRecord {
  /** The public CASE-XXXXXXXX id, so the record links back to the console. */
  caseId: string;
  /** Source URL of the media element this scan came from. */
  mediaUrl: string;
  /** Origin page, for the history list. */
  pageUrl: string;
  mediaType: MediaType;
  status: CaseStatus;
  /** 0-100. Undefined while still processing. */
  riskScore?: number;
  /** One-line reason, lifted from the engine's `rationale` when present. */
  explanation?: string;
  /** Set when the scan never reached the server at all. */
  error?: string;
  /** True when the engine that produced this verdict is not yet validated. */
  experimental?: boolean;
  /**
   * True when this verdict came from the server's cache rather than a fresh
   * analysis — the media had been analysed before, so nothing was downloaded,
   * uploaded, or recomputed. `caseId` then points at the earlier case, which
   * means several scan records can share one case id.
   */
  fromCache?: boolean;
  timestamp: number;
}

export interface ExtensionSettings {
  /** Master switch for DOM scanning. */
  isEnabled: boolean;
  /** Origin of the FastAPI service, WITHOUT the /api suffix. */
  apiUrl: string;
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  isEnabled: true,
  // The backend this extension talks to runs locally by default — the same
  // host:port server/main.py binds to. There is no hosted Deep Truth API.
  apiUrl: 'http://localhost:8000',
};

/* ── Runtime messages ────────────────────────────────────────────────────── */

export interface StartAnalysisPayload {
  mediaUrl: string;
  pageUrl: string;
  mediaType: MediaType;
}

export interface StartAnalysisResponse {
  ok: boolean;
  caseId?: string;
  error?: string;
}
