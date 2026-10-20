export type MediaType = 'image' | 'video' | 'audio';

export type CaseStatus =
  | 'processing'
  | 'authentic'
  | 'manipulated'
  | 'inconclusive'
  | 'failed';

export interface AnalysisRow {
  id: string;
  case_id: string;
  model_name: string;
  confidence: number;
  label: string;
  details?: Record<string, unknown> | null;
  created_at: string;
}

export interface JobState {
  state: 'queued' | 'running' | 'retrying' | 'succeeded' | 'failed' | 'cached';
  position?: number | null;
  attempt: number;
  maxAttempts: number;
  worker?: string | null;
  error?: string | null;
  cacheHit: boolean;
}

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
  sourceCaseId?: string | null;
  analysisResults?: Omit<AnalysisRow, 'id' | 'case_id' | 'created_at'>[];
}

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
  job?: JobState | null;
}

export interface ScanRecord {
  caseId: string;
  mediaUrl: string;
  pageUrl: string;
  mediaType: MediaType;
  status: CaseStatus;
  riskScore?: number;
  explanation?: string;
  error?: string;
  experimental?: boolean;
  fromCache?: boolean;
  timestamp: number;
}

export interface ExtensionSettings {
  isEnabled: boolean;
  apiUrl: string;
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  isEnabled: true,
  apiUrl: 'http://localhost:8000',
};

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
