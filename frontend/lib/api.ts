const API_BASE = typeof window !== "undefined"
  ? (process.env.NEXT_PUBLIC_API_URL || "/api")
  : (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api");

export interface RegistrationConfig {
  engine: "classical" | "deep" | "hybrid";
  illumination_method: "phase_congruency" | "dol" | "wld" | "clahe" | "combined";
  feature_detector: "sift" | "orb" | "akaze" | "rift";
  matcher: string;
  ratio_threshold: number;
  ransac_method: string;
  ransac_threshold: number;
  transform_type: "homography" | "affine" | "rigid" | "similarity" | "tps" | "polynomial";
  grid_size: number;
  max_per_cell: number;
  subpixel_method: "parabolic" | "lucas_kanade" | "phase_correlation" | "combined";
  refine_intensity: boolean;
  multiscale: boolean;
  source_sensor?: string;
  reference_sensor?: string;
}

export const DEFAULT_CONFIG: RegistrationConfig = {
  engine: "classical",
  illumination_method: "phase_congruency",
  feature_detector: "sift",
  matcher: "flann",
  ratio_threshold: 0.75,
  ransac_method: "usac_magsac",
  ransac_threshold: 5.0,
  transform_type: "homography",
  grid_size: 8,
  max_per_cell: 80,
  subpixel_method: "parabolic",
  refine_intensity: true,
  multiscale: false,
};

export interface JobResult {
  job_id: string;
  status: string;
  source_image: string;
  reference_image: string;
  registered_image: string;
  checkerboard: string;
  blended: string;
  match_visualization: string;
  transformation_matrix: number[][] | null;
  transformation_params: Record<string, unknown>;
  metrics: {
    quality_score: number;
    image_metrics: {
      rmse: number;
      mae: number;
      correlation: number;
      ssim: number;
      mutual_info: number;
      gradient_corr: number;
    };
    match_stats: {
      total_matches: number;
      inlier_count: number;
      inlier_ratio: number;
    };
    reprojection: {
      rmse: number;
      mean: number;
      median: number;
      max: number;
      sub_pixel_ratio: number;
      histogram_bins: number[];
      histogram_counts: number[];
    };
    distribution: {
      entropy: number;
      normalized_entropy: number;
      cv: number;
      occupied_cells: number;
      total_cells: number;
      occupancy_ratio: number;
      uniformity_score: number;
      grid: number[][];
    };
    timing: Record<string, number>;
  };
  match_points: Array<{src_x: number; src_y: number; dst_x: number; dst_y: number}>;
  keypoint_counts: {source: number; reference: number};
  subpixel_stats: Record<string, unknown>;
  processing_time: number;
  config: RegistrationConfig;
}

export interface StageEvent {
  stage: string;
  status: string;
  data: Record<string, unknown>;
  timestamp: number;
}

// Submit registration job
export async function submitRegistration(
  sourceFile: File,
  referenceFile: File,
  config: RegistrationConfig,
  sourceSensor?: string,
  referenceSensor?: string,
): Promise<{job_id: string}> {
  const formData = new FormData();
  formData.append("source", sourceFile);
  formData.append("reference", referenceFile);
  formData.append("config_json", JSON.stringify(config));
  if (sourceSensor) formData.append("source_sensor", sourceSensor);
  if (referenceSensor) formData.append("reference_sensor", referenceSensor);

  const res = await fetch(`${API_BASE}/register`, {method: "POST", body: formData});
  if (!res.ok) throw new Error(`Registration failed: ${res.statusText}`);
  return res.json();
}

// Stream job progress via SSE
export function streamJobProgress(
  jobId: string,
  onEvent: (event: StageEvent) => void,
  onComplete: () => void,
  onError: (err: string) => void,
): () => void {
  const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/stream`);

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as StageEvent;
      onEvent(data);
      if (data.stage === "final") {
        eventSource.close();
        onComplete();
      }
    } catch {
      // ignore parse errors
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    onError("SSE connection error");
  };

  return () => eventSource.close();
}

// Get job result
export async function getJobResult(jobId: string): Promise<JobResult> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/result`);
  if (!res.ok) throw new Error(`Failed to get result: ${res.statusText}`);
  return res.json();
}

// Get job status
export async function getJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to get status: ${res.statusText}`);
  return res.json();
}

// Compare two jobs
export async function compareJobs(jobA: string, jobB: string) {
  const res = await fetch(`${API_BASE}/compare?job_a=${jobA}&job_b=${jobB}`);
  if (!res.ok) throw new Error(`Comparison failed: ${res.statusText}`);
  return res.json();
}

// List jobs
export async function listJobs() {
  const res = await fetch(`${API_BASE}/jobs`);
  if (!res.ok) throw new Error(`Failed to list jobs: ${res.statusText}`);
  return res.json();
}

// Download matches
export async function downloadMatches(jobId: string, format: "csv" | "geojson" = "csv") {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/matches?format=${format}`);
  if (!res.ok) throw new Error(`Download failed: ${res.statusText}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `matches_${jobId}.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

// Export report
export async function exportReport(jobId: string, format: "json" | "csv" = "json") {
  const res = await fetch(`${API_BASE}/export/${jobId}?format=${format}`);
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `report_${jobId}.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

// Get sensor list
export async function getSensors() {
  const res = await fetch(`${API_BASE}/sensors`);
  if (!res.ok) throw new Error(`Failed to get sensors: ${res.statusText}`);
  return res.json();
}
