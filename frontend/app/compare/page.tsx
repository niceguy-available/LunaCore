"use client";

import { useEffect, useState } from "react";
import { listJobs, compareJobs } from "@/lib/api";

export default function ComparePage() {
  const [jobs, setJobs] = useState<Array<{ job_id: string; status: string; quality_score?: number }>>([]);
  const [jobA, setJobA] = useState<string>("");
  const [jobB, setJobB] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState<{
    job_a: {
      job_id: string;
      config: { engine?: string; illumination_method?: string; feature_detector?: string; transform_type?: string };
      metrics: {
        quality_score?: number;
        image_metrics?: { rmse?: number; mae?: number; correlation?: number; ssim?: number; mutual_info?: number; gradient_corr?: number };
        match_stats?: { total_matches?: number; inlier_count?: number; inlier_ratio?: number };
        reprojection?: { rmse?: number; sub_pixel_ratio?: number };
        distribution?: { uniformity_score?: number };
      };
      processing_time: number;
      match_visualization?: string;
      registered_image?: string;
      checkerboard?: string;
    };
    job_b: {
      job_id: string;
      config: { engine?: string; illumination_method?: string; feature_detector?: string; transform_type?: string };
      metrics: {
        quality_score?: number;
        image_metrics?: { rmse?: number; mae?: number; correlation?: number; ssim?: number; mutual_info?: number; gradient_corr?: number };
        match_stats?: { total_matches?: number; inlier_count?: number; inlier_ratio?: number };
        reprojection?: { rmse?: number; sub_pixel_ratio?: number };
        distribution?: { uniformity_score?: number };
      };
      processing_time: number;
      match_visualization?: string;
      registered_image?: string;
      checkerboard?: string;
    };
    winner: Record<string, "a" | "b">;
  } | null>(null);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      const data = await listJobs();
      const completed = data.filter((j: { status: string }) => j.status === "complete");
      setJobs(completed);
      if (completed.length >= 2) {
        setJobA(completed[0].job_id);
        setJobB(completed[1].job_id);
      } else if (completed.length === 1) {
        setJobA(completed[0].job_id);
        setJobB(completed[0].job_id);
      }
    } catch {
      // Offline fallback
    }
  };

  const handleCompare = async () => {
    if (!jobA || !jobB) return;
    setLoading(true);
    try {
      const data = await compareJobs(jobA, jobB);
      setComparison(data);
    } catch {
      setComparison(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white tracking-tight">Engine & Pipeline Comparison</h1>
        <p className="text-gray-400 text-sm mt-1">
          Benchmark classical vs. deep-learning engines, or compare different illumination normalization strategies on the same lunar image pair.
        </p>
      </div>

      {/* Selectors */}
      <div className="glass-card p-6 mb-8">
        <div className="grid md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="text-xs font-semibold text-gray-300 mb-1.5 block">Pipeline Run A</label>
            <select
              value={jobA}
              onChange={(e) => setJobA(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500"
            >
              {jobs.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  Run #{j.job_id} {j.quality_score ? `(Score: ${j.quality_score.toFixed(1)})` : ""}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-gray-300 mb-1.5 block">Pipeline Run B</label>
            <select
              value={jobB}
              onChange={(e) => setJobB(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500"
            >
              {jobs.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  Run #{j.job_id} {j.quality_score ? `(Score: ${j.quality_score.toFixed(1)})` : ""}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleCompare}
            disabled={!jobA || !jobB || loading}
            className="btn-primary py-2.5 text-sm flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Comparing...</span>
              </>
            ) : (
              <span>⚖️ Run Side-by-Side Comparison</span>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {comparison && (
        <div className="space-y-8 animate-fade-in">
          {/* Header summary cards */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Run A card */}
            <div className={`glass-card p-6 border ${comparison.winner.overall === "a" ? "border-emerald-500/50 shadow-lg shadow-emerald-500/10" : "border-white/10"}`}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">Run A (Candidate)</span>
                  <h3 className="text-lg font-bold text-white mt-0.5">#{comparison.job_a.job_id}</h3>
                </div>
                {comparison.winner.overall === "a" && (
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    🏆 Overall Winner
                  </span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-2 text-center text-xs mb-4">
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Engine</div>
                  <div className="font-semibold text-white capitalize">{comparison.job_a.config?.engine || "Classical"}</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Detector</div>
                  <div className="font-semibold text-white uppercase">{comparison.job_a.config?.feature_detector || "SIFT"}</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Illum Method</div>
                  <div className="font-semibold text-white">{comparison.job_a.config?.illumination_method || "Phase Cong."}</div>
                </div>
              </div>

              <div className="text-center py-4 bg-black/30 rounded-xl mb-4">
                <div className="text-xs text-gray-400">Overall Quality Score</div>
                <div className="text-4xl font-extrabold font-mono text-emerald-400 mt-1">
                  {comparison.job_a.metrics?.quality_score?.toFixed(1) ?? "N/A"}
                </div>
              </div>

              {comparison.job_a.match_visualization && (
                <div className="rounded-xl overflow-hidden border border-white/5">
                  <img
                    src={`data:image/jpeg;base64,${comparison.job_a.match_visualization}`}
                    alt="Run A Matches"
                    className="w-full object-cover"
                  />
                </div>
              )}
            </div>

            {/* Run B card */}
            <div className={`glass-card p-6 border ${comparison.winner.overall === "b" ? "border-emerald-500/50 shadow-lg shadow-emerald-500/10" : "border-white/10"}`}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-violet-400">Run B (Benchmark)</span>
                  <h3 className="text-lg font-bold text-white mt-0.5">#{comparison.job_b.job_id}</h3>
                </div>
                {comparison.winner.overall === "b" && (
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    🏆 Overall Winner
                  </span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-2 text-center text-xs mb-4">
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Engine</div>
                  <div className="font-semibold text-white capitalize">{comparison.job_b.config?.engine || "Classical"}</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Detector</div>
                  <div className="font-semibold text-white uppercase">{comparison.job_b.config?.feature_detector || "SIFT"}</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02]">
                  <div className="text-gray-500 text-[10px]">Illum Method</div>
                  <div className="font-semibold text-white">{comparison.job_b.config?.illumination_method || "Phase Cong."}</div>
                </div>
              </div>

              <div className="text-center py-4 bg-black/30 rounded-xl mb-4">
                <div className="text-xs text-gray-400">Overall Quality Score</div>
                <div className="text-4xl font-extrabold font-mono text-emerald-400 mt-1">
                  {comparison.job_b.metrics?.quality_score?.toFixed(1) ?? "N/A"}
                </div>
              </div>

              {comparison.job_b.match_visualization && (
                <div className="rounded-xl overflow-hidden border border-white/5">
                  <img
                    src={`data:image/jpeg;base64,${comparison.job_b.match_visualization}`}
                    alt="Run B Matches"
                    className="w-full object-cover"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Metric-by-metric comparison table */}
          <div className="glass-card p-6">
            <h3 className="text-base font-bold text-white mb-4">Quantitative Metric Breakdown</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-gray-400 font-medium">
                    <th className="pb-3">Metric</th>
                    <th className="pb-3 text-center">Run A (#{comparison.job_a.job_id})</th>
                    <th className="pb-3 text-center">Run B (#{comparison.job_b.job_id})</th>
                    <th className="pb-3 text-center">Winner</th>
                    <th className="pb-3">Advantage Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-gray-300">
                  <tr>
                    <td className="py-3 font-semibold text-white">Cross-Correlation (NCC)</td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.correlation === "a" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_a.metrics?.image_metrics?.correlation?.toFixed(4) ?? "-"}
                    </td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.correlation === "b" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_b.metrics?.image_metrics?.correlation?.toFixed(4) ?? "-"}
                    </td>
                    <td className="py-3 text-center font-bold">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300">
                        {comparison.winner.correlation === "a" ? "Run A" : "Run B"}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-[11px]">Superior linear photometric alignment after registration.</td>
                  </tr>

                  <tr>
                    <td className="py-3 font-semibold text-white">Structural Similarity (SSIM)</td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.ssim === "a" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_a.metrics?.image_metrics?.ssim?.toFixed(4) ?? "-"}
                    </td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.ssim === "b" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_b.metrics?.image_metrics?.ssim?.toFixed(4) ?? "-"}
                    </td>
                    <td className="py-3 text-center font-bold">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300">
                        {comparison.winner.ssim === "a" ? "Run A" : "Run B"}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-[11px]">Better preservation of topography and crater edge structures.</td>
                  </tr>

                  <tr>
                    <td className="py-3 font-semibold text-white">Pixel RMSE (Residual Error)</td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.rmse === "a" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_a.metrics?.image_metrics?.rmse?.toFixed(2) ?? "-"}
                    </td>
                    <td className={`py-3 text-center font-mono ${comparison.winner.rmse === "b" ? "text-emerald-400 font-bold" : "text-gray-400"}`}>
                      {comparison.job_b.metrics?.image_metrics?.rmse?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-3 text-center font-bold">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300">
                        {comparison.winner.rmse === "a" ? "Run A" : "Run B"}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-[11px]">Lower absolute pixel residual error across overlapping area.</td>
                  </tr>

                  <tr>
                    <td className="py-3 font-semibold text-white">Inlier Match Count</td>
                    <td className="py-3 text-center font-mono text-gray-300">
                      {comparison.job_a.metrics?.match_stats?.inlier_count ?? 0}
                    </td>
                    <td className="py-3 text-center font-mono text-gray-300">
                      {comparison.job_b.metrics?.match_stats?.inlier_count ?? 0}
                    </td>
                    <td className="py-3 text-center font-bold">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300">
                        {(comparison.job_a.metrics?.match_stats?.inlier_count ?? 0) >= (comparison.job_b.metrics?.match_stats?.inlier_count ?? 0) ? "Run A" : "Run B"}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-[11px]">More robust geometric constraint over the scene.</td>
                  </tr>

                  <tr>
                    <td className="py-3 font-semibold text-white">Execution Latency</td>
                    <td className="py-3 text-center font-mono text-gray-300">
                      {comparison.job_a.processing_time?.toFixed(2)}s
                    </td>
                    <td className="py-3 text-center font-mono text-gray-300">
                      {comparison.job_b.processing_time?.toFixed(2)}s
                    </td>
                    <td className="py-3 text-center font-bold">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300">
                        {comparison.job_a.processing_time <= comparison.job_b.processing_time ? "Run A" : "Run B"}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-[11px]">Processing throughput suitable for real-time or onboard use.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
