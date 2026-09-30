"use client";

import { useEffect, useState } from "react";
import { listJobs, getJobResult, downloadMatches, exportReport, type JobResult } from "@/lib/api";

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Array<{ job_id: string; status: string; created_at: number; quality_score?: number }>>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      const data = await listJobs();
      setJobs(data);
      if (data.length > 0 && !selectedJobId) {
        selectJob(data[0].job_id);
      }
    } catch {
      // API might be offline or no jobs yet
    }
  };

  const selectJob = async (id: string) => {
    setSelectedJobId(id);
    setLoading(true);
    try {
      const res = await getJobResult(id);
      setJobResult(res);
    } catch {
      setJobResult(null);
    } finally {
      setLoading(false);
    }
  };

  const metrics = jobResult?.metrics;
  const distribution = metrics?.distribution;
  const reprojection = metrics?.reprojection;
  const matchStats = metrics?.match_stats;
  const imageMetrics = metrics?.image_metrics;
  const timing = metrics?.timing;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Evaluation Dashboard</h1>
          <p className="text-gray-400 text-sm mt-1">
            Quantitative assessment, spatial uniformity heatmaps, and exportable verification metrics.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadJobs}
            className="px-3 py-2 text-xs rounded-lg glass text-gray-300 hover:text-white transition-all flex items-center gap-1.5"
          >
            <span>🔄</span> Refresh Jobs
          </button>
          {selectedJobId && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => exportReport(selectedJobId, "json")}
                className="px-3 py-2 text-xs rounded-lg bg-indigo-600/30 border border-indigo-500/40 text-indigo-200 hover:bg-indigo-600/50 transition-all flex items-center gap-1.5"
              >
                <span>📄</span> Export JSON
              </button>
              <button
                onClick={() => exportReport(selectedJobId, "csv")}
                className="px-3 py-2 text-xs rounded-lg bg-indigo-600/30 border border-indigo-500/40 text-indigo-200 hover:bg-indigo-600/50 transition-all flex items-center gap-1.5"
              >
                <span>📊</span> Export CSV
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Left column: Run selector */}
        <div className="lg:col-span-1 space-y-4">
          <div className="glass-card p-4">
            <h2 className="text-sm font-semibold text-white mb-3 flex items-center justify-between">
              <span>Registration Runs</span>
              <span className="text-xs text-indigo-400 font-mono">{jobs.length} total</span>
            </h2>
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
              {jobs.length === 0 ? (
                <div className="text-xs text-gray-500 py-8 text-center">
                  No registration runs recorded yet.
                  <br />
                  <a href="/workbench" className="text-indigo-400 hover:underline mt-2 inline-block">
                    Run a job in Workbench →
                  </a>
                </div>
              ) : (
                jobs.map((j) => (
                  <button
                    key={j.job_id}
                    onClick={() => selectJob(j.job_id)}
                    className={`w-full text-left p-3 rounded-xl transition-all border text-xs ${
                      selectedJobId === j.job_id
                        ? "bg-indigo-500/20 border-indigo-500/50 shadow-lg shadow-indigo-500/10 text-white"
                        : "bg-white/[0.02] border-white/5 text-gray-400 hover:bg-white/[0.05] hover:text-gray-200"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-bold text-indigo-300">#{j.job_id}</span>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                        j.status === "complete" ? "bg-emerald-500/20 text-emerald-300" :
                        j.status === "failed" ? "bg-red-500/20 text-red-300" : "bg-amber-500/20 text-amber-300"
                      }`}>
                        {j.status}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-gray-500">
                      <span>Score: {j.quality_score != null ? `${j.quality_score.toFixed(1)}/100` : "N/A"}</span>
                      <span>{new Date(j.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right column: Metrics & Visualizations */}
        <div className="lg:col-span-3 space-y-6">
          {loading ? (
            <div className="glass-card p-16 text-center">
              <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="text-sm text-gray-400">Loading evaluation report...</p>
            </div>
          ) : !jobResult ? (
            <div className="glass-card p-16 text-center">
              <div className="text-4xl mb-3">📈</div>
              <h3 className="text-lg font-semibold text-white mb-1">Select a Run to Inspect</h3>
              <p className="text-sm text-gray-400 max-w-sm mx-auto">
                Detailed quantitative metrics, distribution heatmaps, and error histograms will appear here.
              </p>
            </div>
          ) : (
            <>
              {/* Primary KPI Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="glass-card p-4 text-center">
                  <div className="text-[11px] uppercase tracking-wider text-gray-400 mb-1">Quality Score</div>
                  <div className={`font-mono-data text-3xl font-bold ${
                    (metrics?.quality_score ?? 0) >= 75 ? "text-emerald-400" :
                    (metrics?.quality_score ?? 0) >= 50 ? "text-amber-400" : "text-red-400"
                  }`}>
                    {metrics?.quality_score?.toFixed(1) ?? "0.0"}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">out of 100 max</div>
                </div>

                <div className="glass-card p-4 text-center">
                  <div className="text-[11px] uppercase tracking-wider text-gray-400 mb-1">Inlier Ratio</div>
                  <div className="font-mono-data text-3xl font-bold text-indigo-300">
                    {matchStats ? `${(matchStats.inlier_ratio * 100).toFixed(1)}%` : "0%"}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">
                    {matchStats?.inlier_count ?? 0} / {matchStats?.total_matches ?? 0} verified
                  </div>
                </div>

                <div className="glass-card p-4 text-center">
                  <div className="text-[11px] uppercase tracking-wider text-gray-400 mb-1">Reproj RMSE</div>
                  <div className="font-mono-data text-3xl font-bold text-violet-300">
                    {reprojection?.rmse ? `${reprojection.rmse.toFixed(3)} px` : "0 px"}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">
                    {reprojection?.sub_pixel_ratio ? `${(reprojection.sub_pixel_ratio * 100).toFixed(0)}% sub-pixel` : "0% sub-pixel"}
                  </div>
                </div>

                <div className="glass-card p-4 text-center">
                  <div className="text-[11px] uppercase tracking-wider text-gray-400 mb-1">Uniformity Score</div>
                  <div className="font-mono-data text-3xl font-bold text-teal-300">
                    {distribution?.uniformity_score?.toFixed(1) ?? "0.0"}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">
                    {distribution?.occupied_cells ?? 0} / {distribution?.total_cells ?? 64} cells occupied
                  </div>
                </div>
              </div>

              {/* Spatial Uniformity Heatmap & Reprojection Histogram */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Heatmap */}
                <div className="glass-card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-sm font-bold text-white">Spatial Match Distribution</h3>
                      <p className="text-[11px] text-gray-400">8×8 Image Grid Occupancy (Uniformity requirement)</p>
                    </div>
                    <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded">
                      Entropy: {distribution?.entropy?.toFixed(2) ?? "0.0"}
                    </span>
                  </div>

                  {distribution?.grid && distribution.grid.length > 0 ? (
                    <div className="aspect-square w-full max-w-[280px] mx-auto grid grid-cols-8 gap-1 p-2 rounded-lg bg-black/40 border border-white/5">
                      {distribution.grid.flatMap((row, rIdx) =>
                        row.map((val, cIdx) => {
                          const maxVal = Math.max(...distribution.grid.flat(), 1);
                          const intensity = val / maxVal;
                          return (
                            <div
                              key={`${rIdx}-${cIdx}`}
                              title={`Cell (${rIdx},${cIdx}): ${val} matches`}
                              className="rounded-sm flex items-center justify-center text-[9px] font-mono transition-all hover:scale-110 cursor-help"
                              style={{
                                backgroundColor: val === 0
                                  ? "rgba(255, 255, 255, 0.03)"
                                  : `rgba(99, 102, 241, ${Math.max(0.2, intensity)})`,
                                color: intensity > 0.5 ? "#fff" : "#9898ac",
                              }}
                            >
                              {val > 0 ? val : ""}
                            </div>
                          );
                        })
                      )}
                    </div>
                  ) : (
                    <div className="h-48 flex items-center justify-center text-xs text-gray-500">
                      No grid distribution data available.
                    </div>
                  )}

                  <div className="flex items-center justify-between mt-3 text-[10px] text-gray-400">
                    <span>Low density (0 matches)</span>
                    <div className="w-24 h-2 rounded bg-gradient-to-r from-white/5 to-indigo-600" />
                    <span>High density</span>
                  </div>
                </div>

                {/* Reprojection Error Histogram */}
                <div className="glass-card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-sm font-bold text-white">Reprojection Error Distribution</h3>
                      <p className="text-[11px] text-gray-400">Residual pixel distances ||H·x - x&apos;||</p>
                    </div>
                    <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                      Mean: {reprojection?.mean?.toFixed(3) ?? "0"} px
                    </span>
                  </div>

                  {reprojection?.histogram_counts && reprojection.histogram_counts.length > 0 ? (
                    <div className="h-48 flex items-end gap-1 pt-6 pb-2 px-2 rounded-lg bg-black/40 border border-white/5">
                      {reprojection.histogram_counts.map((count, idx) => {
                        const maxCount = Math.max(...reprojection.histogram_counts, 1);
                        const heightPercent = (count / maxCount) * 100;
                        const binCenter = reprojection.histogram_bins?.[idx]?.toFixed(2) ?? idx;
                        return (
                          <div
                            key={idx}
                            className="flex-1 flex flex-col items-center group relative h-full justify-end"
                          >
                            <div
                              className="w-full rounded-t bg-gradient-to-t from-violet-600 to-indigo-400 hover:from-violet-500 hover:to-indigo-300 transition-all"
                              style={{ height: `${Math.max(4, heightPercent)}%` }}
                            />
                            <div className="absolute -top-6 hidden group-hover:block glass px-1.5 py-0.5 rounded text-[9px] font-mono text-white whitespace-nowrap z-10">
                              {count} pts ({binCenter}px)
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="h-48 flex items-center justify-center text-xs text-gray-500">
                      No reprojection histogram data.
                    </div>
                  )}

                  <div className="flex items-center justify-between mt-3 text-[10px] text-gray-400">
                    <span>0.0 px (Sub-pixel)</span>
                    <span>Max: {reprojection?.max?.toFixed(2) ?? 0} px</span>
                  </div>
                </div>
              </div>

              {/* Comprehensive Metric Table & Timing Breakdown */}
              <div className="grid md:grid-cols-3 gap-6">
                {/* Metric breakdown */}
                <div className="md:col-span-2 glass-card p-5">
                  <h3 className="text-sm font-bold text-white mb-4">Detailed Metrics Table</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 font-medium">
                          <th className="pb-2">Metric</th>
                          <th className="pb-2 font-mono">Value</th>
                          <th className="pb-2">Target / Ideal</th>
                          <th className="pb-2">Interpretation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5 text-gray-300">
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Cross-Correlation (NCC)</td>
                          <td className="py-2.5 font-mono text-emerald-400">{imageMetrics?.correlation?.toFixed(4) ?? "-"}</td>
                          <td className="py-2.5 text-gray-500">&gt; 0.85</td>
                          <td className="py-2.5 text-gray-400">Linear intensity alignment</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Structural Similarity (SSIM)</td>
                          <td className="py-2.5 font-mono text-emerald-400">{imageMetrics?.ssim?.toFixed(4) ?? "-"}</td>
                          <td className="py-2.5 text-gray-500">&gt; 0.70</td>
                          <td className="py-2.5 text-gray-400">Perceptual/topographical match</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Image RMSE</td>
                          <td className="py-2.5 font-mono text-indigo-300">{imageMetrics?.rmse?.toFixed(2) ?? "-"}</td>
                          <td className="py-2.5 text-gray-500">&lt; 35.0</td>
                          <td className="py-2.5 text-gray-400">Mean pixel intensity error</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Mutual Information (MI)</td>
                          <td className="py-2.5 font-mono text-indigo-300">{imageMetrics?.mutual_info?.toFixed(4) ?? "-"}</td>
                          <td className="py-2.5 text-gray-500">&gt; 0.50</td>
                          <td className="py-2.5 text-gray-400">Cross-modality entropy overlap</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Gradient Correlation</td>
                          <td className="py-2.5 font-mono text-indigo-300">{imageMetrics?.gradient_corr?.toFixed(4) ?? "-"}</td>
                          <td className="py-2.5 text-gray-500">&gt; 0.60</td>
                          <td className="py-2.5 text-gray-400">Crater rim / ridge boundary alignment</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 font-semibold text-white">Sub-Pixel Inlier Ratio</td>
                          <td className="py-2.5 font-mono text-emerald-400">
                            {reprojection?.sub_pixel_ratio ? `${(reprojection.sub_pixel_ratio * 100).toFixed(1)}%` : "-"}
                          </td>
                          <td className="py-2.5 text-gray-500">&gt; 70%</td>
                          <td className="py-2.5 text-gray-400">Matches with &lt; 1px reprojection</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Timing & Execution Breakdown */}
                <div className="glass-card p-5">
                  <h3 className="text-sm font-bold text-white mb-4">Pipeline Latency</h3>
                  <div className="space-y-3">
                    {timing ? (
                      Object.entries(timing).map(([stage, dur]) => {
                        if (stage === "total") return null;
                        const totalTime = timing.total || 1;
                        const percent = (dur / totalTime) * 100;
                        return (
                          <div key={stage} className="text-xs">
                            <div className="flex justify-between text-gray-300 mb-1">
                              <span className="capitalize">{stage.replace("_", " ")}</span>
                              <span className="font-mono text-gray-400">{dur.toFixed(3)}s ({percent.toFixed(0)}%)</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                              <div
                                className="h-full bg-indigo-500 rounded-full"
                                style={{ width: `${percent}%` }}
                              />
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-xs text-gray-500">No timing data.</div>
                    )}

                    <div className="pt-3 border-t border-white/10 flex justify-between items-center text-xs font-bold text-white">
                      <span>Total Time</span>
                      <span className="font-mono text-indigo-300">{jobResult.processing_time.toFixed(2)}s</span>
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-white/10">
                    <button
                      onClick={() => downloadMatches(jobResult.job_id, "csv")}
                      className="w-full btn-secondary text-xs py-2 mb-2 flex items-center justify-center gap-1.5"
                    >
                      <span>📥</span> Download Matches CSV
                    </button>
                    <button
                      onClick={() => downloadMatches(jobResult.job_id, "geojson")}
                      className="w-full btn-secondary text-xs py-2 flex items-center justify-center gap-1.5"
                    >
                      <span>🗺️</span> Download Matches GeoJSON
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
