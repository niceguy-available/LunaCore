"use client";

import { useEffect, useState, useRef, useCallback, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getJobResult, downloadMatches, type JobResult } from "@/lib/api";

const METRIC_TOOLTIPS: Record<string, string> = {
  rmse: "Root Mean Square Error — lower is better. Measures average pixel intensity difference.",
  correlation: "Normalized Cross-Correlation — higher is better (max 1.0). Measures linear similarity.",
  ssim: "Structural Similarity Index — higher is better (max 1.0). Measures structural similarity.",
  inlier_ratio: "Inlier Ratio — higher is better. Fraction of matches passing geometric verification.",
  uniformity_score: "Distribution Uniformity — higher is better (max 100). Measures evenness across grid.",
  reproj_rmse: "Reprojection RMSE — lower is better. Error when projecting source points through transform.",
  sub_pixel_ratio: "Sub-Pixel Ratio — higher is better. Percentage of matches with < 1.0 px error.",
};

/* ── Interactive Swipe Compare Slider ──────────────────────── */
function CompareSlider({ imageA, imageB, labelA = "Reference", labelB = "Registered" }: {
  imageA: string; imageB: string; labelA?: string; labelB?: string;
}) {
  const [pos, setPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(0, Math.min(100, x)));
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (dragging.current) handleMove(e.clientX); };
    const onUp = () => { dragging.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [handleMove]);

  return (
    <div ref={containerRef} className="compare-slider relative rounded-xl overflow-hidden cursor-ew-resize select-none border border-[#A2D5C6]/30"
      onMouseDown={(e) => { dragging.current = true; handleMove(e.clientX); }}
      onTouchMove={(e) => handleMove(e.touches[0].clientX)}
    >
      <img src={`data:image/jpeg;base64,${imageA}`} alt={labelA} className="w-full h-auto block" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={`data:image/jpeg;base64,${imageB}`} alt={labelB} className="w-full h-auto block" draggable={false} />
      </div>
      <div className="compare-handle" style={{ left: `${pos}%` }} />
      <span className="absolute top-3 left-3 text-xs bg-black/80 border border-[#A2D5C6]/40 px-2.5 py-1 rounded text-[#F6F6F6]">{labelA}</span>
      <span className="absolute top-3 right-3 text-xs bg-black/80 border border-[#A2D5C6]/40 px-2.5 py-1 rounded text-[#CFFFE2]">{labelB}</span>
    </div>
  );
}

/* ── Metric Card Component ─────────────────────────────────── */
function MetricCard({ label, value, format = "float", tooltip }: {
  label: string; value: number; format?: "float" | "percent" | "int"; tooltip?: string;
}) {
  let display: string;
  let color = "text-[#F6F6F6]";

  switch (format) {
    case "percent":
      display = `${(value * 100).toFixed(1)}%`;
      color = value > 0.5 ? "text-[#CFFFE2]" : value > 0.25 ? "text-[#A2D5C6]" : "text-amber-400";
      break;
    case "int":
      display = Math.round(value).toString();
      color = value > 10 ? "text-[#CFFFE2]" : "text-[#A2D5C6]";
      break;
    default:
      display = value.toFixed(4);
      if (label.toLowerCase().includes("rmse")) {
        color = value < 2.0 ? "text-[#CFFFE2]" : value < 5.0 ? "text-[#A2D5C6]" : "text-amber-400";
      } else {
        color = value > 0.6 ? "text-[#CFFFE2]" : value > 0.3 ? "text-[#A2D5C6]" : "text-[#F6F6F6]";
      }
  }

  return (
    <div className="lunar-card p-5 text-center relative group">
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 p-2 bg-[#000000] border border-[#A2D5C6]/40 rounded-lg text-xs text-[#F6F6F6] whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30 shadow-lg">
          {tooltip}
        </div>
      )}
      <div className={`font-mono text-2xl font-bold ${color}`}>{display}</div>
      <div className="text-xs text-[#F6F6F6]/70 mt-1 uppercase tracking-wider">{label}</div>
    </div>
  );
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const jobId = searchParams.get("job_id");

  const [result, setResult] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [viewMode, setViewMode] = useState<"swipe" | "matches" | "checkerboard" | "blend">("swipe");
  const [sourcePreview, setSourcePreview] = useState<string>("");
  const [referencePreview, setReferencePreview] = useState<string>("");

  useEffect(() => {
    // Load cached preview thumbnails if available
    const cachedSrc = sessionStorage.getItem("luna_source_preview");
    const cachedRef = sessionStorage.getItem("luna_ref_preview");
    if (cachedSrc) setSourcePreview(cachedSrc);
    if (cachedRef) setReferencePreview(cachedRef);

    if (!jobId) {
      // If no job_id in URL, check cached result or redirect
      const cachedResult = sessionStorage.getItem("luna_last_result");
      if (cachedResult) {
        try {
          const parsed = JSON.parse(cachedResult);
          setResult(parsed);
          setLoading(false);
          return;
        } catch {}
      }
      setError("No registration job specified. Please run a job from the workbench.");
      setLoading(false);
      return;
    }

    const fetchResult = async () => {
      try {
        const data = await getJobResult(jobId);
        if (data.status === "failed") {
          setError(data.error || "Registration could not converge on this image pair.");
        } else {
          setResult(data);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load registration result.");
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [jobId]);

  if (loading) {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center text-center px-4">
        <div className="w-12 h-12 border-3 border-[#A2D5C6]/30 border-t-[#A2D5C6] rounded-full animate-spin mb-4" />
        <h2 className="text-2xl font-bold text-[#F6F6F6]">Processing Registration Results...</h2>
        <p className="text-[#F6F6F6]/70 mt-2">Synthesizing multi-modal correspondences and sub-pixel metrics</p>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center text-center px-4 max-w-xl mx-auto">
        <div className="lunar-card p-8 border-red-400/50 w-full text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-[#F6F6F6] mb-3">Registration Notice</h2>
          <p className="text-[#F6F6F6]/80 text-sm mb-6 leading-relaxed">{error}</p>
          <button onClick={() => router.push("/workbench")} className="btn-secondary-outline text-sm">
            ← Return to Workbench
          </button>
        </div>
      </div>
    );
  }

  const quality = result.metrics?.quality_score ?? 0;
  const inliers = result.metrics?.match_stats?.inlier_count ?? 0;

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      {/* Top Header Row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-1 block">
            Analysis Complete
          </span>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-[#F6F6F6] tracking-tight">
            Registration Results
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <Link href="/workbench" className="btn-secondary-outline text-sm px-5 py-2.5">
            ← New Registration
          </Link>
          <button
            onClick={() => downloadMatches(result.job_id, "csv")}
            className="btn-primary text-sm px-5 py-2.5 bg-[#A2D5C6] text-black font-semibold rounded-full hover:bg-[#CFFFE2] transition-colors"
          >
            📥 Export CSV
          </button>
        </div>
      </div>

      {/* ======================================================== */}
      {/* TOP / CENTER: PRIMARY RESULT OUTPUT                      */}
      {/* ======================================================== */}
      <div className="space-y-8 mb-16">
        {/* Quality Score Hero Banner */}
        <div className="lunar-card p-8 text-center bg-gradient-to-b from-[#A2D5C6]/10 to-transparent">
          <div className="text-xs uppercase tracking-widest text-[#A2D5C6] mb-2 font-semibold">
            Overall Quality Score
          </div>
          <div className="font-mono text-6xl sm:text-7xl font-extrabold text-[#CFFFE2] my-2">
            {quality.toFixed(1)}
            <span className="text-2xl text-[#F6F6F6]/50 font-normal"> / 100</span>
          </div>
          <p className="text-sm text-[#F6F6F6]/80 font-mono mt-3">
            {result.processing_time?.toFixed(2)}s runtime · {inliers} verified geometric inliers
          </p>
        </div>

        {/* Primary Interactive Viewer */}
        <div className="lunar-card p-6">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
            <h2 className="text-lg font-bold text-[#CFFFE2]">Co-Registered Visual Output</h2>
            {/* View Mode Tabs */}
            <div className="flex items-center gap-2 bg-[#000000] p-1 rounded-xl border border-[#A2D5C6]/30">
              {(["swipe", "checkerboard", "blend", "matches"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    viewMode === mode
                      ? "bg-[#A2D5C6] text-black shadow-sm"
                      : "text-[#F6F6F6]/70 hover:text-[#F6F6F6] hover:bg-[#A2D5C6]/10"
                  }`}
                >
                  {mode === "swipe" ? "↔️ Swipe Compare" :
                   mode === "checkerboard" ? "⬜ Checkerboard" :
                   mode === "blend" ? "🔀 Blend" : "🔗 Match Lines"}
                </button>
              ))}
            </div>
          </div>

          {/* Viewer Container */}
          <div className="w-full flex justify-center bg-[#000000] rounded-xl overflow-hidden">
            {viewMode === "swipe" && result.reference_image && result.registered_image && (
              <CompareSlider imageA={result.reference_image} imageB={result.registered_image} />
            )}

            {viewMode === "checkerboard" && result.checkerboard && (
              <img
                src={`data:image/jpeg;base64,${result.checkerboard}`}
                alt="Checkerboard alignment"
                className="w-full h-auto rounded-xl border border-[#A2D5C6]/30"
              />
            )}

            {viewMode === "blend" && result.blended && (
              <img
                src={`data:image/jpeg;base64,${result.blended}`}
                alt="Blended overlay"
                className="w-full h-auto rounded-xl border border-[#A2D5C6]/30"
              />
            )}

            {viewMode === "matches" && result.match_visualization && (
              <img
                src={`data:image/jpeg;base64,${result.match_visualization}`}
                alt="Match correspondences"
                className="w-full h-auto rounded-xl border border-[#A2D5C6]/30"
              />
            )}
          </div>
        </div>

        {/* Quantitative Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Correlation" value={result.metrics?.image_metrics?.correlation ?? 0} tooltip={METRIC_TOOLTIPS.correlation} />
          <MetricCard label="SSIM Index" value={result.metrics?.image_metrics?.ssim ?? 0} tooltip={METRIC_TOOLTIPS.ssim} />
          <MetricCard label="Inlier Ratio" value={result.metrics?.match_stats?.inlier_ratio ?? 0} format="percent" tooltip={METRIC_TOOLTIPS.inlier_ratio} />
          <MetricCard label="Uniformity" value={result.metrics?.distribution?.uniformity_score ?? 0} tooltip={METRIC_TOOLTIPS.uniformity_score} />
          <MetricCard label="Reproj RMSE" value={result.metrics?.reprojection?.rmse ?? 0} tooltip={METRIC_TOOLTIPS.reproj_rmse} />
          <MetricCard label="Sub-px Ratio" value={result.metrics?.reprojection?.sub_pixel_ratio ?? 0} format="percent" tooltip={METRIC_TOOLTIPS.sub_pixel_ratio} />
          <MetricCard label="Total Inliers" value={inliers} format="int" />
          <MetricCard label="RMSE" value={result.metrics?.image_metrics?.rmse ?? 0} tooltip={METRIC_TOOLTIPS.rmse} />
        </div>
      </div>

      {/* ======================================================== */}
      {/* BOTTOM: ORIGINAL INPUT IMAGES (HORIZONTAL COMPARISON)     */}
      {/* ======================================================== */}
      <div className="border-t border-[#A2D5C6]/20 pt-12">
        <div className="mb-6 text-center sm:text-left">
          <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold block mb-1">
            Input Imagery
          </span>
          <h2 className="text-2xl font-bold text-[#F6F6F6]">Original Image Pair</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Source Image Box */}
          <div className="lunar-card p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-base font-bold text-[#CFFFE2]">Source Image</span>
              <span className="text-xs bg-[#A2D5C6]/10 text-[#A2D5C6] border border-[#A2D5C6]/30 px-2.5 py-0.5 rounded-full font-mono">
                {result.source_sensor || "Raw Source"}
              </span>
            </div>
            <div className="bg-[#000000] border border-[#A2D5C6]/30 rounded-xl p-2 flex items-center justify-center min-h-[220px]">
              {sourcePreview ? (
                <img src={sourcePreview} alt="Source Preview" className="max-h-64 object-contain rounded-lg" />
              ) : result.registered_image ? (
                <img src={`data:image/jpeg;base64,${result.registered_image}`} alt="Source Preview" className="max-h-64 object-contain rounded-lg opacity-80" />
              ) : (
                <span className="text-xs text-[#F6F6F6]/50">Source preview unavailable</span>
              )}
            </div>
          </div>

          {/* Reference Image Box */}
          <div className="lunar-card p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-base font-bold text-[#CFFFE2]">Reference Image</span>
              <span className="text-xs bg-[#A2D5C6]/10 text-[#A2D5C6] border border-[#A2D5C6]/30 px-2.5 py-0.5 rounded-full font-mono">
                {result.reference_sensor || "Base Reference"}
              </span>
            </div>
            <div className="bg-[#000000] border border-[#A2D5C6]/30 rounded-xl p-2 flex items-center justify-center min-h-[220px]">
              {referencePreview ? (
                <img src={referencePreview} alt="Reference Preview" className="max-h-64 object-contain rounded-lg" />
              ) : result.reference_image ? (
                <img src={`data:image/jpeg;base64,${result.reference_image}`} alt="Reference Preview" className="max-h-64 object-contain rounded-lg" />
              ) : (
                <span className="text-xs text-[#F6F6F6]/50">Reference preview unavailable</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-[80vh] flex items-center justify-center text-center">
        <div className="w-10 h-10 border-3 border-[#A2D5C6]/30 border-t-[#A2D5C6] rounded-full animate-spin" />
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
}
