"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  submitRegistration,
  streamJobProgress,
  getJobResult,
  DEFAULT_CONFIG,
  type RegistrationConfig,
  type StageEvent,
} from "@/lib/api";

const SENSORS = ["OHRC", "TMC2", "IIRS", "LRO_NAC", "LRO_WAC"];

const STAGE_LABELS: Record<string, string> = {
  preprocessing: "1. Normalizing Illumination",
  feature_detection: "2. Detecting Invariant Features",
  feature_matching: "3. Spatial Grid Matching",
  transform_estimation: "4. Solving Surface Transform",
  subpixel_refinement: "5. Sub-Pixel Surface Refinement",
  evaluation: "6. Validating Quality Score",
};

const STAGE_ORDER = [
  "preprocessing",
  "feature_detection",
  "feature_matching",
  "transform_estimation",
  "subpixel_refinement",
  "evaluation",
];

const SAMPLE_PRESETS = [
  {
    id: "pair_sun_angle",
    name: "TMC-2 Sun Angle",
    icon: "☀️",
    source_sensor: "TMC2",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_sun_angle/source.png",
    reference_url: "/samples/pair_sun_angle/reference.png",
    desc: "45° vs 75° solar azimuth with 4° rotation",
  },
  {
    id: "pair_cross_modal",
    name: "OHRC vs TMC-2 Zoom",
    icon: "🔬",
    source_sensor: "OHRC",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_cross_modal/source.png",
    reference_url: "/samples/pair_cross_modal/reference.png",
    desc: "0.25m high-res crop to 5m regional context",
  },
  {
    id: "pair_polar_south",
    name: "South Pole Shadows",
    icon: "🧭",
    source_sensor: "TMC2",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_polar_south/source.png",
    reference_url: "/samples/pair_polar_south/reference.png",
    desc: "Low sun elevation (14°–18°) long shadows",
  },
  {
    id: "pair_mare_plains",
    name: "Mare Basalt Plains",
    icon: "🌑",
    source_sensor: "TMC2",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_mare_plains/source.png",
    reference_url: "/samples/pair_mare_plains/reference.png",
    desc: "Low-contrast smooth regolith basalt plain",
  },
  {
    id: "pair_rugged_crater",
    name: "Rugged Crater Rim",
    icon: "⛰️",
    source_sensor: "OHRC",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_rugged_crater/source.png",
    reference_url: "/samples/pair_rugged_crater/reference.png",
    desc: "Complex crater wall slopes & central peak",
  },
  {
    id: "pair_iirs_multimodal",
    name: "IIRS Hyperspectral",
    icon: "🌈",
    source_sensor: "IIRS",
    reference_sensor: "TMC2",
    source_url: "/samples/pair_iirs_multimodal/source.png",
    reference_url: "/samples/pair_iirs_multimodal/reference.png",
    desc: "Infrared mineral band vs panchromatic",
  },
];

export default function WorkbenchPage() {
  const router = useRouter();

  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string>("");
  const [referencePreview, setReferencePreview] = useState<string>("");
  const [sourceSensor, setSourceSensor] = useState<string>("");
  const [referenceSensor, setReferenceSensor] = useState<string>("");
  const [activePreset, setActivePreset] = useState<string>("");
  const [config, setConfig] = useState<RegistrationConfig>(DEFAULT_CONFIG);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [running, setRunning] = useState(false);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>("");

  // File handler
  const handleFileSelect = (file: File, type: "source" | "reference") => {
    setActivePreset("");
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      if (type === "source") {
        setSourceFile(file);
        setSourcePreview(dataUrl);
        try { sessionStorage.setItem("luna_source_preview", dataUrl); } catch {}
      } else {
        setReferenceFile(file);
        setReferencePreview(dataUrl);
        try { sessionStorage.setItem("luna_ref_preview", dataUrl); } catch {}
      }
    };
    reader.readAsDataURL(file);
  };

  // Quick Preset Loader
  const loadPreset = async (preset: typeof SAMPLE_PRESETS[0]) => {
    try {
      setActivePreset(preset.id);
      setError("");

      const [srcRes, refRes] = await Promise.all([
        fetch(preset.source_url),
        fetch(preset.reference_url),
      ]);

      const srcBlob = await srcRes.blob();
      const refBlob = await refRes.blob();

      const srcF = new File([srcBlob], `${preset.id}_source.png`, { type: "image/png" });
      const refF = new File([refBlob], `${preset.id}_reference.png`, { type: "image/png" });

      setSourceFile(srcF);
      setReferenceFile(refF);
      setSourceSensor(preset.source_sensor);
      setReferenceSensor(preset.reference_sensor);

      // Previews
      const srcUrl = preset.source_url;
      const refUrl = preset.reference_url;
      setSourcePreview(srcUrl);
      setReferencePreview(refUrl);

      // Convert to base64 for persistent session cache
      const toBase64 = (blob: Blob): Promise<string> =>
        new Promise((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result as string);
          r.readAsDataURL(blob);
        });

      const [sBase64, rBase64] = await Promise.all([toBase64(srcBlob), toBase64(refBlob)]);
      sessionStorage.setItem("luna_source_preview", sBase64);
      sessionStorage.setItem("luna_ref_preview", rBase64);
    } catch {
      setError("Failed to load sample dataset preset.");
    }
  };

  // Run registration and redirect to Results page
  const handleRun = async () => {
    if (!sourceFile || !referenceFile || running) return;

    setRunning(true);
    setError("");
    setCurrentStage("preprocessing");
    setProgress(10);

    try {
      if (sourcePreview) sessionStorage.setItem("luna_source_preview", sourcePreview);
      if (referencePreview) sessionStorage.setItem("luna_ref_preview", referencePreview);

      const { job_id } = await submitRegistration(
        sourceFile,
        referenceFile,
        config,
        sourceSensor || undefined,
        referenceSensor || undefined,
      );

      // Stream progress via SSE
      streamJobProgress(
        job_id,
        (event: StageEvent) => {
          setCurrentStage(event.stage);
          const idx = STAGE_ORDER.indexOf(event.stage);
          if (idx >= 0) {
            setProgress(Math.min(95, Math.round(((idx + 1) / STAGE_ORDER.length) * 100)));
          }
        },
        async () => {
          // On Complete — Fetch full result & route to Results page
          try {
            const res = await getJobResult(job_id);
            if (res.status === "failed") {
              setError(res.error || "Registration failed. Try adjusting detector or illumination settings.");
              setRunning(false);
            } else {
              sessionStorage.setItem("luna_last_result", JSON.stringify(res));
              router.push(`/results?job_id=${job_id}`);
            }
          } catch {
            setError("Failed to retrieve final registration data.");
            setRunning(false);
          }
        },
        (err: string) => {
          setError(err);
          setRunning(false);
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Submission failed");
      setRunning(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Page Title & Intro */}
      <div className="text-center max-w-2xl mx-auto mb-8">
        <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-2 block">
          Registration Engine
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#F6F6F6] tracking-tight">
          Registration Workbench
        </h1>
        <p className="text-sm sm:text-base text-[#F6F6F6]/75 mt-2">
          Upload multi-sensor lunar imagery, configure pipeline parameters, and execute sub-pixel alignment.
        </p>
      </div>

      {/* ── Sample Lunar Dataset Quick Loader ─────────────────── */}
      <div className="lunar-card p-5 mb-10 bg-[#000000]">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-bold text-[#A2D5C6] uppercase tracking-wider">
            ⚡ Quick Sample Datasets (Chandrayaan-2 Lunar Benchmarks)
          </span>
          <span className="text-[11px] text-[#F6F6F6]/50">Click any preset to load instantly</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => loadPreset(p)}
              className={`p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                activePreset === p.id
                  ? "border-[#CFFFE2] bg-[#A2D5C6]/15 shadow-sm"
                  : "border-[#A2D5C6]/30 hover:border-[#A2D5C6] bg-[#000000] hover:bg-[#A2D5C6]/5"
              }`}
            >
              <div>
                <div className="text-lg mb-1">{p.icon}</div>
                <div className="text-xs font-bold text-[#F6F6F6] truncate">{p.name}</div>
              </div>
              <div className="text-[10px] text-[#A2D5C6] font-mono mt-1">{p.source_sensor} → {p.reference_sensor}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="lunar-card p-4 border-red-500/60 bg-red-950/20 text-red-300 text-sm mb-8 text-center animate-fade-in">
          ⚠️ {error}
        </div>
      )}

      {/* Processing Modal / Overlay */}
      {running && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4">
          <div className="lunar-card max-w-md w-full p-8 text-center bg-[#000000] border-2 border-[#A2D5C6] shadow-2xl">
            <div className="w-14 h-14 border-3 border-[#A2D5C6]/20 border-t-[#A2D5C6] rounded-full animate-spin mx-auto mb-6" />
            <h3 className="text-xl font-bold text-[#F6F6F6] mb-2">Aligning Lunar Surfaces...</h3>
            <p className="text-xs text-[#A2D5C6] font-mono mb-6">
              {currentStage ? STAGE_LABELS[currentStage] || currentStage : "Initializing pipeline"}
            </p>

            {/* Progress Bar */}
            <div className="h-2 bg-[#000000] border border-[#A2D5C6]/30 rounded-full overflow-hidden mb-3">
              <div
                className="h-full bg-gradient-to-r from-[#A2D5C6] to-[#CFFFE2] transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="font-mono text-xs text-[#F6F6F6]/60">{progress}% complete</span>
          </div>
        </div>
      )}

      {/* ======================================================== */}
      {/* FORM LAYOUT: 2-COLUMN BALANCED GRID                      */}
      {/* ======================================================== */}
      <div className="space-y-8">
        {/* ── ROW 1: Image Uploads (50 / 50 Split) ─────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Source Image Upload Box */}
          <div className="lunar-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-bold text-[#CFFFE2]">Source Image</label>
                <span className="text-xs text-[#F6F6F6]/50 font-mono">Target to align</span>
              </div>
              <div
                className="border-2 border-dashed border-[#A2D5C6]/40 hover:border-[#CFFFE2] rounded-xl p-6 text-center transition-colors cursor-pointer bg-[#000000]"
                onClick={() => document.getElementById("source-input")?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0], "source");
                }}
              >
                {sourcePreview ? (
                  <div className="relative">
                    <img src={sourcePreview} alt="Source" className="max-h-48 mx-auto rounded-lg object-contain" />
                    <span className="block mt-2 text-xs text-[#A2D5C6]">Click to replace</span>
                  </div>
                ) : (
                  <div className="py-8 text-[#F6F6F6]/60 text-sm">
                    <div className="text-3xl mb-2">📤</div>
                    <span className="font-medium text-[#F6F6F6]">Click or drag & drop</span>
                    <span className="block text-xs text-[#F6F6F6]/40 mt-1">PNG, JPG, TIFF (OHRC / TMC-2 / IIRS)</span>
                  </div>
                )}
                <input
                  id="source-input"
                  type="file"
                  accept="image/*,.tif,.tiff"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0], "source")}
                />
              </div>
            </div>

            <div className="mt-4">
              <label className="text-xs text-[#F6F6F6]/70 mb-1.5 block">Sensor Payload (Optional)</label>
              <select
                value={sourceSensor}
                onChange={(e) => setSourceSensor(e.target.value)}
                className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-xl px-3.5 py-2.5 text-sm text-[#F6F6F6] focus:border-[#CFFFE2] outline-none"
              >
                <option value="">Auto-detect sensor</option>
                {SENSORS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Reference Image Upload Box */}
          <div className="lunar-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-bold text-[#CFFFE2]">Reference Image</label>
                <span className="text-xs text-[#F6F6F6]/50 font-mono">Base coordinate grid</span>
              </div>
              <div
                className="border-2 border-dashed border-[#A2D5C6]/40 hover:border-[#CFFFE2] rounded-xl p-6 text-center transition-colors cursor-pointer bg-[#000000]"
                onClick={() => document.getElementById("ref-input")?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0], "reference");
                }}
              >
                {referencePreview ? (
                  <div className="relative">
                    <img src={referencePreview} alt="Reference" className="max-h-48 mx-auto rounded-lg object-contain" />
                    <span className="block mt-2 text-xs text-[#A2D5C6]">Click to replace</span>
                  </div>
                ) : (
                  <div className="py-8 text-[#F6F6F6]/60 text-sm">
                    <div className="text-3xl mb-2">📤</div>
                    <span className="font-medium text-[#F6F6F6]">Click or drag & drop</span>
                    <span className="block text-xs text-[#F6F6F6]/40 mt-1">PNG, JPG, TIFF (TMC-2 / LRO / Map)</span>
                  </div>
                )}
                <input
                  id="ref-input"
                  type="file"
                  accept="image/*,.tif,.tiff"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0], "reference")}
                />
              </div>
            </div>

            <div className="mt-4">
              <label className="text-xs text-[#F6F6F6]/70 mb-1.5 block">Sensor Payload (Optional)</label>
              <select
                value={referenceSensor}
                onChange={(e) => setReferenceSensor(e.target.value)}
                className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-xl px-3.5 py-2.5 text-sm text-[#F6F6F6] focus:border-[#CFFFE2] outline-none"
              >
                <option value="">Auto-detect sensor</option>
                {SENSORS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* ── ROW 2: Engine Selector & Advanced Settings (50 / 50 Split) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
          {/* Left: Engine Selector */}
          <div className="lunar-card p-6 h-full flex flex-col justify-between">
            <div>
              <label className="text-sm font-bold text-[#CFFFE2] mb-1 block">Registration Engine</label>
              <p className="text-xs text-[#F6F6F6]/70 mb-4">Select algorithm pipeline based on payload pairing</p>
              <div className="grid grid-cols-3 gap-3">
                {(["classical", "deep", "hybrid"] as const).map((e) => (
                  <button
                    key={e}
                    type="button"
                    onClick={() => setConfig({ ...config, engine: e })}
                    className={`px-4 py-3 rounded-xl text-xs font-semibold transition-all border text-center ${
                      config.engine === e
                        ? "bg-[#A2D5C6] text-black border-[#A2D5C6] shadow-sm font-bold"
                        : "bg-[#000000] text-[#F6F6F6]/80 border-[#A2D5C6]/30 hover:border-[#A2D5C6]"
                    }`}
                  >
                    {e === "classical" ? "🔬 Classical" : e === "deep" ? "🧠 Deep AI" : "⚡ Hybrid"}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-[#A2D5C6]/15 text-xs text-[#F6F6F6]/60">
              {config.engine === "classical" && "Classical: SIFT + Log-Gabor Phase Congruency & MAGSAC++."}
              {config.engine === "deep" && "Deep: Deep feature descriptor maps with adaptive sub-pixel flow."}
              {config.engine === "hybrid" && "Hybrid: Phase Congruency structural maps fused with multi-scale matching."}
            </div>
          </div>

          {/* Right: Advanced Settings Box */}
          <div className="lunar-card p-6 h-full flex flex-col justify-between">
            <div>
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <div>
                  <h3 className="text-sm font-bold text-[#CFFFE2]">Advanced Parameters</h3>
                  <p className="text-xs text-[#F6F6F6]/70">Fine-tune transforms, illumination filters & thresholds</p>
                </div>
                <span className={`text-[#A2D5C6] text-sm transform transition-transform ${showAdvanced ? "rotate-180" : ""}`}>
                  ▼
                </span>
              </div>

              {showAdvanced && (
                <div className="mt-5 space-y-3 pt-4 border-t border-[#A2D5C6]/20 animate-fade-in">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[11px] text-[#F6F6F6]/70 mb-1 block">Illumination</label>
                      <select
                        value={config.illumination_method}
                        onChange={(e) => setConfig({ ...config, illumination_method: e.target.value as any })}
                        className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-lg px-2.5 py-1.5 text-xs text-[#F6F6F6]"
                      >
                        {["phase_congruency", "clahe", "combined", "dol", "wld"].map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] text-[#F6F6F6]/70 mb-1 block">Feature Detector</label>
                      <select
                        value={config.feature_detector}
                        onChange={(e) => setConfig({ ...config, feature_detector: e.target.value as any })}
                        className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-lg px-2.5 py-1.5 text-xs text-[#F6F6F6]"
                      >
                        {["sift", "orb", "akaze", "rift"].map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[11px] text-[#F6F6F6]/70 mb-1 block">Transform Type</label>
                      <select
                        value={config.transform_type}
                        onChange={(e) => setConfig({ ...config, transform_type: e.target.value as any })}
                        className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-lg px-2.5 py-1.5 text-xs text-[#F6F6F6]"
                      >
                        {["homography", "affine", "rigid", "similarity", "tps", "polynomial"].map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] text-[#F6F6F6]/70 mb-1 block">Sub-Pixel Model</label>
                      <select
                        value={config.subpixel_method}
                        onChange={(e) => setConfig({ ...config, subpixel_method: e.target.value as any })}
                        className="w-full bg-[#000000] border border-[#A2D5C6]/40 rounded-lg px-2.5 py-1.5 text-xs text-[#F6F6F6]"
                      >
                        {["parabolic", "lucas_kanade", "phase_correlation", "combined"].map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {!showAdvanced && (
              <p className="mt-4 pt-4 border-t border-[#A2D5C6]/15 text-xs text-[#F6F6F6]/50">
                Default: Combined Log-Gabor Phase Congruency, SIFT detector, Homography transform, and Parabolic sub-pixel fitting.
              </p>
            )}
          </div>
        </div>

        {/* ── ROW 3: Centered "Run Registration" Button ────────── */}
        <div className="flex flex-col items-center justify-center pt-6">
          <button
            type="button"
            onClick={handleRun}
            disabled={!sourceFile || !referenceFile || running}
            className={`btn-stars text-lg px-12 py-5 ${
              !sourceFile || !referenceFile || running ? "opacity-40 cursor-not-allowed" : ""
            }`}
          >
            Run Registration
            <div className="star-1">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
            <div className="star-2">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
            <div className="star-3">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
            <div className="star-4">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
            <div className="star-5">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
            <div className="star-6">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 784.11 815.53" style={{ shapeRendering: "geometricPrecision", fillRule: "evenodd" }}>
                <path d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z" className="fil0" />
              </svg>
            </div>
          </button>
          {(!sourceFile || !referenceFile) && (
            <p className="text-xs text-[#F6F6F6]/45 mt-3">Upload or pick a sample preset above to begin registration</p>
          )}
        </div>
      </div>
    </div>
  );
}
