"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

/* ── Starfield background (CSS-only, lightweight) ──────────── */
function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const stars = Array.from({ length: 150 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.5 + 0.3,
      a: Math.random(),
      speed: Math.random() * 0.02 + 0.005,
    }));

    let frame = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      stars.forEach((s) => {
        s.a = 0.3 + 0.7 * Math.abs(Math.sin(frame * s.speed));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 200, 230, ${s.a})`;
        ctx.fill();
      });
      frame++;
      requestAnimationFrame(draw);
    };
    draw();
    return () => window.removeEventListener("resize", resize);
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
}

/* ── Pipeline stage diagram ────────────────────────────────── */
const STAGES = [
  { icon: "🔆", label: "Illumination\nNormalization", desc: "Phase congruency, DoL, WLD" },
  { icon: "🔍", label: "Multi-Scale\nFeature Detection", desc: "SIFT/RIFT + Grid NMS" },
  { icon: "🔗", label: "Feature\nMatching", desc: "MAGSAC++ outlier rejection" },
  { icon: "📐", label: "Transform\nEstimation", desc: "Homography / TPS / Polynomial" },
  { icon: "🎯", label: "Sub-Pixel\nRefinement", desc: "Parabolic peak / L-K flow" },
  { icon: "📊", label: "Evaluation\nMetrics", desc: "RMSE, uniformity, inlier ratio" },
];

/* ── Challenge cards ───────────────────────────────────────── */
const CHALLENGES = [
  {
    title: "Sun-Angle Variation",
    icon: "☀️",
    desc: "Same crater looks completely different under varying solar azimuth & elevation — shadow direction and length shift dramatically.",
    color: "from-amber-500/20 to-orange-600/20",
    border: "border-amber-500/20",
  },
  {
    title: "Scale Difference",
    icon: "🔬",
    desc: "Up to 300:1 GSD ratio between OHRC (0.25 m) and IIRS (80 m). A single IIRS pixel covers an entire OHRC image tile.",
    color: "from-blue-500/20 to-cyan-600/20",
    border: "border-blue-500/20",
  },
  {
    title: "Viewpoint Variation",
    icon: "🛰️",
    desc: "Different orbit tracks, look angles, and projections create geometric distortions that must be resolved for accurate co-registration.",
    color: "from-violet-500/20 to-purple-600/20",
    border: "border-violet-500/20",
  },
];

export default function LandingPage() {
  return (
    <div className="relative">
      <Starfield />

      {/* ── Hero Section ───────────────────────────────────── */}
      <section className="relative z-10 min-h-[85vh] flex flex-col items-center justify-center px-4 text-center">
        <h1 className="animate-slide-up text-4xl sm:text-6xl lg:text-7xl font-extrabold text-[#F6F6F6] max-w-5xl leading-tight tracking-tight">
          Multi-Modal Lunar{" "}
          <span className="bg-gradient-to-r from-[#A2D5C6] to-[#CFFFE2] bg-clip-text text-transparent">
            Image Registration
          </span>
        </h1>

        <p className="animate-slide-up delay-200 mt-6 text-lg sm:text-xl text-[#F6F6F6]/85 max-w-2xl opacity-0 leading-relaxed" style={{ animationFillMode: "forwards" }}>
          Intelligent, illumination-invariant orbital mapping and sub-pixel optical image registration
          for lunar exploration—unlocking seamless multi-sensor terrain synthesis.
        </p>

        <div className="animate-slide-up delay-400 mt-10 flex flex-col sm:flex-row items-center gap-5 opacity-0" style={{ animationFillMode: "forwards" }}>
          {/* Primary Action Button with Animated Stars */}
          <Link href="/workbench" className="btn-stars">
            Try the Workbench
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
          </Link>

          {/* Secondary Action Button */}
          <Link href="/methodology" className="btn-secondary-outline">
            View Architecture
          </Link>
        </div>
      </section>

      {/* ── Challenge Cards (Slide 2) ──────────────────────── */}
      <section className="section-slide border-t border-[#A2D5C6]/15">
        <div className="max-w-6xl mx-auto w-full">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-3 inline-block">
              The Problem
            </span>
            <h2 className="text-3xl sm:text-5xl font-extrabold text-[#F6F6F6] tracking-tight mb-4">
              Why Lunar Mapping Is Hard
            </h2>
            <p className="text-lg text-[#F6F6F6]/80 leading-relaxed">
              Taking photos of the Moon from orbit sounds simple—until the sun shifts, cameras change, and orbits tilt.
            </p>
          </div>

          <div className="cards">
            <div className="card">
              <span className="card-icon">☀️</span>
              <p className="tip">Shifting Shadows</p>
              <p className="second-text">
                As the solar angle moves, crater shadows flip completely. The same ridge can look like a deep hole a few hours later.
              </p>
            </div>

            <div className="card">
              <span className="card-icon">🔬</span>
              <p className="tip">Resolution Gap</p>
              <p className="second-text">
                One sensor zooms in on individual rocks; another captures whole regions. Matching microscopic detail to broad plains requires smart feature mapping.
              </p>
            </div>

            <div className="card">
              <span className="card-icon">🛰️</span>
              <p className="tip">Perspective Warp</p>
              <p className="second-text">
                Different satellite passes, look angles, and velocities stretch surface topography like a funhouse mirror across passes.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Pipeline Architecture (Slide 3) ─────────────────── */}
      <section className="section-slide border-t border-[#A2D5C6]/15">
        <div className="max-w-6xl mx-auto w-full">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-3 inline-block">
              The Architecture
            </span>
            <h2 className="text-3xl sm:text-5xl font-extrabold text-[#F6F6F6] tracking-tight mb-4">
              How Lună Fixes It
            </h2>
            <p className="text-lg text-[#F6F6F6]/80 leading-relaxed">
              Six automated steps that turn raw, scrambled orbital imagery into crystal-clear, millimeter-precise maps.
            </p>
          </div>

          <div className="cards">
            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">🔆</span>
                <span className="step-pill">Step 01</span>
              </div>
              <p className="tip">Strip the Shadows</p>
              <p className="second-text">
                Cleans away misleading sunlight and dark shadows to reveal the true topological rock shapes underneath.
              </p>
            </div>

            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">🔍</span>
                <span className="step-pill">Step 02</span>
              </div>
              <p className="tip">Spot Key Landmarks</p>
              <p className="second-text">
                Scans for indestructible landmarks—crater rims, structural ridges, and boulder clusters across the grid.
              </p>
            </div>

            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">🔗</span>
                <span className="step-pill">Step 03</span>
              </div>
              <p className="tip">Filter the Noise</p>
              <p className="second-text">
                AI weeds out false matches so only physically true, verified surface correspondences stay connected.
              </p>
            </div>

            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">📐</span>
                <span className="step-pill">Step 04</span>
              </div>
              <p className="tip">Align the Grid</p>
              <p className="second-text">
                Warps and stretches the images until they snap seamlessly into the exact same physical coordinates.
              </p>
            </div>

            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">🎯</span>
                <span className="step-pill">Step 05</span>
              </div>
              <p className="tip">Sub-Pixel Polish</p>
              <p className="second-text">
                Refines point alignments down to tiny fractions of a single camera pixel for extreme clarity.
              </p>
            </div>

            <div className="card">
              <div className="card-header-row">
                <span className="card-icon">📊</span>
                <span className="step-pill">Step 06</span>
              </div>
              <p className="tip">Score & Validate</p>
              <p className="second-text">
                Calculates strict quality scores, error margins, and export-ready GIS data for scientific workflows.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Why It Matters (Slide 4) ────────────────────────── */}
      <section className="section-slide border-t border-[#A2D5C6]/15">
        <div className="max-w-6xl mx-auto w-full">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-3 inline-block">
              Impact
            </span>
            <h2 className="text-3xl sm:text-5xl font-extrabold text-[#F6F6F6] tracking-tight mb-4">
              Why It Matters
            </h2>
            <p className="text-lg text-[#F6F6F6]/80 leading-relaxed">
              Better lunar mapping directly accelerates the next generation of robotic and human space exploration.
            </p>
          </div>

          <div className="cards cards-4-col">
            <div className="card">
              <span className="card-icon">🚀</span>
              <p className="tip">Pinpoint Landings</p>
              <p className="second-text">
                Guides landers and rovers safely down by spotting hidden slopes and hazardous boulders.
              </p>
            </div>

            <div className="card">
              <span className="card-icon">🕳️</span>
              <p className="tip">Crater History</p>
              <p className="second-text">
                Helps planetary scientists date impact craters and decode billions of years of solar system history.
              </p>
            </div>

            <div className="card">
              <span className="card-icon">⚠️</span>
              <p className="tip">Hazard Detection</p>
              <p className="second-text">
                Builds high-precision 3D elevation maps to keep rovers from getting trapped in permanent shadows.
              </p>
            </div>

            <div className="card">
              <span className="card-icon">🗺️</span>
              <p className="tip">Universal Atlas</p>
              <p className="second-text">
                Fuses photos from different spacecraft and orbital eras into one seamless, unified lunar atlas.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Ready to Register (Slide 5 / Final CTA) ─────────── */}
      <section className="section-slide border-t border-[#A2D5C6]/15 text-center">
        <div className="max-w-4xl mx-auto w-full px-4">
          <span className="text-xs uppercase tracking-widest text-[#A2D5C6] font-semibold mb-3 inline-block">
            Get Started
          </span>
          <h2 className="text-4xl sm:text-6xl font-extrabold text-[#F6F6F6] tracking-tight mb-6">
            Ready to Map the Moon?
          </h2>
          <p className="text-lg sm:text-xl text-[#F6F6F6]/85 max-w-2xl mx-auto mb-12 leading-relaxed">
            Upload any pair of lunar images, watch the pipeline execute in real time, and download verified sub-pixel data.
          </p>

          <div className="flex justify-center">
            {/* Interactive SVG Star Button for CTA */}
            <Link href="/workbench" className="btn-stars text-lg sm:text-xl px-10 py-5">
              Launch Workbench →
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
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
