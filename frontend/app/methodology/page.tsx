import Link from "next/link";

export default function MethodologyPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Header */}
      <div className="mb-12 text-center">
        <span className="px-3 py-1 rounded-full text-xs font-mono text-indigo-400 glass inline-block mb-3">
          Scientific Rationale & Implementation
        </span>
        <h1 className="text-4xl font-extrabold text-white tracking-tight">Methodology & Architecture</h1>
        <p className="text-gray-400 max-w-2xl mx-auto mt-3 text-sm">
          Technical explanation of each stage in the ChandraAlign registration pipeline, algorithm trade-offs, literature grounding, and novelty disclosure.
        </p>
      </div>

      {/* Novelty & Integrity Notice */}
      <div className="glass-card p-6 border-indigo-500/30 mb-12">
        <h2 className="text-sm font-bold text-indigo-300 uppercase tracking-wider mb-2 flex items-center gap-2">
          <span>🛡️</span> SIH 2026 Integrity & Novelty Disclosure
        </h2>
        <p className="text-xs text-gray-300 leading-relaxed mb-3">
          In strict compliance with academic and hackathon evaluation standards, we explicitly distinguish between adapted open-source foundational techniques and our original algorithmic contributions:
        </p>
        <ul className="grid sm:grid-cols-2 gap-3 text-xs text-gray-400">
          <li className="bg-black/30 p-3 rounded-lg border border-white/5">
            <strong className="text-white block mb-1">Implemented Novelties & Custom Pipeline:</strong>
            Log-Gabor Phase Congruency illumination mapping, RIFT-inspired Maximum Index Map descriptors for lunar optical pairs, Grid-based Non-Max Suppression for spatial uniformity enforcement, and multi-scale GSD-informed search narrowing.
          </li>
          <li className="bg-black/30 p-3 rounded-lg border border-white/5">
            <strong className="text-white block mb-1">Standard Classical Formulations:</strong>
            SIFT/ORB feature extractors (Lowe et al., Rublee et al.), USAC/MAGSAC++ geometric verification (Barath et al., 2020), and Lucas-Kanade optical flow formulation (Lucas & Kanade, 1981).
          </li>
        </ul>
      </div>

      {/* Detailed Stages */}
      <div className="space-y-8">
        {/* Stage A */}
        <section className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-300 flex items-center justify-center font-bold text-sm">A</span>
            <div>
              <h2 className="text-lg font-bold text-white">Stage A — Radiometric & Illumination Normalization</h2>
              <span className="text-xs text-gray-400">Overcoming solar azimuth & elevation variations</span>
            </div>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed mb-4">
            In lunar optical imagery, changing sun angles reverse crater illumination—what was a highlighted wall becomes deep shadow. Raw intensity gradients fail under these conditions.
          </p>
          <div className="grid sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-indigo-300 block mb-1">Phase Congruency (PC)</strong>
              Computes feature significance via phase alignment across Log-Gabor filterbanks. Intensity-invariant by mathematical definition:
              <div className="font-mono text-[10px] text-gray-400 mt-2 bg-black/40 p-1.5 rounded">
                PC(x) = E(x) / (Σ A_n(x) + ε)
              </div>
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-indigo-300 block mb-1">Difference-of-Log (DoL)</strong>
              Subtracts a heavily smoothed log image from a lightly smoothed log image to eliminate wide illumination ramps while preserving micro-texture.
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-indigo-300 block mb-1">Weber Excitation (WLD)</strong>
              Differential excitation based on Weber&apos;s Law (ΔI / I), making local feature detection invariant to global illumination scaling.
            </div>
          </div>
        </section>

        {/* Stage B */}
        <section className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="w-8 h-8 rounded-lg bg-blue-500/20 text-blue-300 flex items-center justify-center font-bold text-sm">B</span>
            <div>
              <h2 className="text-lg font-bold text-white">Stage B — Scale-Space & GSD-Informed Pyramid</h2>
              <span className="text-xs text-gray-400">Bridging the ~300:1 resolution gap (OHRC vs. IIRS)</span>
            </div>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed mb-4">
            Matching OHRC (0.25 m/px) directly to IIRS (80 m/px) is intractable in a single step. Our pipeline utilizes two strategies:
          </p>
          <div className="grid sm:grid-cols-2 gap-4 text-xs">
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">1. GSD Prior Scale Estimation:</strong>
              Sensor metadata supplies the native resolution. The pipeline resamples candidate images or initializes the scale parameter:
              <span className="font-mono text-indigo-300 block mt-1">s_prior = GSD_ref / GSD_src</span>
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">2. Coarse-to-Fine Chaining:</strong>
              For extreme multi-modal pairs (OHRC↔IIRS), intermediate registration to TMC-2 (5 m/px) provides a stable bridge:
              <span className="font-mono text-indigo-300 block mt-1">T_total = T_(TMC→IIRS) ∘ T_(OHRC→TMC)</span>
            </div>
          </div>
        </section>

        {/* Stage C & D */}
        <section className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="w-8 h-8 rounded-lg bg-violet-500/20 text-violet-300 flex items-center justify-center font-bold text-sm">C &amp; D</span>
            <div>
              <h2 className="text-lg font-bold text-white">Stage C &amp; D — Spatial NMS, RIFT Descriptors &amp; MAGSAC++</h2>
              <span className="text-xs text-gray-400">Uniform match distribution &amp; robust outlier rejection</span>
            </div>
          </div>
          <div className="space-y-4 text-xs text-gray-300">
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">Grid-Based Non-Max Suppression (Uniform Distribution):</strong>
              Problem statement requirement: &ldquo;uniform distribution across the images&rdquo;. Standard SIFT clusters 90% of keypoints on high-contrast crater rims. We bucket the image into an N×N grid (default 8×8 = 64 cells) and cap keypoints per cell to force global spatial uniformity.
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">RIFT Descriptor Formulation:</strong>
              Instead of intensity gradients, RIFT computes Maximum Index Maps (MIM) from phase congruency orientation bins, building spatial orientation histograms that are radiation and modality invariant.
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">MAGSAC++ Outlier Rejection:</strong>
              Uses marginalizing sample consensus with σ-consensus to estimate homographies without requiring an ad-hoc inlier threshold parameter, providing superior stability on sparse crater correspondences.
            </div>
          </div>
        </section>

        {/* Stage E & F */}
        <section className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-300 flex items-center justify-center font-bold text-sm">E &amp; F</span>
            <div>
              <h2 className="text-lg font-bold text-white">Stage E &amp; F — Sub-Pixel Refinement &amp; Quantitative Metrics</h2>
              <span className="text-xs text-gray-400">Sub-pixel precision &amp; multi-metric evaluation dashboard</span>
            </div>
          </div>
          <div className="grid sm:grid-cols-2 gap-4 text-xs text-gray-300">
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">Parabolic Surface Peak Fitting:</strong>
              Local cross-correlation surfaces around keypoints are fitted to a 2D quadric surface to compute fractional pixel offsets:
              <div className="font-mono text-[10px] text-gray-400 mt-1 bg-black/40 p-1 rounded">
                Δx = 0.5 * (f(x-1) - f(x+1)) / (f(x-1) - 2f(x) + f(x+1))
              </div>
            </div>
            <div className="p-3 bg-white/[0.02] rounded-lg border border-white/5">
              <strong className="text-white block mb-1">Evaluated Metrics Suite:</strong>
              Image RMSE, MAE, Normalized Cross-Correlation (NCC), Structural Similarity Index (SSIM), Mutual Information (MI), Gradient Correlation, Grid Occupancy Entropy, and Inlier Ratio.
            </div>
          </div>
        </section>
      </div>

      {/* Literature References */}
      <div className="glass-card p-6 mt-12">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3">Academic Literature &amp; References</h3>
        <ol className="list-decimal list-inside space-y-1.5 text-xs text-gray-400">
          <li>Li, J., Hu, Q., &amp; Ai, M. (2020). <em>RIFT: Multi-modal image matching based on radiation-variation insensitive feature transform.</em> IEEE TGRS.</li>
          <li>Kovesi, P. (1999). <em>Image features from phase congruency.</em> Videre: Journal of Computer Vision Research, 1(3), 1-26.</li>
          <li>Barath, D., Noskova, J., Ivashechkin, M., &amp; Matas, J. (2020). <em>MAGSAC++, a fast, reliable and accurate robust estimator.</em> CVPR 2020.</li>
          <li>Lowe, D. G. (2004). <em>Distinctive image features from scale-invariant keypoints.</em> International Journal of Computer Vision.</li>
          <li>Sun, J., Shen, Z., Wang, Y., Bao, H., &amp; Zhou, X. (2021). <em>LoFTR: Detector-free local feature matching with transformers.</em> CVPR 2021.</li>
        </ol>
      </div>

      {/* CTA */}
      <div className="mt-12 text-center">
        <Link href="/workbench" className="btn-primary inline-block">
          Test Pipeline on Workbench →
        </Link>
      </div>
    </div>
  );
}
