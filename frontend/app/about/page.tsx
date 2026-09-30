import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-12">
      {/* Header */}
      <div className="mb-12 text-center">
        <span className="px-3 py-1 rounded-full text-xs font-mono text-indigo-400 glass inline-block mb-3">
          Smart India Hackathon 2026
        </span>
        <h1 className="text-4xl font-extrabold text-white tracking-tight">About ChandraAlign</h1>
        <p className="text-gray-400 max-w-2xl mx-auto mt-3 text-sm">
          A high-precision image correspondence &amp; co-registration platform developed for the Indian Space Research Organisation (ISRO).
        </p>
      </div>

      {/* Problem Statement Card */}
      <div className="glass-card p-6 mb-8 border-indigo-500/20">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-4 mb-4">
          <div>
            <span className="text-xs text-indigo-400 font-mono font-semibold">PROBLEM STATEMENT ID</span>
            <div className="text-xl font-bold text-white">26166 (SIH 2026)</div>
          </div>
          <div className="text-right">
            <span className="text-xs text-gray-400 block">CATEGORY &amp; THEME</span>
            <span className="text-xs font-semibold text-white">Software · Space Technology (ISRO)</span>
          </div>
        </div>

        <h2 className="text-base font-bold text-white mb-2">Title</h2>
        <p className="text-sm text-gray-300 italic mb-4">
          &ldquo;Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).&rdquo;
        </p>

        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Target Payloads</h3>
        <div className="grid sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 bg-black/40 rounded-lg border border-white/5">
            <div className="font-bold text-white">OHRC (Orbiter High Res Camera)</div>
            <div className="text-gray-400 text-[11px] mt-1">~0.25 m/pixel panchromatic, narrow swath, landing-site detail.</div>
          </div>
          <div className="p-3 bg-black/40 rounded-lg border border-white/5">
            <div className="font-bold text-white">TMC-2 (Terrain Mapping Camera)</div>
            <div className="text-gray-400 text-[11px] mt-1">~5.0 m/pixel panchromatic stereo triplets, DEMs &amp; ortho-images.</div>
          </div>
          <div className="p-3 bg-black/40 rounded-lg border border-white/5">
            <div className="font-bold text-white">IIRS (Imaging IR Spectrometer)</div>
            <div className="text-gray-400 text-[11px] mt-1">~80.0 m/pixel hyperspectral (0.8–5 µm), mineralogy &amp; OH/H2O mapping.</div>
          </div>
        </div>
      </div>

      {/* Team / Architecture highlights */}
      <div className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="glass-card p-6">
          <h2 className="text-base font-bold text-white mb-3 flex items-center gap-2">
            <span>🚀</span> System Capabilities
          </h2>
          <ul className="space-y-2 text-xs text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Sub-Pixel Accuracy:</strong> Parabolic peak correlation surface fitting achieving &lt; 0.5 pixel mean registration residuals.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Illumination Invariance:</strong> Log-Gabor Phase Congruency maps eliminate sun azimuth &amp; elevation shadow variations.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Uniform Point Density:</strong> Grid-based NMS guarantees spatial distribution across the scene.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Zero Mock Policy:</strong> 100% genuine algorithmic execution on real and synthetic lunar image pairs.</span>
            </li>
          </ul>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-base font-bold text-white mb-3 flex items-center gap-2">
            <span>🛰️</span> Data Sources &amp; Compliance
          </h2>
          <p className="text-xs text-gray-300 leading-relaxed mb-3">
            Designed to ingest PDS4 / GeoTIFF lunar data from:
          </p>
          <ul className="space-y-1.5 text-xs text-gray-400">
            <li>• <strong>ISDC Pradan Portal:</strong> Chandrayaan-2 browse &amp; calibrated products (<a href="https://pradan.issdc.gov.in" target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline">pradan.issdc.gov.in</a>)</li>
            <li>• <strong>ISRO CHMAP:</strong> Planetary Data System browse archive</li>
            <li>• <strong>NASA PDS / LROC:</strong> Lunar Reconnaissance Orbiter NAC/WAC baseline products</li>
          </ul>
          <div className="mt-4 pt-4 border-t border-white/10 text-[11px] text-gray-500">
            Branding disclaimer: Respectful space theme used in accordance with SIH guidelines without unauthorized emblem usage.
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <Link href="/workbench" className="btn-primary inline-block">
          Open Registration Workbench →
        </Link>
      </div>
    </div>
  );
}
