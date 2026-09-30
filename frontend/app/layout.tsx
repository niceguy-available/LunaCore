import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Lună — Lunar Image Registration Platform",
  description: "Intelligent, illumination-invariant orbital mapping and sub-pixel optical image registration for lunar exploration.",
};

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/workbench", label: "Workbench" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/compare", label: "Compare" },
  { href: "/methodology", label: "Methodology" },
  { href: "/about", label: "About" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#000000] text-[#F6F6F6]">
        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 glass-strong">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <Link href="/" className="flex items-center gap-3 group">
                <img
                  src="/ISRO(logo).png"
                  alt="Lună Logo"
                  className="navbar-logo"
                />
                <span className="navbar-title">Lună</span>
              </Link>

              {/* Nav links */}
              <div className="hidden md:flex items-center gap-1">
                {NAV_ITEMS.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="px-3.5 py-2 text-sm text-[#F6F6F6]/80 hover:text-[#CFFFE2] hover:bg-[#A2D5C6]/10 rounded-lg transition-all duration-200"
                  >
                    {item.label}
                  </Link>
                ))}
              </div>

              {/* Mobile menu button */}
              <button className="md:hidden p-2 text-[#F6F6F6]/80 hover:text-white" aria-label="Menu">
                <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="3" y1="6" x2="21" y2="6"/>
                  <line x1="3" y1="12" x2="21" y2="12"/>
                  <line x1="3" y1="18" x2="21" y2="18"/>
                </svg>
              </button>
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="pt-16 min-h-screen">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-[#A2D5C6]/20 bg-[#000000] py-10 px-4">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-[#F6F6F6]/60">
            <p className="flex items-center gap-2">
              <span className="text-[#A2D5C6] font-semibold">Lună</span> — Intelligent Multi-Modal Lunar Image Registration
            </p>
            <p>Advanced optical correspondence engine for planetary science</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
