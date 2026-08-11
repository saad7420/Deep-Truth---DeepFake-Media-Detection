import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import Providers from "./components/providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

/* Display face. Space Grotesk's squared terminals and single-storey `a`
   read as instrument panel rather than marketing page — it carries the
   forensic-console personality that Geist alone doesn't. Used only for
   headings and verdicts, never body copy. */
const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Deep Truth — Active Deepfake Defense System",
    template: "%s · Deep Truth",
  },
  description:
    "Multi-modal forensic verification for audio, video and images. Detect synthetic manipulation with fused visual, audio and noise-residual analysis.",
  applicationName: "Deep Truth",
  authors: [{ name: "Saad Mehmood" }, { name: "Ramish Naseer" }],
  keywords: ["deepfake detection", "digital forensics", "media verification", "synthetic media"],
};

export const viewport: Viewport = {
  themeColor: "#020617",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${geistSans.variable} ${geistMono.variable} ${spaceGrotesk.variable}`}
      suppressHydrationWarning
    >
      <body className="bg-slate-950 text-slate-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}