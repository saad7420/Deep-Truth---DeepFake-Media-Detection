import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
import Providers from "./components/providers";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
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
  themeColor: "#0a101d",
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
      className={`${plexSans.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <body className="bg-slate-950 text-slate-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
