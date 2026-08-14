import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */

  // Lets you open the dev server from your phone (or any other device on
  // the same network) at http://<your-PC's-LAN-IP>:3000. Next.js blocks
  // this by default as a dev-server security measure — the IP below is
  // exactly what Next.js reported blocking, taken from the error message.
  //
  // If this stops working later, it's most likely because your phone (or
  // PC) got a new IP from DHCP — check the new error message for the
  // current blocked address and add it here too.
  allowedDevOrigins: ["172.24.0.1"],
};

export default nextConfig;