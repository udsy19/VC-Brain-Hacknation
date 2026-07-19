import path from "node:path";
import type { NextConfig } from "next";

/**
 * `turbopack.root` is pinned to this directory deliberately.
 *
 * The repo root grew its own package.json + pnpm-lock.yaml so Vercel can detect Next
 * and build the frontend and the Python function from one deployment. That left TWO
 * lockfiles above this app, and Turbopack infers its root by walking up to the nearest
 * one — so it can resolve to the repo root instead of here. When it does, the client
 * manifest is keyed against a different root than the one the server components were
 * compiled under, and the dev server fails with:
 *
 *   Could not find the module "[project]/app/app/page.tsx#default" in the React Client
 *   Manifest.
 *
 * which reads like a bundler bug and is really an ambiguous root. Pinning it makes the
 * resolution explicit rather than dependent on which lockfile is found first.
 */
const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
