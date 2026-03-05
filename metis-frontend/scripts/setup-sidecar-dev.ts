/**
 * Creates the dev sidecar wrapper for metis-web.
 *
 * Tauri needs a real binary at src-tauri/binaries/metis-web-<target-triple>.
 * During development this is a shell script that delegates to `uv run metis-web`.
 * For production bundles, run `bun run build:sidecar` instead (uses PyInstaller).
 *
 * Usage: bun run setup:sidecar
 */
import { $ } from "bun";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { mkdirSync, writeFileSync, chmodSync } from "fs";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");
const binariesDir = join(repoRoot, "metis-frontend/src-tauri/binaries");
const backendDir = join(repoRoot, "backend");

const target = (await $`rustc -vV`.text())
  .split("\n")
  .find((l) => l.startsWith("host:"))
  ?.split(" ")[1]
  ?.trim();

if (!target) throw new Error("Could not determine rustc host triple");

mkdirSync(binariesDir, { recursive: true });

const wrapperPath = join(binariesDir, `metis-web-${target}`);
const script = `#!/usr/bin/env bash\nexec uv run --project "${backendDir}" metis-web "$@"\n`;

writeFileSync(wrapperPath, script);
chmodSync(wrapperPath, 0o755);

console.log(`Dev sidecar wrapper written: ${wrapperPath}`);
