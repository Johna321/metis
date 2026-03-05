/**
 * Builds a standalone metis-web binary with PyInstaller for Tauri bundling.
 *
 * Requirements: `uv add --dev pyinstaller` in the backend project.
 * Usage: bun run build:sidecar
 */
import { $ } from "bun";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { mkdirSync, copyFileSync, chmodSync, existsSync } from "fs";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");
const backendDir = join(repoRoot, "backend");
const binariesDir = join(repoRoot, "metis-frontend/src-tauri/binaries");

const target = (await $`rustc -vV`.text())
  .split("\n")
  .find((l) => l.startsWith("host:"))
  ?.split(" ")[1]
  ?.trim();

if (!target) throw new Error("Could not determine rustc host triple");

console.log(`Building metis-web sidecar for ${target}...`);

await $`uv run pyinstaller \
  --onefile \
  --name metis-web \
  --collect-all sentence_transformers \
  --collect-all pymupdf \
  src/metis/adapters/web.py`.cwd(backendDir);

const src = join(backendDir, "dist/metis-web");
if (!existsSync(src)) throw new Error(`PyInstaller output not found: ${src}`);

mkdirSync(binariesDir, { recursive: true });

const dest = join(binariesDir, `metis-web-${target}`);
copyFileSync(src, dest);
chmodSync(dest, 0o755);

console.log(`Sidecar binary written: ${dest}`);
