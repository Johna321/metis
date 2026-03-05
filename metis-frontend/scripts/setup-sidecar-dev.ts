/**
 * Creates the dev sidecar wrapper for metis-web.
 *
 * Tauri needs a real binary at src-tauri/binaries/metis-web-<target-triple>.
 * During development this delegates to `uv run metis-web`:
 *   - Unix:    a shell script (exec uv run ...)
 *   - Windows: a compiled Bun executable that spawns uv run ...
 *
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

const isWindows = target.includes("windows");

if (isWindows) {
  // Compile a tiny Bun executable that spawns uv as a child and proxies signals.
  const tmpSrc = join(binariesDir, "_metis_web_wrapper.ts");
  writeFileSync(
    tmpSrc,
    [
      `import { spawn } from "child_process";`,
      `const proc = spawn("uv", ["run", "--project", ${JSON.stringify(backendDir)}, "metis-web", ...process.argv.slice(2)], { stdio: "inherit" });`,
      `process.on("SIGTERM", () => proc.kill("SIGTERM"));`,
      `process.on("SIGINT",  () => proc.kill("SIGINT"));`,
      `proc.on("exit", (code) => process.exit(code ?? 0));`,
    ].join("\n")
  );

  const outExe = join(binariesDir, `metis-web-${target}.exe`);
  await $`bun build --compile --target=bun-windows-x64 ${tmpSrc} --outfile ${outExe}`;
  console.log(`Dev sidecar wrapper written: ${outExe}`);
} else {
  const wrapperPath = join(binariesDir, `metis-web-${target}`);
  const script = `#!/usr/bin/env bash\nexec uv run --project "${backendDir}" metis-web "$@"\n`;
  writeFileSync(wrapperPath, script);
  chmodSync(wrapperPath, 0o755);
  console.log(`Dev sidecar wrapper written: ${wrapperPath}`);
}
