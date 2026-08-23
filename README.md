This project is an attempt to create a NLP model using Nengo, specifically NengoSPA. Learn more about it at https://www.nengo.ai/.

Originally, it started as my final project for AMATH 445 (Scientific Machine Learning) at the University of Waterloo. See https://github.com/gnahZnitsuJ/F24-AMATH-445.

## Requirements

The recorded working environment uses Python 3.10.11 on Windows. Python package
versions are pinned in `requirements.txt` because the Nengo, NumPy, and SciPy
versions participate in decoder-cache and checkpoint compatibility.

The full runtime also requires:

- a working OpenCL implementation supplied by your GPU or CPU driver
- the NLTK Reuters corpus, which is data and is therefore not installed by pip

From PowerShell, create an isolated environment and install everything with:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader reuters
```

Verify the package installation and OpenCL discovery before starting an
expensive build:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe model/main.py --check-environment
.\.venv\Scripts\python.exe model/main.py --dry-run
```

`--check-environment` verifies every direct package and pinned version, imports
the installed modules, runs `pip check`, loads the Reuters corpus, and enumerates
OpenCL devices.
Normal model runs perform only a lightweight missing-package bootstrap check;
corpus and OpenCL failures are checked when those resources are first used.

The remaining examples use `python` for readability. Either activate the
environment with `.\.venv\Scripts\Activate.ps1` first or replace `python` with
`.\.venv\Scripts\python.exe`.

## Running

The default behavior now uses a cheaper development loop:

```bash
python model/main.py
```

That default path trains or loads the model, records telemetry, and stops before
evaluation, demo prediction dumps, or interactive mode.

If you want the leanest normal run and do not need a results file, add
`--no-telemetry` to skip telemetry recording entirely.

Use `--full` when you want the original end-to-end behavior:

```bash
python model/main.py --full
```

This runs the whole user-facing workflow in one go:
- train or load a checkpoint
- run evaluation
- print sample next-token predictions
- launch interactive mode

You can now run individual stages during development:

```bash
python model/main.py --train --no-eval --no-demo --no-interactive
python model/main.py --eval --max-examples 50
python model/main.py --demo --max-demo-examples 10 --top-k 3
python model/main.py --interactive --generate --top-k 5 --max-tokens 15
python model/main.py --shell --top-k 5 --max-tokens 15
python model/main.py --train --probe-mode minimal
python model/main.py --train --no-eval --opencl-platform-index 0 --opencl-device-index 0
python model/main.py --dry-run --compile-profile fast-solver
python model/main.py --build-only --probe-mode minimal --compile-profile fast-solver
python model/main.py --inspect-checkpoint --checkpoint-path reuters_checkpoint.pkl
python model/main.py --build-only --inspect-checkpoint --compare-current-architecture
python model/main.py --dry-run --tokenizer bpe-v1
python model/main.py --inspect-decoder-cache
```

- `python model/main.py --train --no-eval --no-demo --no-interactive`
  trains or loads the model, writes telemetry, and exits. This is the most useful
  quick smoke-test path when you want to verify that the model still builds and
  checkpoint loading still works.
- `python model/main.py --eval --max-examples 50`
  loads the checkpoint, evaluates up to `50` next-token prediction examples, prints
  the evaluation result, records telemetry, and exits.
- `python model/main.py --demo --max-demo-examples 10 --top-k 3`
  loads the checkpoint and prints up to `10` human-readable sample predictions with
  the top `3` candidates for each prefix. This is useful for a quick qualitative
  check of model behavior.
- `python model/main.py --interactive --generate --top-k 5 --max-tokens 15`
  loads the checkpoint and opens the realtime prompt. With `--generate`, the model
  continues autoregressively after your prompt, showing up to `5` candidates per
  step and stopping after `15` generated tokens unless you end earlier.
- `python model/main.py --shell --top-k 5 --max-tokens 15`
  loads the checkpoint and opens a persistent developer shell on top of the
  already-compiled runtime. That shell can run status, reset, predict, generate,
  eval, demo, and checkpoint reload commands without paying compile cost again.
- `python model/main.py --train --probe-mode minimal`
  builds the normal workflow with only the required prediction probe. This is
  the lighter instrumentation mode for compile-sensitive development runs.
- `python model/main.py --train --no-eval --opencl-platform-index 0 --opencl-device-index 0`
  runs the normal workflow but pins execution to a specific OpenCL platform and
  device index, which is useful on machines with multiple OpenCL providers.
- `python model/main.py --dry-run --compile-profile fast-solver`
  resolves the workflow, runtime settings, checkpoint target, and compile knobs
  without loading data or building the Nengo model.
- `python model/main.py --build-only --probe-mode minimal --compile-profile fast-solver`
  loads data, builds the Python Nengo model, reports build timing and network
  complexity, and stops before simulator compilation.
- `python model/main.py --inspect-checkpoint --checkpoint-path reuters_checkpoint.pkl`
  prints saved checkpoint metadata, including compile-profile and learned-init
  information, without building or compiling the model.
- `python model/main.py --build-only --inspect-checkpoint --compare-current-architecture`
  combines checkpoint inspection with a current build-only pass so you can
  compare the saved architecture signature against the present build before
  paying OpenCL compile cost.
- `python model/main.py --dry-run --tokenizer bpe-v1`
  selects a versioned tokenizer profile. Available profiles are `word-v1`,
  `bpe-v1`, `unigram-v1`, `character-v1`, and `byte-v1`. Tokenizer changes use
  independent seed-vector caches and are checkpoint-incompatible by design.
- `python model/main.py --inspect-decoder-cache`
  reports the persistent Nengo decoder-cache path, dependency namespace, size,
  and file count without loading data or constructing a simulator.

### Persistent decoder solves

Scientific `random-function` builds now use the committed learned-initialization
seed (`42`) and Nengo's exact disk decoder cache. The first compatible build
performs the normal full solves; later processes and machine restarts reuse the
stored decoder arrays instead of solving them again. Nengo's numerical cache key
remains authoritative, so changes to solver inputs, evaluation points, targets,
or RNG state produce a cache miss and a fresh solve. Canvas also verifies the
stored decoder shape, dtype, and SHA-256 digest before accepting a reuse hit.

Canvas keeps these generated artifacts outside the OneDrive checkout in a
project-specific, dependency-versioned machine-local directory. The default
cache budget is 4 GiB. Set `CANVAS_DECODER_CACHE_DIR` to relocate the root.
This decoder state is independent of PES training checkpoints: the cache holds
the initialized NEF solves, while checkpoints hold learned runtime weights.

Use `--decoder-cache-mode refresh` to clear and rebuild the active Canvas
dependency namespace, or `--decoder-cache-mode off` for a controlled run that
neither reads nor writes decoder artifacts. Normal and benchmark telemetry
records solver reuse/solve counts and per-learned-connection outcomes. Because
the deterministic default establishes a new canonical initialization, older
checkpoints whose learned-init seed was recorded as `None` are intentionally
checkpoint-incompatible.

Compare how profiles segment representative text without building Nengo:

```bash
python scripts/compare_tokenizers.py --text "Café can't cost 12.50€ 👩🏽‍💻."
python scripts/compare_tokenizers.py --reuters-docs 2
```

The comparison reports token counts, unique-token counts, normalized round-trip
behavior, fingerprints, and a token preview. Repeat `--tokenizer PROFILE` to
limit the comparison or add `--json` for machine-readable output.

Compile benchmark modes are available directly from `main.py`:

```bash
python model/main.py --benchmark compile-current
python model/main.py --benchmark compile-components
python model/main.py --benchmark compile-full
python model/main.py --benchmark compile-repeat-current --benchmark-repeats 2
```

- `compile-current`
  benchmarks the current model configuration and records compile/build telemetry for
  the main architecture as it is currently set up.
- `compile-components`
  runs component-level benchmark cases so you can compare the relative cost of
  pieces like `ContextModule`, `BaseComponent`, and related structures.
- `compile-full`
  runs the broader benchmark suite, including scaling-oriented cases, for deeper
  compile-time investigation.
- `compile-repeat-current`
  recompiles the current architecture repeatedly in one process so you can
  compare cold-versus-warm compile behavior and optional post-compile warmup cost.

Benchmark runs now produce both:
- raw timestamped telemetry JSON in `model/results/`
- a timestamped markdown summary in `model/results/` that is easier to paste into notes
- explicit OpenCL platform/device reporting in both console output and saved telemetry
- active probe-mode reporting plus created/skipped probe labels in telemetry

## Named Development Workflows

The transparent PowerShell wrapper keeps the baseline and development
configurations explicit and prints the complete resolved command before running it:

```powershell
./scripts/model_workflows.ps1 plan
./scripts/model_workflows.ps1 build-check
./scripts/model_workflows.ps1 checkpoint-check
./scripts/model_workflows.ps1 architecture-check
./scripts/model_workflows.ps1 compile-baseline
./scripts/model_workflows.ps1 compile-dev
./scripts/model_workflows.ps1 shell-dev
```

The scientific baseline is `debug + full + random-function`. The non-baseline
development configuration is `minimal + fast-solver + zero-nosolver`; it has
different checkpoint compatibility and initial learning conditions. Use
`-ShowOnly` to inspect a workflow without executing it and pass additional model
arguments after the wrapper options when needed.

## Testing

Run the hermetic automated suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The suite combines independently derived behavioral contracts with focused
regression tests. The accepted-contract modules use hand-calculated examples,
state-recording fakes, boundary values, exact JSON round trips, and public CLI
outcomes rather than reproducing implementation tables or call transcripts.

For a quick check that important pure-contract tests can actually detect
plausible faults, run the deterministic mutation sentinels:

```bash
python scripts/verify_test_sentinels.py
```

The script copies only the required source and tests into a temporary directory,
applies seven exact mutations, and requires the owning tests to fail. It never
mutates the working tree. These sentinels cover architecture contracts and
telemetry comparison logic; they do not replace Nengo build, simulator, OpenCL,
checkpoint, or scientific-quality validation.

## Comparing Telemetry

Compare any two or more compile, run, or build-only telemetry files without
loading Nengo:

```bash
python scripts/compare_telemetry.py model/results/before.json model/results/after.json --vary compile_profile --markdown comparison.md --csv comparison.csv
```

The first selected record is the delta reference. The tool prints a concise
console table, checks backend/device/dimension/context/profile/init controls,
warns about unintended differences, and includes absolute and percentage deltas.
Use `--strict` to fail on unexpected control changes, `--since`/`--until` for ISO
timestamp filtering, or repeat `--where FIELD=VALUE` to filter normalized
fingerprint fields. Run `python scripts/compare_telemetry.py --help` for the field
and output options.

Useful flags:

- `--full`
  run the full workflow instead of the cheaper default path
- `--dry-run`
  print the resolved workflow and runtime plan without building the Nengo model
- `--build-only`
  build the Python Nengo model and stop before simulator compilation
- `--inspect-checkpoint`
  inspect checkpoint metadata without building or compiling the model
- `--inspect-decoder-cache`
  inspect persistent decoder-cache location and storage without building or compiling the model
- `--compare-current-architecture`
  with `--build-only --inspect-checkpoint`, compare checkpoint metadata against the current build signature
- `--checkpoint-path PATH`
  choose which checkpoint file to load or write under `model/checkpoints/`
- `--architecture root-context-v1|no-refiner-v1`
  assemble the established baseline or the checkpoint-incompatible mechanical
  variant that exposes the context predictor without the top-level refiner
- `--tokenizer word-v1|bpe-v1|unigram-v1|character-v1|byte-v1`
  select the text unit advanced through the neural model; learned subword
  profiles are fitted deterministically on the selected training partition
- `--tokenizer-vocab-size N`
  set the vocabulary budget for BPE and unigram fitting
- `--tokenizer-normalization NFC|NFKC`
  choose the Unicode normalization policy included in tokenizer identity
- `--tokenizer-max-subword-length N`
  bound candidate piece length for the compact unigram profile
- `--force-retrain`
  ignore an existing checkpoint and retrain from scratch
- `--compile-profile full|fast-solver`
  choose the build profile; `fast-solver` lowers ensemble eval-point counts to reduce solver/setup cost
- `--learned-init-mode random-function|zero-nosolver|seeded-nosolver`
  choose how PES-learned decoded connections are initialized for compile/workflow experiments
- `--learned-init-seed N`
  override the deterministic learned-connection seed (default `42`); changes invalidate decoder reuse and checkpoint compatibility
- `--decoder-cache-mode auto|refresh|off`
  reuse compatible decoder solves, clear and regenerate the active Canvas cache namespace, or bypass cache reads and writes
- `--max-examples N`
  cap how many evaluation examples are processed
- `--max-demo-examples N`
  cap how many demo predictions are printed
- `--top-k N`
  choose how many candidate predictions to show in evaluation, demo output, and interactive mode
- `--generate`
  enable autoregressive generation in interactive mode
- `--max-tokens N`
  limit how many tokens interactive generation may continue for
- `--shell`
  launch the developer runtime shell, which reuses the currently compiled runtime
  for commands like status, prediction, evaluation, and checkpoint reload
- `--probe-mode minimal|debug`
  choose how much build-time probe instrumentation to keep; `minimal` keeps only
  the required prediction probe, while `debug` keeps the richer diagnostic probes
- `--opencl-platform-index N`
  choose which OpenCL platform index to use; defaults to `CANVAS_OPENCL_PLATFORM_INDEX` if set, otherwise `0`
- `--opencl-device-index N`
  choose which device index to use within the selected OpenCL platform; defaults to `CANVAS_OPENCL_DEVICE_INDEX` if set, otherwise `0`
- `--no-telemetry`
  disable telemetry recording and skip writing a `telemetry_*.json` results file for the run
- `--benchmark-repeats N`
  choose how many times `compile-repeat-current` recompiles the current architecture
- `--include-first-run-warmup`
  for `compile-repeat-current`, run one post-compile warmup step per repeat and record its cost
- `--no-eval`
  skip evaluation when using a workflow that would otherwise include it
- `--no-demo`
  skip demo prediction output when using a workflow that would otherwise include it
- `--no-interactive`
  skip the interactive prompt when using a workflow that would otherwise include it

## Interactive Commands

Inside interactive mode, slash commands are supported:

- `/help`
- `/reset`
- `/exit`
- `/quit`

## Developer Shell Commands

Inside the developer shell, slash commands are supported:

- `/help`
  print the available shell commands and their behavior
- `/status`
  show the active checkpoint path, training configuration, compile backend/device,
  and accumulated simulator invocation counters
- `/reset`
  clear the persistent context state so the next prompt starts from a fresh sequence boundary
- `/predict <text>`
  feed the provided text through the current context path and print the next-token candidates
- `/generate <text>`
  feed the provided text through the current context path and continue autoregressively
- `/eval`
  run the standard evaluation routine against the loaded testing set without recompiling
- `/demo`
  print sample next-token predictions from the testing set without recompiling
- `/load`
  reload the configured checkpoint into the current runtime and reset context afterward
- `/exit`
- `/quit`

Bare text in the developer shell is treated as a shorthand prediction request.
Context persists across `/predict`, `/generate`, and bare-text prompts until you
run `/reset`, so sequence boundaries remain explicit.

Telemetry is written locally to `model/results/` as timestamped `telemetry_*.json` files.
These files now include aggregate simulator activity plus explicit
`present_calls`, `reset_context_calls`, and decoder-cache reuse/solve outcomes.
Use `--no-telemetry` to skip this
recording when you want the leanest run possible.

OpenCL selection can also be controlled through environment variables:
- `CANVAS_OPENCL_PLATFORM_INDEX`
- `CANVAS_OPENCL_DEVICE_INDEX`
- `CANVAS_DECODER_CACHE_DIR`

CLI flags take precedence over those environment defaults.
