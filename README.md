This project is an attempt to create a NLP model using Nengo, specifically NengoSPA. Learn more about it at https://www.nengo.ai/.

Originally, it started as my final project for AMATH 445 (Scientific Machine Learning) at the University of Waterloo. See https://github.com/gnahZnitsuJ/F24-AMATH-445.

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
- `--compare-current-architecture`
  with `--build-only --inspect-checkpoint`, compare checkpoint metadata against the current build signature
- `--checkpoint-path PATH`
  choose which checkpoint file to load or write under `model/checkpoints/`
- `--architecture root-context-v1|no-refiner-v1`
  assemble the established baseline or the checkpoint-incompatible mechanical
  variant that exposes the context predictor without the top-level refiner
- `--force-retrain`
  ignore an existing checkpoint and retrain from scratch
- `--compile-profile full|fast-solver`
  choose the build profile; `fast-solver` lowers ensemble eval-point counts to reduce solver/setup cost
- `--learned-init-mode random-function|zero-nosolver|seeded-nosolver`
  choose how PES-learned decoded connections are initialized for compile/workflow experiments
- `--learned-init-seed N`
  provide a deterministic seed for seeded learned-connection initialization
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
`present_calls` and `reset_context_calls`. Use `--no-telemetry` to skip this
recording when you want the leanest run possible.

OpenCL selection can also be controlled through environment variables:
- `CANVAS_OPENCL_PLATFORM_INDEX`
- `CANVAS_OPENCL_DEVICE_INDEX`

CLI flags take precedence over those environment defaults.
