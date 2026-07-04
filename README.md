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
python model/main.py --train --no-eval --opencl-platform-index 0 --opencl-device-index 0
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
- `python model/main.py --train --no-eval --opencl-platform-index 0 --opencl-device-index 0`
  runs the normal workflow but pins execution to a specific OpenCL platform and
  device index, which is useful on machines with multiple OpenCL providers.

Compile benchmark modes are available directly from `main.py`:

```bash
python model/main.py --benchmark compile-current
python model/main.py --benchmark compile-components
python model/main.py --benchmark compile-full
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

Benchmark runs now produce both:
- raw timestamped telemetry JSON in `model/results/`
- a timestamped markdown summary in `model/results/` that is easier to paste into notes
- explicit OpenCL platform/device reporting in both console output and saved telemetry

Useful flags:

- `--full`
  run the full workflow instead of the cheaper default path
- `--checkpoint-path PATH`
  choose which checkpoint file to load or write under `model/checkpoints/`
- `--force-retrain`
  ignore an existing checkpoint and retrain from scratch
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
- `--opencl-platform-index N`
  choose which OpenCL platform index to use; defaults to `CANVAS_OPENCL_PLATFORM_INDEX` if set, otherwise `0`
- `--opencl-device-index N`
  choose which device index to use within the selected OpenCL platform; defaults to `CANVAS_OPENCL_DEVICE_INDEX` if set, otherwise `0`
- `--no-telemetry`
  disable telemetry recording and skip writing a `telemetry_*.json` results file for the run
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
