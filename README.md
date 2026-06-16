This project is an attempt to create a NLP model using Nengo, specifically NengoSPA. Learn more about it at https://www.nengo.ai/.

Originally, it started as my final project for AMATH 445 (Scientific Machine Learning) at the University of Waterloo. See https://github.com/gnahZnitsuJ/F24-AMATH-445.

## Running

The default behavior now uses a cheaper development loop:

```bash
python model/main.py
```

That default path trains or loads the model, records telemetry, and stops before
evaluation, demo prediction dumps, or interactive mode.

Use `--full` when you want the original end-to-end behavior:

```bash
python model/main.py --full
```

You can now run individual stages during development:

```bash
python model/main.py --train --no-eval --no-demo --no-interactive
python model/main.py --eval --max-examples 50
python model/main.py --demo --max-demo-examples 10 --top-k 3
python model/main.py --interactive --generate --top-k 5 --max-tokens 15
```

Compile benchmark modes are available directly from `main.py`:

```bash
python model/main.py --benchmark compile-current
python model/main.py --benchmark compile-components
python model/main.py --benchmark compile-full
```

Useful flags:

- `--full`
- `--checkpoint-path PATH`
- `--force-retrain`
- `--max-examples N`
- `--max-demo-examples N`
- `--top-k N`
- `--generate`
- `--max-tokens N`
- `--no-eval`
- `--no-demo`
- `--no-interactive`

## Interactive Commands

Inside interactive mode, slash commands are supported:

- `/help`
- `/reset`
- `/exit`
- `/quit`

Telemetry is written locally to `model/results/` as timestamped `telemetry_*.json` files.
