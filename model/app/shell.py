"""Interactive and developer-shell surfaces for a compiled model runtime."""

from time import perf_counter

from app.workflow import run_demo_predictions
from utils.eval import evaluate_model


def _print_predictions(predictions):
    """Render top-k predictions in the same compact console format everywhere."""
    if len(predictions) == 0:
        print("No prediction.")
        return

    prediction_text = ", ".join(
        f"{word} ({score:.3f})"
        for word, score in predictions
    )
    print(prediction_text)


def _print_generated(tokens):
    """Render generated output from the autoregressive shell path."""
    print("generated:")
    print(" ".join(tokens))


def _print_timing(label, elapsed):
    """Print shell command timings using the same alignment as the main CLI."""
    print(f"{label + ':':<23}{elapsed:.3f} sec")


def _print_interactive_help():
    """Show the lightweight end-user interactive command surface."""
    print("\nCommands:")
    print("/reset  - clear context memory")
    print("/exit   - quit interactive mode")
    print("/quit   - quit interactive mode")
    print("/help   - show commands")


def _print_shell_help():
    """Show the broader developer-shell command surface."""
    print("\nDeveloper shell commands:")
    print("/status            - show runtime, compile, and invocation state")
    print("/reset             - clear context memory")
    print("/predict <text>    - score the next token after the provided text")
    print("/generate <text>   - continue autoregressively from the provided text")
    print("/eval              - run evaluation with the loaded checkpoint/runtime")
    print("/demo              - print sample next-token predictions")
    print("/load              - reload the configured checkpoint into this runtime")
    print("/exit              - quit the developer shell")
    print("/quit              - quit the developer shell")
    print("/help              - show commands")


def _print_status(runtime, checkpoint_path):
    """Summarize the current compiled runtime so compile-heavy sessions stay legible."""
    snapshot = runtime.runtime_status_snapshot()
    training = snapshot["training"]
    invocations = snapshot["invocations"]
    compile_fingerprint = snapshot["compile_fingerprint"] or {}
    compile_profile = compile_fingerprint.get("compile_profile", {})

    print("\nRuntime status:\n")
    print(f"checkpoint:              {checkpoint_path}")
    print(f"training mode:           {training['training_mode']}")
    print(f"token duration:          {training['token_duration']:.6f} sec")
    print(f"token duration source:   {training['token_duration_source']}")
    print(f"step time:               {snapshot['step_time']:.6f} sec")
    print(f"present calls:           {invocations['present_calls']}")
    print(f"reset_context calls:     {invocations['reset_context_calls']}")
    print(f"sim.run calls:           {invocations['sim_run_count']}")
    print(f"simulated seconds:       {invocations['simulated_seconds']:.3f}")
    print(f"sim.run wall seconds:    {invocations['sim_run_seconds']:.3f}")

    if compile_fingerprint:
        print(f"backend:                 {compile_fingerprint.get('backend')}")
        print(f"OpenCL platform:         {compile_fingerprint.get('opencl_platform')}")
        print(f"OpenCL device:           {compile_fingerprint.get('opencl_device')}")
        print(f"probe mode:              {compile_fingerprint.get('probe_mode')}")
        print(f"compile profile:         {compile_profile.get('name', 'full')}")


def _run_eval(runtime, testing_set, max_examples, top_k):
    """Run evaluation inside the existing compiled runtime and report its timing."""
    start = perf_counter()
    result = evaluate_model(
        runtime,
        testing_set,
        max_examples=max_examples,
        top_k=top_k,
    )
    _print_timing("Evaluation", perf_counter() - start)
    runtime.reset_context()
    print("[evaluation complete; context reset]")
    return result


def _run_demo(runtime, testing_set, max_examples, top_k):
    """Run qualitative demo predictions inside the existing compiled runtime."""
    start = perf_counter()
    run_demo_predictions(
        runtime,
        testing_set,
        max_examples=max_examples,
        top_k=top_k,
    )
    _print_timing("Demo", perf_counter() - start)
    runtime.reset_context()
    print("[demo complete; context reset]")


def _split_command(text):
    """Split a slash command into its normalized verb and optional argument text."""
    command, _, argument = text.partition(" ")
    return command.lower(), argument.strip()


def _handle_interactive_text(runtime, text, top_k, generate, max_tokens):
    """Process free-form user input in the lightweight interactive prompt."""
    if generate:
        output = runtime.generate(
            text,
            max_tokens=max_tokens,
            top_k=top_k,
            reset_context=False,
            verbose=False,
        )
        _print_generated(output)
        return

    predictions = runtime.interactive_predict(
        text,
        top_k=top_k,
        reset_context=False,
    )
    _print_predictions(predictions)


def _handle_shell_text(runtime, text, top_k):
    """Treat bare text in the developer shell as a convenience predict command."""
    predictions = runtime.interactive_predict(
        text,
        top_k=top_k,
        reset_context=False,
    )
    _print_predictions(predictions)


def _handle_slash_command(
    runtime,
    command,
    argument,
    *,
    mode,
    checkpoint_path,
    testing_set,
    top_k,
    max_tokens,
    max_examples,
    max_demo_examples,
):
    """Execute one slash command and report whether the prompt loop should continue."""
    if command in ("/exit", "/quit"):
        return False

    if command == "/reset":
        runtime.reset_context()
        print("[context reset]")
        return True

    if command == "/help":
        if mode == "interactive":
            _print_interactive_help()
        else:
            _print_shell_help()
        return True

    if mode == "interactive":
        print(f"Unknown command: {command}")
        return True

    if command == "/status":
        _print_status(runtime, checkpoint_path)
        return True

    if command == "/predict":
        if len(argument) == 0:
            print("Usage: /predict <text>")
            return True
        _handle_shell_text(runtime, argument, top_k=top_k)
        return True

    if command == "/generate":
        if len(argument) == 0:
            print("Usage: /generate <text>")
            return True
        output = runtime.generate(
            argument,
            max_tokens=max_tokens,
            top_k=top_k,
            reset_context=False,
            verbose=False,
        )
        _print_generated(output)
        return True

    if command == "/eval":
        _run_eval(
            runtime,
            testing_set,
            max_examples=max_examples,
            top_k=top_k,
        )
        return True

    if command == "/demo":
        _run_demo(
            runtime,
            testing_set,
            max_examples=max_demo_examples,
            top_k=top_k,
        )
        return True

    if command == "/load":
        runtime.load_checkpoint(checkpoint_path)
        runtime.reset_context()
        print("[checkpoint reloaded; context reset]")
        return True

    print(f"Unknown command: {command}")
    return True


def _prompt_loop(
    runtime,
    *,
    mode,
    top_k,
    generate,
    max_tokens,
    checkpoint_path=None,
    testing_set=None,
    max_examples=50,
    max_demo_examples=10,
):
    """Run one prompt loop while sharing slash-command behavior across surfaces."""
    while True:
        try:
            prompt = "shell> " if mode == "shell" else ">>> "
            text = input(prompt).strip()
        except KeyboardInterrupt:
            break

        if len(text) == 0:
            continue

        if text.startswith("/"):
            command, argument = _split_command(text)
            should_continue = _handle_slash_command(
                runtime,
                command,
                argument,
                mode=mode,
                checkpoint_path=checkpoint_path,
                testing_set=testing_set,
                top_k=top_k,
                max_tokens=max_tokens,
                max_examples=max_examples,
                max_demo_examples=max_demo_examples,
            )
            if not should_continue:
                break
            continue

        if mode == "shell":
            _handle_shell_text(runtime, text, top_k=top_k)
        else:
            _handle_interactive_text(
                runtime,
                text,
                top_k=top_k,
                generate=generate,
                max_tokens=max_tokens,
            )


def launch_interactive_prompt(runtime, top_k=5, generate=False, max_tokens=20):
    """Launch the lightweight user-facing prompt without recompiling the model."""
    print("\nRealtime interactive mode")
    print("Type '/exit' to quit")
    print("Type '/reset' to clear context")
    print("Type '/help' to show commands\n")

    runtime.reset_context()
    _prompt_loop(
        runtime,
        mode="interactive",
        top_k=top_k,
        generate=generate,
        max_tokens=max_tokens,
    )
    print("\nExiting interactive mode.")


def launch_runtime_shell(
    runtime,
    testing_set,
    checkpoint_path,
    *,
    top_k=5,
    max_tokens=20,
    max_examples=50,
    max_demo_examples=10,
):
    """Launch the developer shell on top of the already-compiled runtime."""
    print("\nDeveloper runtime shell")
    print("Compile cost is already paid for this session.")
    print("Use '/help' for commands.")
    print("Context persists until you run '/reset'.\n")

    runtime.reset_context()
    _prompt_loop(
        runtime,
        mode="shell",
        top_k=top_k,
        generate=False,
        max_tokens=max_tokens,
        checkpoint_path=checkpoint_path,
        testing_set=testing_set,
        max_examples=max_examples,
        max_demo_examples=max_demo_examples,
    )
    print("\nExiting developer shell.")
