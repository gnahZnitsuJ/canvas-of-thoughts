"""Compatibility entrypoint for assembling the selected model architecture."""

from architecture.assembly import AssembledModelResult, assemble_architecture
from architecture.components import default_component_registry
from architecture.contracts import ArchitectureBuildContext
from architecture.variants import architecture_spec
from config import model_parameters as mp
from utils.probes import ProbeRegistry


# Preserve the historical public name for callers that import ModelResult.
ModelResult = AssembledModelResult


def Model(
    sub_lengths,
    model_vocab,
    strict=mp.strict_vocab,
    probe_mode="debug",
    learned_init_mode="random-function",
    learned_init_seed=None,
    compile_profile_name="full",
    compile_profile_settings=None,
    architecture_name="root-context-v1",
):
    """Assemble one validated architecture behind the legacy model facade.

    The default specification reproduces the established root-context path.
    ``sub_lengths`` remains compatibility/telemetry metadata while legacy
    per-component contexts stay deferred.
    """
    probe_registry = ProbeRegistry(mode=probe_mode)
    compile_profile = {
        "name": compile_profile_name,
        "settings": dict(compile_profile_settings or {}),
    }
    context = ArchitectureBuildContext(
        vocab=model_vocab,
        dimensions=model_vocab.dimensions,
        seed=mp.seed,
        strict_vocab=strict,
        probe_registry=probe_registry,
        compile_profile=compile_profile,
        learned_init_mode=learned_init_mode,
        learned_init_seed=learned_init_seed,
    )
    return assemble_architecture(
        architecture_spec(architecture_name),
        context,
        default_component_registry(),
        sub_lengths=sub_lengths,
    )
