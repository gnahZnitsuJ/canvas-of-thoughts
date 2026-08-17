"""Committed policy for persistent, reproducible decoder caching."""

# ``auto`` reuses compatible Nengo decoder solves, ``refresh`` clears the
# Canvas-owned cache before rebuilding it, and ``off`` bypasses disk storage.
DECODER_CACHE_MODES = ("auto", "refresh", "off")
DEFAULT_DECODER_CACHE_MODE = "auto"

# Decoder matrices for the full 256-dimensional model are large enough that
# Nengo's 512 MB user-wide default can evict them between ordinary runs. Keep a
# larger cap for this project's version-scoped cache instead.
DECODER_CACHE_MAX_SIZE_BYTES = 4 * 1024**3

# A machine or CI job can relocate the cache without changing committed model
# configuration. The default resolver keeps it outside the OneDrive checkout.
DECODER_CACHE_ENV_VAR = "CANVAS_DECODER_CACHE_DIR"
DECODER_CACHE_SCHEMA_VERSION = "v2"
