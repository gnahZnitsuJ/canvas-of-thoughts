import os

import pyopencl as cl

OPENCL_PLATFORM_ENV = "CANVAS_OPENCL_PLATFORM_INDEX"
OPENCL_DEVICE_ENV = "CANVAS_OPENCL_DEVICE_INDEX"


def _read_optional_int_env(name):
    value = os.getenv(name)
    if value is None or value == "":
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {value!r}."
        ) from exc


def _resolve_index(cli_value, env_name, default):
    if cli_value is not None:
        return cli_value

    env_value = _read_optional_int_env(env_name)
    if env_value is not None:
        return env_value

    return default


def select_opencl_device(platform_index=None, device_index=None):
    platforms = cl.get_platforms()
    selected_platform_index = _resolve_index(
        platform_index,
        OPENCL_PLATFORM_ENV,
        default=0,
    )

    if not 0 <= selected_platform_index < len(platforms):
        raise IndexError(
            "OpenCL platform index out of range: "
            f"{selected_platform_index}. Available platform indices: "
            f"0..{len(platforms) - 1}"
        )

    platform = platforms[selected_platform_index]
    devices = platform.get_devices()
    selected_device_index = _resolve_index(
        device_index,
        OPENCL_DEVICE_ENV,
        default=0,
    )

    if not 0 <= selected_device_index < len(devices):
        raise IndexError(
            "OpenCL device index out of range: "
            f"{selected_device_index}. Available device indices for "
            f"platform {selected_platform_index}: 0..{len(devices) - 1}"
        )

    device = devices[selected_device_index]
    context = cl.Context([device])

    return {
        "platform": platform,
        "device": device,
        "context": context,
        "platform_index": selected_platform_index,
        "device_index": selected_device_index,
    }


def print_opencl_selection(selection):
    print("\nOpenCL selection:")
    print(
        f"Platform [{selection['platform_index']}]: "
        f"{selection['platform'].name}"
    )
    print(
        f"Device   [{selection['device_index']}]: "
        f"{selection['device'].name}"
    )
