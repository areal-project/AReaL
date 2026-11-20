"""Use the existing Arena/RPC entry with dense Qwen SGLang options."""

import sys

from examples.swe.qwen38_flash_next import train_rl as flash_entry

flash_decorate_controller = flash_entry.decorate_controller


def decorate_dense_controller(controller):
    initialize = controller.initialize
    # Preserve the existing proxy factory override and all other entry logic.
    controller = flash_decorate_controller(controller)

    def initialize_dense(**kwargs):
        server_args = dict(kwargs["server_args"])
        server_args.pop("ple_offload_embedding", None)
        # Keep the SGLang GDN defaults used by the successful dense-model run.
        return initialize(**{**kwargs, "server_args": server_args})

    controller.initialize = initialize_dense
    return controller


def main():
    flash_entry.decorate_controller = decorate_dense_controller
    flash_entry.main(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
