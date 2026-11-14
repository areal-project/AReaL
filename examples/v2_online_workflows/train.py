# SPDX-License-Identifier: Apache-2.0

"""Train with main's native v2 online entrypoint."""

import sys

from examples.hermes.train import main

if __name__ == "__main__":
    main(sys.argv[1:])
