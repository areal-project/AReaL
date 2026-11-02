# SPDX-License-Identifier: Apache-2.0

from areal.elastic.infra.monitor.identity import semantic_digest


def test_measurement_digest_ignores_comments_formatting_and_docstrings():
    """Text-only edits keep measurements applicable, real statements do not."""
    original = '"""old docs"""\ndef value():\n    """old docs"""\n    return 1\n'
    cosmetic = (
        '"""new docs"""\n# comment\ndef value():\n    """new docs"""\n    return (1)\n'
    )
    assert semantic_digest(original) == semantic_digest(cosmetic)
    assert semantic_digest(original) != semantic_digest(
        original.replace("return 1", "return 2")
    )
