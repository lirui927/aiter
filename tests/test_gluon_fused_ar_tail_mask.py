# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
from pathlib import Path


_KERNEL = (
    Path(__file__).parents[1]
    / "aiter/ops/triton/_gluon_kernels/gfx1250/comms/fused"
    / "all_reduce_rmsnorm_quant.py"
)


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def _line_mask(hidden):
    assert hidden % 4 == 0
    n = _next_power_of_2(hidden // 4)
    return n, [(line < 2 * n) and (4 * (line % n) < hidden) for line in range(2 * n)]


def test_hidden_6144_communicates_75_percent_of_padded_lines():
    hidden = 6144
    n, mask = _line_mask(hidden)

    assert n == 2048
    assert len(mask) == 4096
    assert sum(mask) == 3072
    assert sum(mask) / len(mask) == 0.75

    # _lines lays out line h * N + i as the two columns below.
    for line, valid in enumerate(mask):
        i = line % n
        h = line // n
        columns = (4 * i + 2 * h, 4 * i + 2 * h + 1)
        assert valid == all(column < hidden for column in columns)


def test_power_of_two_hidden_keeps_every_line():
    for hidden in (256, 2048, 4096, 8192):
        _, mask = _line_mask(hidden)
        assert all(mask)


def test_sender_and_receiver_share_tail_mask_and_padded_stride():
    tree = ast.parse(_KERNEL.read_text())
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_all_reduce_rmsnorm_kernel"
    )

    valid_line = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "valid_line"
            for target in node.targets
        )
    )
    assert ast.unparse(valid_line.value) == "(line < 2 * N) & (4 * i < HIDDEN)"

    calls = {"_post": [], "_receive": []}
    for node in ast.walk(kernel):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in calls
        ):
            calls[node.func.id].append(node)

    assert all(len(found) == 1 for found in calls.values())
    post = calls["_post"][0]
    receive = calls["_receive"][0]
    assert ast.unparse(post.args[3]) == "valid_line"
    assert ast.unparse(receive.args[2]) == "valid_line"
    assert ast.unparse(post.args[2]) == "2 * HIDDEN_PAD * pid + pos"
    assert ast.unparse(receive.args[1]) == "2 * HIDDEN_PAD * t + pos"
