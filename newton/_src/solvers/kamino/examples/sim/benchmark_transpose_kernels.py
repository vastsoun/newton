#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.kamino.dynamics.delassus import BlockSparseMatrixFreeDelassusOperator
from newton._src.solvers.kamino.linalg import blas
from newton._src.solvers.kamino.models.builders.basics import build_boxes_nunchaku
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers


@dataclass
class BenchResult:
    op: str
    bpt: int
    mean_ms: float
    median_ms: float
    std_ms: float


def _build_fixture(device: wp.context.Device, num_worlds: int, max_contacts: int, seed: int):
    builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)
    model, data, limits, detector, jacobians = make_containers(
        builder=builder,
        max_world_contacts=max_contacts,
        device=device,
        sparse=True,
    )
    update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)
    delassus = BlockSparseMatrixFreeDelassusOperator(model=model, data=data, jacobians=jacobians, device=device)
    bsm = delassus.bsm
    if bsm is None:
        raise RuntimeError("Expected block-sparse Jacobian to be available.")

    rng = np.random.default_rng(seed)
    y_np = rng.standard_normal((bsm.num_matrices, bsm.max_of_max_dims[0]), dtype=np.float32)
    x_np = np.zeros((bsm.num_matrices, bsm.max_of_max_dims[1]), dtype=np.float32)
    y = wp.from_numpy(y_np, dtype=wp.float32, device=device)
    x = wp.from_numpy(x_np, dtype=wp.float32, device=device)
    mask = wp.ones((bsm.num_matrices,), dtype=wp.int32, device=device)
    return bsm, y, x, mask


def _capture_and_time(run_once, device: wp.context.Device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        run_once()
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as capture:
        run_once()
    graph = capture.graph
    if graph is None:
        raise RuntimeError("Failed to capture CUDA graph.")

    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / repeats


def _benchmark_op(
    op: str,
    bsm,
    y,
    x,
    mask,
    device: wp.context.Device,
    bpt: int,
    warmup: int,
    repeats: int,
    samples: int,
) -> BenchResult:
    sample_ms: list[float] = []
    for _ in range(samples):
        if op == "transpose_matvec":
            def run_once():
                blas.block_sparse_transpose_matvec(bsm, y, x, mask, blocks_per_thread=bpt)
        elif op == "transpose_gemv":
            def run_once():
                blas.block_sparse_transpose_gemv(bsm, y, x, alpha=1.0, beta=0.0, matrix_mask=mask, blocks_per_thread=bpt)
        else:
            raise ValueError(f"Unknown op: {op}")
        sample_ms.append(_capture_and_time(run_once, device=device, warmup=warmup, repeats=repeats))

    arr = np.asarray(sample_ms, dtype=np.float64)
    return BenchResult(
        op=op,
        bpt=bpt,
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        std_ms=float(np.std(arr)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark transpose sparse kernels in isolation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--max-contacts", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    wp.set_device(device)

    if args.quick:
        candidates = [8, 12]
        args.samples = 1
        args.repeats = 10
    else:
        candidates = [4, 8, 12, 16]

    bsm, y, x, mask = _build_fixture(
        device=device,
        num_worlds=args.num_worlds,
        max_contacts=args.max_contacts,
        seed=args.seed,
    )

    print(f"Device: {device}")
    print(f"Matrices: {bsm.num_matrices} max_nzb={bsm.max_of_num_nzb}")

    best: dict[str, BenchResult] = {}
    for op in ("transpose_matvec", "transpose_gemv"):
        print(f"\n[{op}]")
        for bpt in candidates:
            os.environ["NEWTON_KAMINO_BPT_TRANSPOSE"] = str(bpt)
            r = _benchmark_op(
                op=op,
                bsm=bsm,
                y=y,
                x=x,
                mask=mask,
                device=device,
                bpt=bpt,
                warmup=args.warmup,
                repeats=args.repeats,
                samples=args.samples,
            )
            is_best = op not in best or r.mean_ms < best[op].mean_ms
            if is_best:
                best[op] = r
            print(
                f"bpt={bpt:>2}: mean={r.mean_ms:.3f} ms, median={r.median_ms:.3f} ms, std={r.std_ms:.3f} ms"
                f"{' [BEST]' if is_best else ''}"
            )

    for op, r in best.items():
        print(f"# MEASURE op={op} best_bpt={r.bpt} mean_ms={r.mean_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
