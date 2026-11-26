"""Workload launched under perf stat."""

import argparse

import numpy as np
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    rng = np.random.default_rng(12345)
    left = rng.random((args.side, args.side), dtype=dtype)
    right = rng.random((args.side, args.side), dtype=dtype)
    output = np.empty((args.side, args.side), dtype=dtype)

    with threadpool_limits(limits=args.threads, user_api="blas"):
        for _ in range(args.warmup):
            np.matmul(left, right, out=output)
        for _ in range(args.repeat):
            np.matmul(left, right, out=output)


if __name__ == "__main__":
    main()
