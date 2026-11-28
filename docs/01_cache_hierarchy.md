# Experiment 1: Cache Hierarchy

## What this experiment does

This experiment compares how quickly the CPU reads data using different memory access patterns.

It tests:

- Sequential reads
- Strided reads
- Random reads
- Pointer chasing

The array size grows from 4 KB to 256 MB, allowing the benchmark to cross the CPU's L1, L2, and L3 cache sizes.

## Access patterns

- **Sequential:** Reads every element in order.
- **Strided:** Reads every `N`th element.
- **Gather:** Reads elements in a random order using an index array.
- **Pointer chase:** Uses the value of one element to choose the next element.

## What is measured

For each run, the experiment records:

- Median execution time
- Minimum and maximum execution time
- Standard deviation
- Individual timing samples
- Time per accessed element
- Useful throughput
- Array size and cache sizes

## Goal

The goal is to show that the way code accesses memory can change performance even when it performs the same amount of work.

It also shows whether performance changes as the working set becomes too large for each cache level.

## How to interpret the results

The results are observations for the recorded CPU. They show trends in memory access performance, but they are not exact measurements of cache latency or DRAM bandwidth.

## Generated artifacts

The CSV and metadata JSON are written to `artifacts/experiment-01/`. The same
folder contains the stride and size-sweep plots.
