import json
import unittest

from experiments._benchmarking import benchmark, cache_csv_fields


class BenchmarkUtilityTests(unittest.TestCase):
    def test_benchmark_preserves_samples_and_reports_median(self):
        evictions = []
        result = benchmark(
            lambda: None,
            repeat=3,
            warmup=2,
            before_each=lambda: evictions.append(True),
        )

        samples = json.loads(result["time_samples_s"])
        self.assertEqual(len(samples), 3)
        self.assertEqual(result["time_s"], sorted(samples)[1])
        self.assertEqual(result["time_min_s"], min(samples))
        self.assertEqual(result["time_max_s"], max(samples))
        self.assertEqual(len(evictions), 3)

    def test_benchmark_rejects_nonpositive_repeat(self):
        with self.assertRaises(ValueError):
            benchmark(lambda: None, repeat=0)

    def test_cache_fields_have_csv_column_names(self):
        cache = {"L1": 32, "L2": 256, "L3": 8192, "cache_line_bytes": 64}
        self.assertEqual(cache_csv_fields(cache), {
            "cache_l1_bytes": 32,
            "cache_l2_bytes": 256,
            "cache_l3_bytes": 8192,
            "cache_line_bytes": 64,
        })


if __name__ == "__main__":
    unittest.main()
