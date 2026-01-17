# Quick Start: Running Benchmarks

## 1. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run a Quick Benchmark (ML vs ScanCode)

```bash
# This will compare your ML classifier against ScanCode on the existing dataset
python benchmarks/run_benchmark.py
```

## 3. View Results

```bash
# View text report
cat benchmark_results/benchmark_report.txt

# Generate visualizations
python benchmarks/visualize.py

# View charts
ls benchmark_results/*.png
```

## Expected Output

```
BENCHMARK SUMMARY
================================================================================
detector          dataset    accuracy  f1_macro
ML Classifier     scancode   0.8269    0.7959
ScanCode          scancode   0.8543    0.8201
================================================================================
```

## Next Steps

- See `BENCHMARKING.md` for complete documentation
- Download additional datasets: `python benchmarks/download_datasets.py`
- Enable more tools in `benchmarks/config.py`
