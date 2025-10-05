# Performance Testing Commands for RecessionRadar

This document outlines the key commands used for profiling and performance testing in the RecessionRadar project.

---

## Installation of Required Packages

```bash
pip install pytest locust memory-profiler psutil line-profiler
pip install snakeviz
```

## API Response Timing with curl

```bash
curl -w "@curl-format.txt" -o /dev/null -s "http://127.0.0.1:8000/api/current-prediction"
python performance_test.py
```

## CPU Profiling with cProfile and Visualization

```bash
python -m cProfile -o profile_results.prof "../api/main.py"
snakeviz profile_results.prof
```

## Function-Level Profiling

```bash
python fuction_profile.py
```

## Memory Profiling with memory_profiler

```bash
mprof run memory_profile.py
mprof plot
```