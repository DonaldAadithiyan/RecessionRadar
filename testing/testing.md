# Performance Testing Commands for RecessionRadar

This document outlines the key commands used for profiling and performance testing in the RecessionRadar project.

---

## Installation of Required Packages

```bash
pip install pytest locust memory-profiler psutil line-profiler
pip install snakeviz
```

## performance testing

### A. Measure Response Time (API-level Profiling)

```bash
curl -w "@curl-format.txt" -o /dev/null -s "http://127.0.0.1:8000/api/current-prediction"
python performance_test.py
```

### B. Profile Backend Code (Function-level Profiling)

```bash
python -m cProfile -o profile_results.prof "../api/main.py"
snakeviz profile_results.prof
```

```bash
python fuction_profile.py
```

### C. Memory Usage Profiling

```bash
mprof run memory_profile.py
mprof plot
```

##  Load testing

### Concurrency Profiling (with Locust)

```bash
locust -f locustfile.py
```

while load test running use `python resource_monitor.py` to monitor performance
