import time, requests

url = "http://127.0.0.1:8000/api/current-prediction"
runs = 10
times = []

for i in range(runs):
    start = time.time()
    res = requests.get(url)
    end = time.time()
    times.append(end - start)
    print(f"Run {i+1}: {end - start:.3f}s, Status: {res.status_code}")

print(f"\nAverage Response Time: {sum(times)/len(times):.3f}s")
