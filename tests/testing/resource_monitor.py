import psutil, os, time, datetime

process = psutil.Process(os.getpid())

print("Monitoring system resource usage (Ctrl + C to stop)\n")
print("{:<20} {:<10} {:<10} {:<10}".format("Timestamp", "CPU (%)", "Memory (MB)", "Threads"))

while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().used / (1024 ** 2)
    threads = len(psutil.pids())
    print("{:<20} {:<10.1f} {:<10.1f} {:<10}".format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        cpu,
        mem,
        threads
    ))
    time.sleep(1)
