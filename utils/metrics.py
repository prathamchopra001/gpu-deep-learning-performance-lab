import time


class ThroughputTracker:
    def __init__(self):
        self.start_time: float | None = None
        self.total_samples = 0

    def start(self):
        self.start_time = time.time()

    def update(self, batch_size):
        self.total_samples += batch_size

    def stop(self):
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self.start_time
        throughput = self.total_samples / elapsed if elapsed > 0 else 0.0
        return throughput, elapsed
