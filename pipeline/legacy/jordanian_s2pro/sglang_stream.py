"""Bounded SSE subscriptions; the existing voice job queue owns generation."""
import json
import queue
import threading


class StreamHub:
    def __init__(self, limit=8, buffer=64):
        self.limit, self.buffer = limit, buffer
        self.lock = threading.Lock()
        self.clients = {}

    def open(self, job_id):
        with self.lock:
            if len(self.clients) >= self.limit:
                raise RuntimeError("Too many open streams; wait for one to finish")
            channel = queue.Queue(maxsize=self.buffer)
            self.clients[job_id] = channel
            return channel

    def close(self, job_id):
        with self.lock:
            self.clients.pop(job_id, None)

    def publish(self, job_id, event):
        with self.lock:
            channel = self.clients.get(job_id)
            if channel is None:
                return
            try:
                channel.put_nowait(event)
            except queue.Full:
                while not channel.empty():
                    channel.get_nowait()
                channel.put_nowait({"event": "error", "error": "Stream client is too slow; the completed WAV will remain in the studio"})
                self.clients.pop(job_id, None)

    def fail_all(self, reason):
        with self.lock:
            for channel in self.clients.values():
                while not channel.empty():
                    channel.get_nowait()
                channel.put_nowait({"event": "error", "error": reason})
            self.clients.clear()


def sse_bytes(event):
    return ("event: " + event["event"] + "\ndata: " +
            json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n\n").encode()
