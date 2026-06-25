import time
import torch
import os
from ultralytics import YOLO
import numpy as np

def benchmark():
    # Performance for the specifically requested model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo11_combined_ds3_shadows", "weights", "best.pt")
    model = YOLO(model_path)
    
    # Warmup
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy_input, verbose=False)
        
    # Measure
    n_frames = 100
    start = time.time()
    for _ in range(n_frames):
        model.predict(dummy_input, verbose=False, device=0)
    end = time.time()
    
    total_time = end - start
    avg_inference = (total_time / n_frames) * 1000
    fps = n_frames / total_time
    
    print(f"BENCHMARK_RESULTS: {avg_inference:.2f} ms | {fps:.2f} FPS")

if __name__ == "__main__":
    benchmark()
