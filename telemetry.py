import os
import time
import json
import psutil
import subprocess
from pathlib import Path

# Config
OUTPUT_FILE = Path("runs/fruit_v3b/telemetry.json")
TEMP_FILE = Path("runs/fruit_v3b/telemetry.tmp.json")
UPDATE_INTERVAL = 2.0

def get_gpu_metrics():
    try:
        # Get GPU Util, VRAM Used (MB), VRAM Total (MB)
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                # We assume 1 GPU for now (index 0)
                gpu_util, mem_used, mem_total = map(int, lines[0].split(","))
                return {
                    "gpu_util_percent": gpu_util,
                    "vram_used_gb": round(mem_used / 1024.0, 2),
                    "vram_total_gb": round(mem_total / 1024.0, 2)
                }
    except Exception as e:
        pass
    
    # Fallback if nvidia-smi fails
    return {
        "gpu_util_percent": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0
    }

def main():
    print("[Telemetry Agent] Starting background hardware monitor...")
    # Ensure directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize CPU percent
    psutil.cpu_percent()

    while True:
        try:
            cpu_util = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used_gb = round(mem.used / (1024**3), 2)
            ram_total_gb = round(mem.total / (1024**3), 2)
            
            gpu_data = get_gpu_metrics()
            
            payload = {
                "timestamp": time.time(),
                "cpu_util_percent": cpu_util,
                "ram_used_gb": ram_used_gb,
                "ram_total_gb": ram_total_gb,
                **gpu_data
            }
            
            # Atomic write to prevent browser fetch from reading a half-written file
            with open(TEMP_FILE, "w") as f:
                json.dump(payload, f)
            
            os.replace(TEMP_FILE, OUTPUT_FILE)
            
        except Exception as e:
            # Silently ignore errors to prevent crashing
            pass
            
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
