import os
import time
import json
import psutil
import argparse
import subprocess
from pathlib import Path

# Config
UPDATE_INTERVAL = 2.0

def get_gpu_metrics():
    try:
        # Get GPU utilization, VRAM, thermals, power, and clocks.
        cmd = (
            "nvidia-smi --query-gpu="
            "utilization.gpu,memory.used,memory.total,temperature.gpu,"
            "power.draw,clocks.current.graphics,clocks.current.memory "
            "--format=csv,noheader,nounits"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                # We assume 1 GPU for now (index 0)
                parts = [p.strip() for p in lines[0].split(",")]
                gpu_util = int(float(parts[0]))
                mem_used = int(float(parts[1]))
                mem_total = int(float(parts[2]))
                gpu_temp = int(float(parts[3]))
                power_draw = float(parts[4]) if parts[4] not in {"[N/A]", "N/A"} else 0.0
                graphics_clock = int(float(parts[5]))
                memory_clock = int(float(parts[6]))
                return {
                    "gpu_util_percent": gpu_util,
                    "vram_used_gb": round(mem_used / 1024.0, 2),
                    "vram_total_gb": round(mem_total / 1024.0, 2),
                    "gpu_temp_c": gpu_temp,
                    "gpu_power_w": round(power_draw, 1),
                    "gpu_graphics_clock_mhz": graphics_clock,
                    "gpu_memory_clock_mhz": memory_clock,
                }
    except Exception as e:
        pass
    
    # Fallback if nvidia-smi fails
    return {
        "gpu_util_percent": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "gpu_temp_c": 0,
        "gpu_power_w": 0.0,
        "gpu_graphics_clock_mhz": 0,
        "gpu_memory_clock_mhz": 0,
    }

def get_active_run_dir() -> Path:
    """Find the newest subdirectory in runs/."""
    runs_root = Path("runs")
    if not runs_root.exists():
        return Path("runs/fruit_v3b")
    
    subdirs = [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subdirs:
        return Path("runs/fruit_v3b")
    
    # Sort by modification time of the directory or results.csv if it exists
    def get_sort_key(p: Path):
        csv = p / "results.csv"
        if csv.exists():
            return csv.stat().st_mtime
        return p.stat().st_mtime

    subdirs.sort(key=get_sort_key, reverse=True)
    return subdirs[0]

def main():
    parser = argparse.ArgumentParser(description="Background hardware telemetry monitor")
    parser.add_argument("--run", type=str, default=None,
                        help="Run folder name under runs/ (e.g. 'fruit_v3_clean'). If omitted, automatically detects the newest active run.")
    args = parser.parse_args()

    # Determine paths dynamically
    if args.run:
        run_dir = Path("runs") / args.run
    else:
        run_dir = get_active_run_dir()
        
    output_file = run_dir / "telemetry.json"
    temp_file = run_dir / "telemetry.tmp.json"
    
    print("=" * 60)
    print("  HARDWARE TELEMETRY AGENT")
    print(f"  Target Run Directory : {run_dir.resolve()}")
    print(f"  Telemetry File       : {output_file.resolve()}")
    print("=" * 60)
    
    # Ensure target directory exists
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CPU percent
    psutil.cpu_percent()

    while True:
        try:
            # If no run was explicitly provided, continuously check if a newer run has started
            if not args.run:
                new_run_dir = get_active_run_dir()
                if new_run_dir != run_dir:
                    print(f"\n  [Telemetry] Detected new active run: {new_run_dir.name}")
                    run_dir = new_run_dir
                    output_file = run_dir / "telemetry.json"
                    temp_file = run_dir / "telemetry.tmp.json"
                    run_dir.mkdir(parents=True, exist_ok=True)

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
            with open(temp_file, "w") as f:
                json.dump(payload, f)
            
            os.replace(temp_file, output_file)
            
        except Exception as e:
            # Silently ignore errors to prevent crashing
            pass
            
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
