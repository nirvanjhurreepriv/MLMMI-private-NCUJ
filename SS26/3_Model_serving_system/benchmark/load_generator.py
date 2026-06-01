import asyncio
import json
import time
import random
import statistics
from typing import List, Dict, Any
import httpx
import matplotlib.pyplot as plt
import pandas as pd

class LoadGenerator:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def send_request(self, client: httpx.AsyncClient, text: str, model_id: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = await client.post(
                f"{self.base_url}/predict",
                json={"text": text, "model_id": model_id},
                timeout=30.0
            )
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": response.status_code,
                "latency_ms": latency_ms,
                "success": response.status_code == 200
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": 0,
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            }
    
    async def run_benchmark(
        self,
        target_rps: float,
        duration_s: int = 30,
        model_id: str = "logreg",
        seed: int = 42
    ) -> Dict[str, Any]:
        random.seed(seed)
        
        sample_texts = [
            "BTC",
            "SPCX",
            "NVIDIA",
            "Anthropix",
            "REstate",
            "HFinance",
            "BER",
            "TUB"
        ]
        
        num_requests = int(target_rps * duration_s)
        inter_arrival_times = [random.expovariate(target_rps) for _ in range(num_requests)]
        
        print(f"Starting benchmark: {target_rps} RPS for {duration_s}s ({num_requests} requests)")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            tasks = []
            current_time = 0.0
            
            for i, wait_time in enumerate(inter_arrival_times):
                current_time += wait_time
                await asyncio.sleep(wait_time)
                
                text = random.choice(sample_texts)
                task = asyncio.create_task(self.send_request(client, text, model_id))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
        
        successful = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if not latencies:
            return {
                "target_rps": target_rps,
                "actual_rps": 0,
                "mean_latency_ms": 0,
                "p95_latency_ms": 0,
                "success_rate": 0,
                "total_requests": num_requests
            }
        
        mean_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        success_rate = len(successful) / len(results) * 100
        actual_rps = len(successful) / total_time
        
        return {
            "target_rps": target_rps,
            "actual_rps": round(actual_rps, 2),
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "success_rate": round(success_rate, 2),
            "total_requests": len(results),
            "successful_requests": len(successful)
        }
    
    def run_rps_sweep(
        self,
        rps_values: List[float] = None,
        duration_s: int = 20,
        model_id: str = "logreg"
    ) -> pd.DataFrame:
        if rps_values is None:
            rps_values = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100]
        
        all_results = []
        
        for rps in rps_values:
            result = asyncio.run(self.run_benchmark(target_rps=rps, duration_s=duration_s, model_id=model_id))
            all_results.append(result)
            print(f"  RPS {rps}: mean={result['mean_latency_ms']:.1f}ms, p95={result['p95_latency_ms']:.1f}ms, success={result['success_rate']:.1f}%")
        
        return pd.DataFrame(all_results)
    
    def plot_results(self, df: pd.DataFrame, output_file: str = "benchmark_results.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(df['target_rps'], df['mean_latency_ms'], 'o-', label='Mean Latency', linewidth=2)
        plt.plot(df['target_rps'], df['p95_latency_ms'], 's--', label='P95 Latency', linewidth=2)
        plt.xlabel('Target RPS', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.title('Latency vs Request Rate - Baseline Server', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Plot saved to {output_file}")

def main():
    print("=" * 70)
    print("Task 2: RPS-Sweep Benchmark (Baseline Server)")
    print("=" * 70)
    
    generator = LoadGenerator()
    
    rps_values = [1, 2, 5, 10, 15, 20, 30, 40, 50]
    
    results_df = generator.run_rps_sweep(
        rps_values=rps_values,
        duration_s=20,
        model_id="logreg"
    )
    
    print("\n" + "=" * 70)
    print("Benchmark Results Summary:")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    generator.plot_results(results_df, "baseline_benchmark.png")
    
    results_df.to_csv("baseline_benchmark.csv", index=False)
    print("\nResults saved to baseline_benchmark.csv")

if __name__ == "__main__":
    main()