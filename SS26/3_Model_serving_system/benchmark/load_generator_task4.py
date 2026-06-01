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
        duration_s: int,
        model_id: str,
        repeat_rate: float,
        seed: int
    ) -> Dict[str, Any]:
        random.seed(seed)
        
        unique_texts = [
            "BTC",
            "SPCX",
            "NVIDIA",
            "Anthropix",
            "REstate",
            "HFinance",
            "BER",
            "TUB"
            "Masterchef Hells Kit",
            "Michelin Starred Restos"
        ]
        
        hot_pool = unique_texts[:3]
        
        num_requests = int(target_rps * duration_s)
        inter_arrival_times = [random.expovariate(target_rps) for _ in range(num_requests)]
        
        print(f"  Running: {target_rps} RPS, {repeat_rate*100:.0f}% repeat, {duration_s}s")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            tasks = []
            current_time = 0.0
            
            for i, wait_time in enumerate(inter_arrival_times):
                current_time += wait_time
                await asyncio.sleep(wait_time)
                
                if random.random() < repeat_rate and hot_pool:
                    text = random.choice(hot_pool)
                else:
                    text = random.choice(unique_texts)
                
                task = asyncio.create_task(self.send_request(client, text, model_id))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
        
        successful = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if not latencies:
            return {"target_rps": target_rps, "mean_latency_ms": 0, "p95_latency_ms": 0, "success_rate": 0}
        
        mean_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        success_rate = len(successful) / len(results) * 100
        
        return {
            "target_rps": target_rps,
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "success_rate": round(success_rate, 2),
        }
    
    def run_rps_sweep(
        self,
        repeat_rate: float,
        rps_values: List[float] = None,
        duration_s: int = 20,
        model_id: str = "logreg"
    ) -> pd.DataFrame:
        if rps_values is None:
            rps_values = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        
        all_results = []
        
        for rps in rps_values:
            result = asyncio.run(self.run_benchmark(
                target_rps=rps,
                duration_s=duration_s,
                model_id=model_id,
                repeat_rate=repeat_rate,
                seed=42
            ))
            all_results.append(result)
            print(f"    RPS {rps}: mean={result['mean_latency_ms']:.1f}ms, p95={result['p95_latency_ms']:.1f}ms")
        
        return pd.DataFrame(all_results)
    
    def plot_comparison(
        self,
        results_0: pd.DataFrame,
        results_10: pd.DataFrame,
        results_20: pd.DataFrame,
        output_file: str = "task4_comparison.png"
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        repeat_rates = [0, 10, 20]
        results_list = [results_0, results_10, results_20]
        
        for ax, results, repeat in zip(axes, results_list, repeat_rates):
            ax.plot(results['target_rps'], results['mean_latency_ms'], 'o-', label='Mean', linewidth=2)
            ax.plot(results['target_rps'], results['p95_latency_ms'], 's--', label='P95', linewidth=2)
            ax.set_xlabel('Target RPS')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'{repeat}% Repeat Rate')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Task 4: Latency vs RPS at Different Repeat Rates', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {output_file}")

def main():
    print("=" * 70)
    print("Task 4: RPS-Sweep at Different Repeat Rates")
    print("=" * 70)
    
    generator = LoadGenerator()
    
    print("\n--- 0% Repeat Rate ---")
    results_0 = generator.run_rps_sweep(repeat_rate=0.0, duration_s=15)
    
    print("\n--- 10% Repeat Rate ---")
    results_10 = generator.run_rps_sweep(repeat_rate=0.1, duration_s=15)
    
    print("\n--- 20% Repeat Rate ---")
    results_20 = generator.run_rps_sweep(repeat_rate=0.2, duration_s=15)
    
    print("\n" + "=" * 70)
    print("Saving results and plots...")
    print("=" * 70)
    
    results_0.to_csv("task4_0pct_repeat.csv", index=False)
    results_10.to_csv("task4_10pct_repeat.csv", index=False)
    results_20.to_csv("task4_20pct_repeat.csv", index=False)
    
    generator.plot_comparison(results_0, results_10, results_20, "task4_comparison.png")
    
    print("\nTask 4 complete. Files saved:")
    print("  - task4_0pct_repeat.csv")
    print("  - task4_10pct_repeat.csv")
    print("  - task4_20pct_repeat.csv")
    print("  - task4_comparison.png")

if __name__ == "__main__":
    main()