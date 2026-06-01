import asyncio
import time
import random
import statistics
from typing import List, Dict, Any
import httpx
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
            return {"latency_ms": latency_ms, "success": response.status_code == 200}
        except Exception:
            latency_ms = (time.time() - start_time) * 1000
            return {"latency_ms": latency_ms, "success": False}
    
    async def run_benchmark(self, target_rps: float, duration_s: int, model_id: str, repeat_rate: float, seed: int) -> Dict[str, Any]:
        random.seed(seed)
        unique_texts = [f"Topic {i} news and discussion" for i in range(10)]
        hot_pool = unique_texts[:3]
        
        num_requests = int(target_rps * duration_s)
        inter_arrival_times = [random.expovariate(target_rps) for _ in range(num_requests)]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            for wait_time in inter_arrival_times:
                await asyncio.sleep(wait_time)
                text = random.choice(hot_pool) if random.random() < repeat_rate else random.choice(unique_texts)
                tasks.append(asyncio.create_task(self.send_request(client, text, model_id)))
            
            results = await asyncio.gather(*tasks)
        
        successful = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if not latencies: return {"mean_latency_ms": 0, "p95_latency_ms": 0}
        
        return {
            "target_rps": target_rps,
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
        }

    def run_sweep(self, repeat_rate: float) -> pd.DataFrame:
        print(f"Running sweep for {repeat_rate*100:.0f}% repeat...")
        results = []
        rps_values = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        for rps in rps_values:
            res = asyncio.run(self.run_benchmark(rps, 15, "logreg", repeat_rate, 42))
            results.append(res)
            print(f"  RPS {rps}: mean={res['mean_latency_ms']:.1f}ms")
        return pd.DataFrame(results)

def main():
    gen = LoadGenerator()
    
    df_0 = gen.run_sweep(0.0)
    df_0.to_csv("task5_0pct_preproc.csv", index=False)
    
    df_10 = gen.run_sweep(0.1)
    df_10.to_csv("task5_10pct_preproc.csv", index=False)
    
    df_20 = gen.run_sweep(0.2)
    df_20.to_csv("task5_20pct_preproc.csv", index=False)
    
    print("Task 5 benchmark complete. Files saved.")

if __name__ == "__main__":
    main()