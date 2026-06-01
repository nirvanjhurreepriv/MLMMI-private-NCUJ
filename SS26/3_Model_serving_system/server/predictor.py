""" import asyncio
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from .registry import ModelRegistry

class Predictor:
    def __init__(self, registry: ModelRegistry, encoder_name: str = "all-MiniLM-L6-v2"):
        self.registry = registry
        self.encoder = SentenceTransformer(encoder_name)
        self.queue = asyncio.Queue()
    
    async def predict(self, text: str, model_id: str) -> Dict[str, Any]:
        request = {"text": text, "model_id": model_id}
        future = asyncio.Future()
        
        await self.queue.put((request, future))
        return await future
    
    async def process_queue(self):
        while True:
            request, future = await self.queue.get()
            try:
                text = request["text"]
                model_id = request["model_id"]
                
                embedding = self.encoder.encode([text], convert_to_numpy=True)[0]
                model = self.registry.get_model(model_id)
                prediction = model.predict([embedding])[0]
                
                result = {
                    "prediction": int(prediction),
                    "model_id": model_id,
                    "text": text
                }
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.queue.task_done() """

#-----------------------------------------------------------------------

""" import asyncio
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List
from sentence_transformers import SentenceTransformer
from .registry import ModelRegistry

class LRUCache:
    #Thread-safe LRU cache for predictions
    def __init__(self, capacity: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: Tuple[str, str]) -> Optional[Any]:
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    async def put(self, key: Tuple[str, str], value: Any):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": self.hits / total if total > 0 else 0.0,
            "cache_size": len(self.cache),
        }

class Predictor:
    def __init__(
        self,
        registry: ModelRegistry,
        encoder_name: str = "all-MiniLM-L6-v2",
        max_batch_size: int = 8,
        max_wait_ms: int = 10,
        cache_capacity: int = 1000,
    ):
        self.registry = registry
        self.encoder = SentenceTransformer(encoder_name)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.cache = LRUCache(capacity=cache_capacity)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = True
    
    async def predict(self, text: str, model_id: str) -> Dict[str, Any]:
        cache_key = (model_id, text)
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        future = asyncio.Future()
        await self.queue.put((text, model_id, future))
        result = await future
        
        await self.cache.put(cache_key, result)
        return result
    
    async def process_queue(self):
        while self.running:
            batch = []
            start_time = time.time()
            
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= self.max_wait_ms:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=(self.max_wait_ms - elapsed_ms) / 1000
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                await asyncio.sleep(0.001)
                continue
            
            texts = [t for t, _, _ in batch]
            unique_texts = list(dict.fromkeys(texts))
            embeddings = self.encoder.encode(
                unique_texts,
                batch_size=len(unique_texts),
                convert_to_numpy=True
            )
            text_to_emb = dict(zip(unique_texts, embeddings))
            
            for text, model_id, future in batch:
                try:
                    emb = text_to_emb[text]
                    model = self.registry.get_model(model_id)
                    prediction = model.predict([emb])[0]
                    result = {
                        "prediction": int(prediction),
                        "model_id": model_id,
                        "text": text,
                    }
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
    
    def stop(self):
        self.running = False
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        return self.cache.get_metrics() """


#-----------------------------------------------------------------------


import asyncio
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List, Set
from sentence_transformers import SentenceTransformer
import numpy as np
from .registry import ModelRegistry

class LRUCache:
    #Generic LRU Cache with metrics
    def __init__(self, capacity: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: Tuple) -> Optional[Any]:
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    async def put(self, key: Tuple, value: Any):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "size": len(self.cache),
        }

class Predictor:
    def __init__(
        self,
        registry: ModelRegistry,
        encoder_name: str = "all-MiniLM-L6-v2",
        max_batch_size: int = 8,
        max_wait_ms: int = 10,
        cache_capacity: int = 1000,
    ):
        self.registry = registry
        self.encoder = SentenceTransformer(encoder_name)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        # Task 3: Prediction cache keyed by (model_id, text)
        self.prediction_cache = LRUCache(capacity=cache_capacity)
        # Task 5: Embedding cache keyed by (text) alone
        self.embedding_cache = LRUCache(capacity=cache_capacity)
        
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = True
    
    async def predict(self, text: str, model_id: str) -> Dict[str, Any]:
        # 1. Check prediction cache
        pred_key = (model_id, text)
        cached_pred = await self.prediction_cache.get(pred_key)
        if cached_pred is not None:
            return cached_pred
        
        # 2. Check embedding cache (Task 5)
        emb_key = (text,)
        cached_emb = await self.embedding_cache.get(emb_key)
        if cached_emb is not None:
            # Skip encoder, run head directly
            model = self.registry.get_model(model_id)
            prediction = model.predict([cached_emb])[0]
            result = {"prediction": int(prediction), "model_id": model_id, "text": text}
            await self.prediction_cache.put(pred_key, result)
            return result
        
        # 3. Queue request
        future = asyncio.Future()
        await self.queue.put((text, model_id, future))
        result = await future
        
        # Cache result for future
        await self.prediction_cache.put(pred_key, result)
        return result
    
    async def process_queue(self):
        while self.running:
            batch = []
            start_time = time.time()
            
            # Gather batch
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= self.max_wait_ms:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=(self.max_wait_ms - elapsed_ms) / 1000
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                await asyncio.sleep(0.001)
                continue
            
            # Task 5: In-batch deduplication logic
            texts = [t for t, _, _ in batch]
            unique_texts = list(dict.fromkeys(texts)) # Preserve order, remove dups
            
            # Check embedding cache for unique texts
            embeddings_map = {}
            texts_to_encode = []
            
            # lock the cache for reading/writing during batch processing
            async with self.embedding_cache.lock:
                for text in unique_texts:
                    key = (text,)
                    if key in self.embedding_cache.cache:
                        self.embedding_cache.cache.move_to_end(key)
                        self.embedding_cache.hits += 1
                        embeddings_map[text] = self.embedding_cache.cache[key]
                    else:
                        self.embedding_cache.misses += 1
                        texts_to_encode.append(text)
            
            # Encode missing texts
            if texts_to_encode:
                new_embeddings = self.encoder.encode(
                    texts_to_encode,
                    batch_size=len(texts_to_encode),
                    convert_to_numpy=True
                )
                # Add to map and cache
                for text, emb in zip(texts_to_encode, new_embeddings):
                    embeddings_map[text] = emb
                    await self.embedding_cache.put((text,), emb)
            
            # Process each request in batch
            for text, model_id, future in batch:
                try:
                    emb = embeddings_map[text]
                    model = self.registry.get_model(model_id)
                    prediction = model.predict([emb])[0]
                    result = {
                        "prediction": int(prediction),
                        "model_id": model_id,
                        "text": text,
                    }
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
    
    def stop(self):
        self.running = False
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "prediction_cache": self.prediction_cache.get_metrics(),
            "embedding_cache": self.embedding_cache.get_metrics(),
            "queue_size": self.queue.qsize(),
        }