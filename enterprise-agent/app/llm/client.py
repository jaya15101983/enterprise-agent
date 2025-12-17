"""
Production LLM client with rate limiting, retries, and circuit breaker.
Using Anthropic Claude Haiku 4.5.
"""

import asyncio
import random
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic, RateLimitError, APITimeoutError
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Prevents cascade failures during outages."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0, init=False)
    half_open_calls: int = field(default=0, init=False)
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN


class ProductionLLMClient:
    """Rate-limited, retry-enabled LLM client using Claude Haiku 4.5."""
    
    def __init__(
        self, 
        max_concurrent: int = 30,
        model: str = "claude-haiku-4-5-20251001"
    ):
        self.client = AsyncAnthropic()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model = model
        self.circuit_breaker = CircuitBreaker()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def complete(self, messages: list[dict]) -> str:
        """Complete with rate limiting, retries, and circuit breaker."""
        
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker OPEN - service unavailable")
        
        try:
            async with self.semaphore:
                # Jitter prevents thundering herd
                await asyncio.sleep(random.uniform(0, 0.1))
                
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=messages
                )
                
                self.circuit_breaker.record_success()
                return response.content[0].text
                
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def batch_complete(self, message_batches: list[list[dict]]) -> list[str]:
        """Process multiple requests with controlled concurrency."""
        
        tasks = [self.complete(messages) for messages in message_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if isinstance(r, str) else f"Error: {type(r).__name__}"
            for r in results
        ]
