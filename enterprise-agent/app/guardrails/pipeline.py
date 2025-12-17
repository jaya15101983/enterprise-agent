"""
Guardrails pipeline with PII detection and prompt injection protection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import re
import time


@dataclass
class GuardrailResult:
    """Result from a guardrail check."""
    passed: bool
    action: Literal["allow", "flag", "block"]
    confidence: float
    reason: str
    guardrail_name: str
    latency_ms: float


class BaseGuardrail(ABC):
    """Abstract base class for guardrails."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def check(self, content: str, context: dict) -> GuardrailResult:
        pass


class PIIGuardrail(BaseGuardrail):
    """Fast regex-based PII detection."""
    
    name = "pii_detector"
    
    def __init__(self):
        self.patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        }
    
    async def check(self, content: str, context: dict) -> GuardrailResult:
        start = time.time()
        
        detected = []
        for pii_type, pattern in self.patterns.items():
            if pattern.search(content):
                detected.append(pii_type)
        
        latency_ms = (time.time() - start) * 1000
        
        if detected:
            return GuardrailResult(
                passed=False,
                action="block",
                confidence=0.95,
                reason=f"Detected PII: {', '.join(detected)}",
                guardrail_name=self.name,
                latency_ms=latency_ms
            )
        
        return GuardrailResult(
            passed=True,
            action="allow",
            confidence=1.0,
            reason="No PII detected",
            guardrail_name=self.name,
            latency_ms=latency_ms
        )


class PromptInjectionGuardrail(BaseGuardrail):
    """Detect common prompt injection patterns."""
    
    name = "injection_detector"
    
    def __init__(self):
        self.patterns = [
            re.compile(r'ignore\s+(all\s+)?(previous|prior|above)', re.I),
            re.compile(r'disregard\s+(all\s+)?(previous|prior|above)', re.I),
            re.compile(r'forget\s+(all\s+)?(previous|prior|above)', re.I),
            re.compile(r'you\s+are\s+now\s+(a|an)', re.I),
            re.compile(r'new\s+instructions?:', re.I),
            re.compile(r'system\s*:\s*', re.I),
            re.compile(r'\[INST\]|\[\/INST\]', re.I),
            re.compile(r'<\|im_start\|>|<\|im_end\|>', re.I),
        ]
    
    async def check(self, content: str, context: dict) -> GuardrailResult:
        start = time.time()
        
        matches = []
        for pattern in self.patterns:
            if pattern.search(content):
                matches.append(pattern.pattern)
        
        latency_ms = (time.time() - start) * 1000
        
        if matches:
            return GuardrailResult(
                passed=False,
                action="flag",  # Flag for review, don't block
                confidence=0.7,
                reason="Potential prompt injection detected",
                guardrail_name=self.name,
                latency_ms=latency_ms
            )
        
        return GuardrailResult(
            passed=True,
            action="allow",
            confidence=0.9,
            reason="No injection patterns detected",
            guardrail_name=self.name,
            latency_ms=latency_ms
        )


class GuardrailPipeline:
    """Execute guardrails in priority order."""
    
    def __init__(self, guardrails: list[BaseGuardrail]):
        self.guardrails = guardrails
    
    async def evaluate(
        self, 
        content: str, 
        context: dict
    ) -> tuple[bool, list[GuardrailResult]]:
        """Run all guardrails, return overall pass/fail and details."""
        
        results = []
        overall_action = "allow"
        
        for guardrail in self.guardrails:
            try:
                result = await guardrail.check(content, context)
                results.append(result)
                
                if result.action == "block":
                    overall_action = "block"
                elif result.action == "flag" and overall_action == "allow":
                    overall_action = "flag"
                    
            except Exception as e:
                # Fail OPEN
                results.append(GuardrailResult(
                    passed=True,
                    action="allow",
                    confidence=0.0,
                    reason=f"Guardrail error (failing open): {e}",
                    guardrail_name=guardrail.name,
                    latency_ms=0
                ))
        
        passed = overall_action == "allow"
        return passed, results


def create_default_pipeline() -> GuardrailPipeline:
    """Create pipeline with standard guardrails."""
    return GuardrailPipeline([
        PIIGuardrail(),
        PromptInjectionGuardrail(),
    ])
