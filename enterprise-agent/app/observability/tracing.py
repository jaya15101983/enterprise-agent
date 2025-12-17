"""
OpenTelemetry tracing setup with GenAI semantic conventions.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from functools import wraps
import time
import os


def setup_tracing(
    service_name: str = "enterprise-agent",
    otlp_endpoint: str | None = None
) -> trace.Tracer:
    """Initialize OpenTelemetry with OTLP exporter."""
    
    endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", 
        "http://localhost:4317"
    )
    
    resource = Resource.create({"service.name": service_name})
    
    provider = TracerProvider(resource=resource)
    
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer(service_name)


# Global tracer
tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    """Get or initialize the global tracer."""
    global tracer
    if tracer is None:
        tracer = setup_tracing()
    return tracer


def trace_agent_execution(agent_name: str):
    """Decorator for tracing agent execution with GenAI semantic conventions."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            t = get_tracer()
            
            with t.start_as_current_span(
                f"agent.{agent_name}",
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": agent_name,
                }
            ) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    span.set_attribute("gen_ai.response.finish_reason", "success")
                    
                    # Extract token usage if available
                    if hasattr(result, 'usage'):
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", 
                            result.usage.input_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", 
                            result.usage.output_tokens
                        )
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("gen_ai.response.finish_reason", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("gen_ai.latency_ms", duration_ms)
        
        return wrapper
    return decorator


def trace_llm_call(model: str, operation: str = "chat"):
    """Decorator for tracing individual LLM calls."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            t = get_tracer()
            
            with t.start_as_current_span(
                f"llm.{operation}",
                attributes={
                    "gen_ai.system": "anthropic",
                    "gen_ai.request.model": model,
                    "gen_ai.operation.name": operation,
                }
            ) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if hasattr(result, 'usage'):
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", 
                            result.usage.input_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", 
                            result.usage.output_tokens
                        )
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("gen_ai.latency_ms", duration_ms)
        
        return wrapper
    return decorator
