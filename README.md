# Enterprise Agent

Production-grade multi-agent AI system using LangGraph and Claude Haiku 4.5.

## What's included

- **Multi-agent orchestration** with supervisor pattern
- **Rate limiting** via asyncio Semaphore (not sleep)
- **Retry logic** with tenacity (exponential backoff)
- **Circuit breaker** to prevent cascade failures
- **OpenTelemetry tracing** with GenAI semantic conventions
- **Guardrails pipeline** (PII detection, prompt injection)
- **PostgreSQL checkpointing** for state persistence

## Quick Start

```bash
# Clone
git clone https://github.com/jaya15101983/enterprise-agent.git
cd enterprise-agent

# Start infrastructure
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Run
uvicorn app.main:app --reload
```

## Architecture

```
API Gateway
    ↓
FastAPI (Rate Limit → Guardrails → Router)
    ↓
LangGraph Orchestrator
    ├── Supervisor Agent
    ├── Researcher Agent
    ├── Analyst Agent
    └── Executor Agent
    ↓
PostgreSQL (checkpoints) + Claude Haiku 4.5 (rate-limited)
    ↓
OpenTelemetry → Jaeger/Langfuse
```

## Key Patterns

### Rate Limiting
```python
semaphore = asyncio.Semaphore(30)
async with semaphore:
    await asyncio.sleep(random.uniform(0, 0.1))  # jitter
    response = await client.messages.create(...)
```

### Circuit Breaker
5 failures → circuit opens → 30s recovery timeout → half-open test

### Guardrails
Regex-based (~0.5ms) for blocking, LLM-based for flagging only.

## Blog Post

Full technical deep-dive: [Why 90% of AI Agent Projects Fail Before They Reach Production](https://jaybigdataiscool.medium.com/why-90-of-ai-agent-projects-fail-before-they-reach-production-4b948d804567)

## License

MIT
