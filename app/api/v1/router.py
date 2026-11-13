"""Aggregated router for API v1.

Endpoint routers are attached here as their phases land:
- Phase 3: ``ingest`` (POST /ingest)
- Phase 4: ``chat`` (POST /chat)
"""

from fastapi import APIRouter

api_router: APIRouter = APIRouter()
