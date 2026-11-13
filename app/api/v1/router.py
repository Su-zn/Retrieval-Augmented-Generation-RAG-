"""Aggregated router for API v1.

Endpoint routers are attached here as their phases land:
- Phase 3: ``ingest`` (POST /ingest)
- Phase 4: ``chat`` (POST /chat)
"""

from fastapi import APIRouter

from app.api.v1.endpoints import chat, ingest

api_router: APIRouter = APIRouter()
api_router.include_router(ingest.router)
api_router.include_router(chat.router)
