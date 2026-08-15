from threading import RLock
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .inference import RecRecInference


app = FastAPI(
    title="MCM Recommendation API",
    version="1.0.0",
)


# 422 Validation Error 디버깅용
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    body = await request.body()

    print("\n=== 422 VALIDATION ERROR ===")
    print("URL:", request.url)
    print("BODY:", body.decode("utf-8"))
    print("ERRORS:", exc.errors())
    print("============================\n")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )


# 서버 시작할 때 모델 1번만 로드
recommender = RecRecInference()


class InteractionRequest(BaseModel):
    productId: int
    interactionType: str


class ZoneInteractionRequest(BaseModel):
    zone: str
    category: str
    dwellSeconds: float


class PreferenceInitializeRequest(BaseModel):
    arSessionId: int
    zoneInteractions: List[ZoneInteractionRequest] = Field(default_factory=list)


class PreferenceInitializeResponse(BaseModel):
    arSessionId: int
    initialized: bool


class RecommendationRequest(BaseModel):
    arSessionId: int

    interactions: List[InteractionRequest] = Field(
        default_factory=list
    )

    category: str

    topK: int = Field(
        default=6,
        gt=0,
    )


class RecommendationItem(BaseModel):
    productId: int
    score: float


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]


ZONE_ALIASES = {
    "NEW_COLLECTION": "NEW",
}


# Process-local MVP storage. Preferences are reset whenever the server restarts.
preferences: Dict[int, Dict[str, Dict[str, float]]] = {}
preferences_lock = RLock()


def store_preferences(ar_session_id, zone_interactions):
    dwell_by_category = {}
    for interaction in zone_interactions:
        category = interaction.category.strip().upper()
        zone = interaction.zone.strip().upper()
        zone = ZONE_ALIASES.get(zone, zone)
        dwell_seconds = max(0.0, interaction.dwellSeconds)
        category_dwell = dwell_by_category.setdefault(category, {})
        category_dwell[zone] = category_dwell.get(zone, 0.0) + dwell_seconds

    category_preferences = {}
    for category, zone_dwell in dwell_by_category.items():
        total_dwell = sum(zone_dwell.values())
        category_preferences[category] = {
            zone: dwell / total_dwell if total_dwell > 0 else 0.0
            for zone, dwell in zone_dwell.items()
        }

    with preferences_lock:
        preferences[ar_session_id] = category_preferences


def get_zone_scores(ar_session_id, category):
    with preferences_lock:
        scores = preferences.get(ar_session_id, {}).get(category)
        return dict(scores) if scores is not None else None


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/preferences/initialize",
    response_model=PreferenceInitializeResponse,
)
def initialize_preferences(request: PreferenceInitializeRequest):
    if not request.zoneInteractions:
        raise HTTPException(
            status_code=400,
            detail="zoneInteractions cannot be empty",
        )

    store_preferences(request.arSessionId, request.zoneInteractions)
    return {
        "arSessionId": request.arSessionId,
        "initialized": True,
    }


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
):
    category = request.category.strip().upper()
    zone_scores = get_zone_scores(request.arSessionId, category)

    interactions = [
        interaction.model_dump()
        for interaction in request.interactions
    ]

    try:
        recommendations = recommender.recommend(
            interactions=interactions,
            zone_scores=zone_scores,
            category=category,
            top_k=request.topK,
            exclude_seen=True,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "recommendations": recommendations
    }
