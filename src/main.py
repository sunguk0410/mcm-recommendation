from threading import RLock
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .avatar_look import select_avatar_look_products
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


class AvatarLookRequest(BaseModel):
    arSessionId: int


class AvatarLookProduct(BaseModel):
    productId: int


class AvatarLookResponse(BaseModel):
    arSessionId: int
    styleIdentityTitle: str
    products: List[AvatarLookProduct]


ZONE_ALIASES = {
    "NEW_COLLECTION": "NEW",
}

# Process-local MVP storage. Preferences are reset whenever the server restarts.
preferences: Dict[int, Dict[str, Dict[str, float]]] = {}
preferences_lock = RLock()

# Latest successful recommendation interactions per session. Like preferences,
# these process-local values are reset whenever the server restarts.
session_interactions: Dict[int, List[dict]] = {}
session_interactions_lock = RLock()


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


def store_session_interactions(ar_session_id, interactions):
    with session_interactions_lock:
        session_interactions[ar_session_id] = [dict(item) for item in interactions]


def get_session_interactions(ar_session_id):
    with session_interactions_lock:
        return [
            dict(item)
            for item in session_interactions.get(ar_session_id, [])
        ]


def get_style_identity_title():
    return "Your Signature Look"


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

    store_session_interactions(request.arSessionId, interactions)

    return {
        "recommendations": recommendations
    }


@app.post(
    "/recommendations/avatar-look",
    response_model=AvatarLookResponse,
)
def recommend_avatar_look(request: AvatarLookRequest):
    interactions = get_session_interactions(request.arSessionId)
    categories = list(dict.fromkeys(
        product.category
        for product in recommender.products
    ))

    scored_products = []
    for category in categories:
        category_product_count = sum(
            product.category == category
            for product in recommender.products
        )
        recommendations = recommender.recommend(
            interactions=interactions,
            zone_scores=get_zone_scores(request.arSessionId, category),
            category=category,
            top_k=category_product_count,
            exclude_seen=True,
        )
        scored_products.extend(
            {
                **item,
                "category": category,
            }
            for item in recommendations
        )

    selected_products = select_avatar_look_products(scored_products)

    return {
        "arSessionId": request.arSessionId,
        "styleIdentityTitle": get_style_identity_title(),
        "products": [
            {"productId": item["productId"]}
            for item in selected_products
        ],
    }
