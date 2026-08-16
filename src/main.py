import logging
from pathlib import Path
from threading import RLock
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .avatar_look import (
    calculate_product_active_states,
    explicitly_removed_without_positive_state,
    select_category_anchors,
    select_category_complement,
)
from .background_removal import (
    BackgroundRemovalError,
    ImageDownloadError,
    ImageSaveError,
    ImageTooLargeError,
    UnsupportedImageError,
    remove_background,
)
from .contrastive_refresh import (
    has_new_interactions,
    select_contrastive_products,
)
from .dataset import BEHAVIOR_TO_ID
from .inference import RecRecInference


app = FastAPI(
    title="MCM Recommendation API",
    version="1.0.0",
)
logger = logging.getLogger(__name__)
generated_image_directory = Path("generated").resolve()
app.mount(
    "/images/generated",
    StaticFiles(directory=generated_image_directory, check_dir=False),
    name="generated-images",
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


class MemberInteractionRequest(BaseModel):
    productId: int
    action: Literal["WISHLIST"]


class PreferenceInitializeRequest(BaseModel):
    arSessionId: int
    zoneInteractions: List[ZoneInteractionRequest] = Field(default_factory=list)
    memberInteractions: List[MemberInteractionRequest] = Field(default_factory=list)


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


class RefreshRecommendationRequest(BaseModel):
    interactions: Optional[List[InteractionRequest]] = None


class AvatarLookRequest(BaseModel):
    arSessionId: int


class AvatarLookProduct(BaseModel):
    productId: int


class AvatarLookResponse(BaseModel):
    arSessionId: int
    styleIdentityTitle: str
    products: List[AvatarLookProduct]


class CategoryRankingValidationRequest(BaseModel):
    arSessionId: int
    productIds: List[int] = Field(min_length=1)


class CategoryRankingItem(BaseModel):
    productId: int
    name: str
    category: str
    categoryRank: int
    categorySize: int
    score: float


class CategoryRankingValidationResponse(BaseModel):
    arSessionId: int
    anchorRankings: List[CategoryRankingItem]


class RemoveBackgroundRequest(BaseModel):
    imageUrl: str = Field(min_length=1, pattern=r"^https?://")


class RemoveBackgroundResponse(BaseModel):
    imageUrl: str


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

# Latest result and interaction snapshot per session/category. Process-local,
# matching the lifetime of the existing preference and interaction stores.
recent_category_recommendations: Dict[tuple, Dict[str, List[dict]]] = {}
recent_category_recommendations_lock = RLock()


def store_preferences(ar_session_id, zone_interactions, member_interactions):
    dwell_by_category = {}
    for interaction in zone_interactions:
        category = interaction.category.strip().upper()
        zone = interaction.zone.strip().upper()
        zone = ZONE_ALIASES.get(zone, zone)
        dwell_seconds = max(0.0, interaction.dwellSeconds)
        category_dwell = dwell_by_category.setdefault(category, {})
        category_dwell[zone] = category_dwell.get(zone, 0.0) + dwell_seconds

    zone_preferences = {}
    for category, zone_dwell in dwell_by_category.items():
        total_dwell = sum(zone_dwell.values())
        zone_preferences[category] = {
            zone: dwell / total_dwell if total_dwell > 0 else 0.0
            for zone, dwell in zone_dwell.items()
        }

    member_scores = (
        recommender.build_wishlist_preference_scores(member_interactions)
        if member_interactions
        else {}
    )
    has_zone_preferences = bool(zone_interactions)
    has_member_preferences = bool(member_interactions)
    category_preferences = {}

    for product in recommender.products:
        zone_score = zone_preferences.get(product.category, {}).get(product.zone, 0.0)
        member_score = member_scores.get(product.product_id, 0.0)

        if has_zone_preferences and has_member_preferences:
            score = 0.7 * zone_score + 0.3 * member_score
        elif has_zone_preferences:
            score = zone_score
        else:
            score = member_score

        category_preferences.setdefault(product.category, {})[product.product_id] = score

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


def store_recent_category_recommendations(
    ar_session_id,
    category,
    recommendations,
    interactions,
):
    with recent_category_recommendations_lock:
        recent_category_recommendations[(ar_session_id, category)] = {
            "recommendations": [dict(item) for item in recommendations],
            "interactions": [dict(item) for item in interactions],
        }


def get_recent_category_recommendations(ar_session_id, category):
    with recent_category_recommendations_lock:
        recent = recent_category_recommendations.get((ar_session_id, category))
        if recent is None:
            return None
        return {
            "recommendations": [dict(item) for item in recent["recommendations"]],
            "interactions": [dict(item) for item in recent["interactions"]],
        }


def get_style_identity_title():
    return "Your Signature Look"


def score_category_products(ar_session_id, interactions, category, exclude_seen):
    category_product_count = sum(
        product.category == category
        for product in recommender.products
    )
    recommendations = recommender.recommend(
        interactions=interactions,
        zone_scores=get_zone_scores(ar_session_id, category),
        category=category,
        top_k=category_product_count,
        exclude_seen=exclude_seen,
    )
    return sorted(
        recommendations,
        key=lambda item: (-item["score"], item["productId"]),
    )


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
    if not request.zoneInteractions and not request.memberInteractions:
        raise HTTPException(
            status_code=400,
            detail="zoneInteractions and memberInteractions cannot both be empty",
        )

    try:
        store_preferences(
            request.arSessionId,
            request.zoneInteractions,
            request.memberInteractions,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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
    store_recent_category_recommendations(
        request.arSessionId,
        category,
        recommendations,
        interactions,
    )

    return {
        "recommendations": recommendations
    }


@app.post(
    "/api/recommendations/ar-sessions/{arSessionId}/categories/{categoryCode}/refresh",
    response_model=RecommendationResponse,
)
def refresh_recommendations(
    arSessionId: int,
    categoryCode: str,
    request: Optional[RefreshRecommendationRequest] = None,
):
    category = categoryCode.strip().upper()
    if request is not None and request.interactions is not None:
        interactions = [item.model_dump() for item in request.interactions]
    else:
        interactions = get_session_interactions(arSessionId)

    recent = get_recent_category_recommendations(arSessionId, category)
    interaction_added = (
        recent is None
        or has_new_interactions(interactions, recent["interactions"])
    )

    try:
        if interaction_added:
            recommendations = recommender.recommend(
                interactions=interactions,
                zone_scores=get_zone_scores(arSessionId, category),
                category=category,
                top_k=6,
                exclude_seen=True,
            )
        else:
            category_product_count = sum(
                product.category == category
                for product in recommender.products
            )
            scored_products = recommender.recommend(
                interactions=interactions,
                zone_scores=get_zone_scores(arSessionId, category),
                category=category,
                top_k=category_product_count,
                exclude_seen=True,
            )
            previous_ids = {
                item["productId"]
                for item in recent["recommendations"]
            }
            candidate_count = sum(
                item["productId"] not in previous_ids
                for item in scored_products
            )
            recommendations = select_contrastive_products(
                scored_products,
                previous_ids,
            )
            logger.info(
                "Contrastive refresh applied. arSessionId=%s, categoryCode=%s, "
                "previousCount=%s, candidateCount=%s, resultCount=%s",
                arSessionId,
                category,
                len(previous_ids),
                candidate_count,
                len(recommendations),
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    store_session_interactions(arSessionId, interactions)
    store_recent_category_recommendations(
        arSessionId,
        category,
        recommendations,
        interactions,
    )
    return {"recommendations": recommendations}


@app.post(
    "/images/remove-background",
    response_model=RemoveBackgroundResponse,
)
def remove_image_background(request: RemoveBackgroundRequest):
    try:
        filename = remove_background(
            request.imageUrl,
            generated_image_directory,
        )
    except ImageTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except UnsupportedImageError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except ImageDownloadError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except BackgroundRemovalError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ImageSaveError as error:
        raise HTTPException(status_code=507, detail=str(error)) from error

    return {"imageUrl": f"/images/generated/{filename}"}


@app.post(
    "/recommendations/validation",
    response_model=CategoryRankingValidationResponse,
)
def validate_category_rankings(request: CategoryRankingValidationRequest):
    requested_product_ids = list(dict.fromkeys(request.productIds))
    unknown_product_ids = [
        product_id
        for product_id in requested_product_ids
        if product_id not in recommender.product_by_id
    ]
    if unknown_product_ids:
        raise HTTPException(
            status_code=404,
            detail={"unknownProductIds": unknown_product_ids},
        )

    interactions = get_session_interactions(request.arSessionId)
    products = [
        recommender.product_by_id[product_id]
        for product_id in requested_product_ids
    ]
    categories = list(dict.fromkeys(product.category for product in products))

    rankings_by_product_id = {}
    try:
        for category in categories:
            recommendations = score_category_products(
                request.arSessionId,
                interactions,
                category,
                exclude_seen=False,
            )
            category_size = sum(
                product.category == category
                for product in recommender.products
            )
            rankings_by_product_id.update({
                item["productId"]: {
                    "categoryRank": rank,
                    "categorySize": category_size,
                    "score": item["score"],
                }
                for rank, item in enumerate(recommendations, start=1)
            })
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    anchor_rankings = []
    for product in products:
        ranking = rankings_by_product_id.get(product.product_id)
        if ranking is None:
            raise HTTPException(
                status_code=500,
                detail=f"Product {product.product_id} was not scored in {product.category}",
            )
        anchor_rankings.append({
            "productId": product.product_id,
            "name": product.name,
            "category": product.category,
            **ranking,
        })

    return {
        "arSessionId": request.arSessionId,
        "anchorRankings": anchor_rankings,
    }


@app.post(
    "/recommendations/avatar-look",
    response_model=AvatarLookResponse,
)
def recommend_avatar_look(request: AvatarLookRequest):
    interactions = get_session_interactions(request.arSessionId)
    valid_interaction_count = sum(
        interaction["productId"] in recommender.mapper.product_to_index
        and interaction["interactionType"] in BEHAVIOR_TO_ID
        for interaction in interactions
    )
    logger.info(
        "Avatar Look request. arSessionId=%s, storedInteractionCount=%s, "
        "validInteractionCount=%s, uniqueProductIdCount=%s, catalogProductCount=%s",
        request.arSessionId,
        len(interactions),
        valid_interaction_count,
        len({int(interaction["productId"]) for interaction in interactions}),
        len(recommender.products),
    )

    current_category = None
    try:
        categories = list(dict.fromkeys(
            product.category
            for product in recommender.products
        ))

        recommendations_by_category = {}
        scores_by_product_id = {}
        for category in categories:
            current_category = category
            recommendations = score_category_products(
                request.arSessionId,
                interactions,
                category,
                exclude_seen=False,
            )
            recommendations_by_category[category] = recommendations
            scores_by_product_id.update({
                item["productId"]: item["score"]
                for item in recommendations
            })
            logger.info(
                "Avatar Look category. arSessionId=%s, category=%s, "
                "excludeSeen=false, scoredProductCount=%s",
                request.arSessionId,
                category,
                len(recommendations),
            )

        current_category = None
        product_states = calculate_product_active_states(interactions)
        anchors_by_category = select_category_anchors(
            product_states,
            recommender.product_by_id,
            scores_by_product_id,
        )
        excluded_complements = explicitly_removed_without_positive_state(
            product_states
        )
        selected_products = []
        selected_product_ids = set()
        for category in categories:
            selected = anchors_by_category.get(category)
            if selected is None:
                selected = select_category_complement(
                    recommendations_by_category[category],
                    excluded_complements | selected_product_ids,
                )
            if selected is not None and selected["productId"] not in selected_product_ids:
                selected_products.append(selected)
                selected_product_ids.add(selected["productId"])

        logger.info(
            "Avatar Look selection. arSessionId=%s, anchors=%s, "
            "excludedComplements=%s, finalProductIds=%s",
            request.arSessionId,
            {
                category: item["productId"]
                for category, item in anchors_by_category.items()
            },
            sorted(excluded_complements),
            [item["productId"] for item in selected_products],
        )
    except Exception as error:
        logger.exception(
            "Avatar Look failed. arSessionId=%s, category=%s, error=%s",
            request.arSessionId,
            current_category,
            error,
        )
        raise

    return {
        "arSessionId": request.arSessionId,
        "styleIdentityTitle": get_style_identity_title(),
        "products": [
            {"productId": item["productId"]}
            for item in selected_products
        ],
    }
