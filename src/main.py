import logging
from pathlib import Path
from threading import RLock
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .avatar_look import select_avatar_look_products
from .background_removal import (
    BackgroundRemovalError,
    ImageDownloadError,
    ImageSaveError,
    ImageTooLargeError,
    UnsupportedImageError,
    initialize_background_removal,
    remove_background,
)
from .contrastive_refresh import (
    has_new_interactions,
    select_contrastive_products,
)
from .dataset import BEHAVIOR_TO_ID
from .direct_interest import score_direct_interest
from .evaluation import evaluate_personas
from .inference import RecRecInference
from .preference import build_product_preference_scores
from .style_identity import generate_style_identity_title


app = FastAPI(
    title="MCM Recommendation API",
    version="1.0.0",
)


@app.on_event("startup")
def warm_up_background_removal():
    initialize_background_removal()


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


class EvaluationZoneInteractionRequest(ZoneInteractionRequest):
    sequenceNo: int = Field(gt=0)


class EvaluationInteractionRequest(InteractionRequest):
    interactionType: Literal[
        "PRODUCT_SELECT",
        "FITTING",
        "WISHLIST_ADD",
        "WISHLIST_REMOVE",
    ]
    sequenceNo: int = Field(gt=0)


class EvaluationMemberWishlistRequest(BaseModel):
    productId: int


class EvaluationExpectedRecommendationRequest(BaseModel):
    productId: int
    relevance: int = Field(ge=1, le=5)


class EvaluationGroundTruthRequest(BaseModel):
    recommendations: List[EvaluationExpectedRecommendationRequest] = Field(min_length=1)


class EvaluationPersonaRequest(BaseModel):
    personaId: str = Field(min_length=1)
    personaType: Literal["CONFIDENT", "EXPLORATORY"]
    zoneInteractions: List[EvaluationZoneInteractionRequest] = Field(default_factory=list)
    arInteractions: List[EvaluationInteractionRequest] = Field(min_length=1)
    memberWishlists: List[EvaluationMemberWishlistRequest] = Field(default_factory=list)
    groundTruth: EvaluationGroundTruthRequest


class RecommendationEvaluationRequest(BaseModel):
    personas: List[EvaluationPersonaRequest] = Field(min_length=1)


class PreferenceInitializeRequest(BaseModel):
    arSessionId: int
    zoneInteractions: List[ZoneInteractionRequest] = Field(default_factory=list)
    memberInteractions: List[MemberInteractionRequest] = Field(default_factory=list)


class PreferenceInitializeResponse(BaseModel):
    arSessionId: int
    initialized: bool


class RecommendationRequest(BaseModel):
    arSessionId: int

    gender: Literal["MALE", "FEMALE"]

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
    interactions: List[InteractionRequest]


class AvatarLookProduct(BaseModel):
    productId: int


class AvatarLookResponse(BaseModel):
    arSessionId: int
    styleIdentityTitle: str
    products: List[AvatarLookProduct]


class RemoveBackgroundRequest(BaseModel):
    imageUrl: str = Field(min_length=1, pattern=r"^https?://")


class RemoveBackgroundResponse(BaseModel):
    imageUrl: str


# Process-local MVP storage. Preferences are reset whenever the server restarts.
preferences: Dict[int, Dict[str, Dict[str, float]]] = {}
preferences_lock = RLock()

# Raw online wishlist product IDs are retained separately for deterministic
# Avatar Look interest scoring. The existing preference scores remain unchanged.
member_wishlists: Dict[int, List[int]] = {}
member_wishlists_lock = RLock()

# Raw store movement is retained for Style Identity generation without
# changing the existing Spring request contract.
zone_interaction_contexts: Dict[int, List[dict]] = {}
zone_interaction_contexts_lock = RLock()

# Latest successful recommendation interactions per session. Like preferences,
# these process-local values are reset whenever the server restarts.
session_interactions: Dict[int, List[dict]] = {}
session_interactions_lock = RLock()

# Gender used for recommendation candidate filtering in each AR session.
session_genders: Dict[int, str] = {}
session_genders_lock = RLock()

# Latest result and interaction snapshot per session/category. Process-local,
# matching the lifetime of the existing preference and interaction stores.
recent_category_recommendations: Dict[tuple, Dict[str, List[dict]]] = {}
recent_category_recommendations_lock = RLock()


def store_preferences(ar_session_id, zone_interactions, member_interactions):
    member_scores = (
        recommender.build_wishlist_preference_scores(member_interactions)
        if member_interactions
        else {}
    )
    product_scores = build_product_preference_scores(
        recommender.products,
        zone_interactions,
        member_scores,
    )
    category_preferences = {}

    for product in recommender.products:
        category_preferences.setdefault(product.category, {})[product.product_id] = (
            product_scores[product.product_id]
        )

    with preferences_lock:
        preferences[ar_session_id] = category_preferences
    with member_wishlists_lock:
        member_wishlists[ar_session_id] = [
            int(interaction.productId)
            for interaction in member_interactions
        ]
    with zone_interaction_contexts_lock:
        zone_interaction_contexts[ar_session_id] = [
            interaction.model_dump()
            for interaction in zone_interactions
        ]


def get_zone_scores(ar_session_id, category):
    with preferences_lock:
        scores = preferences.get(ar_session_id, {}).get(category)
        return dict(scores) if scores is not None else None


def get_member_wishlist_product_ids(ar_session_id):
    with member_wishlists_lock:
        return list(member_wishlists.get(ar_session_id, []))


def get_zone_interaction_contexts(ar_session_id):
    with zone_interaction_contexts_lock:
        return [
            dict(item)
            for item in zone_interaction_contexts.get(ar_session_id, [])
        ]


def store_session_interactions(ar_session_id, interactions):
    with session_interactions_lock:
        session_interactions[ar_session_id] = [dict(item) for item in interactions]


def get_session_interactions(ar_session_id):
    with session_interactions_lock:
        return [
            dict(item)
            for item in session_interactions.get(ar_session_id, [])
        ]


def store_session_gender(ar_session_id, gender):
    with session_genders_lock:
        session_genders[ar_session_id] = gender


def get_session_gender(ar_session_id):
    with session_genders_lock:
        return session_genders.get(ar_session_id)


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
            gender=request.gender,
            top_k=request.topK,
            preference_product_ids=get_member_wishlist_product_ids(
                request.arSessionId
            ),
            exclude_seen=True,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    store_session_interactions(request.arSessionId, interactions)
    store_session_gender(request.arSessionId, request.gender)
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
    gender = get_session_gender(arSessionId)
    if gender is None:
        raise HTTPException(
            status_code=400,
            detail="Call /recommend with gender before refreshing recommendations",
        )
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
                gender=gender,
                top_k=6,
                preference_product_ids=get_member_wishlist_product_ids(
                    arSessionId
                ),
                exclude_seen=True,
            )
        else:
            category_product_count = sum(
                product.category == category
                and (
                    category == "BAG"
                    or product.gender in {gender, "UNISEX"}
                )
                for product in recommender.products
            )
            scored_products = recommender.recommend(
                interactions=interactions,
                zone_scores=get_zone_scores(arSessionId, category),
                category=category,
                gender=gender,
                top_k=category_product_count,
                preference_product_ids=get_member_wishlist_product_ids(
                    arSessionId
                ),
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
    "/recommendations/avatar-look",
    response_model=AvatarLookResponse,
)
def recommend_avatar_look(request: AvatarLookRequest):
    interactions = [
        interaction.model_dump()
        for interaction in request.interactions
    ]
    valid_interaction_count = sum(
        interaction["productId"] in recommender.mapper.product_to_index
        and interaction["interactionType"] in BEHAVIOR_TO_ID
        for interaction in interactions
    )
    unique_product_ids = {
        int(interaction["productId"])
        for interaction in interactions
    }
    logger.info(
        "Avatar Look request. arSessionId=%s, receivedInteractionCount=%s, "
        "validInteractionCount=%s, uniqueProductIdCount=%s, catalogProductCount=%s",
        request.arSessionId,
        len(interactions),
        valid_interaction_count,
        len(unique_product_ids),
        len(recommender.products),
    )

    try:
        scored_products = score_direct_interest(
            products=recommender.products,
            online_wishlist_product_ids=get_member_wishlist_product_ids(
                request.arSessionId
            ),
            interactions=interactions,
        )
        logger.info(
            "Avatar Look direct interest scoring complete. arSessionId=%s, "
            "scoredProductsCount=%s",
            request.arSessionId,
            len(scored_products),
        )
        selected_products = select_avatar_look_products(
            scored_products,
            ar_session_id=request.arSessionId,
        )
        style_identity_title = generate_style_identity_title(
            selected_products=selected_products,
            zone_interactions=get_zone_interaction_contexts(request.arSessionId),
        )
    except Exception as error:
        logger.exception(
            "Avatar Look failed. arSessionId=%s, error=%s",
            request.arSessionId,
            error,
        )
        raise

    return {
        "arSessionId": request.arSessionId,
        "styleIdentityTitle": style_identity_title,
        "products": [
            {"productId": item["productId"]}
            for item in selected_products
        ],
    }


@app.post("/evaluations/recommendations")
def evaluate_recommendations(request: RecommendationEvaluationRequest):
    persona_ids = [persona.personaId for persona in request.personas]
    if len(persona_ids) != len(set(persona_ids)):
        raise HTTPException(status_code=400, detail="personaId must be unique")

    try:
        return evaluate_personas(request.personas, recommender)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
