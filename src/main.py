from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import RecRecInference


app = FastAPI(
    title="MCM Recommendation API",
    version="1.0.0",
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


class RecommendationRequest(BaseModel):
    zoneInteractions: List[
        ZoneInteractionRequest
    ] = Field(default_factory=list)

    interactions: List[
        InteractionRequest
    ] = Field(default_factory=list)

    category: str
    
    topK: int = Field(default=6, gt=0)


class RecommendationItem(BaseModel):
    productId: int
    score: float


class RecommendationResponse(BaseModel):
    recommendations: List[
        RecommendationItem
    ]


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
):

    if not request.interactions and not request.zoneInteractions:
        raise HTTPException(
            status_code=400,
            detail="interactions and zoneInteractions cannot both be empty",
        )

    interactions = [
        interaction.model_dump()
        for interaction
        in request.interactions
    ]

    zone_interactions = [
        interaction.model_dump()
        for interaction
        in request.zoneInteractions
    ]

    try:
        recommendations = (
            recommender.recommend(
                interactions=interactions,
                zone_interactions=zone_interactions,
                category=request.category,
                top_k=request.topK,
                exclude_seen=True,
            )
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "recommendations": (
            recommendations
        )
    }
