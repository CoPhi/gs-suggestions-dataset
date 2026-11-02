from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Annotated, List, Optional

from backend.api.models import (
    FeedbackCreateRequest,
    FeedbackResponse,
    PredictionCount,
    PredictionsResponse,
)
from backend.api.services.suggestions_service import SuggestionsService

router = APIRouter(tags=["Predictions"])


def get_service() -> SuggestionsService:
    return SuggestionsService()


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@router.get(
    "/predictions",
    response_model=PredictionsResponse,
    responses={
        200: {
            "description": "Predictions",
            "content": {
                "application/json": {"example": {"predictions": {"type": "array"}}}
            },
        },
        204: {"description": "No content"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_predictions(
    model_id: Annotated[str, Query(title="ID", description="ID del modello")],
    context: Annotated[
        str,
        Query(
            title="Contesto di generazione",
            description="Deve contenere una lacuna da riempire, indicata tramite `[...]`",
            max_length=512,
        ),
    ],
    num_predictions: Annotated[
        PredictionCount,
        Query(
            title="Numero di predizioni da generare",
            description="Valori consentiti: 1, 5, 10, 20",
        ),
    ],
    num_tokens: Annotated[
        Optional[int],
        Query(
            title="Numero di token",
            description="Solo per modelli di tipo `Ngrams`",
            ge=1,
            le=10,
        ),
    ] = None,
    human_feedback: Annotated[
        Optional[str],
        Query(
            title="Indicazioni filologiche (Human-in-the-loop)",
            description="Vincoli o indicazioni libere del filologo (es. dialetto, morfologia, semantica) per raffinare la generazione",
            max_length=512,
        ),
    ] = None,
    rejected_candidates: Annotated[
        Optional[List[str]],
        Query(
            title="Supplementi scartati",
            description="Lista di supplementi già scartati dal filologo da escludere nella rigenerazione",
        ),
    ] = None,
    service: SuggestionsService = Depends(get_service),
):
    # Mantiene la chiamata compatibile con eventuali mock/spy nei test che non espongono
    # i parametri opzionali HITL quando non vengono passati nella query HTTP.
    extra_kwargs = {}
    if human_feedback is not None:
        extra_kwargs["human_feedback"] = human_feedback
    if rejected_candidates is not None:
        extra_kwargs["rejected_candidates"] = rejected_candidates

    predictions = await service.get_predictions(
        model_id=model_id,
        context=context,
        num_tokens=num_tokens,
        num_predictions=num_predictions,
        **extra_kwargs,
    )
    return JSONResponse(status_code=200, content={"predictions": predictions})


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    summary="Registra la validazione o lo scarto di un supplemento da parte del filologo (HITL Fase 1)",
)
async def create_feedback(
    payload: FeedbackCreateRequest,
    service: SuggestionsService = Depends(get_service),
):
    result = await service.save_feedback(payload)
    return JSONResponse(status_code=201, content=result)


@router.get(
    "/feedback",
    summary="Recupera i feedback filologici registrati",
)
async def list_feedback(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    is_accepted: Annotated[Optional[bool], Query()] = None,
    service: SuggestionsService = Depends(get_service),
):
    items = await service.get_feedback_list(limit=limit, is_accepted=is_accepted)
    return JSONResponse(status_code=200, content={"feedback": items})
