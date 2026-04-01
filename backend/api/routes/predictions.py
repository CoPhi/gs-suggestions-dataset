from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Annotated, Optional

from backend.api.exceptions import InvalidContextError, ModelNotFoundError
from backend.api.models import PredictionCount, PredictionsResponse
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
    service: SuggestionsService = Depends(get_service),
):
    try:
        predictions = await service.get_predictions(
            model_id=model_id,
            context=context,
            num_tokens=num_tokens,
            num_predictions=num_predictions,
        )
        return JSONResponse(status_code=200, content={"predictions": predictions})

    except InvalidContextError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except ModelNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
