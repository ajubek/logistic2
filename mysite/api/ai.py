from fastapi import APIRouter, HTTPException

from ..chain.search_chain import run_search
from ..database.schema import SearchInputSchema, SearchOutputSchema

ai_router = APIRouter(prefix="/ai", tags=["AI"])


@ai_router.post("/chat", response_model=SearchOutputSchema)
def chat(data: SearchInputSchema):
    try:
        return run_search(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")