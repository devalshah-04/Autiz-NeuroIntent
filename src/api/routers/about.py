from fastapi import APIRouter

from routers._common import pipeline
from schemas import AboutResponse

router = APIRouter()


@router.get("/about", response_model=AboutResponse)
def about():
    return pipeline.get_about()
