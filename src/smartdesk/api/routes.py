"""HTTP route definitions; endpoint behavior pending."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok"}
