from uuid import uuid4

from fastapi import FastAPI

from app.models import CompositionLayer, ThumbnailRequest, ThumbnailResponse

app = FastAPI(title="Thumbnail Alchemist API", version="0.1.0")


@app.post("/thumbnails/generate", response_model=ThumbnailResponse)
async def generate_thumbnail(payload: ThumbnailRequest) -> ThumbnailResponse:
    demo_id = str(uuid4())

    return ThumbnailResponse(
        request_id=demo_id,
        status="draft",
        recommended_title=payload.title_hint
        or "The AI That Designs Thumbnails For You",
        thumbnail_url="https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f"
        "?auto=format&fit=crop&w=1200&q=80",
        selected_frame_url="https://images.unsplash.com/photo-1521737604893-d14cc237f11d"
        "?auto=format&fit=crop&w=900&q=80",
        profile_variant_url="https://images.unsplash.com/photo-1494790108377-be9c29b29330"
        "?auto=format&fit=crop&w=800&q=80",
        layers=[
            CompositionLayer(
                kind="background",
                description="High-energy gradient with subtle glow to lift the subject.",
                asset_url="https://images.unsplash.com/photo-1502082553048-f009c37129b9"
                "?auto=format&fit=crop&w=1200&q=80",
            ),
            CompositionLayer(
                kind="subject",
                description="Cutout of creator with key light on the right shoulder.",
                asset_url="https://images.unsplash.com/photo-1469474968028-56623f02e42e"
                "?auto=format&fit=crop&w=800&q=80",
            ),
            CompositionLayer(
                kind="title",
                description="Bold, uppercase typography with yellow stroke and drop shadow.",
                asset_url=None,
            ),
            CompositionLayer(
                kind="callout",
                description="Arrow pointing at the laptop screen to anchor viewer focus.",
                asset_url=None,
            ),
        ],
        summary=(
            "Draft composition combining your sources with a bold, high-contrast layout. "
            "Static assets are returned for now; a later iteration will run the full "
            "agent pipeline to select frames, pose the subject, and render the final thumbnail."
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
