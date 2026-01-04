import httpx
import pytest

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_generate_thumbnail_happy_path():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/thumbnails/generate",
            json={"content_sources": {"image_paths": ["https://example.com/frame.jpg"]}},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "draft"
    assert data["recommended_title"]
    assert data["thumbnail_url"]
    assert isinstance(data["layers"], list)


@pytest.mark.anyio
async def test_generate_thumbnail_requires_media():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/thumbnails/generate", json={})

    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"] == "Invalid request payload"
    assert body["errors"][0]["field"] == "content_sources"
