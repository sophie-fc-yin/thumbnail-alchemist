"""Test script to validate vision media processing with GCS URLs."""

import asyncio

from app.models import SourceMedia
from app.vision_media import extract_candidate_frames, validate_and_load_content


async def test_vision_media_processing():
    """Test validation and frame extraction with real GCS URLs."""

    # Your actual GCS URLs
    test_sources = SourceMedia(
        video_path="https://storage.googleapis.com/thumbail-alchemist-media/source-media/videos/0713-copy.mp4",
        image_paths=[
            "https://storage.googleapis.com/thumbail-alchemist-media/source-media/images/DSC02907.JPG"
        ],
    )

    print("=" * 60)
    print("Testing Media Validation and Frame Extraction")
    print("=" * 60)

    # Test 1: Validate content
    print("\n[1/2] Testing validate_and_load_content...")
    try:
        metadata = await validate_and_load_content(test_sources)
        print("✅ Validation successful!")
        print(f"   Video source: {metadata['video']['source']}")
        print(f"   Total files: {metadata['total_files']}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    # Test 2: Extract frames
    print("\n[2/2] Testing extract_candidate_frames...")
    print("   This will use ffmpeg to extract frames from the video URL...")
    try:
        frames = await extract_candidate_frames(
            test_sources, project_id="test-vision", max_frames=5
        )
        print("✅ Frame extraction successful!")
        print(f"   Extracted {len(frames)} frames:")
        for i, frame_path in enumerate(frames, 1):
            print(f"   {i}. {frame_path}")
    except Exception as e:
        print(f"❌ Frame extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_vision_media_processing())
