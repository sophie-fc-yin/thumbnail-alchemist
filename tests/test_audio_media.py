"""Integration script (skipped in unit test runs).

This file uses real media URLs, cloud credentials, and external models.
It is not suitable for CI/unit test environments.
"""

import pytest

pytest.skip("integration-only test; skipped in unit test runs", allow_module_level=True)

import asyncio  # noqa: E402

from app.audio_media import (  # noqa: E402
    analyze_audio_features,
    extract_audio_from_video,
    transcribe_and_analyze_audio,
)
from app.models import SourceMedia  # noqa: E402


async def test_audio_media_processing():
    """Test audio extraction, feature analysis, and transcription with real GCS URLs."""

    # Test video source
    test_sources = SourceMedia(
        video_path="https://storage.googleapis.com/thumbail-alchemist-media/source-media/videos/0713-copy.mp4",
        image_paths=[],
    )
    project_id = "test-audio"

    print("=" * 60)
    print("Testing Audio Media Processing")
    print("=" * 60)

    # Test 1: Extract audio from video
    print("\n[1/3] Testing extract_audio_from_video...")
    print("   This will extract audio from the video URL without loading the entire file...")
    try:
        audio_path = await extract_audio_from_video(
            test_sources,
            project_id=project_id,
            max_duration_seconds=60,  # Only extract first 60 seconds for testing
        )
        if audio_path:
            print("✅ Audio extraction successful!")
            print(f"   Extracted audio: {audio_path}")
            print(f"   File size: {audio_path.stat().st_size / 1024:.2f} KB")
        else:
            print("❌ Audio extraction failed: No audio path returned")
            return
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        return

    # Test 2: Analyze audio features
    print("\n[2/3] Testing analyze_audio_features...")
    print("   Extracting prosody, pitch, energy, tempo, and music features...")
    try:
        audio_features = await analyze_audio_features(audio_path)
        print("✅ Audio feature analysis successful!")
        print(f"   Duration: {audio_features['times'][-1]:.2f} seconds")
        print(
            f"   Average pitch: {sum(p for p in audio_features['pitch'] if p > 0) / max(1, sum(1 for p in audio_features['pitch'] if p > 0)):.2f} Hz"
        )
        print(
            f"   Average energy: {sum(audio_features['energy']) / len(audio_features['energy']):.4f}"
        )
        print(f"   Tempo: {audio_features['tempo']:.2f} BPM")
        print(f"   Detected beats: {len(audio_features['beat_times'])}")
    except Exception as e:
        print(f"❌ Audio feature analysis failed: {e}")
        return

    # Test 3: Transcribe and analyze with GPT-4o
    print("\n[3/3] Testing transcribe_and_analyze_audio...")
    print("   Using Whisper for transcription and GPT-4o for semantic analysis...")
    try:
        result = await transcribe_and_analyze_audio(
            audio_path, project_id=project_id, language="en"
        )
        print("✅ Transcription and analysis successful!")

        # Show timeline JSON path if saved
        if "timeline_path" in result:
            print(f"\n   Timeline saved to: {result['timeline_path']}")

        print("\n   Transcript preview:")
        transcript_preview = (
            result["transcript"][:200] + "..."
            if len(result["transcript"]) > 200
            else result["transcript"]
        )
        print(f'   "{transcript_preview}"')

        print("\n   Timeline Summary:")
        print(f"   - Total events: {len(result['timeline'])} (KEY MOMENTS ONLY)")
        print(f"   - Speakers detected: {len(result['speakers'])}")
        print(f"   - Duration: {result['duration_seconds']:.2f} seconds")

        # Print timeline breakdown by event type
        event_types = {}
        for event in result["timeline"]:
            event_type = event.get("type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("\n   Timeline Event Breakdown:")
        for event_type, count in sorted(event_types.items()):
            print(f"   - {event_type}: {count}")

        # Show sample events for each type
        print("\n   Sample Timeline Events:")
        shown_types = set()
        for event in result["timeline"][:20]:  # Show first 20 events
            event_type = event.get("type")
            if event_type not in shown_types:
                shown_types.add(event_type)
                if event_type == "segment":
                    print(
                        f"   [{event_type}] {event['start']:.1f}s: \"{event['text'][:50]}...\" (energy: {event.get('avg_energy', 0):.3f})"
                    )
                elif event_type == "energy_peak":
                    print(
                        f"   [{event_type}] {event['time']:.1f}s: High energy moment (energy: {event['energy']:.3f})"
                    )
                elif event_type == "pause":
                    print(
                        f"   [{event_type}] {event['start']:.1f}s: Significant pause ({event['duration']:.1f}s)"
                    )
                elif event_type == "emphasis":
                    print(
                        f"   [{event_type}] {event.get('time', 0):.1f}s: \"{event.get('text', '')}\" ({event.get('emphasis_type', 'unknown')})"
                    )
                elif event_type == "tone_change":
                    print(
                        f"   [{event_type}] {event.get('time', 0):.1f}s: Tone shift to {event.get('tone', 'unknown')}"
                    )
                elif event_type == "music_section":
                    print(
                        f"   [{event_type}] {event['start']:.1f}s-{event['end']:.1f}s: Background music (intensity: {event.get('intensity', 0):.3f})"
                    )
                elif event_type == "speaker_turn":
                    print(
                        f"   [{event_type}] {event['start']:.1f}s-{event['end']:.1f}s: Speaker {event.get('speaker', 'unknown')}"
                    )
                else:
                    print(f"   [{event_type}] {event}")

        # Print speech tone
        print("\n   Speech characteristics:")
        print(f"   - Avg pitch: {result['speech_tone']['avg_pitch']:.2f} Hz")
        print(f"   - Avg energy: {result['speech_tone']['avg_energy']:.4f}")
        print(f"   - Tempo: {result['speech_tone']['tempo']:.2f} BPM")

        # Highlight key moments for thumbnail selection
        print("\n   🎯 Key Moments for Thumbnail Selection:")
        energy_peaks = [e for e in result["timeline"] if e.get("type") == "energy_peak"]
        speaker_turns = [e for e in result["timeline"] if e.get("type") == "speaker_turn"]
        music_sections = [e for e in result["timeline"] if e.get("type") == "music_section"]

        if energy_peaks:
            print(f"   - {len(energy_peaks)} high-energy moments detected")
            for peak in energy_peaks[:3]:  # Show top 3
                print(f"     • {peak['time']:.1f}s (energy: {peak['energy']:.3f})")

        if speaker_turns:
            print(f"   - {len(speaker_turns)} speaker transitions detected")
            for turn in speaker_turns[:3]:  # Show top 3
                print(
                    f"     • {turn['time']:.1f}s: {turn.get('from_speaker', 'unknown')} → {turn.get('to_speaker', 'unknown')}"
                )

        if music_sections:
            print(f"   - {len(music_sections)} music sections detected")
            for music in music_sections[:3]:  # Show top 3
                print(
                    f"     • {music['start']:.1f}s-{music['end']:.1f}s (intensity: {music['intensity']:.3f})"
                )

        print("\n   💡 Timeline Efficiency:")
        print(f"   - Clean timeline with only {len(result['timeline'])} key events")
        print("   - No word-level pollution (removed thousands of events)")
        print("   - Perfect for thumbnail frame selection")

    except Exception as e:
        print(f"❌ Transcription and analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("All audio tests passed! 🎉")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up temporary files...")
    try:
        if audio_path and audio_path.exists():
            # Remove the audio file and its parent directories if empty
            audio_path.unlink()
            print(f"   Removed: {audio_path}")

            # Try to remove parent directories if they're empty
            parent = audio_path.parent
            while parent.name and parent.exists():
                try:
                    parent.rmdir()
                    print(f"   Removed empty directory: {parent}")
                    parent = parent.parent
                except OSError:
                    # Directory not empty or can't be removed
                    break
    except Exception as e:
        print(f"   Warning: Cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_audio_media_processing())
