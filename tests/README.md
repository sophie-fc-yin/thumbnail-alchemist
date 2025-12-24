# Tests

Test scripts for validating media processing functionality.

## Available Tests

### Vision Media Tests (`test_vision_media.py`)
Tests video frame extraction and image validation:
- GCS URL validation
- Frame extraction from video streams
- Image metadata processing

**Run:**
```bash
python -m tests.test_vision_media
```

### Audio Media Tests (`test_audio_media.py`)
Tests audio extraction, analysis, and transcription:
- Audio extraction from video (streaming, no full download)
- Audio feature analysis (pitch, energy, tempo, spectral features)
- GPT-4o Transcribe Diarize for automatic transcription and speaker diarization
- Timeline assembly with key moments for thumbnail selection

**Run:**
```bash
python -m tests.test_audio_media
```

## Requirements

**Environment Variables:**
- `OPENAI_API_KEY` - For GPT-4o Transcribe Diarize (transcription + speaker diarization)
- `GOOGLE_APPLICATION_CREDENTIALS` - For GCS access (if using private buckets)

**Test Data:**
Tests use real GCS URLs from `thumbail-alchemist-media` bucket. Update the URLs in test files if using different sources.

## What's Tested

### Vision Media
- ✅ URL validation (GCS, HTTP, local paths)
- ✅ Streaming frame extraction with ffmpeg
- ✅ Metadata extraction
- ✅ Error handling

### Audio Media
- ✅ Streaming audio extraction (WAV, 16kHz, mono)
- ✅ Audio feature analysis with librosa:
  - Pitch tracking
  - Energy/RMS
  - Spectral features
  - Tempo and beat detection
- ✅ GPT-4o Transcribe Diarize:
  - Automatic transcription with timestamps
  - Speaker diarization (who speaks when)
- ✅ Timeline assembly (KEY MOMENTS only):
  - Segments with speaker labels and audio features
  - Speaker transitions
  - Energy peaks (top 10%)
  - Significant pauses (>1s)
  - Music sections (>3s)

## Notes

- Audio tests extract only 60 seconds by default to save time/costs
- Tests include automatic cleanup of temporary files
- All tests work with streaming (no full file downloads required)
- Timeline is optimized for thumbnail selection (20-50 key events, not word-level)
