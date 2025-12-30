# Thumbnail Selection Agent - Example Output

This document shows what the enhanced reasoning output looks like.

## Example Selection Result

```json
{
  "selected_frame": 2,
  "selected_frame_number": 2,
  "selected_frame_path": "/path/to/frame_12800ms.jpg",
  "selected_frame_url": "gs://bucket/projects/123/frames/frame_12800ms.jpg",
  "selected_frame_timestamp": 12.8,
  "confidence": 0.87,

  "reasoning": {
    "summary": "Frame 2 is the clear winner because it captures a genuine moment of surprise with perfect expression intensity that will immediately grab attention in crowded YouTube feeds.",

    "visual_analysis": "The frame shows a wide-eyed expression with raised eyebrows and slightly open mouth - a textbook 'surprise' reaction. The lighting is bright and even, keeping the face clearly visible. The subject is perfectly centered with good headroom, leaving space for title text overlay at the top. The background is slightly blurred, keeping focus on the facial expression.",

    "score_alignment": "My visual assessment strongly aligns with the quantitative scores. Frame 2 scored highest in psychology (0.82) and face quality (0.91), which matches what I see - the expression intensity is genuinely high and the emotion is unmistakable. The aesthetic score (0.75) is appropriate given the 'tech reviews' niche doesn't prioritize soft lighting over clarity.",

    "niche_fit": "For tech reviews, this frame works perfectly because it combines clarity with emotion. The sharp focus and bright lighting meet the niche's technical standards, while the genuine surprise expression adds the human element that tech channels need to avoid feeling sterile. The centered composition is standard for tech thumbnails but the expression makes it stand out.",

    "goal_optimization": "For maximizing CTR, this frame excels because the surprise expression triggers the 'curiosity gap' - viewers will want to know what caused this reaction. The expression is intense enough (0.91) to pattern-interrupt as viewers scroll, and the genuine emotion creates an immediate connection that makes people pause and click.",

    "psychology_triggers": "This frame activates three powerful triggers: (1) Curiosity Gap - the expression begs the question 'what happened?', (2) Emotional Contagion - viewers mirror the surprise emotion, making them feel invested before even clicking, and (3) Pattern Interrupt - among typical tech thumbnails showing products, this human reaction breaks the pattern and demands attention."
  },

  "key_strengths": [
    "Genuine surprise expression with 0.91 intensity - immediately eye-catching",
    "Perfect centering with space for title text overlay at top",
    "Bright, even lighting ensures visibility in all contexts (mobile, desktop, dark mode)",
    "Activates curiosity gap + emotional contagion psychology triggers"
  ],

  "comparative_analysis": {
    "runner_up": "Frame 1 was second with similar surprise expression (0.82 intensity) but the timing at 5.2s showed a less peak moment - the expression was building rather than fully formed. Frame 4 scored high on joy (0.88) but for a 'maximize CTR' goal with 'surprise' target emotion, joy doesn't create the same curiosity gap.",

    "score_vs_visual": "The highest-scoring frame (Frame 2) was indeed selected, showing strong alignment between quantitative analysis and visual judgment. The scoring system correctly identified that expression intensity and psychology triggers matter most for this use case.",

    "weaknesses_avoided": "Frame 3's 'serious' expression (0.65 intensity) would underperform for CTR because it lacks the pattern-interrupt quality needed in competitive feeds. Frame 1's slightly lower expression intensity means it would be less effective at stopping the scroll. By selecting Frame 2, we avoid the risk of a 'good but not great' thumbnail that gets overlooked."
  },

  "creator_message": "This frame will maximize your CTR because the genuine surprise expression creates an immediate curiosity gap that makes viewers want to know what caused your reaction. The bright, centered composition leaves perfect space for your title text at the top third. Consider placing your text overlay above the eyebrows to keep the full facial expression visible - the wide eyes are your strongest asset here.",

  "quantitative_scores": {
    "frame_number": 2,
    "timestamp": 12.8,
    "total_score": 0.804,
    "creator_alignment": 0.85,
    "aesthetic_score": 0.75,
    "psychology_score": 0.82,
    "face_quality": 0.91,
    "originality": 0.75,
    "composition": 0.70,
    "technical": 0.80,
    "detected_triggers": ["surprise", "curiosity_gap", "emotional_contagion"],
    "emotion": "surprise",
    "expression_intensity": 0.91
  },

  "gemini_model": "gemini-2.5-flash",
  "cost_usd": 0.0023
}
```

## Key Improvements in New Format

### Before (Old Format)
```json
{
  "reasoning": "This frame has a strong surprise expression that will create curiosity and the lighting is good for the tech niche."
}
```

### After (New Format)
```json
{
  "reasoning": {
    "summary": "Frame 2 is the clear winner because...",
    "visual_analysis": "Wide-eyed expression with raised eyebrows...",
    "score_alignment": "My visual assessment strongly aligns...",
    "niche_fit": "For tech reviews, this frame works perfectly...",
    "goal_optimization": "For maximizing CTR, this frame excels...",
    "psychology_triggers": "Activates curiosity gap, emotional contagion..."
  }
}
```

## How to Use the Reasoning in Your Application

### Display to Creators
```python
result = await selector.select_best_thumbnail(...)

# Show the creator a comprehensive breakdown:
print("🎯 Why This Frame Was Selected:")
print(f"\n{result['reasoning']['summary']}\n")

print("👁️ Visual Analysis:")
print(f"{result['reasoning']['visual_analysis']}\n")

print(f"✅ Why It Works for {profile['niche']}:")
print(f"{result['reasoning']['niche_fit']}\n")

print(f"📈 How It Achieves Your Goal ({brief['primary_goal']}):")
print(f"{result['reasoning']['goal_optimization']}\n")

print("🧠 Psychology Triggers:")
print(f"{result['reasoning']['psychology_triggers']}\n")

print("💡 Recommendation:")
print(f"{result['creator_message']}")
```

### Store in Database
```python
# Save structured reasoning for later analysis
thumbnail_result = {
    "project_id": project_id,
    "selected_frame_url": result["selected_frame_url"],
    "timestamp": result["selected_frame_timestamp"],
    "confidence": result["confidence"],

    # Store full reasoning breakdown
    "reasoning_summary": result["reasoning"]["summary"],
    "visual_analysis": result["reasoning"]["visual_analysis"],
    "niche_fit_explanation": result["reasoning"]["niche_fit"],
    "goal_optimization_explanation": result["reasoning"]["goal_optimization"],
    "psychology_triggers_explanation": result["reasoning"]["psychology_triggers"],

    # Store comparative analysis
    "runner_up_explanation": result["comparative_analysis"]["runner_up"],
    "weaknesses_avoided": result["comparative_analysis"]["weaknesses_avoided"],

    # Store quantitative scores
    "scores": result["quantitative_scores"],

    "created_at": datetime.utcnow()
}
```

### API Response
```python
# Return to frontend with structured reasoning
return {
    "selected_thumbnail": {
        "url": result["selected_frame_url"],
        "timestamp": result["selected_frame_timestamp"],
        "confidence": result["confidence"]
    },
    "explanation": {
        "summary": result["reasoning"]["summary"],
        "why_this_niche": result["reasoning"]["niche_fit"],
        "how_it_helps": result["reasoning"]["goal_optimization"],
        "psychology": result["reasoning"]["psychology_triggers"],
        "creator_tip": result["creator_message"]
    },
    "alternatives": {
        "runner_up": result["comparative_analysis"]["runner_up"],
        "why_not_others": result["comparative_analysis"]["weaknesses_avoided"]
    }
}
```

## Benefits of Structured Reasoning

1. **Transparency**: Creators understand exactly why each frame was chosen
2. **Education**: Helps creators learn what makes effective thumbnails for their niche
3. **Trust**: Detailed explanations build confidence in AI decisions
4. **Actionable**: Includes specific suggestions (e.g., text overlay placement)
5. **Debuggable**: Structured format makes it easy to identify if the agent is making good decisions
6. **Analytics**: Can analyze reasoning patterns across many selections to improve the system
