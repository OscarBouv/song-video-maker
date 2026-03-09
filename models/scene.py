from pydantic import BaseModel, Field


class Scene(BaseModel):
    index: int
    start_time: float               # seconds in source video
    end_time: float                 # seconds in source video
    frames: list[str] = Field(default_factory=list)  # paths to sampled frame images

    # ── Set by scene_analyzer ─────────────────────────────────────────────────
    description: str = ""           # narrative summary: who + what + dramatic feel
    characters_present: list[str] = Field(default_factory=list)  # named characters visible
    emotion: str = ""               # dominant emotional register (e.g. "longing, quiet joy")
    shot_type: str = ""             # framing: "close-up", "medium", "wide", "over-the-shoulder", etc.
    lighting: str = ""              # lighting quality: "warm golden", "dim/moody", "harsh daylight", etc.
    visual_power: int = 3           # 1–5 iconic/cinematic strength (5 = music-video gold, 1 = weak)
    is_film_related: bool = True    # False if not from the target film
    is_aesthetic: bool = True       # False if overlaid text, watermarks, explicit content, heavy blur, etc.
    confidence: float = 1.0        # confidence in is_film_related (0.0–1.0)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_usable(self) -> bool:
        """Scene passes both film-relation and aesthetic checks."""
        return self.is_film_related and self.is_aesthetic

    def __repr__(self) -> str:
        status = "✓" if self.is_usable else "✗"
        return (
            f"Scene({self.index} [{self.start_time:.1f}s-{self.end_time:.1f}s] "
            f"{status} conf={self.confidence:.2f}: {self.description[:60]})"
        )
