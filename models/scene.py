from pydantic import BaseModel, Field


class Scene(BaseModel):
    index: int
    start_time: float               # seconds in source video
    end_time: float                 # seconds in source video
    frames: list[str] = Field(default_factory=list)  # paths to sampled frame images
    description: str = ""           # filled by scene_analyzer
    is_film_related: bool = True    # filled by scene_analyzer
    confidence: float = 1.0        # filled by scene_analyzer (0.0–1.0)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        status = "✓" if self.is_film_related else "✗"
        return (
            f"Scene({self.index} [{self.start_time:.1f}s-{self.end_time:.1f}s] "
            f"{status} conf={self.confidence:.2f}: {self.description[:60]})"
        )
