from pydantic import BaseModel, Field


class LyricLine(BaseModel):
    text: str
    start_time: float   # seconds from song start (after optional trim)
    end_time: float     # seconds from song start (after optional trim)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"[{self.start_time:.2f}s-{self.end_time:.2f}s] {self.text}"


class MatchedSegment(BaseModel):
    scene_index: int
    lyric_lines: list[LyricLine] = Field(default_factory=list)
    scene_trim_start: float = 0.0   # offset within scene to start from (seconds)
    scene_trim_end: float = -1.0    # offset within scene to end at (-1 = use scene end)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (used when saving plan.json)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "MatchedSegment":
        """Deserialize from a plain dict (used when loading plan.json)."""
        return cls.model_validate(data)
