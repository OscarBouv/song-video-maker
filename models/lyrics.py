from dataclasses import dataclass, field


@dataclass
class LyricLine:
    text: str
    start_time: float   # seconds from song start (after optional trim)
    end_time: float     # seconds from song start (after optional trim)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"[{self.start_time:.2f}s-{self.end_time:.2f}s] {self.text}"


@dataclass
class MatchedSegment:
    scene_index: int
    lyric_lines: list[LyricLine] = field(default_factory=list)
    scene_trim_start: float = 0.0   # offset within scene to start from
    scene_trim_end: float = -1.0    # -1 means use full scene end

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "scene_trim_start": self.scene_trim_start,
            "scene_trim_end": self.scene_trim_end,
            "lyric_lines": [
                {
                    "text": l.text,
                    "start_time": l.start_time,
                    "end_time": l.end_time,
                }
                for l in self.lyric_lines
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatchedSegment":
        return cls(
            scene_index=data["scene_index"],
            scene_trim_start=data.get("scene_trim_start", 0.0),
            scene_trim_end=data.get("scene_trim_end", -1.0),
            lyric_lines=[
                LyricLine(
                    text=l["text"],
                    start_time=l["start_time"],
                    end_time=l["end_time"],
                )
                for l in data.get("lyric_lines", [])
            ],
        )
