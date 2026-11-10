from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Game:
    game_id: str
    title: str
    genres: tuple[str, ...]
    tags: tuple[str, ...]
    description: str
    positive_ratings: int
    negative_ratings: int
    release_year: int

    @property
    def rating_count(self) -> int:
        return self.positive_ratings + self.negative_ratings

    @property
    def positive_ratio(self) -> float:
        if self.rating_count == 0:
            return 0.0
        return self.positive_ratings / self.rating_count


@dataclass(frozen=True)
class Interaction:
    user_id: str
    game_id: str
    signal: float
    timestamp: str


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split("|") if item.strip())


def load_games(path: str | Path) -> list[Game]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [
            Game(
                game_id=row["game_id"],
                title=row["title"],
                genres=_split_values(row["genres"]),
                tags=_split_values(row["tags"]),
                description=row["description"],
                positive_ratings=int(row["positive_ratings"]),
                negative_ratings=int(row["negative_ratings"]),
                release_year=int(row["release_year"]),
            )
            for row in csv.DictReader(file)
        ]


def load_interactions(path: str | Path) -> list[Interaction]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [
            Interaction(
                user_id=row["user_id"],
                game_id=row["game_id"],
                signal=float(row["signal"]),
                timestamp=row["timestamp"],
            )
            for row in csv.DictReader(file)
        ]
