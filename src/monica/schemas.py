"""Pydantic request/response validation for POST /v1/systemone (Monica System One)."""

from __future__ import annotations

import os
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTIONS = int(os.environ.get("MONICA_MAX_QUESTIONS", "64"))
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 16


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1, max_length=2000)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: Dict[str, str] = Field(min_length=2)

    @field_validator("criteria")
    @classmethod
    def _keys_nonempty(cls, v):
        if any(not k.strip() or not s.strip() for k, s in v.items()):
            raise ValueError("choice criteria keys and descriptions must be non-empty")
        return v


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: List[str] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)

    @field_validator("criteria")
    @classmethod
    def _levels_ordered_unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("score level descriptions must be unique")
        if any(not s.strip() for s in v):
            raise ValueError("score level descriptions must be non-empty")
        return v


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class StateAudio(StrictModel):
    audio: str = ""


class JevRequest(StrictModel):
    model: str = ""
    state: StateAudio
    questions: Dict[str, Question] = Field(default_factory=dict, max_length=MAX_QUESTIONS)

    @field_validator("questions")
    @classmethod
    def _names(cls, v):
        if any(not k.strip() for k in v):
            raise ValueError("question names must be non-empty")
        return v

    def to_plain(self) -> dict:
        return {
            "model": self.model,
            "state": {"audio": self.state.audio},
            "questions": {k: q.model_dump() for k, q in self.questions.items()},
        }


class ErrorResponse(BaseModel):
    error: Dict[str, str]
