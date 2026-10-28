"""schemas.py - Pydantic models for structured LLM output (chapter 21)."""
from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class Priority(str, Enum):
    lead_time = "lead_time"
    price = "price"
    unknown = "unknown"

class RfqLine(BaseModel):
    mpn: str = Field(min_length=3, max_length=40,
                     description="Manufacturer part number, copied exactly")
    quantity: int = Field(gt=0, le=10_000_000)
    needed_by: Optional[date] = Field(
        default=None, description="ISO date, null if not stated")

    @field_validator("mpn")
    @classmethod
    def no_spaces(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("mpn must not contain spaces")
        return v.strip()

class RfqExtraction(BaseModel):
    customer_company: Optional[str] = None
    lines: list[RfqLine] = Field(min_length=1)
    priority: Priority
    language: str = Field(pattern=r"^(de|en|fa|mixed)$")

class BomLine(BaseModel):
    designators: list[str] = Field(default_factory=list)
    mpn: Optional[str] = None
    manufacturer: Optional[str] = None
    quantity: int = Field(gt=0)
    description: Optional[str] = None
    dnp: bool = False          # "do not populate"

if __name__ == "__main__":
    import json
    schema = RfqExtraction.model_json_schema()
    print(json.dumps(schema, indent=1)[:1200])
