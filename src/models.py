"""
This module defines the data structures for the application using Pydantic.

These models ensure that the extracted data conforms to a specific, validated
schema before being output as the final JSON. They provide type hinting,
validation, and serialization for the investor data.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Commitment(BaseModel):
    """
    Represents the commitment made by an investor.
    """
    amount: Optional[int] = Field(None, description="The commitment amount in the relevant currency.")
    percent: Optional[float] = Field(None, description="The commitment as a percentage of the total issue.")


class Investor(BaseModel):
    """
    Represents an underwriter or guarantor.
    """
    name: str = Field(..., description="The name of the investor.")
    commitment: Commitment = Field(..., description="The commitment details.")
    investor_level: int = Field(
        ...,
        description="0 for underwriter, 1 for the lowest guarantor level, 2 for the next, and so on."
    )


class Meta(BaseModel):
    """
    Contains metadata about the extraction process.
    """
    source: str = Field(..., description="The name of the source PDF file.")

    extracted_at: datetime = Field(default_factory=datetime.now, description="The timestamp of the extraction.")
    source_page: Optional[Union[int, str]] = Field(None, description="The page number(s) where the information was found.")
    confidence: Optional[str] = Field(None, description="The confidence level of the extraction (e.g., 'high', 'medium', 'low').")


class ExtractionResult(BaseModel):
    """
    The root model for the structured extraction output.
    """
    meta: Meta = Field(..., description="Metadata about the extraction.")
    investors: List[Investor] = Field(default_factory=list, description="A list of extracted investors.")
