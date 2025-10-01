"""
This module defines the Pydantic data models for the application.

These models serve as the "single source of truth" for the output data structure.
They provide strict type hinting, validation, and serialization, ensuring that
the final JSON output is always consistent and conforms to the defined schema.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Commitment(BaseModel):
    """
    Represents the commitment made by an investor.
    """
    amount: Optional[Union[int, float, str]] = Field(None, description="The commitment amount. Can be an integer, float, or a string for ranges.")
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


class ExtractionResult(BaseModel):
    """
    The root model for the structured extraction output.
    """
    meta: Meta = Field(..., description="Metadata about the extraction.")
    investors: List[Investor] = Field(default_factory=list, description="A list of extracted investors.")
