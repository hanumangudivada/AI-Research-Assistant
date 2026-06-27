from pydantic import BaseModel, Field
from typing import Dict,List

class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""
    answer :str=Field(description="The answer to the question")
    confidence:str=Field(description="high, medium, or low based on source quality")
    sources:List[str]=Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")
    

                         