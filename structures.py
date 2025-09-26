from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import List, Optional

class CandidateSummary(BaseModel):
    core_skills: List[str] = Field(default_factory=list, description="Key technical skills")
    seniority: Optional[str] = Field(default=None, description="e.g., Junior/Mid/Senior/Lead")
    domains: List[str] = Field(default_factory=list, description="Business/tech domains")
    notable_projects: List[str] = Field(default_factory=list)
    leadership: Optional[str] = Field(default=None, description="Leadership/mentoring highlights")
    tools: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)