from pydantic import BaseModel
from typing import List

class VariableImportance(BaseModel):
    variable: str
    importance: float

class AnalyzeResponse(BaseModel):
    importances: List[VariableImportance] 