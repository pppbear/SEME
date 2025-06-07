# auth_service/app/schemas/response.py
from typing import Any, Optional
from pydantic import BaseModel
from fastapi import status

class AuthResponse(BaseModel):
    code: int = status.HTTP_200_OK
    message: str = "success"
    data: Optional[Any] = None