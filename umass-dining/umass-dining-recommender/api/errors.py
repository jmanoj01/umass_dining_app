from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional

class AppError(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
                "details": exc.details,
            }
        },
    )

# Common error classes
class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            status_code=404,
            message=f"{resource} with id {resource_id} not found",
            error_code="NOT_FOUND",
            details={"resource": resource, "id": resource_id}
        )

class ValidationError(AppError):
    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(
            status_code=422,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class DatabaseError(AppError):
    def __init__(self, operation: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            message=f"Database error during {operation}",
            error_code="DATABASE_ERROR",
            details=details
        )