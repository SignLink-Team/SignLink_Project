from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from bson import ObjectId

from app.auth.jwt import decode_access_token
from app.database import get_database
from app.services.counters import next_sequence

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    medical_id = payload.get("medical_id")
    if not medical_id:
        database = get_database()
        user = await database.user.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        medical_id = user.get("medical_id")
        if not medical_id:
            medical_id = await next_sequence("medical_id", 1000)
            await database.user.update_one({"_id": user["_id"]}, {"$set": {"medical_id": medical_id}})

    return {
        "id": user_id,
        "medical_id": int(medical_id),
        "email": payload.get("email"),
        "role": payload.get("role"),
    }


def require_role(*roles: Literal["doctor", "patient"]) -> Callable[..., dict]:
    async def role_checker(
        current_user: Annotated[dict, Depends(get_current_user)],
    ) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_checker
