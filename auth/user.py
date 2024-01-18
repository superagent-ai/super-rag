import logging
import jwt

from decouple import config
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from superagent.client import AsyncSuperagent

logger = logging.getLogger(__name__)
security = HTTPBearer()


def generate_jwt(data: dict):
    token = jwt.encode({**data}, config("JWT_SECRET"), algorithm="HS256")
    return token


def decode_jwt(token: str):
    return jwt.decode(token, config("JWT_SECRET"), algorithms=["HS256"])


async def get_current_api_user(
    authorization: HTTPAuthorizationCredentials = Security(security),
):
    token = authorization.credentials
    print("TOKEN", token)
    decoded_token = decode_jwt(token)
    superagent = AsyncSuperagent(
        base_url="https://api.beta.superagent.sh", token=decoded_token
    )
    api_user = superagent.api_user.get()
    if not api_user:
        raise HTTPException(status_code=401, detail="Invalid token or expired token")
    return api_user
