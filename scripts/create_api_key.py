#!/usr/bin/env python3
"""Create a user (if needed) and an API key. The plaintext key is printed ONCE.

Usage:
    python scripts/create_api_key.py --email you@example.com --name "local dev"
"""

from __future__ import annotations

import argparse
import asyncio
import secrets

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models import Base
from app.repositories.database import Database
from app.repositories.user import ApiKeyRepository, UserRepository
from app.utils.security import generate_api_key, hash_api_key


async def main(email: str, name: str) -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    database = Database(settings.database)
    await database.initialize()
    # Dev convenience: ensure schema exists (migrations are the real path).
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with database.session() as session:
        users = UserRepository(session)
        user = await users.get_by_email(email)
        if user is None:
            user = await users.create(
                email=email, hashed_password=secrets.token_urlsafe(32), full_name=name
            )
        raw_key = generate_api_key()
        await ApiKeyRepository(session).create(
            user_id=user.id, key_hash=hash_api_key(raw_key), name=name
        )
        await session.commit()

    await database.dispose()
    print("API key created. Store it securely — it is shown only once:")
    print(raw_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="api key")
    args = parser.parse_args()
    asyncio.run(main(args.email, args.name))