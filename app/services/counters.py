from pymongo import ReturnDocument

from app.database import get_database


async def next_sequence(name: str, start: int = 1) -> int:
    database = get_database()
    doc = await database.counters.find_one_and_update(
        {"_id": name},
        [{"$set": {"value": {"$add": [{"$ifNull": ["$value", start - 1]}, 1]}}}],
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["value"])
