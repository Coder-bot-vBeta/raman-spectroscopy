import sys; sys.path.insert(0, '.')
from database.reference_store import get_database, invalidate_cache
invalidate_cache()
matrix, names, metadata = get_database()
print(f"Total classes in DB: {len(names)}")
print(f"Matrix shape: {matrix.shape}")
sources = {}
for v in metadata.values():
    s = v["source"]; sources[s] = sources.get(s, 0) + 1
print(f"Sources: {sources}")
