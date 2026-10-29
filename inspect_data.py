import json
from pathlib import Path

for path in (Path("data/structured_records.json"), Path("data/chunks.json")):
    print("\n" + "=" * 70 + f"\n{path}\n" + "=" * 70)
    if not path.exists():
        print("Not found; run uv run python ingest.py")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    print("Count:", len(data))
    for item in data[:10]:
        md = item.get("metadata", item)
        text = item.get("page_content", item.get("content", ""))
        print(f"\nPages {md.get('page_start')}-{md.get('page_end')} | {md.get('section_path') or md.get('section_title')}")
        print(text[:300])
