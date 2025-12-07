# preprocess_and_label.py
import re
from db import get_conn, init_db

KEYWORDS = {
    "calm": ["quiet", "peaceful", "calm", "serene", "less crowded"],
    "adventurous": ["trek", "hike", "trail", "adventure", "cliff", "rafting"],
    "photogenic": ["viewpoint", "sunrise", "sunset", "scenic", "photogenic", "instagram"],
    "family_friendly": ["family", "kids", "children", "park", "zoo"],
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def has_any(text: str, words) -> int:
    return int(any(w in text for w in words))

def run():
    init_db()
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("SELECT id, description FROM places")
        rows = cur.fetchall()

        for pid, desc in rows:
            desc_clean = clean_text(desc or "")
            calm = has_any(desc_clean, KEYWORDS["calm"])
            adv = has_any(desc_clean, KEYWORDS["adventurous"])
            photo = has_any(desc_clean, KEYWORDS["photogenic"])
            fam = has_any(desc_clean, KEYWORDS["family_friendly"])

            cur.execute("""
            UPDATE places
            SET description_clean = ?, calm = ?, adventurous = ?,
                photogenic = ?, family_friendly = ?
            WHERE id = ?
            """, (desc_clean, calm, adv, photo, fam, pid))
        con.commit()
    print(f"Preprocessed and labeled {len(rows)} places")

if __name__ == "__main__":
    run()
