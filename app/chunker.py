import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page: int
    section: str
    kind: str  # "text" или "table"


def clean_markdown(text: str) -> str:
    """Убираем переносы строк HTML и комментарии картинок от pymupdf."""
    text = text.replace("<br>", " ").replace("<br/>", " ")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()


def chunk_page(
    md: str,
    page: int,
    section: str = "",
    max_size: int = 1000,
    size: int | None = None,
    overlap: int | None = None,
) -> tuple[list[Chunk], str]:
    if size is not None:
        max_size = size
    md = clean_markdown(md)
    chunks: list[Chunk] = []
    current_text = ""

    for block in md.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            section = block.lstrip("# ").strip()
            continue

        if block.startswith("|"):
            if current_text:
                chunks.append(
                    Chunk(current_text.strip(), page, section, "text")
                )
                current_text = ""
            chunks.append(Chunk(block, page, section, "table"))
            continue

        clean_block = " ".join(block.splitlines())

        if current_text and (len(current_text) + len(clean_block) > max_size):
            chunks.append(Chunk(current_text.strip(), page, section, "text"))
            current_text = clean_block
        else:
            current_text += (" " if current_text else "") + clean_block

    if current_text.strip():
        chunks.append(Chunk(current_text.strip(), page, section, "text"))

    return chunks, section