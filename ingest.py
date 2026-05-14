import json
import re
import base64
from io import BytesIO
from pathlib import Path

from config import config

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

from pdf2image import convert_from_path


# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = config["pdf_path"]
OLLAMA_URL = config["ollama_url"]

JSON_PATH = "./processed_docs.json"
OCR_MODEL = "qwen3.5:35b-a3b-q8_0"


# -----------------------------
# OCR MODEL
# -----------------------------
ocr_llm = ChatOllama(
    model=OCR_MODEL,
    base_url=OLLAMA_URL,
    temperature=0
)


# -----------------------------
# PDF → IMAGES
# -----------------------------
print("Converting PDF to images...")

pages = convert_from_path(PDF_PATH, dpi=150)

print(f"Loaded {len(pages)} pages")


# -----------------------------
# CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if re.match(r"^\d+$", line):
            continue

        if "GE Appliances" in line:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def extract_section(text: str) -> str:

    lines = text.splitlines()

    for line in lines:

        stripped = line.strip()

        if (
            4 < len(stripped) < 80
                and (
                stripped.isupper()
                or re.match(r"^\d+\.", stripped)
            )
        ):
            return stripped

    return "Unknown"


# -----------------------------
# SPLITTER
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=[
        "\n# ",
        "\n## ",
        "\n\n",
        "\n",
        ". ",
        "•",
        " "
    ]
)


# -----------------------------
# IMAGE ENCODING
# -----------------------------
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------
# PROCESSING LOOP
# -----------------------------
json_docs = []
chunk_counter = 0

for page_num, image in enumerate(pages):

    print(f"OCR page {page_num + 1}/{len(pages)}...")

    image_b64 = pil_to_base64(image)

    prompt = """
Extract all readable text from this image.

Rules:
- Return exact text only
- Preserve headings and bullet points
- No summaries
- No descriptions
- No interpretation
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{image_b64}"
            }
        ]
    )

    response = ocr_llm.invoke([message])

    raw_text = response.content if hasattr(response, "content") else str(response)

    cleaned = clean_text(raw_text)

    cleaned = remove_boilerplate(cleaned)

    if not cleaned:
        continue

    section = extract_section(cleaned)

    chunks = splitter.split_text(cleaned)

    for chunk in chunks:

        json_docs.append({
            "chunk_id": chunk_counter,
            "page": page_num + 1,
            "section": section,
            "content": chunk,
            "source": Path(PDF_PATH).name
        })

        chunk_counter += 1


# -----------------------------
# SAVE JSON
# -----------------------------
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_docs, f, indent=2, ensure_ascii=False)

print(f"Saved JSON to {JSON_PATH}")
print(f"Created {len(json_docs)} chunks")
