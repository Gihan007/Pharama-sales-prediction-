from copy import deepcopy
from pathlib import Path
from typing import Optional

from docx import Document
from docx.oxml import OxmlElement


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CSCI 43018- final structured thesis CS_2020_015.docx"
OUTPUT = ROOT / "CSCI 43018- final structured thesis CS_2020_015 - with reference 7.docx"

CITATION_OLD = (
    "LightGBM is another gradient boosting framework that is suitable for efficient "
    "model training and structured forecasting tasks."
)
CITATION_NEW = (
    "LightGBM is another gradient boosting framework that is suitable for efficient "
    "model training and structured forecasting tasks [7]."
)
REFERENCE = (
    "[7]  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "
    "“LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” in Advances in "
    "Neural Information Processing Systems 30 (NIPS 2017), Long Beach, CA, USA, "
    "pp. 3146–3154, 2017."
)


def replace_text_preserving_runs(paragraph, old: str, new: str) -> None:
    full_text = paragraph.text
    if old not in full_text:
        raise RuntimeError("The expected LightGBM sentence was not found.")

    replaced = full_text.replace(old, new, 1)
    first_run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    formatting = deepcopy(first_run._r.rPr) if first_run._r.rPr is not None else None

    for run in paragraph.runs:
        run._element.getparent().remove(run._element)
    run = paragraph.add_run(replaced)
    if formatting is not None:
        run._r.insert(0, formatting)


def insert_paragraph_before(paragraph, text: str = "", style: Optional[str] = None):
    element = OxmlElement("w:p")
    paragraph._p.addprevious(element)
    inserted = paragraph._parent.add_paragraph()
    inserted._p.getparent().remove(inserted._p)
    element.addnext(inserted._p)
    if style:
        inserted.style = style
    if text:
        inserted.add_run(text)
    return inserted


def main() -> None:
    document = Document(SOURCE)

    citation_matches = [p for p in document.paragraphs if CITATION_OLD in p.text]
    if len(citation_matches) != 1:
        raise RuntimeError(f"Expected one citation location; found {len(citation_matches)}.")
    replace_text_preserving_runs(citation_matches[0], CITATION_OLD, CITATION_NEW)

    reference_8 = next((p for p in document.paragraphs if p.text.strip().startswith("[8]")), None)
    if reference_8 is None:
        raise RuntimeError("Reference [8] was not found.")
    insert_paragraph_before(reference_8, REFERENCE, "Bibliography")
    insert_paragraph_before(reference_8, "", "Normal")

    document.save(OUTPUT)

    check = Document(OUTPUT)
    if sum("tasks [7]." in p.text for p in check.paragraphs) != 1:
        raise RuntimeError("Citation verification failed.")
    if sum(p.text.strip().startswith("[7]") for p in check.paragraphs) != 1:
        raise RuntimeError("Reference verification failed.")
    print(OUTPUT)


if __name__ == "__main__":
    main()
