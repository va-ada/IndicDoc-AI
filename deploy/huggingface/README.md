---
title: IndicDoc Verify
emoji: 📄
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Pre-print checking of Indian question papers and translations
---

# IndicDoc Verify

Check an Indian question paper, and its translations, **before it is printed**: numbering, duplicate questions,
options, marks, and every number, option and mark that differs between the language versions. Bilingual papers
printed page by page (CBSE: a Hindi page, then the same questions in English) are recognised by themselves.

Every paper is read twice, independently: by a deep-learning layout model (DocLayout-YOLO, fine-tuned on IndicDLP)
with OCR, and without the layout model from the PDF's own text (or a second OCR pass). A finding is confirmed only
when both readings support it; where they disagree the question is read a third time.

Open the **Verify Paper** tab and pick an example paper to see it work, or upload your own PDF.

Code: github.com/va-ada/IndicDoc-AI
