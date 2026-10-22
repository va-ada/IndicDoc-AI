# Next plan: IndicDoc AI → **Sulabh Path** (accessible Indian-language study material)

> Status: plan agreed 2026-10-22. Track progress in the checklist at the bottom.

## 1. Who it is for (the USP)

**Blind and low-vision students who study from state-board textbooks and exam papers in Indian languages,
and the teachers / NGOs who convert that material for them.**

> *Turn any scanned state-board page into screen-reader-, Braille- and audio-ready text in its own language,
> in the right reading order, fully offline.*

### Why this need is real (research, Oct 2026)
- ~2.86 lakh low-vision and ~49 thousand blind children are enrolled in schools (UDISE+ 2024,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12199054/).
- 64% of sampled NCERT and 95% of sampled Tamil Nadu / Telangana SCERT textbooks are inaccessible; 68% of DIKSHA
  links fail with screen readers (Vidhi, https://vidhilegalpolicy.in/blog/inaccessibility-plagues-digital-education/).
- RPwD Act 2016 requires study material and question papers in accessible formats; Delhi HC issued notice on a
  plea for time-bound accessible books (Oct 2026, https://theprint.in/india/sc-delhi-hc-notices-spotlight-persistent-accessibility-gaps-in-india-decade-after-disabilities-law/3066524/).
- Kerala: no Braille textbooks after Class 10 (Onmanorama, Aug 2025); Assam blind students flag shortages (Assam
  Tribune, Aug 2026). Conversion today is mostly manual NGO work (DAISY Forum of India, Sugamya Pustakalaya).

### Why existing tools do not solve it
| Tool | Gap for this user |
|---|---|
| Bhashini / CVIT OCR | text only: no headings, tables, reading order |
| Sarvam Vision 2.1 | cloud only, paid, 10 pages per job; exam papers are confidential before the exam |
| Bodhan AI + AI4Bharat IndicOCR (Sep 2026) | general parser; no accessible output (DAISY/EPUB/Braille), weak on dense reading order (self-reported) |
| Google / Azure / AWS | no Odia/Assamese (Google), Punjabi only in Arabic script (Azure), no Indic at all (AWS) |
| Vision-language models | rewrite / hallucinate text; a blind reader cannot check it |

**The manual step we automate.** Xavier's Resource Centre (Mumbai) production standards: ABBYY supports no
Indian language, so regional books are OCR'd page by page in Google Drive, pasted into Word, and staff then
rebuild headings, tables, page numbers, contents and image labels by hand
(https://www.xrcvc.org/docs/XRCVC's%20Accessible%20Book%20Production%20Standards%201.3.pdf). Typing a Telugu page
costs ~₹30 (CIS). Closest prior work, IIT Delhi's RAVI, only handles born-digital PDFs, not scans. We found no tool
that turns a *scanned* Indic page into tagged EPUB3/DAISY offline. Blind users navigate by headings first (71.6%,
WebAIM survey #10). Copyright Act s.52(1)(zb) lets non-profits convert any book without permission.

**Be honest about:** maths and diagrams (we only add placeholders) and Indic TTS quality; present output as
"structured text for EPUB/DAISY", not a finished audiobook.

**Our edge:** layout-aware (headings, lists, tables, question numbers, captions, reading order) + offline +
12 languages + faithful (every word traceable to a box on the page) + accessible output formats.

## 2. What we build (in order)

| # | Feature | Why it matters for the user |
|---|---|---|
| 1 | **Accessible HTML + EPUB3 export** (semantic headings, lists, tables, `lang` attribute, figure placeholders, page furniture separated) | Screen readers (NVDA, TalkBack) navigate by heading and read the right language |
| 2 | **Multi-page books** (PDF → one EPUB with page markers and a table of contents) | Students need whole chapters, not single pages |
| 3 | **Exam-paper mode**: question numbers, options and marks kept as structured items; one question per heading | Blind students can jump question-to-question without a scribe |
| 4 | **Better OCR engine** behind the same interface: IndicOCR (Bodhan/AI4Bharat) or CVIT, Tesseract as fallback | Tesseract is the weakest stage (mis-orders vowel signs in several scripts) |
| 5 | **Braille** (.brf via liblouis Indic tables) and **audio** (offline TTS, e.g. AI4Bharat Indic-TTS) | The two formats blind students actually use |
| 6 | **Review screen**: page image + extracted text side by side, edit before export | A teacher fixes OCR errors once; every student benefits |
| 7 | **Accessibility evaluation**: reading-order accuracy and "screen-reader success" on 20 real pages (Balbharati Marathi, TN SCERT Tamil, Odia, Punjabi) | Turns the USP into a measured claim for the paper |

**Demo to aim for:** 5-10 scanned pages each from a TN SCERT (Tamil) and a Balbharati (Marathi) textbook plus one
two-column question paper → EPUB3 → compare with the Google Drive OCR workflow on reading-order errors, heading
recall, table accuracy and minutes of manual cleanup per page; record an NVDA/TalkBack session navigating by
heading (ideally reviewed by XRCVC or DAISY Forum of India).

## 3. Research angle (paper v2)
"Can layout-aware OCR make Indian-language textbooks accessible?" — measure: reading-order accuracy, heading
recall, table preservation, and time saved vs. manual conversion, on public state-board PDFs. Keep the earlier
study (consistency, robustness) as background.

## 4. Checklist
- [x] 1. Accessible HTML + EPUB3 export (`ml/accessible.py`, tests, app download buttons) — API endpoint still to add
- [x] 2. Multi-page PDF → one EPUB (`scripts/pdf_to_epub.py`, CLI; app button still to add)
- [ ] 3. Exam-paper mode
- [ ] 4. Pluggable OCR engine (IndicOCR / CVIT)
- [ ] 5. Braille + audio
- [ ] 6. Review / edit screen
- [ ] 7. Accessibility evaluation on 20 state-board pages
- [ ] Unseen-script experiment (`scripts/study_script_transfer.py`, ~2 h GPU, needs charger)
