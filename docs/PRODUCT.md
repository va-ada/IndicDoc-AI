# IndicDoc Verify: pre-print checking of multilingual exam papers

> Catch the errors in a question paper and in its translations **before** the paper is printed: offline,
> in 12 Indian languages, pointing to the exact question and page.

## 1. The problem nobody automates

Indian exams are set in English (or the state language) and translated into Hindi and regional languages. NEET-UG is
offered in 13 languages; UPSC, the state Public Service Commissions, school boards and universities print bilingual
papers. Every version is checked by hand. When that check fails, the cost is a re-exam, grace marks for lakhs of
candidates, or years in court.

| Exam | What went wrong | Consequence | Check in IndicDoc Verify |
|---|---|---|---|
| NEET-UG 2018 (CBSE) | Errors in **49 questions** of the Tamil version | Madras HC ordered 196 grace marks (July 2018); Supreme Court set it aside and asked for bilingual subject experts (Nov 2018) | Parity: numbers, options, marks between versions |
| NEET-UG 2021 (NTA) | A term ("amplitude of current") dropped in the Hindi version of a physics question | NTA told the Supreme Court it would examine it | Meaning check (optional cross-lingual model); flagged for the vetter |
| KPSC KAS prelims, Aug 2024 | **59 errors** in the Kannada paper; machine translation suspected | Exam re-held in Dec 2024, errors found again; ~2 lakh candidates | Parity + structure |
| APPSC Group-I prelims | **51 of 120** questions alleged mistranslated into Telugu | Supreme Court notice to the Commission | Parity |
| UGC-NET June 2026 (NTA) | Factual, typographical and translation errors; **questions repeated from earlier papers** | Re-test of English, Commerce and Sociology on 9-10 Sep 2026 | Repeat check against a question bank + parity |
| MBBS (Dr NTR UHS), 2026 | **The same MCQ printed twice** in one paper | AP High Court ordered a compensatory mark (July 2026) | Duplicate-question check |

Sources: The News Minute (NEET 2018, Madras HC and Supreme Court), Careers360 (NEET 2021), The South First /
Deccan Herald / ETV Bharat (KPSC 2024-25), India Legal (APPSC), The Week / SCC Online / ETV Bharat (UGC-NET 2026),
Medical Dialogues (MBBS 2026). Links in the lab report references.

## 2. What exists today, and the gap

| Category | Examples | Why it does not solve this |
|---|---|---|
| Manual vetting | Paper-setter, moderator, translator, vetter with a checklist (e.g. Cambridge's vetter role) | Slow, tiring, and it is exactly the step that failed in every case above |
| Answer-sheet evaluation | Eklavvya, GradeLab, E-Valuate | Grade the candidates' answers **after** the exam; never look at the paper |
| Translation quality tools | LanguageCheck.ai; MT quality estimation such as CometKiwi | Work on plain sentences: no idea of question numbers, options, marks or figures; cloud services; not built for typeset or scanned Indic PDFs |
| Question-bank / paper generators | various ed-tech tools | Create papers; do not compare the language versions that come back from translators |
| Court e-filing defect detection | Supreme Court + IIT Madras tool; Nyaay.AI in 16 High Courts | Already deployed. We looked at it and **chose not to compete there** |

We found **no product** that reads a typeset or scanned paper in Indian languages, understands its structure
(questions, sub-questions, options, marks, figures) and compares its language versions, offline.

**Why offline is not optional.** Before the exam, a question paper is the most confidential document an exam body
has; it cannot be uploaded to a cloud service. IndicDoc Verify runs in one Docker image with no internet access.

## 3. The product

Input: the paper as it will be printed (PDF or scans), optionally its source-language version, a bank of earlier
papers and a format profile. Output: a verdict ("ready for printing" / "check before printing" / "do not print"),
every finding with its question and page, the pages with each problem outlined, as HTML and JSON.

| Check | Catches |
|---|---|
| Numbering | gaps, repeats, out-of-order question numbers |
| Duplicates | the same question printed twice |
| Options | a question with fewer options than the rest, an option printed twice, an empty option |
| Marks | per-question marks that do not add up to the printed maximum, missing marks |
| **Translation parity** | a question missing or added in a version; different option counts, marks or sub-questions; **any number that differs** in a stem or an option (an answer key that no longer holds); missing figures, tables or formulas |
| Repeats | questions that repeat an earlier paper (question bank) |
| Format profiles | any document type described as a JSON rule file: required elements, their order, page numbers, bilingual text (e.g. an Office Memorandum per CSMOP, Official Languages Act s.3(3)) |

**How it works.** DocLayout-YOLO, fine-tuned on IndicDLP by its authors, finds the questions, sub-questions, options
and answers on each page (0.81-0.90 AP@50 on these labels, measured on 682 IndicDLP test pages). Tesseract reads the
text; a second **number pass** with the English model fixes the Latin digits that the Indic models misread (Hindi
reads "1947" as "947"). The parser builds the questions; the checks compare them. Translation parity needs **no
translation**: numbers, option counts, marks and structure are the same in every language.

**USP in one line:** the only offline tool that checks an Indian question paper and its translations question by
question, in 12 languages, before it is printed.

## 4. Who pays

| Segment | Examples | Need |
|---|---|---|
| National exam agencies | NTA (NEET-UG in 13 languages, JEE Main, CUET, UGC-NET) | Parity across many languages at scale |
| Recruitment commissions | UPSC, state PSCs, SSC, recruitment boards | Bilingual papers, high litigation risk |
| School boards | CBSE, state boards | Many papers each season, regional languages |
| Universities | Controllers of examinations | Many papers, few vetters; repeat check |
| Test-series publishers | Coaching institutes printing bilingual mock tests | Quality at volume (no confidentiality limit, so a hosted plan works) |
| Government departments | Official-language cells | Format and bilingual-compliance profiles for circulars and orders |

**Pricing (proposal, to be validated with pilots).** On-premise annual licence per exam body, unlimited papers;
per-paper pricing for universities and publishers; a hosted plan only for non-confidential material (mock tests).
The comparison that matters to a buyer is not another tool but the cost of one re-conducted exam.

## 5. Honest limits

- It checks structure, numbers, marks and repeats. It does **not** decide whether a question is correct, and it
  catches a change of meaning only through the optional cross-lingual model; every finding goes to a human vetter.
- Number words ("two" vs "three") and mathematics inside formulas are not compared yet.
- OCR on badly scanned pages adds false alarms; born-digital PDFs (the normal pre-print case) work best.
- Measured so far on a controlled benchmark (rendered papers with injected errors, real OCR); a pilot with real
  pre-print papers of an exam body is the next step.
