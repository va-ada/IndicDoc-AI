# Dual-reader consensus

IndicDoc Verify reads every paper twice, independently, and reports only what the two readings agree on. This page
describes how; the code is in `ml/verify/reader.py`, `ml/verify/consensus.py` and `ml/verify/pipeline.py`.

## Why

A checker of question papers is only as good as its reading of the paper. OCR of Indian-language pages misreads
digits (`120` read as `।20`, `16.` as `6.`), option markers (`(A)` read as `(8)`), Latin units and formulas, and a
layout detector sometimes misses or merges a question. A checker that trusts one reading turns every such slip into a
"finding": a real CBSE bilingual paper run through one reader gave 39 errors and 72 warnings, and none was a real
error in the paper. A vetter who sees that stops trusting the tool.

Two human vetters who read a paper independently and compare notes catch each other's slips. The consensus does
the same with two machine readers whose mistakes are of different kinds.

## The two readers

| | Reader A | Reader B |
|---|---|---|
| Structure | DocLayout-YOLO layout regions (questions, sub-questions, options, answers) | the typography of exam papers: a number at a column's left edge starts a question; cells that open with a marker are options; '(a)', '(i)' lines and lines ending in '[1]' are parts; 'SECTION - A' / 'खण्ड - क' headings; instruction blocks; 'OR' / 'अथवा'; running headers and footers (lines near an edge that repeat on other pages); two columns (a gutter no line crosses) |
| Text | Tesseract on the page (psm 3, rendered size) with the English number pass, margin numbers read separately | the PDF's own text layer (born-digital file, when it holds the page's script); otherwise Tesseract with another configuration (psm 4, page enlarged 1.5 times), English number pass |
| Fails by | missed / merged / mislabelled regions; OCR slips | garbled line starts; OCR slips on scans (none on a text layer) |

Both produce pages in the same form, so the same parser and checks run on each (`parse_paper`, `check_paper`,
`compare_versions`). Bilingual files are split into their language versions first (`ml/verify/versions.py`), for
each reader alike.

## Pairing and fusion

The two readings' questions are paired by a global alignment (Needleman-Wunsch): printed number (+2 when equal, -1
when different), text similarity (character 3-grams, x2), the numbers in the question (Jaccard), and the page
(+0.5 same, -1 different); a gap costs 0.7. Each pair is compared on four fields: **number**, **option count**,
**the numbers in the stem and options**, and **marks**.

- Fields both read alike are confirmed.
- When reader B read a trustworthy text layer, it decides every text field (it recognised nothing, so it cannot have
  misread); figures and tables come from reader A, which sees them.
- Otherwise, a disagreement (or confidences more than 0.25 apart) triggers a **third reading**: only that question's
  region (the union of both readers' boxes, enlarged) is OCR'd again (psm 6, enlarged) and parsed alone. The
  majority of the three readings decides each field; a tie goes to the more confident reading and the field is marked
  uncertain.
- A question only one reader found is kept when the third reading (or the text layer) shows a real question start,
  or when it has no readable number and the agreed numbering has exactly that number missing; otherwise it is dropped
  as a fragment.

The checks then run once more on the agreed ("fused") paper.

## Verifying number findings

Both readers of a scanned Hindi or Tamil page use Tesseract's Indic model, which misreads Latin digits and units in
the same way ('2 A' as '2 ·', '120 °C' as '1:20'): such slips survive the agreement of two readers. So every number
that differs between two versions (NUMBER_MISMATCH, OPTION_NUMBER_MISMATCH) is checked once more by the model that
reads digits best: the question is re-read in both versions by the English OCR model (digits and SI units are
printed the same in every language). The finding stands unless both re-readings agree on every disputed number;
then it is dismissed as a misreading. A text-layer question is its own re-reading. A real change (300 -> 164, a
number added or removed) is read as such by the English model too, so it stands.

## Statuses

Every finding of the fused paper gets a `consensus` record: `status`, `reader_a`, `reader_b` (did each reader raise
it on its own), `confidence_a`, `confidence_b` (the readers' confidence in that question: mean OCR word confidence,
times the layout score for reader A; 0.99 for a text layer), `confidence`, `third_reading`.

| Status | Meaning | Severity |
|---|---|---|
| confirmed | both readers raised it independently, with confidences within 0.25 | as the check sets it |
| resolved | the readers disagreed; the third reading or the text layer decided, and supports it | as the check sets it |
| review | it rests on a value the readers could not agree on, even after the third reading | not a finding: listed as an *uncertain reading* (severity info), for a look by eye; the verdict becomes "check before printing" |

Findings that one reader raised and the fused paper does not reproduce are returned in `consensus.dismissed` with
the reader that raised them (and those the English re-reading disproved, `raised_by: consensus`), so nothing is
hidden. `consensus.stats` gives, per version, the questions both read,
those only one read, fragments dropped, the share of fields read alike, third readings, questions decided by the text
layer, uncertain questions and each reader's mean confidence.

## Cost

Born-digital PDFs: reader B needs no OCR (the text layer), so the consensus costs almost nothing. Scans: one more
OCR pass per page, plus a small crop for each question the readers disagree on.

## Measuring it

`scripts/evaluate_consensus.py` (CBSE-format bilingual papers, `scripts/verify_bench/cbse.py`) and
`scripts/evaluate_verify.py --consensus` (the original English-Hindi/Tamil pairs) report reader A alone, reader B
alone and the consensus on the same papers: share of planted errors found at the right question, and errors and
warnings per clean paper. Results are in the README.
