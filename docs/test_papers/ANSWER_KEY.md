# Test papers for IndicDoc Verify

Each folder has an English source paper and its Hindi or Tamil version (18 questions, 3 pages). The version has one planted error, or none. Upload them in the app's **Verify Paper** tab (or `POST /verify`) and check that the report points at the question below. Set the language of the paper (Hindi / Tamil) and of the source (English), or leave auto-detect.

These papers come from the test set of the verification benchmark (`scripts/verify_bench/`), on which the system found 98% of the planted errors. The report can also list a few other findings (warnings, occasional false alarms): the vetter confirms each one.

"Other findings" is how many more the report listed for that pair in the benchmark run.

| Folder | Language | Kind | Planted error | Expected finding | Other findings | How to check |
|---|---|---|---|---|---|---|
| `01_hindi_no_error` | Hindi | born-digital | None: a correct translation. Anything the report lists is a false alarm. | none | 2 | Verify Paper → "Translation against its source": paper = `hindi_version.pdf`, source = `english_source.pdf` |
| `02_hindi_number_changed` | Hindi | born-digital | Q15: a number in the question was changed (x: 45 -> 102). | NUMBER_MISMATCH | 2 | Verify Paper → "Translation against its source": paper = `hindi_version.pdf`, source = `english_source.pdf` |
| `03_hindi_option_number_changed` | Hindi | born-digital | Q3: a number in an option was changed (option 2: 320 -> 330). | OPTION_NUMBER_MISMATCH | 0 | Verify Paper → "Translation against its source": paper = `hindi_version.pdf`, source = `english_source.pdf` |
| `04_hindi_question_missing` | Hindi | born-digital | Q16 is missing from the version (number gap left). | MISSING_IN_TRANSLATION or NUMBER_GAP or QUESTION_COUNT | 4 | Verify Paper → "Translation against its source": paper = `hindi_version.pdf`, source = `english_source.pdf` |
| `05_hindi_marks_changed` | Hindi | born-digital | Q17: the marks differ from the source (marks 2 -> 6). | MARKS_MISMATCH or MARKS_TOTAL | 3 | Verify Paper → "Translation against its source": paper = `hindi_version.pdf`, source = `english_source.pdf` |
| `06_hindi_repeated_from_old_paper` | Hindi | born-digital | Q14 repeats a question of an earlier paper (repeats Model paper 2025 set 2, Q5). | REPEATED_FROM_BANK | 1 | Verify Paper → "Translation against its source" with `hindi_version.pdf` and `english_source.pdf`; under "More checks" upload `question_bank_hi.json` as the question bank |
| `07_tamil_no_error` | Tamil | born-digital | None: a correct translation. Anything the report lists is a false alarm. | none | 0 | Verify Paper → "Translation against its source": paper = `tamil_version.pdf`, source = `english_source.pdf` |
| `08_tamil_option_missing` | Tamil | born-digital | Q9: an option is missing (option 4 removed). | OPTION_COUNT or OPTION_COUNT_MISMATCH | 0 | Verify Paper → "Translation against its source": paper = `tamil_version.pdf`, source = `english_source.pdf` |
| `09_tamil_question_printed_twice` | Tamil | born-digital | Q13 is printed twice (repeats question 12). | DUPLICATE_QUESTION | 0 | Verify Paper → "One paper": paper = `tamil_version.pdf` (the error is inside this paper; the English source is there to compare too) |
| `10_tamil_wrong_maximum_marks` | Tamil | born-digital | The printed maximum marks are wrong (maximum marks 24 -> 19). | MARKS_TOTAL or TOTAL_MISMATCH | 0 | Verify Paper → "One paper": paper = `tamil_version.pdf` (the error is inside this paper; the English source is there to compare too) |
| `11_tamil_numbering_error` | Tamil | born-digital | Question numbering is wrong (question 16 printed as 15). | NUMBER_GAP or NUMBER_ORDER or NUMBER_REPEATED | 0 | Verify Paper → "One paper": paper = `tamil_version.pdf` (the error is inside this paper; the English source is there to compare too) |
| `12_tamil_scanned_number_changed` | Tamil | scanned | Q5: a number in the question was changed (d: 240 -> 619). | NUMBER_MISMATCH | 0 | Verify Paper → "Translation against its source": paper = `tamil_version.pdf`, source = `english_source.pdf` |

API example (test 02):

```bash
curl -F file=@02_hindi_number_changed/hindi_version.pdf -F source=@02_hindi_number_changed/english_source.pdf \
     "http://localhost:8000/verify?lang=hin&source_lang=eng&output=html" > report.html
```

Rebuild: `python scripts/verify_bench/export_test_papers.py` (needs `data/verify_bench`, made by the benchmark scripts).
