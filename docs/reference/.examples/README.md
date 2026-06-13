# Captured ground-truth for the reference manual examples

These files are raw output captured from a populated demo catalog
(`create_demo_catalog(create_features=True, create_datasets=True)`,
deriva-ml v1.46.x) on 2026-06-13. The reference docs
(`fk-traversal.md`, `bag-export.md`, `denormalization.md`) paste
verified excerpts from these — every numeric/structural claim in a
"Worked examples" section traces here.

- `traversal.txt` — reached-table sets for dataset 5D0 (with/without terminal tables)
- `term_paths.txt` — FK paths to Execution_Asset (shows terminal made no
  reached-set difference on THIS catalog; the eye-ai explosion topology
  is not reproducible here — see fk-traversal.md's illustrative note)
- `export.txt` — anchors / excluded-empty-associations / member counts for flat dataset 5D0
- `nested_b2.txt` — nested datasets 5CT/5CM: recursive anchors + the
  descendant-member rule (5CM has 0 own Image members but Dataset_Image
  is NOT excluded because descendants have images)
- `denorm.txt` — bag tables + denormalized frame (shape, columns, rows) for 5D0
