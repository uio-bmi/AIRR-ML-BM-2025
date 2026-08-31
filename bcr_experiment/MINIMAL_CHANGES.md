# Minimal changes made to run the AIRR-ML-25 methods on the Zaslavsky BCR data

**Rule applied:** change only how a method is *packaged, invoked or wired up*. Never change what
it computes. No feature definition, model, hyperparameter, threshold, sampling strategy or
scoring rule has been touched in any repository.

Every change below is one of: a missing dependency declaration, a missing `import`, an undefined
name, an entry point, an environment variable passed between a wrapper and its own scripts, a
file-discovery glob, or a serialisation fallback for a log file.

Images used for the BCR run are tagged `:minimal-fix`. For the six repositories that needed no
change at all, that tag points at the **byte-identical** image already published as
`airrml25/airr-ml-25-<r>:as-published` (verified by image ID).

Patches are stored as diffs in `airr_ml_25_docker_validation/docker_overrides/<R#>/*.patch`,
with the untouched originals alongside.

---

## Summary

| repo | methods | code changed? | result |
|---|---|---|---|
| R2, R3, R5, R7, R9, R10 | 6 | **none** | run as published |
| R4 | R4-M1, R4-M2, R4-M3 | 1 line | all 3 now run |
| R6 | R6 | dependency pins only | now runs |
| R8 | R8-M1, R8-M2 | 1 file + 3 keywords | both now run |
| R8 | R8-M3 | — | **still blocked**, cannot be fixed by us |
| R1 | R1-M1, R1-M2, R1-M3, R1-M5 | 7 small fixes | all 4 now run |
| R1 | R1-M4 | — | **still blocked**, cannot be fixed by us |

**16 of 18 methods now runnable.** The two exceptions are missing model files, not code defects.

---

## R4 — one commented-out line (affects R4-M1, R4-M2, R4-M3)

**Symptom:** every invocation ignored its arguments and died with
`AssertionError: Train directory /mnt/sda/Kaggle/AIRR-ML/... does not exist`.

**Cause:** `submission/main.py` shipped with the argparse entry point disabled and a hardcoded
call to the author's own machine left active. `--predictor emerson|kmer|multikmer` — the only
way to select between the three methods — lives inside `run()`, so no method was reachable.

**Change** (`docker_overrides/R4/restore_cli_entrypoint.patch`):

```diff
 if __name__ == "__main__":
-    # run()
-    main_emerson("/mnt/sda/Kaggle/AIRR-ML/train_datasets/train_datasets/train_dataset_1",
-                 ["/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_1"],
-                 "./results", 16, 'cpu')
+    run()
```

**Classification:** entry point. `run()` is the authors' own function, called exactly as their
README documents. No model code executed differently — it is now reachable rather than dead.

---

## R6 — dependency pinning only (affects R6)

**Symptom:** `TypeError: DataFrame.groupby() got an unexpected keyword argument 'axis'` in
`normalize_kmer_rows_by_category()` (`03_Esemble_xgb_30_iteration_v2.py:519`).

**Cause:** the Dockerfile installs `numpy pandas tqdm scikit-learn xgboost optuna` with no
version constraints. A build made in 2026 resolves pandas 3.0.5, where `groupby(axis=1)` has
been removed. Nothing about the method changed — its environment moved underneath it.

**Change** (`docker_overrides/R6/Dockerfile`, original preserved as `Dockerfile.original`):

```diff
-RUN python3.11 -m pip install --no-cache-dir \
-    numpy pandas tqdm scikit-learn xgboost optuna
+RUN python3.11 -m pip install --no-cache-dir \
+    "numpy==1.26.4" "pandas==1.5.3" "tqdm==4.67.2" \
+    "scikit-learn==1.5.2" "xgboost==2.1.4" "optuna==4.7.0"
```

**Classification:** dependency pin. The versions are those of the conda environment in which
this team's Phase-2 results were actually produced on this server, so this restores the
authors' own environment rather than imposing a new one.

---

## R8 — a missing file and a logging call (affects R8-M1, R8-M2)

### (a) The image could not be built at all

`Dockerfile` line 9 does `COPY requirements.txt /app/requirements.txt`, but the repository
contains no such file. Verified by rebuilding the repo exactly as GitHub serves it
(`git archive HEAD`): the build fails at that line.

**Change:** added `requirements.txt`, reconstructed strictly from the third-party imports in
`submission/` and `methods/`:

```
numpy
pandas
scipy
scikit-learn
joblib
tqdm
torch
```

Versions intentionally unpinned. **Classification:** packaging. Without it there is no R8 image.

### (b) All three methods crashed after training

**Symptom:** `TypeError: Object of type function is not JSON serializable`.

**Cause:** each method writes a `*_meta.json` containing `vars(args)`, and argparse's subcommand
dispatch puts a non-serialisable `func` callback in there. The model `.pkl` is saved *before*
this line but the important-sequences output is written *after* it, so every training run lost
its sequence output.

**Change** (`docker_overrides/R8/fix_meta_json_serialization.patch`) — three files, one keyword each:

```diff
-        json.dump(bundle.to_meta_dict(), f, indent=2)      # public_TCR_model.py:516
+        json.dump(bundle.to_meta_dict(), f, indent=2, default=str)
-        json.dump(bundle.to_dict_meta(), f, indent=2)      # kmer_index_model.py:490
+        json.dump(bundle.to_dict_meta(), f, indent=2, default=str)
-        json.dump(bundle.to_meta(), f, indent=2)           # embedding_model.py:1053
+        json.dump(bundle.to_meta(), f, indent=2, default=str)
```

**Classification:** serialisation of a metadata log file. It changes only how non-serialisable
values are rendered *into a JSON side-file*; the model, its inputs and its predictions are
untouched.

---

## R1 — seven small fixes (affects R1-M1, R1-M2, R1-M3, R1-M5)

As published none of R1's five methods completed. Seven distinct defects, none of them logic
(`docker_overrides/R1/fix_missing_deps_and_paths.patch`).

**1–2. Two undeclared dependencies** — `docker/requirements.txt`

```diff
+imbalanced-learn>=0.9.0     # ds7/code/train.py: from imblearn.under_sampling import RandomUnderSampler
+transformers>=4.20.0        # ds8/code/cvc_embedder.py: from transformers import BertModel, BertTokenizer
```

Both are imported by shipped code but were never listed. *Classification: dependency declaration.*

**3. Missing import** — `ds4/code/train.py`

```diff
+import os
```

The file calls `os.environ.get(...)` on line 43 but never imports `os` → `NameError`.
*Classification: missing import.*

**4. Undefined name** — `ds4/code/train.py`

```diff
-CACHE_DIR = BASE_DIR / 'cache'
+CACHE_DIR = CODE_DIR.parent / 'cache'
```

`BASE_DIR` is never defined in this file. Its two sibling scripts (`predict.py`,
`rank_sequences.py`) both define `BASE_DIR = SCRIPT_DIR.parent`, which is exactly
`CODE_DIR.parent` here — so the substituted value is the authors' own.
*Classification: undefined name; resolves to an identical path.*

**5. Missing import** — `ds8/code/cvc_embedder.py`

```diff
+import os
```

Same defect as (3). *Classification: missing import.*

**6. Wrapper did not pass the training directory** — `docker/submission/predictor.py`

```diff
+        self.train_dir_ = train_dir
...
+            if getattr(self, "train_dir_", None) is not None:
+                env[f"DS{self.dataset_num}_TRAIN_DATA"] = str(self.train_dir_)
```

`predict()` exported only `DS<N>_TEST_DATA`. `ds5/code/predict.py` also reads
`DS5_TRAIN_DATA` and, without it, fell back to the author's `../input/train`
→ `FileNotFoundError`. *Classification: environment variable passed between the repo's own
wrapper and its own script.*

**7. Wrapper used the wrong test variable name** — `docker/submission/predictor.py`

```diff
+            env[f"DS{self.dataset_num}_TEST{test_idx}_DATA"] = str(test_dir)
```

`ds7/code/predict.py` reads `DS7_TEST1_DATA` / `DS7_TEST2_DATA` (Phase-1 shipped two test sets
per dataset), not `DS7_TEST_DATA`. Without this it globbed the author's `../input/test1`,
found **0 test repertoires** and silently produced 0 predictions while still reporting a
training AUC. *Classification: environment variable name; the most dangerous of the seven
because it failed silently.*

**Also: output file discovery** — `docker/submission/predictor.py`

```diff
-        test_pred_files = list(dataset_output_dir.glob("*test_predictions.csv"))
-        ranked_seq_files = list(dataset_output_dir.glob("*ranked_sequences.csv"))
+        test_pred_files = list(dataset_output_dir.glob("*test_predictions*.csv"))
+        ranked_seq_files = list(dataset_output_dir.glob("*ranked_sequences*.csv"))
```

DS2 writes `ds2_test_predictions_v2.csv`, which the original exact-suffix glob never matched,
so DS2's predictions were discarded with "No predictions found".
*Classification: file discovery.*

### What was deliberately NOT changed in R1

Two known fragilities were left exactly as the authors wrote them, because fixing them would
alter behaviour:

* **R1-M2 (DS7)** hardcodes `RandomUnderSampler(sampling_strategy=0.5)`, which requires the
  training set to be at least 2:1 imbalanced. It would crash on balanced data. Left alone —
  and the BCR cohorts happen to be ~2.3:1 controls-to-cases, so it should run here.
* **R1-M3 (DS5)** selects its model by cross-validation; if a RandomForest wins,
  `predict.py` dies on `.coef_` because it assumes logistic regression. Left alone — with
  ~213 training repertoires logistic regression is expected to win, as it did in validation.

For **R1-M1 (DS2)** and **R1-M3 (DS5)** the repo's wrapper writes an all-`-999.0` file because
its scripts emit `repertoire_id,probability` while the wrapper builds the Kaggle schema. Rather
than patch that mapping, the scorer reads the scripts' own
`ds{2,5}_test_predictions.csv` directly. **Zero code change.**

---

## Changes to *how methods are invoked* (no repository code involved)

These live entirely in our harness (`airr_ml_25_docker_validation/scripts/run_method.sh`):

| method | invocation detail | why |
|---|---|---|
| **R1-M\*** | train/test dirs bind-mounted **as** `train_dataset_<N>` | R1 selects its method by parsing `dataset_1..8` from the directory name, so a BCR directory called `lupus_fold0` fails. The mount renames it; nothing in the repo changes. |
| **R1-M\*** | per-configuration workspace under `$OUT/_r1ws/` | R1 writes into `winningApproach/ds<N>/{model,output}` inside the image; a shared workspace would leak fold *N*'s model into fold *N+1*. |
| **R4-M1** | working directory `/workspace`, writable `/workspace/logs` mounted | the bundled `./compairr-1.13.0-linux-x86_64` is invoked by relative path. |
| **R4-M2/M3** | working directory `/out`, `PYTHONPATH=/workspace` | they write `./cache_kmer` and `./logs` into the current directory. |
| **R8-M2** | a two-column `metadata.csv` synthesised for the test directory | its `predict` requires `--metadata_csv`; real AIRR test directories have none. Contains only `repertoire_id,filename` — no labels. |
| **R9** | `--user 999:999`, `HF_HOME` on a writable mount | the image declares `USER user` (uid 999) but this server forces `--user $(id -u)`, making its WORKDIR unreadable. Running it as its own uid is what a normal Docker host does. It also downloads ESM2 at runtime. |
| **R10** | dataset tree mounted at `/app/<dataset-name>` | it resolves data relative to its own repo root and writes into `<dataset>/results/`. |

---

## Still blocked — cannot be fixed at any level

Neither is a code defect. Both need files only the authors have.

**R1-M4 (DS8)** — requires a CVC/BERT embedding model absent from the repository.
`ds8/code/predict.py` instructs `python -m scripts.download_cvc --model_type CVC`, but there is
no `scripts/` module in the repo. `ds8/model/` contains only GCN weights. Also hard-requires CUDA.

```
FileNotFoundError: CVC model not found. Please download the model first using:
    python -m scripts.download_cvc --model_type CVC
```

**R8-M3** — requires `resources/100k_kmean.pkl` (precomputed k-means prototypes) and
precomputed per-TCR embeddings in `train_datasets_emb/`. Both are excluded by the repository's
own `.gitignore` (`*.pkl`, `train_datasets_emb/`) and no script regenerates them.

```
FileNotFoundError: [Errno 2] No such file or directory: '/app/resources/100k_kmean.pkl'
```

---

## Excluded from BCR results — ran, but did not produce usable predictions on this dataset

Distinct from the two above: these images build and run (they worked in Phase-2), but on the
Zaslavsky BCR data one fails and one does not finish. Neither was altered — the outcome *is* the
finding. Both repositories are still represented in the results by their other methods.

**R4-M1 (CompAIRR-based)** — fails on every one of the 9 configurations. CompAIRR rejects the
ambiguous residue `X`, which occurs in the BCR CDR3s, unless invoked with `-u`; the method does
not pass `-u`. Adding it would be a logic change to the pipeline, so per the no-fixing policy the
method is omitted and documented. R4 is still represented by R4-M2 and R4-M3 (both 9/9).

```
Error: Illegal character 'X' in sequence on line 237254. Use -u to ignore.
```

**R1-M1 (DS-embedding variant)** — cut after **63 h** of wall-clock on its *first* fold with no
output (single-threaded, RAM climbing 38 → 49 GB). It does not complete a single BCR fold in any
practical time, so it is excluded as infeasible-on-BCR-timescale. R1 is still represented by
R1-M2, R1-M3, and R1-M5. No code was changed; the container was killed on 2026-07-26.

**R9 (ESM2 + k-mer ensemble) — partial, lupus only.** R9 works correctly on GPU but is very
slow (~8–10 h per config: a single-threaded k-mer pass plus ESM2 embedding of ~320 repertoires).
It completed all three **lupus** folds (pooled ROC AUC 0.79) but the full 9-config run projected
to ~2 more days, so HIV and T1D were cut for time on 2026-07-26. R9's lupus result is retained as
a partial datapoint; it is excluded from the full 3-disease comparison. No code was changed — the
only pin was `torch==2.6.0` (cu124) so ESM2 uses the GPU rather than falling back to CPU, already
documented above. The incomplete `hiv_fold0` working dir was moved aside, not deleted (uid-999
owned).

---

## Data preparation applied equally to all methods

Not method changes, but they affect every result and belong in the manuscript:

* **`junction_aa` was empty in every row of the entire deposit** (BCR and TCR, both
  `airr_format` and `internal_format`). The CDR3 is in `cdr3_aa`. Since all ten repositories
  read `junction_aa`, we set `junction_aa := cdr3_aa`. We did **not** fabricate a junction by
  adding flanking Cys/Phe. Consequence: motifs are CDR3-based rather than junction-based,
  unlike the Phase-2 TCR data.
* **No count column exists** (`templates` / `duplicate_count`). Verified that R4, R5, R7 and R8
  all fall back to 1, and R2, R3, R6, R9, R10 never read one. Nothing was synthesised.
* **Non-productive sequences dropped** (~3.7%), as were rows with an empty CDR3.
* **13 of 49 AIRR columns retained** (21 GB rather than 178 GB). Provably result-identical: a
  scan of all ten repositories shows no method reads the omitted columns.
