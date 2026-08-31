# AIRR-ML-25 — exact Docker commands used for the Zaslavsky BCR experiment

Every method was run through its published Docker image, one container per
train/test configuration. This document lists the exact command for each method.

## Conventions

- **Images** live under one Docker Hub user: <https://hub.docker.com/u/airrml25>,
  named `airrml25/airr-ml-25-r<N>` and keeping the same `R#` scheme as the paper.
- **Tags**
  - `:as-published` — byte-identical to the image the participants published
    (R2, R3, R5, R7, R9, R10).
  - `:minimal-fix` — the participant image plus the packaging/interface fixes
    documented in `MINIMAL_CHANGES.md` (R1, R4, R6, R8); **no method logic is
    changed**. For the six repositories that needed no change, `:minimal-fix`
    points at the same image ID as `:as-published`.
- **Placeholders** used below (one substitution per configuration):
  - `<TRAIN>` = `bcr_datasets/<config>/train`   (holds `metadata.csv` + `*.tsv`)
  - `<TEST>`  = `bcr_datasets/<config>/test`    (`*.tsv` only — no labels)
  - `<OUT>`   = `bcr_results/<method>/<config>`
  - `<config>` ∈ {`lupus_fold0..2`, `hiv_fold0..2`, `t1d_fold0..2`} — 9 per method
  - `n_jobs = 8`
---

## R1 — R1-M2, R1-M3, R1-M5   

R1 picks its sub-method from the **dataset number in the mount path**, so the
train/test dirs are bind-mounted *as* `train_dataset_<N>` / `test_dataset_<N>`:
`R1-M2→N=7`, `R1-M3→N=5`, `R1-M5→N=4`. A per-configuration workspace
`<OUT>/_r1ws/ds<N>/{model,output,cache}` isolates each fold; `model` is seeded
once from the image:

```bash
cid=$(docker create airrml25/airr-ml-25-r1:minimal-fix)
docker cp "$cid:/app/winningApproach/ds<N>/model" "<OUT>/_r1ws/ds<N>/model"
docker rm -f "$cid"
```

```bash
docker run --rm \
  -v <TRAIN>:/data/train_dataset_<N>:ro \
  -v <TEST>:/data/test_dataset_<N>:ro \
  -v <OUT>:/out \
  -v <OUT>/_r1ws/ds<N>/model:/app/winningApproach/ds<N>/model \
  -v <OUT>/_r1ws/ds<N>/output:/app/winningApproach/ds<N>/output \
  -v <OUT>/_r1ws/ds<N>/cache:/app/winningApproach/ds<N>/cache \
  airrml25/airr-ml-25-r1:minimal-fix \
  --train_dir /data/train_dataset_<N> --test_dir /data/test_dataset_<N> \
  --out_dir /out --n_jobs 8 --device cpu
```

*R1-M1 (N=2) and R1-M4 (N=8) are not in the results — see "Excluded" below.*

---

## R2   (`airrml25/airr-ml-25-r2:as-published`)

```bash
docker run --rm \
  -v <TRAIN>:/data/train_dataset_1:ro -v <TEST>:/data/test_dataset_1:ro \
  -v <OUT>:/out --entrypoint python airrml25/airr-ml-25-r2:as-published \
  run_all_datasets.py --train_datasets_dir /data --test_datasets_dir /data \
  --out_dir /out --n_jobs 8 --device cpu
```

## R3   (`airrml25/airr-ml-25-r3:as-published`)

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  airrml25/airr-ml-25-r3:as-published \
  --train_dir /data/train --test_dirs /data/test --out_dir /out --n_jobs 8 --device cpu
```

---

## R4 — R4-M2, R4-M3   (`airrml25/airr-ml-25-r4:as-published`)

Sub-method chosen by `--predictor`: `R4-M2→kmer`, `R4-M3→multikmer`
(`R4-M1→emerson`, excluded — see below). These two write cache/logs into the CWD,
so `-w /out` and `PYTHONPATH=/workspace`:

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  -w /out -e PYTHONPATH=/workspace --entrypoint bash \
  airrml25/airr-ml-25-r4:minimal-fix -lc \
  "source /opt/conda/etc/profile.d/conda.sh && conda activate kaggle_ml && \
   python -m submission.main --train_dir /data/train --test_dirs /data/test \
   --out_dir /out --predictor <kmer|multikmer> --n_jobs 8 --device cpu"
```

---

## R5   (`airrml25/airr-ml-25-r5:as-published`)

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  airrml25/airr-ml-25-r5:as-published \
  --train_dir /data/train --test_dirs /data/test --out_dir /out --n_jobs 8
```

## R6   (`airrml25/airr-ml-25-r6:as-published`)

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  -w /out --entrypoint python3.11 airrml25/airr-ml-25-r6:minimal-fix \
  /app/script/03_Esemble_xgb_30_iteration_v2.py \
  --train_dir /data/train --test_dir /data/test --out_dir /out --n_jobs 8
```

## R7   (`airrml25/airr-ml-25-r7:as-published`)

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  airrml25/airr-ml-25-r7:as-published \
  --train_dir /data/train --test_dir /data/test --out_dir /out --n_jobs 8 --is_synthetic
```

---

## R8 — R8-M1, R8-M2   (`airrml25/airr-ml-25-r8:as-published`)

**R8-M1** (public-TCR model) — three steps in one container:

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  -w /out --entrypoint bash airrml25/airr-ml-25-r8:minimal-fix -c "set -e
  python3 /app/methods/public_TCR_model.py make-candidates --train_meta /data/train/metadata.csv \
    --train_dir /data/train --test_dir /data/test --out_tsv /out/candidate_tcrs.tsv
  python3 /app/methods/public_TCR_model.py train --train_meta /data/train/metadata.csv \
    --train_dir /data/train --candidate_tcr_tsv /out/candidate_tcrs.tsv --out_dir /out \
    --dataset_name R8-M1 --top_tcr_out_csv /out/R8-M1_important_sequences.csv
  python3 /app/methods/public_TCR_model.py predict \
    --model_bundle_pkl /out/publictcr_fisher_lr_bundle.pkl --test_dir /data/test \
    --dataset_name R8-M1 --out_csv /out/R8-M1_test_predictions.csv"
```

**R8-M2** (k-mer index model) — its `predict` needs a test `metadata.csv`, which
real AIRR test dirs lack, so a two-column (`repertoire_id,filename`, **no labels**)
file is synthesised from the `.tsv` filenames:

```bash
docker run --rm -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  -w /out --entrypoint bash airrml25/airr-ml-25-r8:minimal-fix -c "set -e
  python3 -c \"import os,csv;d='/data/test';rows=[(f[:-4],f) for f in sorted(os.listdir(d)) if f.endswith('.tsv')];w=csv.writer(open('/out/_test_metadata.csv','w',newline=''));w.writerow(['repertoire_id','filename']);w.writerows(rows)\"
  python3 /app/methods/kmer_index_model.py train --metadata_csv /data/train/metadata.csv \
    --tsv_dir /data/train --out_dir /out --dataset_name R8-M2 \
    --top_tcr_out_csv /out/R8-M2_important_sequences.csv
  python3 /app/methods/kmer_index_model.py predict --model_bundle_pkl /out/motif_lr_bundle.pkl \
    --metadata_csv /out/_test_metadata.csv --tsv_dir /data/test \
    --dataset_name R8-M2 --out_csv /out/R8-M2_test_predictions.csv"
```

---

## R9   (`airrml25/airr-ml-25-r9:as-published`, GPU) — lupus folds only

Image declares `USER user` (uid 999); run as that uid, with a writable HF cache
for the runtime ESM2 download:

```bash
docker run --rm --user 999:999 --gpus all \
  -v <TRAIN>:/data/train:ro -v <TEST>:/data/test:ro -v <OUT>:/out \
  -e HF_HOME=/out/hf_cache airrml25/airr-ml-25-r9:as-published \
  --train_dir /data/train --test_dirs /data/test --out_dir /out \
  --n_jobs 8 --no-reproduce --no-topseq --num_gpus 1 --esm_batch_size 1024
```

## R10   (`airrml25/airr-ml-25-r10:as-published`)

Resolves data relative to its own repo root, so the dataset tree is mounted at
`/app/AIRR_ML_25_Phase2_data`:

```bash
docker run --rm \
  -v <OUT>/_r10/AIRR_ML_25_Phase2_data:/app/AIRR_ML_25_Phase2_data \
  -v <TRAIN>:/app/AIRR_ML_25_Phase2_data/train_datasets/train_dataset_1:ro \
  -v <TEST>:/app/AIRR_ML_25_Phase2_data/test_datasets/test_dataset_1:ro \
  --entrypoint python3 airrml25/airr-ml-25-r10:as-published \
  src/execute.py --dataset AIRR_ML_25_Phase2_data \
  --model_type both --model_name 4mer-logreg --vj_model_name vj-logreg \
  --classifier_type logistic --save_important_sequences \
  --parallel_jobs 1 --per_job_n_jobs 8 --dataset-ids 1
```

---

## Excluded methods (attempted, no usable predictions — documented, not fixed)

| method | image / N | outcome |
|---|---|---|
| **R1-M1** | r1, N=2 | ran but produced no output after 63 h on one fold; infeasible at this scale, cut |
| **R1-M4** | r1, N=8 | **blocked** — CVC/BERT embedding model absent from repo (also hard-requires CUDA) |
| **R4-M1** | r4, `--predictor emerson` | runs but fails all 9 folds: CompAIRR rejects the `X` residue in BCR CDR3s (needs `-u`, not passed) |
| **R8-M3** | r8 | **blocked** — `resources/100k_kmean.pkl` and precomputed embeddings absent from repo |


