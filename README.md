# amp-design-colab

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdulrahman-Neamah/amp-design-colab/blob/main/notebooks/amp_pipeline.ipynb)
   [![tests](https://github.com/Abdulrahman-Neamah/amp-design-colab/actions/workflows/tests.yml/badge.svg)](https://github.com/Abdulrahman-Neamah/amp-design-colab/actions/workflows/tests.yml)
   ![license](https://img.shields.io/badge/license-MIT-blue)

A small, end-to-end pipeline for designing **antimicrobial peptides (AMPs)** with a generative model, built to run entirely in Google Colab.

This is a learning project: a break from protein design to find out how peptide design differs, and to get comfortable with Colab and reproducible machine-learning workflows. It is a computational exercise. Nothing here has been synthesized or tested in a lab.

## What it does

```
known AMPs + measured MICs (e.g. GRAMPA)
        |
        v
1. train a small LSTM on the peptides (family-balanced, so big families don't dominate)
2. sample 300,000 new sequences
3. filter: valid letters, length, unique, not a known peptide, complexity,
           repeats, charge, not a near-copy of a known peptide
4. rank survivors with a random forest trained on measured MICs,
           optionally for ONE organism (e.g. E. coli)
5. pick a top-100 that is diverse: caps per cluster, per family and per
           known neighbor, a charge cap, and a nearest-known-peptide check
        |
        v
library.csv, top100.csv, report.txt, saved weights
```

`report.txt` is the main output: gate-by-gate survival counts, novelty checks against held-out real peptides, ranker accuracy, and a property comparison of the generated library with real peptides.

## Quick start (Colab)

1. Open `notebooks/amp_pipeline.ipynb` with the badge above and set the runtime to **GPU (T4)**.
2. Get a table of peptides with measured MICs (see [Data](#data)) and upload it when the data cell asks.
3. Set the three column names in the data cell, then run the cells from top to bottom. A full run takes about 2-3 minutes on a T4.
4. The last cell zips the results folder for download. Colab deletes its files when the runtime resets, so download before closing.

## Quick start (command line)

```bash
pip install -r requirements.txt
python amp_baseline.py --demo                      # smoke test on synthetic data (not real AMPs)

python amp_baseline.py --data my_amps_org.csv --out run1 \
    --organism "e. coli,escherichia coli" --no_cys --max_overlap 0.9 \
    --train_cap_per_cluster 5 --max_per_neighbor 2 --max_family 10 \
    --top_max_len 30 --top_max_charge 8
```

Run `python amp_baseline.py --help` for every option. The ones that mattered most:

| option | what it does |
|---|---|
| `--organism` | train the potency ranker on one organism's MIC data only |
| `--train_cap_per_cluster` | family-balanced generator training (at most N peptides per similarity cluster) |
| `--max_overlap` | drop generated sequences that share too many 5-mers with known peptides |
| `--max_per_neighbor`, `--max_family` | top-100 caps: per known neighbor, and per family of related picks |
| `--top_max_len`, `--top_max_charge` | top-100 only: length and net-charge limits |
| `--weights` | skip training and resample from saved generator weights |

## Data

No data is included. The pipeline reads a CSV with these columns:

| column | meaning |
|---|---|
| `sequence` | one-letter amino acid sequence |
| `mic_um` | MIC in micromolar |
| `bacterium` (or `organism`, `species`, `strain`) | needed for `--organism` |

I used **GRAMPA** (about 51,000 measurements, 6,218 unique peptides after filtering). The data cell in the notebook converts GRAMPA's log10(µM) values into the format above. Please get the dataset from its authors and check its terms before redistributing it:

> Witten J, Witten Z. *Deep learning regression model for antimicrobial peptide design.* bioRxiv (2019). https://doi.org/10.1101/692681

By default the notebook keeps modified peptides (for example C-terminal amidation) in the data. The data cell can drop them (`DROP_MODIFIED`) once you have looked at the modification columns in your copy.

## Results snapshot

One run, *E. coli* target (`examples/train_a_report.txt` is the full report):

| | |
|---|---|
| Training peptides | 5,597 (3,202 after family balancing; 621 held out) |
| Raw samples | 300,000; 97.9% unique; 0.7% exact copies of a training peptide |
| After all filters | 148,822 sequences, from which a 50,000-sequence library is drawn |
| Potency ranker | random forest on 4,748 *E. coli* peptides. Spearman 0.69 on a random split, **0.57 when whole peptide families are held out** |
| Top-100 | 67 families (largest 10), 91 clusters, median nearest-known identity 0.60, none at 0.80 or above |

The generated library is more cationic and more amphipathic than real peptides (mean charge 4.2 against 3.9, hydrophobic moment 1.29 against 1.11), which is worth knowing before trusting it.

## What I learned

- **A random split flatters the model.** Training data from peptide databases is full of near-duplicate analogs, so a random test set leaks. The ranker's Spearman fell from 0.69 to 0.57 once whole families were held out. The same effect showed up in the all-organism ranker (0.64 to 0.46).
- **Which organism you train on changes what you design.** With an all-organism ranker, 26 of the top 100 picks were tryptophan/proline-rich peptides. With an *E. coli*-specific ranker, only 1 was.
- **Databases are dominated by a few families, and generators copy them.** Before balancing, roughly 8-11% of generated sequences were exact copies of training peptides. After capping peptides per cluster it was under 1%. Filtering alone just pushed the picks to sit under the similarity cutoff; the bigger improvements came upstream, in the training data, and from a cap on family size in the final selection.
- **Reproducibility is detective work.** Three separate things silently changed results between identical runs: Python's randomized string hashing (which changed my clustering), floating-point sums in a parallel random-forest prediction (which reordered near-ties), and GPU nondeterminism. After fixing them, two independent training runs gave bit-identical weights, library and top-100 on a Colab T4. I also once misread a comparison where both runs had loaded the same weights file, so it could not have tested training at all.
- **Colab practicalities:** the disk is temporary, so every result folder gets zipped with a manifest and an environment record; a notebook is easiest to debug when it can be checked cell by cell; and a saved weights file with its settings next to it is the only exact way to reproduce a result.

## Limitations

- Everything is predicted. There is no wet-lab validation, and predicted MICs for the top picks are optimistic (they are the best of 50,000 according to a noisy model).
- Toxicity (hemolysis) and stability are not modeled. A charge cap is only a rough guard.
- Different strains and test conditions are pooled in the MIC averages. Coverage in GRAMPA is uneven across organisms.
- The similarity clustering and "family" definition are simple k-mer and identity heuristics, not proper alignments.
- Tested on a Colab T4 only.

Not medical advice, and not a source of therapeutic candidates.

## Reproducibility

The environment for the example run is in `examples/environment.txt` (Python 3.13, PyTorch 2.11, scikit-learn 1.6, Tesla T4). Seeds are fixed, PyTorch deterministic mode is on, and sampling uses its own seed, so the same weights plus the same settings always give the same library and top-100. Training from scratch repeated exactly in the same environment; a different GPU or library version may not. Each run saves `lstm_best.pt` and `lstm_best.json` (architecture and a fingerprint of the training set), and `--weights` reuses them and warns if the training set differs.

Unit tests (no GPU or PyTorch needed):

```bash
python -m unittest discover -s tests -v
```

## Repo layout

```
amp_baseline.py                  the whole pipeline (one file, one command)
notebooks/amp_pipeline.ipynb     the same code as a step-by-step Colab notebook
tests/test_helpers.py            filters, clustering determinism, top-100 caps, organism selection
examples/                        a full report and the environment from one run
```

## Acknowledgments

Developed with help from Claude (Anthropic) as a coding and debugging partner. The data comes from GRAMPA (Witten and Witten, 2019).

## License

MIT. See `LICENSE`.
