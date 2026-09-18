# Using the trained emulators

This page is for readers who want to **run** ML-IAM rather than train it:
what the published models are, what input they expect, how to get predictions
for new scenarios, how to reproduce the paper's test metrics, and how to
retrain on your own data.

## 1. What is published

Three trained runs accompany the paper. Each is a small archive of weights and
summary statistics, listed in `metadata/published_models.json`.

| Run | Model | Needs target history | Timesteps needed | Leading steps without a prediction |
|---|---|---|---|---|
| `xgb_85` | XGBoost, one booster per target, autoregressive | yes, first 3 timesteps | 4 | 3 |
| `lstm_89` | LSTM with Region and Model_Family embeddings | no | 3 | 2 |
| `tft_95` | Temporal Fusion Transformer, two-window forecast | no | 15 | 3 |

The archives contain **no AR6 data**. They hold the weights, the feature and
target lists, the category vocabularies, the scalers, the imputation medians,
the hyperparameters, the test metrics, and the train/val/test assignment of
series identifiers (Model, Scenario, Region and a split label, no values).
The scenario data belong to the
[AR6 Scenarios Database](https://data.ece.iiasa.ac.at/ar6/) hosted by IIASA;
download them there under their own terms.

## 2. Quick start

```bash
git clone https://github.com/YenShin1891/ml-iam && cd ml-iam
pip install -r requirements.txt            # add requirements-advanced.txt for LSTM and TFT
cp configs/paths-template.py configs/paths.py

make fetch-models                          # all three, or MODELS=tft_95
make predict RUN_ID=tft_95 DESCRIBE=1      # what this run expects
make predict RUN_ID=tft_95 INPUT=my_scenarios.csv OUTPUT=emulated.csv
```

`fetch-models` checks every archive against its published SHA-256 and unpacks
it under `RESULTS_PATH/<model>/<run_id>/`. If you downloaded an archive by
hand, pass it with `ARCHIVE=path/to/tft_95.tar.gz`.

Prediction runs on the CPU and needs neither a GPU nor the AR6 data.

## 3. Input format

A CSV in the wide layout of the IAMC template, which is also the layout
`make process-data` writes:

| Model | Scenario | Region | Variable | 2020 | 2025 | 2030 | ... |
|---|---|---|---|---|---|---|---|
| MESSAGEix-GLOBIOM 1.0 | MyPolicy | World | Price\|Carbon | 0 | 35 | 80 | ... |
| MESSAGEix-GLOBIOM 1.0 | MyPolicy | World | GDP\|PPP | ... | ... | ... | ... |
| MESSAGEix-GLOBIOM 1.0 | MyPolicy | World | Population | ... | ... | ... | ... |

- **Required columns**: `Model`, `Scenario`, `Region`, `Variable`, and one
  column per year. A (Model, Scenario, Region) triple is one series.
- **Optional columns**: `Model_Family` (otherwise the leading token of `Model`,
  e.g. `MESSAGEix-GLOBIOM 1.0` gives `MESSAGEix`), `Region_Scale` (otherwise
  derived from the region name) and `Scenario_Category` (not a model input).
- **Variables**: the 42 input Variables that `--describe` lists, with IAMC
  names and the units of the AR6 database. Variables you do not have can be
  left out. They enter as "not reported", exactly as for an AR6 scenario that
  omits them: XGBoost reads a missing value, the LSTM and TFT read the
  training median for that region scale plus a missingness indicator. Other
  Variables in the file are ignored and reported in a note.
- **Timesteps**: every year in which a series reports at least one input is a
  timestep. Leave out years you do not want emulated. The models were trained
  on AR6 spacing, mostly 5-year and 10-year steps.
- **Regions and model families are closed vocabularies.** The models learn one
  embedding per label, so an unseen region or family cannot be emulated and is
  refused with the list of known labels. For an IAM version that is not in
  AR6, set `Model_Family` to the family it belongs to.
- **Target history (XGBoost only).** XGBoost predicts each timestep from the
  two before it, so add rows for the nine target Variables with values at the
  first three timesteps of each series. Later target values are ignored. A
  target with no history enters as not reported. The LSTM and TFT never read
  target values, so target rows can be omitted for them.

Output is one row per series and year with a column per target, in the
training units (`--describe` lists them). `--format iamc` writes the wide IAMC
layout instead. The leading timesteps in the table above carry no prediction,
because the models use them as context.

## 4. What the emulator can and cannot tell you

- It emulates the **AR6 ensemble of IAMs**, not the energy system. A
  prediction answers "what would an IAM of this family report for these
  inputs", within the range of inputs the AR6 scenarios span.
- Inputs outside that range are extrapolation, and tree ensembles in
  particular return the nearest value they have seen. Compare your inputs with
  the AR6 ranges before trusting the output.
- Accuracy differs by region and variable. `metrics/performance_by_region.csv`
  in each bundle and Appendix F of the paper give the numbers. Country-level
  series are the weakest.
- The TFT was trained with a per-target flag marking whether the source IAM
  reported that target at that timestep, which the masked loss requires. For a
  new scenario every flag is set to "reported". On the AR6 test set this
  lowers the per-target average R² from 0.984 to 0.979, so expect accuracy at
  the lower figure.

## 5. Reproducing the paper's test metrics

The bundles carry each run's split assignment, so the same 2,346 test series
(1,738 of them long enough for the TFT) can be rebuilt from your own AR6
download:

```bash
make process-data                          # prints the dataset version it wrote
python scripts/train.py --model xgb --resume preprocess --run_id xgb_85 --dataset <that version>
python scripts/train.py --model xgb --resume test       --run_id xgb_85 --dataset <that version>
```

`preprocess` rebuilds the run's cached data and reuses the published split.
`test` rewrites `metrics/performance.csv`, which should match the copy in the
bundle. The dashboard (`make dashboard RUN_ID=xgb_85`) works after the same
two steps. Use `--model lstm` or `--model tft` for the other runs. The TFT
test phase expects a GPU.

## 6. Retraining

Retraining is the normal pipeline described in the README: edit
`configs/data.py` to change inputs, targets or filters, run
`make process-data`, then `make train RUN=configs/runs/<model>_example.yaml`.
A retrained run is used exactly like a published one: pass its run id to
`scripts/predict.py`. For the LSTM and TFT the imputation medians are written
on first use from the run's cached training data.

To publish your own run, `python scripts/export_run_bundle.py --run_id <id>
--out release/` packs the same whitelist of files, strips optimizer and
callback state from the checkpoint, writes a model card and checksums, and
refuses to write an archive that contains a local file path.
