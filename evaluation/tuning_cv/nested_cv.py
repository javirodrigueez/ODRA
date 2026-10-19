"""Nested (repeated, patient-grouped) CV for the resistance hyperparameters.

Selection: in each fold pick the config with the highest mean Overall CTRS on the 4 training folds
(tie-break: first config listed). Evaluation: out-of-fold (OOF) score of the selected configs.
CACTUS profiles idx 3p, 3p+1, 3p+2 are the same patient p under 3 attitudes, so folds and the
bootstrap are built over patients (idx // 3), never over individual profiles.
"""
import argparse
import glob
import json
import os

import numpy as np

CONFIGS = [  # order = tie-break priority
  ("C1 [-0.05,+0.10] a=0.15", "/outs/hyperparam_tuning/qwen_bounds_005_010_recovery_015"),
  ("C2 [-0.10,+0.15] a=0.05", "/outs/hyperparam_tuning/qwen_bounds_010_015_recovery_005"),
  ("C3 [-0.10,+0.15] a=0.15", "/outs/hyperparam_tuning/qwen_bounds_010_015_recovery_015"),
  ("C4 [-0.10,+0.15] a=0.30", "/outs/hyperparam_tuning/qwen_bounds_010_015_recovery_030"),
  ("C5 [-0.20,+0.30] a=0.15", "/outs/hyperparam_tuning/qwen_bounds_020_030_recovery_015"),
]
ITEMS = ["CBT_1_guided_discovery", "CBT_2_focus", "CBT_3_strategy",
         "general_1_understanding", "general_2_interpersonal_effectiveness", "general_3_collaboration"]


def ctrs_dir(run_dir):
  for sub in ["eval/ctrs", "evaluation/ctrs"]:
    if os.path.isdir(os.path.join(run_dir, sub)):
      return os.path.join(run_dir, sub)
  raise FileNotFoundError(f"No eval/ctrs or evaluation/ctrs in {run_dir}")


def load(configs):
  # returns profile ids, attitudes and scores S with shape (n_configs, n_profiles, 6)
  per = []
  for _, d in configs:
    m = {}
    for f in glob.glob(ctrs_dir(d) + "/*.json"):
      _, idx, att = os.path.basename(f)[:-5].split("_")
      m[int(idx)] = (att, [json.load(open(f))[k] for k in ITEMS])
    per.append(m)
  ids = sorted(per[0])
  for (name, _), m in zip(configs, per):
    assert sorted(m) == ids, f"{name}: profile ids differ from {configs[0][0]}"
  att = np.array([per[0][i][0] for i in ids])
  S = np.array([[m[i][1] for i in ids] for m in per], dtype=float)
  return np.array(ids), att, S


def patient_folds(pid, seed, k=5):
  # shuffle patients with the seed and deal them round-robin into k folds
  rng = np.random.default_rng(seed)
  patients = rng.permutation(np.unique(pid))
  pfold = dict(zip(patients, np.arange(len(patients)) % k))
  fold = np.array([pfold[p] for p in pid])
  return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def patient_boot_idx(pid, n, rng):
  # bootstrap over patients, keeping all profiles of each resampled patient
  upid = np.unique(pid)
  rows = {p: np.flatnonzero(pid == p) for p in upid}
  pick = rng.choice(upid, (n, len(upid)))
  return np.array([np.concatenate([rows[p] for p in r]) for r in pick])


def report(name, X, boot_idx):
  ov = X.mean(1)
  lo, hi = np.percentile(ov[boot_idx].mean(1), [2.5, 97.5])
  it_lo, it_hi = np.percentile(X[boot_idx].mean(1), [2.5, 97.5], axis=0)
  print(f"\n{name}: Overall {ov.mean():.4f} [{lo:.4f}, {hi:.4f}]")
  for j, it in enumerate(ITEMS):
    print(f"    {it:40s} {X[:, j].mean():.4f} [{it_lo[j]:.4f}, {it_hi[j]:.4f}]")
  return ov.mean()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Patient-grouped nested CV + OOF evaluation of resistance hyperparameters.")
  parser.add_argument("--c3-dir", type=str, default=CONFIGS[2][1], help="Run dir for C3 (bounds [-0.10,+0.15], a=0.15).")
  parser.add_argument("--reps", type=int, default=10, help="Repetitions (seeds 0..reps-1).")
  parser.add_argument("--folds", type=int, default=5)
  parser.add_argument("--boot", type=int, default=10000, help="Bootstrap resamples (over patients).")
  parser.add_argument("--show-folds", action="store_true", help="Print seed-0 fold membership and per-config train/test means.")
  args = parser.parse_args()

  configs = list(CONFIGS)
  configs[2] = (CONFIGS[2][0], args.c3_dir)
  names = [c[0] for c in configs]
  ids, att, S = load(configs)
  pid = ids // 3
  ov = S.mean(2)  # (n_configs, n_profiles)
  nC, nP = ov.shape
  print("configs:", *[f"{n} <- {d}" for n, d in configs], sep="\n  ")
  print("profiles", nP, "patients", len(np.unique(pid)), "attitudes", {a: int((att == a).sum()) for a in sorted(set(att))})

  print("\nOverall CTRS on all profiles:")
  for c in range(nC):
    print(f"  {names[c]}: {ov[c].mean():.4f}  items", S[c].mean(0).round(3))

  if args.show_folds:
    print("\nSeed 0 folds:")
    for k, (tr, te) in enumerate(patient_folds(pid, 0, args.folds)):
      print(f"  fold{k}: patients {sorted(set(pid[te].tolist()))} n={len(te)} "
            f"att={ {a: int((att[te] == a).sum()) for a in sorted(set(att))} } "
            f"train/test patient overlap={len(set(pid[te]) & set(pid[tr]))}")
      print(f"    TRAIN " + " ".join(f"C{c+1}={ov[c, tr].mean():.4f}" for c in range(nC)))
      print(f"    TEST  " + " ".join(f"C{c+1}={ov[c, te].mean():.4f}" for c in range(nC)))

  sel_counts = np.zeros(nC, int)
  oof_items = np.zeros((args.reps, nP, len(ITEMS)))
  print("\nSelected config : held-out score (rows=seed, cols=fold)")
  for seed in range(args.reps):
    cells = []
    for k, (tr, te) in enumerate(patient_folds(pid, seed, args.folds)):
      tr_means = ov[:, tr].mean(1)
      c = int(np.flatnonzero(tr_means == tr_means.max())[0])  # tie-break: first listed
      sel_counts[c] += 1
      oof_items[seed, te] = S[c, te]
      cells.append(f"C{c+1}:{ov[c, te].mean():.3f}")
    print(f"  seed{seed} " + "  ".join(cells) + f"  | OOF={oof_items[seed].mean(1).mean():.4f}")

  rng = np.random.default_rng(0)
  boot_idx = patient_boot_idx(pid, args.boot, rng)
  oof_mean = report("OOF seed-averaged (per-profile mean over reps, bootstrap over patients)", oof_items.mean(0), boot_idx)
  report("OOF canonical seed=0", oof_items[0], boot_idx)

  print(f"\nSelection frequency over {sel_counts.sum()} selections:")
  for c in range(nC):
    print(f"  {names[c]}: {sel_counts[c]}")

  naive_c = int(np.argmax(ov.mean(1)))
  naive = report(f"Naive (best on all profiles = {names[naive_c]})", S[naive_c], boot_idx)
  d = ov[naive_c] - oof_items.mean(0).mean(1)
  lo, hi = np.percentile(d[boot_idx].mean(1), [2.5, 97.5])
  print(f"\nGap naive - OOF: {naive - oof_mean:.4f} [{lo:.4f}, {hi:.4f}]")
  worst = int(np.argmin(ov.mean(1)))
  print(f"Worst config: {names[worst]} {ov[worst].mean():.4f}")
