"""Unit tests for the parts of the pipeline that do not need PyTorch or a GPU.

Run:  python -m unittest discover -s tests -v
"""
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import amp_baseline as m  # noqa: E402


def args(**kw):
    base = dict(min_len=8, max_len=50, min_entropy=2.0, no_cys=False, min_charge=2, max_selfrep=0.2)
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestFilters(unittest.TestCase):
    def test_gates_remove_what_they_should(self):
        good = "KWKLFKKIGAVLKVLTTGLPAL"
        raw = [good, good,                     # duplicate
               "KKKKKKKKKK",                   # homopolymer
               "ACDE",                         # too short
               "KWKLFKKIGAVLKVLTTGLPALX",      # invalid letter
               "DDDDEEEEDDDD" + "AGSTQ"]       # net negative charge
        kept, steps = m.filter_chain(raw, set(), args())
        self.assertEqual(kept, [good])
        self.assertTrue(all(n >= 1 for _, n in steps))

    def test_exact_training_peptides_are_dropped(self):
        known = "KWKLFKKIGAVLKVLTTGLPAL"
        kept, _ = m.filter_chain([known], {known}, args())
        self.assertEqual(kept, [])

    def test_internal_repeat_fraction(self):
        unit = "AKRHHGYKRKFH"
        self.assertGreater(m.selfrep_frac(unit * 3), 0.3)
        self.assertEqual(m.selfrep_frac("KWKLFKKIGAVLKVLTTGLPAL"), 0.0)

    def test_net_charge(self):
        self.assertEqual(m.net_charge("KKRDE"), 1)


class TestClusteringIsDeterministic(unittest.TestCase):
    """Clustering used to depend on Python's per-process string hashing; it must not any more."""

    def test_same_result_under_different_hash_seeds(self):
        code = (
            "import sys, random, hashlib; sys.path.insert(0, %r); import amp_baseline as m\n"
            "r = random.Random(0); base = [''.join(r.choices(m.AA, k=r.randint(10, 25))) for _ in range(200)]\n"
            "seqs = []\n"
            "for b in base:\n"
            "    for _ in range(6):\n"
            "        q = list(b); q[r.randrange(len(q))] = r.choice(m.AA); seqs.append(''.join(q))\n"
            "print(hashlib.md5(str(m.cluster_seqs(seqs)).encode()).hexdigest())\n" % ROOT
        )
        outs = set()
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                    text=True, check=True).stdout.strip())
        self.assertEqual(len(outs), 1)


class TestTop100Selection(unittest.TestCase):
    def setUp(self):
        r = random.Random(1)
        alphabet = "KRLIVFAGSTQ"
        base = "KWKSFLKTFKSALKTVLHTALKAISS"

        def variant():
            q = list(base)
            for pos in r.sample(range(len(q)), 9):
                q[pos] = r.choice(alphabet)
            return "".join(q)

        self.series = list(dict.fromkeys(variant() for _ in range(400)))[:200]
        self.others = ["".join(r.choices(alphabet + "WPYHN", k=r.randint(12, 28))) for _ in range(800)]

    def pick(self, n_others, cap):
        cand = self.series + self.others[:n_others]       # the series holds the best-scored candidates
        chosen, note = m.pick_top(cand, list(range(len(cand))), 100, 3, 0.7, None, 0, cap, 0.6)
        return [cand[i] for i in chosen], note

    def test_family_cap_limits_the_largest_family(self):
        picks, note = self.pick(800, 10)
        self.assertEqual(len(picks), 100)
        self.assertLessEqual(m.family_sizes(picks, 0.6)[0], 10)
        self.assertIsNone(note)

    def test_without_cap_one_series_dominates(self):
        picks, _ = self.pick(800, 0)
        self.assertGreater(m.family_sizes(picks, 0.6)[0], 50)

    def test_relaxes_with_a_warning_when_candidates_run_out(self):
        picks, note = self.pick(40, 10)
        self.assertIn("WARNING", note)


class TestOrganismSelectionAndRanker(unittest.TestCase):
    def make_csv(self):
        r = random.Random(3)
        rows = []
        for _ in range(600):
            q = "".join(r.choices("KKRRLLIVFAGST", k=r.randint(10, 25)))
            rows.append((q, "E. coli", 2 ** r.uniform(0, 7)))
            rows.append((q, "S. aureus", 2 ** r.uniform(0, 7)))
        path = os.path.join(tempfile.mkdtemp(), "amps.csv")
        pd.DataFrame(rows, columns=["sequence", "bacterium", "mic_um"]).to_csv(path, index=False)
        return path

    def test_organism_filter_keeps_only_matching_rows(self):
        df, rows = m.load_data(self.make_csv(), 8, 50)
        sub = m.select_organism(rows, "e. coli,escherichia coli")
        self.assertEqual(len(sub), rows[rows.organism == "e. coli"].sequence.nunique())

    def test_unknown_organism_exits_with_a_helpful_message(self):
        df, rows = m.load_data(self.make_csv(), 8, 50)
        with self.assertRaises(SystemExit) as ctx:
            m.select_organism(rows, "pseudomonas")
        self.assertIn("Most common values", str(ctx.exception))

    def test_ranker_scores_are_exactly_repeatable(self):
        df, rows = m.load_data(self.make_csv(), 8, 50)
        sub = m.select_organism(rows, "e. coli")
        seqs = ["".join(random.Random(i).choices("KKRRLLIVFAGST", k=15)) for i in range(300)]
        a = m.score(seqs, m.fit_ranker(sub, 0)[0])
        b = m.score(seqs, m.fit_ranker(sub, 0)[0])
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
