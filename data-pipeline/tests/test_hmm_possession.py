import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.hmm_possession import FEATURES, _sequences, _viterbi, add_hmm_possession

PARAMS = {
    "self_transition": 0.99,
    "n_iter": 40,
    "tol": 0.01,
    "min_covar": 0.01,
    "random_state": 0,
    "min_samples": 100,
    "min_anchor_samples": 10,
}


def sample():
    rng = np.random.default_rng(42)
    centres = np.repeat([[1, 8, 2, 2], [8, 1, 2, 2], [5, 6, 0.3, 0.3]], 100, axis=0)
    values = np.maximum(centres + rng.normal(0, 0.15, centres.shape), 0.01)
    df = pd.DataFrame(values, columns=[
        "min_team_a_dist", "min_team_b_dist",
        "team_centroid_speed_barcelona", "team_centroid_speed_atletic_club",
    ])
    df["period_id"] = 1
    df["video_frame"] = np.arange(len(df))
    df["sp_kind"] = [None] * 200 + ["free_kick"] * 100
    for feature in FEATURES:
        if feature not in df:
            df[feature] = 0.0
    df["possession_advantage"] = df["min_team_b_dist"] - df["min_team_a_dist"]
    df.loc[45:49, list(FEATURES)] = np.nan
    df.loc[40, "ball_dx"] = np.nan
    return df


class HMMTests(unittest.TestCase):
    def test_viterbi_matches_library_and_marginalizes_missing_frames(self):
        model = GaussianHMM(n_components=2, covariance_type="diag", init_params="")
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])
        model.means_ = np.array([[0.0], [5.0]])
        model.covars_ = np.array([[1.0], [1.0]])
        x = np.array([[0.0], [1.0], [5.0], [5.0]])
        np.testing.assert_array_equal(_viterbi(model, x, np.ones(4, dtype=bool)), model.predict(x))
        invalid = np.array([True, False, False, True])
        changed = x.copy()
        changed[~invalid] = 1000
        np.testing.assert_array_equal(_viterbi(model, x, invalid), _viterbi(model, changed, invalid))

    def test_sequence_boundaries(self):
        df = pd.DataFrame({"period_id": [1, 1, 1, 2, 2], "video_frame": [1, 2, 5, 6, 7]})
        self.assertEqual([x.tolist() for x in _sequences(df)], [[0, 1], [2], [3, 4]])

    def test_training_preserves_order_input_and_missing_abstentions(self):
        df = sample().sample(frac=1, random_state=5)
        original = df.copy(deep=True)
        result = add_hmm_possession(df, PARAMS)
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(result.drop(columns="hmm_possession"), original)
        missing_frames = df.loc[df.loc[:, FEATURES].isna().all(axis=1), "video_frame"]
        missing_labels = result.loc[
            result.video_frame.isin(missing_frames), "hmm_possession"
        ]
        self.assertTrue((missing_labels == "X").all())
        self.assertNotEqual(result.loc[result.video_frame == 40, "hmm_possession"].iloc[0], "X")
        observed_labels = result.loc[
            ~result.video_frame.isin(missing_frames), "hmm_possession"
        ]
        self.assertFalse((observed_labels == "X").any())
        for start, label in [(0, "H"), (100, "A"), (200, "D")]:
            section = result[(result.video_frame >= start) & (result.video_frame < start + 100)]
            self.assertGreater((section.hmm_possession == label).sum(), 80)
        pd.testing.assert_series_equal(
            result.hmm_possession, add_hmm_possession(df, PARAMS).hmm_possession
        )

    def test_rejects_bad_input_and_insufficient_anchors(self):
        df = sample()
        with self.assertRaisesRegex(ValueError, "missing required"):
            add_hmm_possession(df.drop(columns="sp_kind"), PARAMS)
        with self.assertRaisesRegex(ValueError, "unique"):
            add_hmm_possession(pd.concat([df, df.iloc[:1]]), PARAMS)
        df["sp_kind"] = None
        with self.assertRaisesRegex(ValueError, "initialization"):
            add_hmm_possession(df, PARAMS)


if __name__ == "__main__":
    unittest.main()
