"""Focused checks for dataset-level NvvMix concurrency planning."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.alternatives.step_4_metadata_multiple import (
    build_candidate_mixture,
    fill_speaker_region,
    natural_speaker_count_ratios,
    plan_dataset_trajectories,
    run_metadata_dir,
    summarize_dataset_overlap,
    validate_source_activity,
)
from pipeline.step_4_metadata import ActiveSegment, SpeakerGroup


def source_groups(count: int) -> list[SpeakerGroup]:
    return [
        SpeakerGroup(
            str(index),
            [ActiveSegment(str(index), f"/{index}.wav", str(index), str(index), 0, 0.0, 100.0)],
            100.0,
        )
        for index in range(count)
    ]


class DatasetTrajectoryTest(unittest.TestCase):
    def test_exact_global_quotas_and_rare_mixtures(self) -> None:
        plans, quotas = plan_dataset_trajectories(
            [1000] * 1000, natural_speaker_count_ratios(4),
            4, 25, 250, 1, 3, random.Random(42),
        )
        actual = {
            order: sum(event["duration_frames"] for plan in plans for event in plan
                       if event["num_active"] == order)
            for order in quotas
        }
        self.assertEqual(actual, quotas)
        peak_positions = []
        reversals = {3: 0, 4: 0}
        repeated_peak_reversals = 0
        sustained_ascents = 0
        for plan in plans:
            self.assertEqual((plan[0]["num_active"], plan[-1]["num_active"]), (1, 1))
            self.assertTrue(all(abs(left["num_active"] - right["num_active"]) <= 1
                                for left, right in zip(plan, plan[1:])))
            orders = [event["num_active"] for event in plan]
            for level in reversals:
                reversals[level] += sum(
                    orders[index:index + 3] == [level, level - 1, level]
                    for index in range(len(orders) - 2)
                )
            repeated_peak_reversals += sum(
                orders[index:index + 5] == [4, 3, 4, 3, 4]
                for index in range(len(orders) - 4)
            )
            cursor = 0
            for index, event in enumerate(plan):
                if event["num_active"] == 4:
                    peak_positions.append((cursor + event["duration_frames"] / 2) / 1000)
                    ascent = plan[:index]
                    last_single = max(
                        position for position, previous in enumerate(ascent)
                        if previous["num_active"] == 1
                    )
                    if any(previous["event_type"] == "normal" and previous["num_active"] > 1
                           for previous in ascent[last_single + 1:]):
                        sustained_ascents += 1
                cursor += event["duration_frames"]
        self.assertLess(sum(any(event["num_active"] == 4 for event in plan) for plan in plans), 200)
        self.assertGreater(len(peak_positions), 0)
        self.assertLess(min(peak_positions), 0.5)
        self.assertGreater(max(peak_positions), 0.5)
        self.assertGreater(reversals[3], 0)
        self.assertGreater(reversals[4], 0)
        self.assertGreater(repeated_peak_reversals, 0)
        self.assertEqual(sustained_ascents, len(peak_positions))

    def test_generic_five_speaker_bridges_and_reversals(self) -> None:
        plans, _ = plan_dataset_trajectories(
            [1000] * 100, [0.70, 0.10, 0.08, 0.07, 0.05],
            6, 25, 250, 1, 3, random.Random(42),
        )
        plan = next(plan for plan in plans if any(event["num_active"] == 5 for event in plan))
        for level in (3, 4, 5):
            self.assertTrue(any(
                [event["num_active"] for event in session[index:index + 3]]
                == [level, level - 1, level]
                for session in plans for index in range(len(session) - 2)
            ))
        self.assertTrue(any(event["num_active"] == 5 and event["event_type"] == "normal"
                            for event in plan))
        self.assertTrue(all(event["duration_frames"] >= 25
                            for session in plans for event in session
                            if event["event_type"] == "normal"))
        transition_lengths = {
            event["duration_frames"] for session in plans for event in session
            if event["event_type"] == "transition" and event["num_active"] > 1
        }
        self.assertGreater(len(transition_lengths), 1)
        record = build_candidate_mixture(
            "", source_groups(6), random.Random(42), [0.70, 0.10, 0.08, 0.07, 0.05],
            20_000, 20_000, max_speakers=6, max_speakers_per_frame=5,
            planned_events=plan,
        )
        self.assertEqual(record["speaker_count"], 6)
        for previous, current in zip(record["events"], record["events"][1:]):
            before, after = set(previous["speakers"]), set(current["speakers"])
            delta = current["num_active"] - previous["num_active"]
            if delta == 1:
                self.assertTrue(before < after)
            elif delta == -1:
                self.assertTrue(after < before)
            else:
                self.assertEqual(before, after)

    def test_distinct_speakers_are_independent_of_frame_limit(self) -> None:
        inputs = ([1000] * 40, [0.8, 0.2], 5, 25, 250, 1, 3)
        plans, quotas = plan_dataset_trajectories(*inputs, random.Random(42))
        self.assertEqual((plans, quotas), plan_dataset_trajectories(*inputs, random.Random(42)))
        self.assertNotEqual(plans, plan_dataset_trajectories(*inputs, random.Random(43))[0])
        record = build_candidate_mixture(
            "", source_groups(5), random.Random(42), [0.8, 0.2], 20_000, 20_000,
            max_speakers=5, max_speakers_per_frame=2, planned_events=plans[0],
        )
        self.assertEqual(record["speaker_count"], 5)
        self.assertEqual(len({source["speaker"] for source in record["sources"]}), 5)
        self.assertLessEqual(max(event["num_active"] for event in record["events"]), 2)
        validate_source_activity(record["speaker_regions"], record["sources"])

    def test_zero_supported_ratio_and_impossible_configuration(self) -> None:
        plans, quotas = plan_dataset_trajectories(
            [1000] * 20, [0.8, 0.2, 0.0, 0.0], 4, 25, 250, 1, 3, random.Random(42),
        )
        summary = summarize_dataset_overlap([
            {"frame_resolution": 0.02,
             "target_frame_quota": {str(order): quotas[order] for order in quotas},
             "events": plan}
            for plan in plans
        ])
        self.assertEqual((summary[3]["frames"], summary[4]["frames"]), (0, 0))
        with self.assertRaisesRegex(ValueError, "two-speaker quota"):
            plan_dataset_trajectories([1000], [1.0, 0.0], 5, 25, 250, 1, 3, random.Random(42))
        with self.assertRaisesRegex(ValueError, "Transition frames"):
            plan_dataset_trajectories([1000], [0.8, 0.2], 2, 25, 250, 0, 3, random.Random(42))

    def test_hidden_source_gap_is_rejected(self) -> None:
        queue = [
            ActiveSegment("first", "/A.wav", "A", "A", 0, 0.0, 0.25),
            ActiveSegment("second", "/A.wav", "A", "A", 1, 0.25, 0.50),
        ]
        sources = []
        fill_speaker_region(queue, "A", 0.0, 0.5, sources, {}, random.Random(42))
        self.assertEqual(len(sources), 2)
        self.assertAlmostEqual(sources[0]["mix_end"], sources[1]["mix_start"])
        validate_source_activity([{"speaker": "A", "start": 0.0, "end": 0.5}], sources)

        sources = [
            {"speaker": "A", "mix_start": 0.0, "mix_end": 0.2, "orig_duration": 0.2},
            {"speaker": "A", "mix_start": 0.25, "mix_end": 0.5, "orig_duration": 0.25},
        ]
        with self.assertRaisesRegex(ValueError, "source gap"):
            validate_source_activity([{"speaker": "A", "start": 0.0, "end": 0.5}], sources)

    def test_runner_shuffles_anonymous_records_before_numbering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mix.json"
            args = argparse.Namespace(
                vad_dir=Path(directory), out=output, n_mixtures=50,
                min_segment_duration=500, min_mixture_duration=20_000,
                max_mixture_duration=20_000, frame_resolution=20,
                min_event_duration=500, max_event_duration=5000,
                min_transition_frames=1, max_transition_frames=3,
                seed=42, prefix="mix", max_speakers=4, max_speakers_per_frame=4,
                max_speakers_per_frame_ratio=["natural"],
                min_overlap_order_duration=500, min_duration_ratio_threshold=0.05,
                max_session_duration=60_000, force=True,
            )
            generated: list[list[tuple[int, int]]] = []

            def build_anonymously(*positional, **keywords):
                self.assertEqual(keywords["uri"], "")
                generated.append([
                    (event["num_active"], event["duration_frames"])
                    for event in keywords["planned_events"]
                ])
                return build_candidate_mixture(*positional, **keywords)

            with patch("pipeline.alternatives.step_4_metadata_multiple.collect_groups",
                       return_value=source_groups(4)), \
                 patch("pipeline.alternatives.step_4_metadata_multiple.build_candidate_mixture",
                       side_effect=build_anonymously), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_metadata_dir(args)
            first = json.loads(output.read_text())
            self.assertEqual([record["uri"] for record in first],
                             [f"mix_{index:08d}" for index in range(50)])
            self.assertEqual(len({record["generation_signature"] for record in first}), 1)
            self.assertNotEqual(generated, [
                [(event["num_active"], event["duration_frames"]) for event in record["events"]]
                for record in first
            ])
            with patch("pipeline.alternatives.step_4_metadata_multiple.collect_groups",
                       return_value=source_groups(4)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_metadata_dir(args)
            self.assertEqual(first, json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
