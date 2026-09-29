# our realisation in simple_evol

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Mapping

import numpy as np

from . import code_manipulation
from .funsearch_utils import make_signature, reduce_score
from .prompt_builder import build_prompt_code

ScoresPerTest = Mapping[Any, float]


@dataclasses.dataclass(frozen=True)
class Prompt:
    code: str
    version_generated: int
    island_id: int


@dataclasses.dataclass
class ProgramRecord:
    function: code_manipulation.Function
    score: float
    signature: tuple[float, ...]
    scores_per_test: dict[str, float]
    island_id: int
    source_path: str | None = None
    raw_code: str | None = None
    experiment_id: int | None = None


class Cluster:
    def __init__(self, score: float, record: ProgramRecord):
        self.score = score
        self.signature = record.signature
        self.records: list[ProgramRecord] = [record]

    def add(self, record: ProgramRecord) -> None:
        self.records.append(record)

    def sample_program(self, rng: np.random.Generator) -> ProgramRecord:
        lengths = np.array([len(str(r.function)) for r in self.records], dtype=float)
        norm = lengths - lengths.min()
        probs = np.exp(-norm)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(self.records), p=probs))
        return self.records[idx]


class Island:
    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
        rng: np.random.Generator,
    ):
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.functions_per_prompt = functions_per_prompt
        self.cluster_sampling_temperature_init = max(cluster_sampling_temperature_init, 1e-6)
        self.cluster_sampling_temperature_period = max(int(cluster_sampling_temperature_period), 1)
        self.rng = rng
        self.clusters: dict[tuple[float, ...], Cluster] = {}
        self.num_registered = 0

    def register_program(self, record: ProgramRecord) -> None:
        cluster = self.clusters.get(record.signature)
        if cluster is None:
            self.clusters[record.signature] = Cluster(record.score, record)
        else:
            cluster.add(record)
        self.num_registered += 1

    def _current_temperature(self) -> float:
        phase = self.num_registered % self.cluster_sampling_temperature_period
        temperature = self.cluster_sampling_temperature_init * (
            1.0 - phase / self.cluster_sampling_temperature_period
        )
        return max(temperature, 1e-6)

    def _sample_clusters(self, k: int) -> list[Cluster]:
        clusters = list(self.clusters.values())
        if not clusters:
            raise ValueError("Cannot sample from empty island.")
        scores = np.array([c.score for c in clusters], dtype=float)
        temp = self._current_temperature()
        logits = scores - scores.max()
        probs = np.exp(logits / temp)
        probs = probs / probs.sum()
        k = min(len(clusters), k)
        idxs = self.rng.choice(len(clusters), size=k, replace=True, p=probs)
        return [clusters[int(i)] for i in np.atleast_1d(idxs)]

    def get_prompt(self) -> tuple[str, int]:
        sampled_clusters = self._sample_clusters(self.functions_per_prompt)
        sampled_records = [cluster.sample_program(self.rng) for cluster in sampled_clusters]
        sampled_records.sort(key=lambda r: r.score, reverse=True)
        sampled_functions = [copy.deepcopy(r.function) for r in sampled_records]
        return build_prompt_code(self.template, sampled_functions, self.function_to_evolve)

    def get_best_record(self) -> ProgramRecord | None:
        if not self.clusters:
            return None
        best_cluster = max(self.clusters.values(), key=lambda c: c.score)
        return max(best_cluster.records, key=lambda r: r.score)

    def clear(self) -> None:
        self.clusters.clear()
        self.num_registered = 0


class ProgramsDatabase:
    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        num_islands: int = 8,
        functions_per_prompt: int = 2,
        cluster_sampling_temperature_init: float = 0.1,
        cluster_sampling_temperature_period: int = 30,
        obj_type: str = "min",
        use_signature_clustering: bool = True,
        seed: int = 0,
    ):
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.num_islands = int(num_islands)
        self.obj_type = obj_type
        self.use_signature_clustering = use_signature_clustering
        self.rng = np.random.default_rng(seed)
        self.islands = [
            Island(
                template=template,
                function_to_evolve=function_to_evolve,
                functions_per_prompt=functions_per_prompt,
                cluster_sampling_temperature_init=cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=cluster_sampling_temperature_period,
                rng=self.rng,
            )
            for _ in range(self.num_islands)
        ]

    def register_program(
        self,
        function: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
        source_path: str | None = None,
        raw_code: str | None = None,
        experiment_id: int | None = None,
    ) -> ProgramRecord:
        signature = make_signature(scores_per_test) if self.use_signature_clustering else tuple([round(float(list(scores_per_test.values())[0]), 8)])
        score = reduce_score(scores_per_test, obj_type=self.obj_type)
        record = ProgramRecord(
            function=copy.deepcopy(function),
            score=score,
            signature=signature,
            scores_per_test={str(k): float(v) for k, v in scores_per_test.items()},
            island_id=island_id,
            source_path=source_path,
            raw_code=raw_code,
            experiment_id=experiment_id,
        )
        self.islands[island_id].register_program(record)
        return record

    def get_prompt(self) -> Prompt:
        non_empty = [i for i, island in enumerate(self.islands) if island.clusters]
        if not non_empty:
            raise ValueError("ProgramsDatabase is empty.")
        island_id = int(self.rng.choice(non_empty))
        code, version_generated = self.islands[island_id].get_prompt()
        return Prompt(code=code, version_generated=version_generated, island_id=island_id)

    def get_best_record(self) -> ProgramRecord | None:
        best = None
        for island in self.islands:
            rec = island.get_best_record()
            if rec is None:
                continue
            if best is None or rec.score > best.score:
                best = rec
        return best

    def reset_islands(self, keep_ratio: float = 0.5) -> None:
        keep_ratio = min(max(keep_ratio, 0.0), 1.0)

        ranked: list[tuple[int, float]] = []
        for i, island in enumerate(self.islands):
            rec = island.get_best_record()
            score = -float("inf") if rec is None else rec.score
            noisy_score = score + float(self.rng.normal(0.0, 1e-6))
            ranked.append((i, noisy_score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        keep_n = max(1, int(round(len(ranked) * keep_ratio)))
        survivors = ranked[:keep_n]
        weak = ranked[keep_n:]
        if not weak:
            return

        survivor_records = [self.islands[i].get_best_record() for i, _ in survivors]
        survivor_records = [r for r in survivor_records if r is not None]
        if not survivor_records:
            return

        for island_id, _ in weak:
            self.islands[island_id].clear()
            seed_record = copy.deepcopy(
                survivor_records[int(self.rng.integers(len(survivor_records)))]
            )
            seed_record.island_id = island_id
            self.islands[island_id].register_program(seed_record)
