import os
from pathlib import Path
from typing import List, Optional

import attrs as attrs
from attr import field

from classes.classes import ModelConfig, ModelType


@attrs.define(frozen=True, slots=False)
class WorkloadRuns:
    train_workload_runs: List[Path] = field(default=None)
    test_workload_runs: Optional[List[Path]] = field(default=None)
    target_test_csv_paths: List[Path] = []
    split_manifest: Optional[Path] = None
    alignment_manifest: Optional[Path] = None
    experiment_protocol: str = "legacy"

    def update_test_workloads(self, target_dir: Path, seed: int) -> None:
        if self.test_workload_runs:
            for test_path in self.test_workload_runs:
                test_workload = os.path.basename(test_path).replace('.json', '')
                self.target_test_csv_paths.append(Path(target_dir) / f'{test_workload}_{seed}')
        else:
            # When no test paths are given, this is a workload driven model,
            # and we use the training workload as test workload
            for test_path in self.train_workload_runs:
                test_workload = os.path.basename(test_path).replace('.json', '')
                self.target_test_csv_paths.append(Path(target_dir) / f'{test_workload}_{seed}')

    def check_if_done(self, model_name: str) -> bool:
        if all([os.path.exists(p) for p in self.target_test_csv_paths]) and self.target_test_csv_paths:
            short_paths = {}
            for path in self.target_test_csv_paths:
                key = (path.parts[-3], path.parts[-2])
                value = path.parts[-1]
                short_paths.setdefault(key, []).append(value)
            print(
                f"{'Model '}{model_name:<16}{' was already trained and evaluated for '}{str(len(self.target_test_csv_paths))}{' queries: '}{str(short_paths)}")
            return True
        return False

    def check_model_compability(self, model_config: ModelConfig, mode: str):
        if mode == "train":
            if model_config.type == ModelType.WL_DRIVEN:
                assert len(self.train_workload_runs) == 1, "One training workload run is supported for workload driven models"
                assert len(self.test_workload_runs) == 0, "No test workload runs are allowed for workload driven models, as the test workload is the same as the training workload"

            if model_config.type == ModelType.WL_AGNOSTIC:
                assert len(self.train_workload_runs) >= 1, "Model need to be trained more than 1 database"
                assert len(self.test_workload_runs) == 1, "Only one test workload run is supported for workload agnostic models"
        return
