#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pprint
import re
import sys
from collections import Counter
from typing import Any

import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_search_env import QASearchEnv

TASK_NAME = "qa_search"
STOP_STRINGS = ["</search>"]


def _install_nemolab_memory_tracker_workaround() -> None:
    """Keep optional Ray diagnostics from aborting an otherwise healthy run."""
    if not os.environ.get("NEMOLAB_ENABLED"):
        return

    from nemo_rl.utils.memory_tracker import MemoryTrackerDataPoint

    def get_snapshot_str(self) -> str:
        return (
            f"Driver CPU memory tracker for {self.stage}:\n"
            f"- Mem usage before                  "
            f"{self.memory_used_before_stage_gb:>7.2f} GB\n"
            f"- Mem usage after                   "
            f"{self.memory_used_after_stage_gb:>7.2f} GB\n"
            f"- Mem usage diff (after - before)   {self.mem_used_diff_gb:>+7.2f} GB\n"
            f"- New variables: {self.new_variables}\n"
            "- Ray memory snapshot skipped on NeMoLab"
        )

    MemoryTrackerDataPoint.get_snapshot_str = get_snapshot_str


def parse_args():
    parser = argparse.ArgumentParser(description="QA document-search GRPO training")
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("query") or not row.get("expected_answer"):
                raise ValueError(f"invalid QA row at {path}:{line_number}")
            rows.append(row)
    return rows


def _question_type(expected_answer: str) -> str:
    match = re.match(r"\s*\[([^]]+)]", expected_answer)
    return match.group(1) if match else "unknown"


class QAAgentDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        input_key: str,
        output_key: str,
        system_prompt: str | None,
    ) -> None:
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def type_counts(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    _question_type(str(row[self.output_key])) for row in self.rows
                ).items()
            )
        )

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected_answer = str(row[self.output_key])

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        ).strip()
        token_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected_answer,
                "query": query,
                "num_searches": 0,
                "num_turns": 0,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": STOP_STRINGS,
        }


def main() -> None:
    _install_nemolab_memory_tracker_workaround()
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("QA data directory is not configured")
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    if not os.path.isfile(train_path) or not os.path.isfile(val_path):
        raise FileNotFoundError(f"missing train.jsonl or val.jsonl under {data_dir}")

    dataset_options = {
        "input_key": str(data_cfg.get("input_key", "query")),
        "output_key": str(data_cfg.get("output_key", "expected_answer")),
        "system_prompt": data_cfg.get("system_prompt") or None,
    }
    train_dataset = QAAgentDataset(train_path, tokenizer, **dataset_options)
    val_dataset = QAAgentDataset(val_path, tokenizer, **dataset_options)
    print(
        "QA data:",
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_types": train_dataset.type_counts,
            "val_types": val_dataset.type_counts,
        },
        flush=True,
    )

    train_env_cfg = dict(config.env[TASK_NAME]["cfg"])
    val_env_cfg = dict(train_env_cfg)
    val_env_cfg["first_search_reward"] = 0.0
    train_env = QASearchEnv.options(num_gpus=0).remote(cfg=train_env_cfg)
    val_env = QASearchEnv.options(num_gpus=0).remote(cfg=val_env_cfg)
    print("Document index:", ray.get(train_env.get_stats.remote()), flush=True)
    train_task_to_env = {TASK_NAME: train_env}
    val_task_to_env = {TASK_NAME: val_env}

    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        train_task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
