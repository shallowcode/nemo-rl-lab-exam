from __future__ import annotations

import os
import re
from typing import Any, TypedDict

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from common.environments.qa_search import MarkdownSearchIndex, QASearchRunner


class QASearchMetadata(TypedDict):
    expected_answer: str
    query: str
    num_searches: int
    num_turns: int


@ray.remote  # pragma: no cover
class QASearchEnv(EnvironmentInterface[QASearchMetadata]):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        docs_dir = os.environ.get("QA_DOCS_DIR") or cfg.get("docs_dir", "/data/docs")
        self.index = MarkdownSearchIndex(
            docs_dir,
            chunk_chars=int(cfg.get("chunk_chars", 1000)),
            overlap_chars=int(cfg.get("overlap_chars", 150)),
            max_file_bytes=int(cfg.get("max_file_bytes", 2_000_000)),
            k1=float(cfg.get("k1", 1.5)),
            b=float(cfg.get("b", 0.75)),
        )

        if bool(cfg.get("use_judge", False)):
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            reward_fn = qa_rule_reward_fn
        from common.rewards.qa_reward import FORMAT_PENALTY, extract_boxed

        self.runner = QASearchRunner(
            self.index,
            reward_fn=reward_fn,
            boxed_extractor=extract_boxed,
            top_k=int(cfg.get("top_k", 3)),
            max_result_chars=int(cfg.get("max_result_chars", 2600)),
            max_searches=int(cfg.get("max_searches", 2)),
            max_turns=int(cfg.get("max_turns", 3)),
            format_penalty=float(cfg.get("format_penalty", FORMAT_PENALTY)),
            first_search_reward=float(cfg.get("first_search_reward", 0.0)),
        )

    def get_stats(self) -> dict[str, int | str]:
        return self.index.stats

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QASearchMetadata],
    ) -> EnvironmentReturn[QASearchMetadata]:
        results = [
            self.runner.process_turn(log, item)
            for log, item in zip(message_log_batch, metadata, strict=False)
        ]
        observations, rewards, terminateds, stops, next_metadata, answers = zip(
            *results, strict=False
        )
        return EnvironmentReturn(
            observations=list(observations),
            metadata=list(next_metadata),
            next_stop_strings=list(stops),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=list(answers),
        )

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, float]]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]), dtype=torch.float32)
        ).float()
        logs = batch.get("message_log", [])
        search_counts = []
        for log in logs:
            assistant_text = "\n".join(
                str(message.get("content", ""))
                for message in log
                if message.get("role") == "assistant"
            )
            search_counts.append(len(re.findall(r"<search>", assistant_text, re.IGNORECASE)))

        if len(rewards) == 0:
            return batch, {}
        search_tensor = torch.tensor(search_counts, dtype=torch.float32)
        metrics = {
            "qa_mean_reward": rewards.mean().item(),
            "qa_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_format_penalty_rate": (rewards < 0).float().mean().item(),
            "qa_search_usage_rate": (search_tensor > 0).float().mean().item(),
            "qa_avg_searches": search_tensor.mean().item(),
        }
        return batch, metrics
