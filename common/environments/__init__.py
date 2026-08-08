"""跨实验复用的自定义 GRPO 环境。"""

from typing import Any

__all__ = ["QARewardEnv"]


def __getattr__(name: str) -> Any:
    if name == "QARewardEnv":
        from common.environments.qa_env import QARewardEnv

        return QARewardEnv
    raise AttributeError(name)
