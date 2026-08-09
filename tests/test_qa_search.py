from __future__ import annotations

from common.environments.qa_search import (
    MarkdownSearchIndex,
    QASearchRunner,
    extract_search_query,
    resolve_search_query,
)


def test_chinese_document_ranking(tmp_path):
    (tmp_path / "implant.md").write_text(
        "# 离子注入系统\n离子注入系统由离子源、分析磁场、加速管、扫描器、法拉第和反应室组成。",
        encoding="utf-8",
    )
    (tmp_path / "server.md").write_text(
        "# Server Room\nServer Room 通过 SQL Server 与 Clean Room 连接。",
        encoding="utf-8",
    )
    index = MarkdownSearchIndex(tmp_path, chunk_chars=200, overlap_chars=20)

    hits = index.search("离子注入系统组成", top_k=2)

    assert hits
    assert hits[0].chunk.source == "implant.md"
    assert index.stats["files"] == 2


def test_result_format_respects_budget(tmp_path):
    (tmp_path / "long.md").write_text("# Test\n" + "技术资料" * 300, encoding="utf-8")
    index = MarkdownSearchIndex(tmp_path, chunk_chars=300, overlap_chars=20)
    hits = index.search("技术资料", top_k=3)

    result = index.format_results("技术资料", hits, max_chars=500)

    assert result.startswith("<search_results>")
    assert result.endswith("</search_results>")
    assert len(result) <= 520


def test_extracts_last_search_query():
    text = "<search>旧关键词</search>\n继续思考\n<search>离子注入 系统组成</search>"
    assert extract_search_query(text) == "离子注入 系统组成"


def test_placeholder_search_falls_back_to_question_stem():
    question = (
        "下面是一道填空题。\n\n"
        "题目：SERVER ROOM 通过【1】与Clean room进行连接\n\n"
        "请作答。"
    )

    query = resolve_search_query("简洁关键词", question)

    assert query == "SERVER ROOM 通过【1】与Clean room进行连接"
    assert "简洁关键词" not in query


def test_specific_search_is_anchored_to_question():
    query = resolve_search_query(
        "污染风险上报",
        "题目：当 MO case 发生时应如何处理？\n\n选项：\nA. 立即上报",
    )

    assert query.startswith("污染风险上报 ")
    assert "MO case" in query


def test_runner_search_then_answer(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# 控制图\nExclude 功能可以永久排除 Sample。", encoding="utf-8"
    )
    index = MarkdownSearchIndex(tmp_path, chunk_chars=200, overlap_chars=20)

    def reward_fn(queries, completions, expected_answers):
        return [1.0 if "\\boxed{A}" in completions[0] else 0.0]

    def extract_boxed(text):
        return "A" if "\\boxed{A}" in text else None

    runner = QASearchRunner(
        index,
        reward_fn=reward_fn,
        boxed_extractor=extract_boxed,
        max_searches=2,
        max_turns=3,
    )
    metadata = {
        "query": "哪个功能可以永久排除 Sample？",
        "expected_answer": "[single] A",
        "num_searches": 0,
        "num_turns": 0,
    }

    search_result = runner.process_turn(
        [{"role": "assistant", "content": "<search>永久排除 Sample</search>"}],
        metadata,
    )
    assert search_result[2] is False
    assert "Exclude" in search_result[0]["content"]
    assert search_result[4]["num_searches"] == 1

    answer_result = runner.process_turn(
        [{"role": "assistant", "content": "因此选择 A。\\boxed{A}"}],
        search_result[4],
    )
    assert answer_result[1] == 1.0
    assert answer_result[2] is True
    assert answer_result[5] == "A"


def test_runner_recovers_placeholder_and_allows_final_turn(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Server Room\nServer Room 通过 SQL Server 与 Clean Room 连接。",
        encoding="utf-8",
    )
    index = MarkdownSearchIndex(tmp_path, chunk_chars=200, overlap_chars=20)
    runner = QASearchRunner(
        index,
        reward_fn=lambda *_args: [1.0],
        boxed_extractor=lambda text: "SQL Server" if "\\boxed" in text else None,
        max_searches=2,
        max_turns=4,
    )
    metadata = {
        "query": "题目：SERVER ROOM 通过【1】与Clean room进行连接",
        "expected_answer": "[fill] SQL Server",
        "num_searches": 0,
        "num_turns": 1,
    }

    first_search = runner.process_turn(
        [{"role": "assistant", "content": "<search>简洁关键词</search>"}],
        metadata,
    )
    assert "SQL Server" in first_search[0]["content"]
    assert "系统已改用题目内容检索" in first_search[0]["content"]

    second_search = runner.process_turn(
        [{"role": "assistant", "content": "<search>Server Room 连接</search>"}],
        first_search[4],
    )
    assert second_search[2] is False

    final_answer = runner.process_turn(
        [{"role": "assistant", "content": "\\boxed{SQL Server}"}],
        second_search[4],
    )
    assert final_answer[1] == 1.0
    assert final_answer[2] is True


def test_runner_penalizes_missing_final_answer_on_last_turn(tmp_path):
    (tmp_path / "guide.md").write_text("# Test\n测试资料", encoding="utf-8")
    index = MarkdownSearchIndex(tmp_path, chunk_chars=200, overlap_chars=20)
    runner = QASearchRunner(
        index,
        reward_fn=lambda *_args: [0.0],
        boxed_extractor=lambda _text: None,
        max_turns=1,
        format_penalty=-0.5,
    )
    metadata = {
        "query": "test",
        "expected_answer": "[single] A",
        "num_searches": 0,
        "num_turns": 0,
    }

    result = runner.process_turn(
        [{"role": "assistant", "content": "没有按协议输出"}], metadata
    )

    assert result[1] == -0.5
    assert result[2] is True
