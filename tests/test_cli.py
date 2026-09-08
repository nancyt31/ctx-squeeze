import io
import json

import pytest

from ctx_squeeze.cli import main


def test_main_document_mode_writes_squeezed_text_to_stdout(tmp_path, capsys):
    path = tmp_path / "doc.txt"
    path.write_text("first paragraph about the release\n\nsecond paragraph about the rollout")

    exit_code = main([str(path), "--budget", "1000"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out == "first paragraph about the release\n\nsecond paragraph about the rollout\n"


def test_main_messages_mode_writes_json_array_to_stdout(tmp_path, capsys):
    history = [{"role": "user", "content": "hello there"}]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history))

    exit_code = main([str(path), "--budget", "1000", "--messages"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == history


def test_main_document_mode_stats_go_to_stderr(tmp_path, capsys):
    path = tmp_path / "doc.txt"
    path.write_text("only paragraph in the document")

    main([str(path), "--budget", "1000", "--stats"])

    err = capsys.readouterr().err
    assert "kept 1 of 1 segments" in err


def test_main_messages_mode_stats_go_to_stderr(tmp_path, capsys):
    history = [{"role": "user", "content": "hi"}]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history))

    main([str(path), "--budget", "1000", "--messages", "--stats"])

    err = capsys.readouterr().err
    assert "kept 1 of 1 messages" in err


def test_main_document_mode_json_flag_emits_report(tmp_path, capsys):
    path = tmp_path / "doc.txt"
    path.write_text("only paragraph in the document")

    main([str(path), "--budget", "1000", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["text"] == "only paragraph in the document"
    assert report["segments_in"] == 1
    assert report["segments_out"] == 1


def test_main_messages_mode_json_flag_emits_report(tmp_path, capsys):
    history = [{"role": "user", "content": "hi"}]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history))

    main([str(path), "--budget", "1000", "--messages", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert report["messages"] == history
    assert report["pinned_tool_results"] == []


def test_main_reads_from_stdin_when_path_is_dash(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("a document read from stdin"))

    main(["-", "--budget", "1000"])

    out = capsys.readouterr().out
    assert out == "a document read from stdin\n"


def test_main_writes_to_output_file_when_o_given(tmp_path, capsys):
    src = tmp_path / "doc.txt"
    src.write_text("a document written to a file")
    dest = tmp_path / "out.txt"

    exit_code = main([str(src), "--budget", "1000", "-o", str(dest)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert dest.read_text() == "a document written to a file\n"


def test_main_missing_input_file_exits_with_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises(SystemExit):
        main([str(missing), "--budget", "1000"])

    assert "No such file" in capsys.readouterr().err


def test_main_messages_mode_rejects_non_array_json(tmp_path, capsys):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"role": "user", "content": "not a list"}))

    with pytest.raises(SystemExit):
        main([str(path), "--budget", "1000", "--messages"])

    assert "expects a JSON array" in capsys.readouterr().err
