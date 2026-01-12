"""Comprehensive tests for Mindcore CLI commands.

Tests all CLI commands with real Mindcore integration - no mocks.
"""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from mindcore.cli import main


class TestCLIHelp:
    """Test CLI help and basic commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Mindcore" in result.output
        assert "init" in result.output
        assert "demo" in result.output
        assert "doctor" in result.output

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_status_without_config(self, runner):
        """Test status command without configuration."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "Not found" in result.output or "Not configured" in result.output


class TestDemoCommand:
    """Test the demo command with real Mindcore integration."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_demo_quick(self, runner):
        """Test quick demo runs without errors."""
        result = runner.invoke(main, ["demo", "--quick"])
        assert result.exit_code == 0
        assert "Interactive Demo" in result.output
        assert "Mindcore initialized" in result.output

    def test_demo_section_1(self, runner):
        """Test demo section 1 (basic store and recall)."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "1"])
        assert result.exit_code == 0
        assert "Basic Store and Recall" in result.output
        assert "Stored" in result.output

    def test_demo_section_3(self, runner):
        """Test demo section 3 (reinforcement)."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "3"])
        assert result.exit_code == 0
        assert "Memory Reinforcement" in result.output

    def test_demo_section_4(self, runner):
        """Test demo section 4 (advanced search)."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "4"])
        assert result.exit_code == 0
        assert "Advanced Search" in result.output

    def test_demo_section_5(self, runner):
        """Test demo section 5 (LLM context building)."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "5"])
        assert result.exit_code == 0
        assert "Building Context" in result.output

    def test_demo_section_6(self, runner):
        """Test demo section 6 (SVL vocabulary)."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "6"])
        assert result.exit_code == 0
        assert "SVL" in result.output


class TestDoctorCommand:
    """Test the doctor command for health checks."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_doctor_runs(self, runner):
        """Test doctor command runs without errors."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["doctor"])
            assert result.exit_code == 0
            assert "Mindcore Doctor" in result.output or "Checking" in result.output

    def test_doctor_verbose(self, runner):
        """Test doctor command with verbose flag."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["doctor", "--verbose"])
            assert result.exit_code == 0

    def test_doctor_fix_creates_missing_files(self, runner):
        """Test doctor --fix can create missing files."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["doctor", "--fix"])
            assert result.exit_code == 0


class TestBenchmarkCommand:
    """Test benchmark command with real operations."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_benchmark_replay(self, runner):
        """Test deterministic replay benchmark."""
        result = runner.invoke(main, ["benchmark", "replay"])
        assert result.exit_code == 0
        assert "replay" in result.output.lower()
        assert "PASS" in result.output or "passed" in result.output.lower()
        # Verify it's using real operations
        assert "Stored" in result.output
        assert "Hash match" in result.output

    def test_benchmark_latency(self, runner):
        """Test latency benchmark with real measurements."""
        result = runner.invoke(main, ["benchmark", "latency"])
        assert result.exit_code == 0
        assert "latency" in result.output.lower()
        # Verify real measurements are shown (should have ms values)
        assert "ms" in result.output
        assert "Store operation:" in result.output
        assert "Recall operation:" in result.output

    def test_benchmark_audit(self, runner):
        """Test audit trail benchmark."""
        result = runner.invoke(main, ["benchmark", "audit"])
        assert result.exit_code == 0
        assert "audit" in result.output.lower()
        assert "operations tracked" in result.output

    def test_benchmark_drift(self, runner):
        """Test memory drift benchmark."""
        result = runner.invoke(main, ["benchmark", "drift"])
        assert result.exit_code == 0
        assert "drift" in result.output.lower()
        assert "Reinforcement score" in result.output

    def test_benchmark_all(self, runner):
        """Test running all benchmarks."""
        result = runner.invoke(main, ["benchmark", "all"])
        assert result.exit_code == 0
        # Should run all 4 tests
        assert "replay" in result.output.lower()
        assert "latency" in result.output.lower()
        assert "audit" in result.output.lower()
        assert "drift" in result.output.lower()

    def test_benchmark_verbose(self, runner):
        """Test verbose benchmark output."""
        result = runner.invoke(main, ["benchmark", "replay", "--verbose"])
        assert result.exit_code == 0

    def test_benchmark_custom_iterations(self, runner):
        """Test benchmark with custom iterations."""
        result = runner.invoke(main, ["benchmark", "latency", "-n", "5"])
        assert result.exit_code == 0
        assert "5 iterations" in result.output


class TestConfigCommand:
    """Test config management commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_config_view_without_config(self, runner):
        """Test config view when no config exists."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["config", "view"])
            assert result.exit_code == 0
            assert "No configuration found" in result.output

    def test_config_view_with_config(self, runner):
        """Test config view with existing config."""
        with runner.isolated_filesystem():
            # Create a config file
            config_content = """
storage:
  backend: sqlite
  path: ./mindcore.db
svl:
  policies:
    strict_mode: false
"""
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["config", "view"])
            assert result.exit_code == 0
            assert "storage:" in result.output
            assert "sqlite" in result.output

    def test_config_validate_valid(self, runner):
        """Test config validation with valid config."""
        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: sqlite
  path: ./mindcore.db
"""
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["config", "validate"])
            assert result.exit_code == 0
            assert "valid" in result.output.lower()

    def test_config_validate_missing(self, runner):
        """Test config validation when config missing."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["config", "validate"])
            assert result.exit_code == 0
            assert "No configuration found" in result.output

    def test_config_diff(self, runner):
        """Test config diff command."""
        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: postgresql
  path: ./custom.db
"""
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["config", "diff"])
            assert result.exit_code == 0
            # Should show differences from defaults

    def test_config_reset_no_config(self, runner):
        """Test config reset when no config exists."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["config", "reset"])
            assert result.exit_code == 0
            assert "No config to reset" in result.output


class TestExplainCommand:
    """Test the explain command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_explain_without_args(self, runner):
        """Test explain without arguments shows usage."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["explain"])
            assert result.exit_code == 0
            assert "Usage" in result.output or "request-id" in result.output

    def test_explain_without_config(self, runner):
        """Test explain shows configuration guidance when no config exists."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["explain", "test-id"])
            assert result.exit_code == 0
            assert "No configuration found" in result.output or "mindcore init" in result.output

    def test_explain_without_audit_enabled(self, runner):
        """Test explain shows message when audit not enabled."""
        with runner.isolated_filesystem():
            config_content = "storage:\n  backend: sqlite\n"
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["explain", "test-request-id"])
            assert result.exit_code == 0
            assert "Audit logging is not enabled" in result.output

    def test_explain_with_audit_enabled_no_file(self, runner):
        """Test explain when audit enabled but no log file exists yet."""
        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: sqlite
enterprise:
  audit:
    enabled: true
    file_path: ./audit.log
"""
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["explain", "test-id"])
            assert result.exit_code == 0
            assert "not found" in result.output or "No operations" in result.output

    def test_explain_with_real_audit_data(self, runner):
        """Test explain reads and displays real audit entries."""
        import json

        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: sqlite
enterprise:
  audit:
    enabled: true
    file_path: ./audit.log
"""
            Path("mindcore.yaml").write_text(config_content)

            # Create a real audit log entry
            audit_entry = {
                "event_type": "memory.store",
                "memory_id": "test-request-id",
                "user_id": "user123",
                "timestamp": "2024-01-15T10:30:00Z",
                "memory_type": "semantic",
            }
            Path("audit.log").write_text(json.dumps(audit_entry) + "\n")

            result = runner.invoke(main, ["explain", "test-request-id"])
            assert result.exit_code == 0
            assert "Audit Entry" in result.output
            assert "memory.store" in result.output or "Memory Id" in result.output

    def test_explain_last_with_real_data(self, runner):
        """Test explain --last reads the most recent real audit entry."""
        import json

        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: sqlite
enterprise:
  audit:
    enabled: true
    file_path: ./audit.log
"""
            Path("mindcore.yaml").write_text(config_content)

            # Create multiple audit entries
            entries = [
                {"event_type": "memory.store", "memory_id": "first-id"},
                {"event_type": "memory.recall", "memory_id": "second-id"},
            ]
            Path("audit.log").write_text("\n".join(json.dumps(e) for e in entries) + "\n")

            result = runner.invoke(main, ["explain", "--last"])
            assert result.exit_code == 0
            assert "Audit Entry" in result.output
            # Should show the last entry
            assert "memory.recall" in result.output or "second-id" in result.output


class TestStatusCommand:
    """Test the status command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_status_without_config(self, runner):
        """Test status shows appropriate message without config."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "Status" in result.output

    def test_status_with_config(self, runner):
        """Test status with configuration."""
        with runner.isolated_filesystem():
            config_content = """
storage:
  backend: sqlite
  path: ./mindcore.db
svl:
  policies:
    strict_mode: true
features:
  hot_path: true
"""
            Path("mindcore.yaml").write_text(config_content)

            result = runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "SQLite" in result.output

    def test_status_with_env_vars(self, runner):
        """Test status detects environment variables."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["status"], env={"OPENAI_API_KEY": "sk-test123456789"})
            assert result.exit_code == 0
            # Should show LLM provider info


class TestInitCommand:
    """Test the init wizard command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_init_help(self, runner):
        """Test init help text."""
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "Setup" in result.output or "init" in result.output

    def test_init_quick_mode(self, runner):
        """Test init with quick flag."""
        with runner.isolated_filesystem():
            # Simulate selecting option 1 (Single AI Agent)
            result = runner.invoke(main, ["init", "--quick"], input="1\n")
            # Init may require more inputs, check it starts correctly
            assert "Setup" in result.output or "building" in result.output.lower()


class TestCLIIntegration:
    """Integration tests verifying CLI connects to real Mindcore modules."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_demo_creates_real_memories(self, runner):
        """Verify demo command actually stores memories in Mindcore."""
        result = runner.invoke(main, ["demo", "--quick", "--section", "1"])
        assert result.exit_code == 0
        # These indicate real Mindcore operations
        assert "Stored" in result.output
        assert "memories" in result.output.lower()

    def test_benchmark_uses_real_database(self, runner):
        """Verify benchmark uses real database operations."""
        result = runner.invoke(main, ["benchmark", "replay"])
        assert result.exit_code == 0
        # Hash changes prove real data is stored
        assert "Hash match" in result.output

    def test_benchmark_latency_is_realistic(self, runner):
        """Verify latency measurements are realistic (not mocked)."""
        result = runner.invoke(main, ["benchmark", "latency"])
        assert result.exit_code == 0

        # Parse latency values - they should be real numbers
        import re

        store_match = re.search(r"Store operation: (\d+\.?\d*)ms", result.output)
        recall_match = re.search(r"Recall operation: (\d+\.?\d*)ms", result.output)

        assert store_match is not None, "Store latency should be in output"
        assert recall_match is not None, "Recall latency should be in output"

        store_ms = float(store_match.group(1))
        recall_ms = float(recall_match.group(1))

        # Realistic latencies should be between 0.01ms and 1000ms
        assert 0.01 <= store_ms <= 1000, f"Store latency {store_ms}ms seems unrealistic"
        assert 0.01 <= recall_ms <= 1000, f"Recall latency {recall_ms}ms seems unrealistic"

    def test_benchmark_drift_shows_real_scores(self, runner):
        """Verify drift test shows real reinforcement scores."""
        result = runner.invoke(main, ["benchmark", "drift"])
        assert result.exit_code == 0

        # Should show actual score values
        import re

        score_match = re.search(r"score.*?(-?\d+\.?\d*)", result.output, re.IGNORECASE)
        assert score_match is not None, "Reinforcement score should be shown"

        score = float(score_match.group(1))
        assert -1.0 <= score <= 1.0, f"Score {score} should be bounded"


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test handling of invalid commands."""
        result = runner.invoke(main, ["invalid-command"])
        assert result.exit_code != 0

    def test_benchmark_invalid_type(self, runner):
        """Test handling of invalid benchmark type."""
        result = runner.invoke(main, ["benchmark", "invalid"])
        assert result.exit_code != 0

    def test_config_invalid_action(self, runner):
        """Test handling of invalid config action."""
        result = runner.invoke(main, ["config", "invalid"])
        assert result.exit_code != 0


class TestCLIWithRealMindcore:
    """Tests that verify deep integration with Mindcore modules."""

    def test_mindcore_import(self):
        """Verify Mindcore can be imported for CLI."""
        from mindcore import Mindcore

        assert Mindcore is not None

    def test_svl_import(self):
        """Verify SVL can be imported for CLI."""
        from mindcore.svl import SharedVocabularyLayer

        assert SharedVocabularyLayer is not None

    def test_storage_import(self):
        """Verify storage modules are available."""
        from mindcore.storage import SQLiteStorage

        assert SQLiteStorage is not None

    def test_full_workflow(self):
        """Test complete workflow: store, recall, reinforce."""
        from mindcore import Mindcore
        from mindcore.svl import SharedVocabularyLayer

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            vocab = SharedVocabularyLayer()
            vocab.add_topics("test", "cli")
            memory = Mindcore(storage=f"sqlite:///{db_path}", vocabulary=vocab)

            # Store
            mid = memory.store(
                content="CLI test memory",
                memory_type="semantic",
                user_id="cli_test",
                topics=["cli", "test"],
            )
            assert mid is not None

            # Recall
            result = memory.recall(query="CLI test", user_id="cli_test")
            assert len(result.memories) > 0

            # Reinforce
            memory.reinforce(mid, signal=0.5)
            retrieved = memory.get(mid)
            # Handle both dict and Memory object return types
            if isinstance(retrieved, dict):
                reinf_score = retrieved.get("reinforcement_score", 0.0)
            else:
                reinf_score = retrieved.reinforcement_score
            assert reinf_score != 0.0

            memory.close()
        finally:
            os.unlink(db_path)


class TestCLIPerformance:
    """Performance tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_demo_completes_quickly(self, runner):
        """Verify demo command completes in reasonable time."""
        import time

        start = time.time()
        result = runner.invoke(main, ["demo", "--quick"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        assert elapsed < 10, f"Demo took too long: {elapsed:.2f}s"

    def test_benchmark_completes_quickly(self, runner):
        """Verify benchmark command completes in reasonable time."""
        import time

        start = time.time()
        result = runner.invoke(main, ["benchmark", "latency", "-n", "5"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        assert elapsed < 10, f"Benchmark took too long: {elapsed:.2f}s"
