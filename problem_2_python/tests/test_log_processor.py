import os
import sys
# Add parent directory to path so we can import log_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from log_processor import LogProcessor

@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture that creates a temporary log file for testing."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "test_transactions.log"
    return str(log_file)

def test_context_manager_and_streaming(temp_log_file):
    # Create mock logs
    mock_logs = [
        "usr_001,15000.0,ERROR,Database failure\n",       # Flagged: ERROR and >10000
        "usr_002,5000.00,ERROR,Timeout\n",                # Skipped: ERROR but <=10000
        "usr_003,12000.0,INFO,All good\n",                 # Skipped: No ERROR
        "usr_004,20000.50,ERROR,Auth failure\n",           # Flagged: ERROR and >10000
        "\n",                                             # Skipped: Empty line
        "usr_005,ERROR,Malformed line\n",                 # Skipped: Malformed amount
        "usr_006,25000.00,ERROR\n",                        # Flagged: ERROR and >10000 (missing description is fine)
    ]
    
    with open(temp_log_file, "w", encoding="utf-8") as f:
        f.writelines(mock_logs)

    # Process logs
    flagged = []
    with LogProcessor(temp_log_file) as processor:
        for entry in processor.stream_flagged_transactions():
            flagged.append(entry)

    # Assertions
    assert len(flagged) == 3
    
    # First flagged: usr_001
    assert flagged[0]["user"] == "usr_001"
    assert flagged[0]["amount"] == 15000.0
    assert flagged[0]["status"] == "flagged"
    assert flagged[0]["line_num"] == 1

    # Second flagged: usr_004
    assert flagged[1]["user"] == "usr_004"
    assert flagged[1]["amount"] == 20000.5
    assert flagged[1]["status"] == "flagged"
    assert flagged[1]["line_num"] == 4

    # Third flagged: usr_006
    assert flagged[2]["user"] == "usr_006"
    assert flagged[2]["amount"] == 25000.0
    assert flagged[2]["status"] == "flagged"
    assert flagged[2]["line_num"] == 7

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        with LogProcessor("non_existent_file.log") as processor:
            list(processor.stream_flagged_transactions())

def test_malformed_lines_handling(temp_log_file):
    # Create logs with various malformed structures
    malformed_logs = [
        "ERROR,15000.0,ERROR\n",                           # Malformed: User ID is empty or equals "ERROR"
        ",20000.0,ERROR,Empty user\n",                    # Malformed: User ID is empty
        "usr_001,,ERROR,Empty amount\n",                  # Malformed: Amount is empty
        "usr_002,not_a_float,ERROR,Invalid amount\n",     # Malformed: Amount is string
        "usr_003,15000.0\n",                               # Malformed: Missing ERROR keyword
        "usr_004,18000.00,ERROR,Valid line\n",             # Flagged: ERROR and >10000
    ]
    
    with open(temp_log_file, "w", encoding="utf-8") as f:
        f.writelines(malformed_logs)

    # Process logs (should not raise exception and should complete)
    flagged = []
    with LogProcessor(temp_log_file) as processor:
        for entry in processor.stream_flagged_transactions():
            flagged.append(entry)

    # Assertions
    assert len(flagged) == 1
    assert flagged[0]["user"] == "usr_004"
    assert flagged[0]["amount"] == 18000.0
