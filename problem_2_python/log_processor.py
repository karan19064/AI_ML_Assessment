import os
import logging
from typing import Generator, Dict, Any, Union

# Set up logging for production
logger = logging.getLogger("LogProcessor")
logger.setLevel(logging.INFO)

# Create console handler if not already present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class LogProcessor:
    """
    Production-ready log processor to process large transactional log files.
    Utilizes context managers and generators to process files of arbitrary size
    (e.g., 50GB) in O(1) space complexity.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_handle = None

    def __enter__(self) -> 'LogProcessor':
        """
        Context manager entry. Ensures the file is opened safely.
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Log file not found at: {self.file_path}")
            # Open file with standard buffer size (or default)
            self.file_handle = open(self.file_path, 'r', encoding='utf-8', errors='replace')
            logger.info(f"Successfully opened log file for processing: {self.file_path}")
            return self
        except Exception as e:
            logger.error(f"Failed to open log file {self.file_path}: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context manager exit. Ensures the file is closed safely,
        preventing resource leaks even if exceptions occur.
        """
        if self.file_handle:
            try:
                self.file_handle.close()
                logger.info("Closed log file handle.")
            except Exception as e:
                logger.error(f"Error while closing log file: {str(e)}")
            finally:
                self.file_handle = None
        # Return False to propagate any exceptions raised within the context block
        return False

    def stream_flagged_transactions(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator function that streams flagged transactions line-by-line.
        Filters for log lines containing "ERROR", splits fields, and checks
        if the transaction amount exceeds 10,000.
        
        Yields:
            Dict[str, Any]: Dictionary containing 'user', 'amount', and 'status'
        """
        if not self.file_handle:
            raise RuntimeError("File is not open. Use the LogProcessor inside a 'with' context manager.")

        line_count = 0
        malformed_count = 0
        flagged_count = 0

        for line in self.file_handle:
            line_count += 1
            # Clean newline characters
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Core filtering condition (same as legacy code)
            if "ERROR" in stripped_line:
                try:
                    # Expected format: USER_ID,AMOUNT,LOG_LEVEL,MESSAGE...
                    # Example: usr_100,12500.50,ERROR,Database connection failed
                    parts = stripped_line.split(",")
                    
                    if len(parts) < 2:
                        raise ValueError(f"Line does not contain enough comma-separated fields (found {len(parts)})")

                    user_id = parts[0].strip()
                    amount_str = parts[1].strip()
                    
                    if not user_id:
                        raise ValueError("User ID field is empty")
                        
                    if user_id.upper() in ("ERROR", "INFO", "WARNING", "WARN", "DEBUG"):
                        raise ValueError(f"Invalid User ID (matches log level keyword: '{user_id}')")

                    amount = float(amount_str)
                    
                    # Flag transactions exceeding the $10,000 threshold
                    if amount > 10000.0:
                        flagged_count += 1
                        yield {
                            "user": user_id,
                            "amount": amount,
                            "status": "flagged",
                            "line_num": line_count
                        }

                except ValueError as ve:
                    malformed_count += 1
                    # Log parsing failures at WARNING level to keep console uncluttered but trackable
                    if malformed_count <= 10:  # Prevent log flooding for huge files
                        logger.warning(f"Malformed log line at line {line_count}: '{stripped_line}'. Error: {str(ve)}")
                    elif malformed_count == 11:
                        logger.warning("Further malformed line warnings suppressed.")
                except Exception as e:
                    logger.error(f"Unexpected error parsing line {line_count}: {str(e)}")

        logger.info(f"Log processing completed. Total lines read: {line_count}, "
                    f"Flagged transactions: {flagged_count}, Malformed lines skipped: {malformed_count}")

"""
================================================================================
COMPLEXITY ANALYSIS (Big-O)
================================================================================

1. TIME COMPLEXITY:
   - Legacy Code: O(N) where N is the number of lines in the log file. It reads all 
     lines into memory and loops through them sequentially.
   - Refactored Code: O(N) where N is the number of lines in the log file. It still
     processes each line exactly once.
   - Comparison: Both are theoretically O(N) in time, but the refactored code has 
     a lower practical CPU footprint as it avoids the massive memory-allocation 
     overhead of keeping all lines in RAM.

2. SPACE COMPLEXITY:
   - Legacy Code: O(N) where N is the number of lines in the log file. 
     The f.readlines() method reads the ENTIRE file into a list of strings in RAM.
     For a 50GB log file, this attempts to allocate >50GB of RAM, leading to an 
     immediate Out-Of-Memory (OOM) crash in production.
   - Refactored Code: O(1) auxiliary space. 
     By processing the file line-by-line using a generator and a buffered file
     reader, only a single line is held in memory at any given instant. The memory
     footprint remains constant (typically < 10MB) regardless of whether the log 
     file is 100KB or 50GB.
================================================================================
"""
