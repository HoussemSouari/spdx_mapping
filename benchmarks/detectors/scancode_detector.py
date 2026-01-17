"""
ScanCode Toolkit license detector.
"""

import subprocess
import json
import tempfile
from pathlib import Path
from benchmarks.base_detector import BaseLicenseDetector


class ScanCodeDetector(BaseLicenseDetector):
    """ScanCode Toolkit wrapper."""
    
    def __init__(self):
        super().__init__("ScanCode Toolkit")
        
    def setup(self) -> bool:
        """Check if scancode is installed."""
        try:
            result = subprocess.run(
                ["scancode", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.is_available = True
                return True
            else:
                self.setup_error = "ScanCode command failed"
                return False
        except FileNotFoundError:
            self.setup_error = "ScanCode not installed. Run: pip install scancode-toolkit"
            return False
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license using ScanCode."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        # Write text to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_file = f.name
        
        try:
            # Run scancode
            result = subprocess.run(
                [
                    "scancode",
                    "--license",
                    "--json-pp", "-",
                    "--quiet",
                    temp_file
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return "UNKNOWN"
            
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Extract license from results
            if data.get("files") and len(data["files"]) > 0:
                licenses = data["files"][0].get("licenses", [])
                if licenses:
                    # Return highest score license
                    licenses.sort(key=lambda x: x.get("score", 0), reverse=True)
                    spdx_key = licenses[0].get("spdx_license_key")
                    if spdx_key:
                        return spdx_key
            
            return "UNKNOWN"
            
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return "ERROR"
        finally:
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
