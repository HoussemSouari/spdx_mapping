"""
ScanCode API-based license detector (direct Python API, not CLI).
"""

from benchmarks.base_detector import BaseLicenseDetector
from typing import Optional


class ScanCodeAPIDetector(BaseLicenseDetector):
    """ScanCode license detection using Python API (not subprocess)."""
    
    def __init__(self):
        super().__init__("ScanCode API")
        self.license_index = None
        
    def setup(self) -> bool:
        """Initialize ScanCode license index."""
        try:
            # Import ScanCode's licensedcode module
            from licensedcode.cache import get_index
            
            print("    Loading ScanCode license index...")
            self.license_index = get_index()
            print(f"    Loaded {len(self.license_index.dictionary)} license rules")
            
            self.is_available = True
            return True
            
        except ImportError as e:
            self.setup_error = f"ScanCode licensedcode module not found: {e}"
            return False
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license using ScanCode API."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        try:
            # Use ScanCode's match function
            matches = list(self.license_index.match(text=text))
            
            if not matches:
                return "UNKNOWN"
            
            # Sort by score and get best match
            matches.sort(key=lambda m: m.score(), reverse=True)
            best_match = matches[0]
            
            # Get SPDX identifier from the matched rule
            rule = best_match.rule
            
            # Try to get SPDX key
            if hasattr(rule, 'license_expression'):
                # Parse license expression to get primary license
                license_expr = rule.license_expression
                
                # Simple parsing: take first license key
                # (for complex expressions like "gpl-2.0 AND mit", takes "gpl-2.0")
                if ' ' in license_expr:
                    # Has operators (AND, OR, WITH)
                    parts = license_expr.replace('(', '').replace(')', '').split()
                    # Get first non-operator part
                    for part in parts:
                        if part.lower() not in ['and', 'or', 'with']:
                            license_key = part
                            break
                    else:
                        license_key = parts[0]
                else:
                    license_key = license_expr
                
                # Convert to SPDX ID format
                spdx_id = self._to_spdx_id(license_key)
                return spdx_id
            
            return "UNKNOWN"
            
        except Exception as e:
            print(f"    Error in ScanCode API detect: {e}")
            return "ERROR"
    
    def _to_spdx_id(self, license_key: str) -> str:
        """
        Convert ScanCode license key to SPDX ID.
        
        ScanCode uses lowercase keys with hyphens (e.g., 'mit', 'apache-2.0')
        SPDX uses specific capitalization (e.g., 'MIT', 'Apache-2.0')
        """
        # Common mappings
        spdx_map = {
            'mit': 'MIT',
            'apache-2.0': 'Apache-2.0',
            'apache-1.1': 'Apache-1.1',
            'gpl-2.0': 'GPL-2.0-only',
            'gpl-2.0-plus': 'GPL-2.0-or-later',
            'gpl-3.0': 'GPL-3.0-only',
            'gpl-3.0-plus': 'GPL-3.0-or-later',
            'lgpl-2.1': 'LGPL-2.1-only',
            'lgpl-2.1-plus': 'LGPL-2.1-or-later',
            'lgpl-3.0': 'LGPL-3.0-only',
            'lgpl-3.0-plus': 'LGPL-3.0-or-later',
            'bsd-new': 'BSD-3-Clause',
            'bsd-simplified': 'BSD-2-Clause',
            'mpl-2.0': 'MPL-2.0',
            'isc': 'ISC',
            'cc0-1.0': 'CC0-1.0',
            'unlicense': 'Unlicense',
        }
        
        license_key_lower = license_key.lower()
        
        # Try direct mapping
        if license_key_lower in spdx_map:
            return spdx_map[license_key_lower]
        
        # Try to capitalize intelligently
        # GPL-2.0-only -> GPL-2.0-only
        # mit -> MIT
        parts = license_key.split('-')
        capitalized = []
        for part in parts:
            if part.replace('.', '').isdigit():
                # Version number
                capitalized.append(part)
            elif part in ['only', 'or', 'later', 'plus']:
                # Special keywords
                if part == 'plus':
                    capitalized.append('or-later')
                else:
                    capitalized.append(part)
            else:
                # License name
                capitalized.append(part.upper())
        
        result = '-'.join(capitalized)
        
        # Handle some special cases
        result = result.replace('-ONLY', '-only')
        result = result.replace('-OR-LATER', '-or-later')
        
        return result
