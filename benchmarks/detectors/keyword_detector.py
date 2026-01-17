"""
Keyword-based license detector (simple baseline).
"""

from benchmarks.base_detector import BaseLicenseDetector
from typing import Dict, List
import re


class KeywordDetector(BaseLicenseDetector):
    """Simple keyword-based license detection."""
    
    def __init__(self):
        super().__init__("Keyword Matching")
        self.keywords = {}
        
    def setup(self) -> bool:
        """Setup keyword patterns for common licenses."""
        # Define keywords that strongly indicate specific licenses
        self.keywords = {
            "MIT": [
                r"permission is hereby granted, free of charge",
                r"mit license",
                r"without restriction.*use, copy, modify, merge, publish"
            ],
            "Apache-2.0": [
                r"apache license.*version 2\.0",
                r"licensed under the apache license",
                r"www\.apache\.org/licenses/LICENSE-2\.0"
            ],
            "GPL-2.0-only": [
                r"gnu general public license.*version 2",
                r"gpl.*version 2.*only",
                r"free software foundation.*version 2"
            ],
            "GPL-2.0-or-later": [
                r"gnu general public license.*version 2.*or.*later",
                r"either version 2.*or.*any later version"
            ],
            "GPL-3.0-only": [
                r"gnu general public license.*version 3",
                r"gpl.*version 3.*only"
            ],
            "GPL-3.0-or-later": [
                r"gnu general public license.*version 3.*or.*later",
                r"either version 3.*or.*any later version"
            ],
            "BSD-2-Clause": [
                r"redistribution and use in source and binary forms",
                r"bsd.*2-clause",
                r"without specific prior written permission"
            ],
            "BSD-3-Clause": [
                r"redistribution and use in source and binary forms",
                r"bsd.*3-clause",
                r"neither the name.*nor the names"
            ],
            "LGPL-2.1-only": [
                r"gnu lesser general public license.*version 2\.1",
                r"lgpl.*version 2\.1.*only"
            ],
            "LGPL-2.1-or-later": [
                r"gnu lesser general public license.*version 2\.1.*or.*later"
            ],
            "LGPL-3.0-only": [
                r"gnu lesser general public license.*version 3",
                r"lgpl.*version 3.*only"
            ],
            "LGPL-3.0-or-later": [
                r"gnu lesser general public license.*version 3.*or.*later"
            ],
            "MPL-2.0": [
                r"mozilla public license.*version 2\.0",
                r"mpl.*2\.0"
            ],
            "ISC": [
                r"isc license",
                r"permission to use, copy, modify.*distribute this software"
            ],
            "CC0-1.0": [
                r"creative commons.*public domain dedication",
                r"cc0.*universal"
            ],
            "Unlicense": [
                r"this is free and unencumbered software",
                r"unlicense"
            ],
            "AGPL-3.0-only": [
                r"gnu affero general public license.*version 3",
                r"agpl.*version 3.*only"
            ],
            "AGPL-3.0-or-later": [
                r"gnu affero general public license.*version 3.*or.*later"
            ]
        }
        
        # Compile patterns
        self.patterns = {}
        for license_id, keywords in self.keywords.items():
            self.patterns[license_id] = [
                re.compile(pattern, re.IGNORECASE | re.DOTALL)
                for pattern in keywords
            ]
        
        self.is_available = True
        return True
    
    def detect(self, text: str) -> str:
        """Detect license using keyword matching."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        text_lower = text.lower()
        scores = {}
        
        # Score each license based on keyword matches
        for license_id, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text_lower):
                    score += 1
            if score > 0:
                scores[license_id] = score
        
        # Return license with highest score
        if scores:
            best_license = max(scores.items(), key=lambda x: x[1])
            return best_license[0]
        
        return "UNKNOWN"
