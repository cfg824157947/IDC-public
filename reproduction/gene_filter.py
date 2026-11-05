"""Gene filter v2 (2026-10-31): genes excluded before IDC fitting (the within-donor test still tests all genes).
MT- (mitochondrially encoded) and cytosolic ribosomal-protein genes (same RPL/RPS pattern as the 20261029 filter; MRPL/MRPS not matched).
Nuclear respiratory-chain genes (NDUF/COX/ATP5/UQCR/SDH) are kept (excluded by the 20261029 filter)."""
import re
EXCLUDE = re.compile(r"^(MT-.*|RP[LS]\d+[ALXY]{0,2}\d*)$")
excluded = lambda symbols: [bool(EXCLUDE.match(str(s))) for s in symbols]
