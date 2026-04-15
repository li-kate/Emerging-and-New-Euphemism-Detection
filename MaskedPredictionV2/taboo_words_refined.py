# ============================================================================
# TABOO ANCHORS — DRUG-RELATED ONLY
# ============================================================================
#
# TABOO_ANCHORS for masked_prediction.py (exec) and first_pass.load_taboo_anchors.
# Single category "drugs": literal/clinical names + misuse-related concepts.
# Skipped very polysemous tokens (e.g. weed, ice, speed, pot) where possible.
#
# ============================================================================

TABOO_ANCHORS = {
    "drugs": [
        "addiction",
        "amphetamine",
        "benzodiazepine",
        "cannabis",
        "cocaine",
        "crack",
        "dealer",
        "drug",
        "ecstasy",
        "fentanyl",
        "hallucinogen",
        "heroin",
        "intoxication",
        "ketamine",
        "lsd",
        "marijuana",
        "mdma",
        "methadone",
        "methamphetamine",
        "morphine",
        "narcotic",
        "opioid",
        "opiate",
        "overdose",
        "oxycodone",
        "oxycontin",
        "paraphernalia",
        "pcp",
        "relapse",
        "suboxone",
        "syringe",
        "thc",
        "trafficking",
        "withdrawal",
        "xanax",
    ],
}
