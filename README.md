# Emerging and New Euphemism Detection
This repository is a data collection and learning pipeline to detect emerging and new euphemisms.

## Data Collection
Streams through Reddit zst files to collect word instances. There are 3 txt categories, anchor and baseline drug terms (e.g., cocaine, fentanyl, molly, coke), candidate euphemisms (e.g., “fenty,” “tranq,”, “gray death”), and comparison words with drug-
adjacent but non-euphemistic usage (e.g., “needle,” “pharmacy”). All occurrences are identified via the Aho-Corasick algorithm in a single pass over each dump file, and each match is saved with its surrounding context, timestamp, subreddit, and permalink. Spelling variants are normalized through an alias map (e.g., “grey death” → “gray death”) so that orthographic variation does not fragment downstream signals.