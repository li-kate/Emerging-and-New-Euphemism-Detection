Change:
    - in drug_centroid_embeddings.py, change OUTPUT_DIR
    - in drug_centroid_merge_results.py, change RESULTS_DIR (should be equal to OUTPUT_DIR from previous file)

To run:
    chmod +x submit_drug_centroid_pipeline.sh
    ./submit_drug_centroid_pipeline.sh

For future:
 - move drug_synonyms.txt out of taboo_list_creation and fix file paths
 - move drug_subreddit_list.txt out of cosine_similarity_drift and fix file paths

The drug centroid will just be gotten from drug subreddits to get the most accurate representation possible.
