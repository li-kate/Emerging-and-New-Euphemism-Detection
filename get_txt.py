from data_sources import stream_csv, write_sentences_to_txt, stream_common_crawl

stream = stream_csv("data.csv")
write_sentences_to_txt(stream, "csv_sentences.txt")

stream_cc = stream_common_crawl("CC-MAIN-2024-10", max_files=2)
write_sentences_to_txt(stream_cc, "cc_sentences.txt")