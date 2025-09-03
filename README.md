# AI-NER Historical Text Systen With LLMs

The AI-NER Historical Text System processes historical text records in in multiple languages (Old Norse, Latin, 
Middle Norse, Middle Dutch) with unstandardized orthography using Large Language Models (LLMs) to extract named entities
and generate structured metadata.

## System Architecture Overview



## Setup

Run 

`uv sync`

## Run 

eg 1. Claude API:

```Bash
uv run src/ai_ner_system/main.py \ 
      --client claude \
      --output-text output/annotated_output_claude_batch_13R_B2_async_incremental.txt \
      --output-table output/metadata_table_claude_batch_13R_B2_async_incremental.txt \
      --output-stats output/stats_claude_batch_13R_B2_async_incremental.txt \
      --batch-size 2 --async --incremental-output
```

eg 2. Ollama/OPENWEBUI:

```Bash
uv run src/ai_ner_system/main.py \ 
      --model ollama \
      --output-text output/annotated_output_gemma_batch_10R_B2.txt \
      --output-table output/metadata_table_gemma_batch_10R_B2.txt \
      -l DEBUG
      --use-batch --batch-size 2
```

```Bash
uv run src/ai_ner_system/main.py \ 
       --client ollama \ 
       --output-text output/annotated_output_gemma_batch_13R_B1.txt \
       --output-table output/metadata_table_gemma_batch_13R_B1.txt 
       -l DEBUG
```


## Note: Individual Processing vs Batch Processing

### Individual processing: each record is processed separately

For 10 records, Ollama used 22:21 minutes, Claude used 06:15 minutes with US$0.31 

Then, for 18,559 Records:

* total time for individual processing: 6:15 * 18559 / 10 = 194:00 hours (8,1 days)


### Batch Processing: each batch of records is processed together (eg: 10 records together)

For 10 records, batch size 3, Claude used 05:55 minutes with US$0.42, input token: 1912 + 2363 + 1587 + 764 (=6626), output token: 6569 + 5854 + 4627 + 1185 (=18235)

For 10 records, batch size 5, Claude used 05:26 minutes with US$0.29, input token: 2406 + 3248 (=5654), output token: 8161 + 9957 (=18118)
 
For 10 records, batch size 10, Claude used 05:23 minutes with US$0.29, input token: 5168, output token: 18177

- improvement over individual processing:
    1. ~14% faster than individual processing (5:23 vs 6:15)
    2. Cost Reduction: 6.5% cheaper (0.29 vs0.31)


Then, for 18,559 Records:

* total time for batch processing: 5:23 * 18559 / 10 = 166.5 hours (6.9 days)


Furthermore: for the new async implementation, 
1. for 10 records, batch size 10, claude used 1.56 minutes with US$0.17, total time for batch processing 1:34 * 18559 / 10 = 48.5 hours (2.02 days)
2. for 50 records, batch size 50, claude used 2.04 minutes with US$1.18, total time for batch processing 2:04 * 18559 / 50 = 12.79 hours (0.53 days)
3. for 100 records, batch size 100, claude used 2.04 minutes with US$1.75, total time for batch processing 2:35 * 18559 / 100 = 7.99 hours (0.33 days)

4. for 103 records, batch size 10, currently handle 5 batches each time, claude used 3:37 for 100 records and 1:03 for 3 records, total time for batch processing will be about 11 hours 

5. for 551 records, batch size 100 currently handle 5 batches each time, claude used 827 seconds and about 12 usd for 551 records, then total time for batch processing 18559 records will be about 827 * 18559 / 551 / 3600 = 7.73 hours (0.32 days), 12 * 18559 / 551 = 404.19 USD