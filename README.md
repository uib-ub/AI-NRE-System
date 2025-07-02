# Medieval Text Annotation With LLMs

## Setup

1. Run 

`uv sync`

2. Run 

eg 1. Claude API:

```Bash
uv run process-medieval-llm \
      --model claude \
      --output-text output/annotated_output_claude_batch_10R_B10.txt \
      --output-table output/metadata_table_claude_batch_10R_B10.txt \
      -v \
      --use-batch --batch-size 10
```

eg 2. Ollama/OPENWEBUI:

```Bash
uv run process-medieval-llm \
      --model ollama \
      --output-text output/annotated_output_gemma_batch_10R_B2.txt \
      --output-table output/metadata_table_gemma_batch_10R_B2.txt \
      -v \
      --use-batch --batch-size 2
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
