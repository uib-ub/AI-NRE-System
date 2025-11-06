# Scripts Directory

This directory contains helper scripts for the AI-NER Historical Text System.

## Structure

```
scripts/
├── examples/          # Example run scripts demonstrating different modes
│   ├── run-async-batch-example.sh    # Async batch processing with Claude
│   ├── run-sync-batch-example.sh     # Sync batch processing with Ollama
│   └── run-sync-example.sh           # Single record processing with Ollama
└── utils/             # Utility scripts for development and maintenance
    ├── sort-records.sh               # Sort input records by Bindnr and Brevid
    └── test-security.sh              # Test security scanning tools locally
```

## Usage

### Example Scripts

All example scripts should be run from the project root or directly:

```bash
# From project root
bash scripts/examples/run-async-batch-example.sh

# Or directly (scripts change to project root automatically)
cd scripts/examples
./run-async-batch-example.sh
```

**Available examples:**

1. **run-async-batch-example.sh** - Demonstrates async batch processing with Claude API
   - Uses incremental output mode
   - Batch size: 2 records
   - Ideal for large datasets

2. **run-sync-batch-example.sh** - Demonstrates synchronous batch processing with Ollama
   - Processes multiple records per batch
   - Batch size: 2 records
   - Debug logging enabled

3. **run-sync-example.sh** - Demonstrates single record processing
   - Processes one record at a time
   - Useful for testing and debugging
   - Debug logging enabled

### Utility Scripts

**sort-records.sh** - Sort input records
```bash
bash scripts/utils/sort-records.sh
```
Sorts records in `examples/__DN__AI.txt` by Bindnr and Brevid, output to `examples/Brevid-DN-AI-sorted.txt`.

**test-security.sh** - Test security scanning locally
```bash
bash scripts/utils/test-security.sh
```
Runs Bandit and Safety checks to verify security scanning configuration before CI/CD.

## Making Scripts Executable

To run scripts without `bash` prefix:

```bash
chmod +x scripts/examples/*.sh
chmod +x scripts/utils/*.sh
```

Then run directly:
```bash
./scripts/examples/run-async-batch-example.sh
```

## Notes

- All scripts automatically change to the project root directory
- Output files are written to the `output/` directory
- Input files are read from the `input/` or `examples/` directory
- Scripts use `uv run` to ensure proper virtual environment activation
