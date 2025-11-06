# Scripts Directory

This directory contains helper scripts for the AI-NER Historical Text System.

## Structure

```
scripts/
└── utils/             # Utility scripts for development and maintenance
    ├── sort-records.sh               # Sort input records by Bindnr and Brevid
    └── test-security.sh              # Test security scanning tools locally
```

## Usage

### Utility Scripts

**sort-records.sh** - Sort input records
```bash
bash scripts/utils/sort-records.sh
```
Sorts records in `examples/__DN__AI.txt` by Bindnr and Brevid, output to
`examples/Brevid-DN-AI-sorted.txt`. Create the `examples/` directory and source
file before running this utility—the sample data is not tracked in the
repository.

**test-security.sh** - Test security scanning locally
```bash
bash scripts/utils/test-security.sh
```
Runs Bandit and Safety checks to verify security scanning configuration before CI/CD.

## Making Scripts Executable

To run scripts without `bash` prefix:

```bash
chmod +x scripts/utils/*.sh
```

## Notes

- All scripts automatically change to the project root directory
- Scripts use `uv run` to ensure proper virtual environment activation
