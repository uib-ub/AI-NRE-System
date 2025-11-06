#!/usr/bin/env bash
# Test script to verify security scanning commands work locally
# Run this before pushing changes to GitHub
# Usage: ./test-security.sh

set -e  # Exit on error

# Change to project root directory
cd "$(dirname "$0")/../.." || exit 1

echo "🔍 Testing Security Scanning Commands..."
echo ""

# Test Bandit
echo "1️⃣  Testing Bandit..."
uv run bandit -r src/ \
    -f json \
    -o bandit-report.json \
    -ll \
    --exit-zero

if [ -f bandit-report.json ]; then
    echo "✅ Bandit report created: bandit-report.json"
    SIZE=$(wc -c < bandit-report.json)
    echo "   File size: $SIZE bytes"
else
    echo "❌ Bandit report not found"
    exit 1
fi

echo ""

# Test Safety
echo "2️⃣  Testing Safety..."
uv run safety check \
    --save-json safety-report.json \
    --output json \
    || true  # Don't fail on vulnerabilities found

if [ -f safety-report.json ]; then
    echo "✅ Safety report created: safety-report.json"
    SIZE=$(wc -c < safety-report.json)
    echo "   File size: $SIZE bytes"
else
    echo "❌ Safety report not found"
    exit 1
fi

echo ""
echo "🎉 All security scanning commands work correctly!"
echo ""
echo "📊 Summary:"
echo "   - bandit-report.json: $(wc -c < bandit-report.json) bytes"
echo "   - safety-report.json: $(wc -c < safety-report.json) bytes"
echo ""
echo "💡 Next steps:"
echo "   1. Review the reports for any critical issues"
echo "   2. Commit the updated workflow files"
echo "   3. Push to GitHub to test CI/CD"
