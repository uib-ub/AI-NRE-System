Look for security problems in the target named at the end (a file, folder or module); if no target is named, in the whole repository.

Check for secrets or keys in the code, injection (SQL, shell, templates), unsafe deserialization or eval, path traversal, unsafe use of subprocesses or temporary files, missing input validation where untrusted data comes in, and risky settings in CI workflows. Do not try exploits, and do not modify anything.

Reply in Markdown: concrete findings, most serious first, each with the file path and line number, a severity (high, medium or low) and a fix, or say clearly that you found none. Never print the value of a secret; name its file and line only.
