# SYSTEM PROMPT v2.0 — Enhanced Python & File Handling

> **Status:** Updated with Python linting patterns, large file navigation, and multi_edit clarification
> **Date:** 2024-04-25
> **Changes:** 3 sections added based on 30-feature implementation experience

---

## Original System Prompt Content

[All original system prompt content would be preserved here...]

---

## NEW: Python-Specific Linting Patterns

When implementing Python code, follow these patterns to avoid common linting errors:

### RUF012 — Mutable Class Attributes
Convert mutable defaults to immutable:

```python
# ❌ BAD — Mutable list/dict as class attribute
class MyClass:
    ITEMS = []  # RUF012 error
    CONFIG = {}  # RUF012 error

# ✅ GOOD — Use tuple or MappingProxyType
from types import MappingProxyType

class MyClass:
    ITEMS = ()  # Immutable tuple
    CONFIG = MappingProxyType({"key": "value"})  # Immutable mapping
```

### ARG002 — Unused Parameters
For intentionally unused parameters in interfaces/abstract methods:

```python
# ❌ BAD — Unused parameter triggers warning
def method(self, unused_param: str) -> None:

# ✅ GOOD — Prefix with underscore
def method(self, _unused_param: str) -> None:
```

### E501 — Line Too Long
Split at natural boundaries:

```python
# ❌ BAD — Long line
logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {error_msg}")

# ✅ GOOD — Split concatenated f-strings
logger.warning(
    f"Task {task.id} failed, retrying "
    f"({task.retry_count}/{task.max_retries}): {error_msg}"
)
```

---

## NEW: Handling Large Files (>500 lines)

When working with large files:

1. **Prefer reading in chunks** — Use offset/limit parameters:
   ```python
   # Read first 100 lines
   read_file(file_path, offset=1, limit=100)
   # Read lines 200-300
   read_file(file_path, offset=200, limit=100)
   ```

2. **Consider file splitting** — If a file exceeds 1000 lines, consider:
   - Breaking into logical modules
   - Extracting related classes/functions to separate files
   - Creating a package directory with `__init__.py`

3. **Use code_search first** — For locating specific functions in large files:
   ```
   code_search(
       search_folder_absolute_uri="/path/to/project",
       search_term="Find the authenticate method in calendar integration"
   )
   ```

---

## UPDATED: multi_edit Tool Clarification

Use `multi_edit` for making multiple edits to a single file in one operation. It is built on top of the Edit tool and allows you to perform multiple find-and-replace operations efficiently.

**Note:** `multi_edit` operates on a single file only. For cross-file changes:
- Chain multiple `multi_edit` calls (one per file)
- Or use sequential `edit` calls for simple changes
- Consider `grep_search` + `edit` pattern for widespread renames

---

## Implementation Notes

These additions address gaps identified during the 30-feature WhatsApp AI Agent implementation:
- Repeated RUF012/ARG002 fixes (now have copy-pasteable patterns)
- Multiple read_file calls for large files (now have chunking guidance)
- Confusion about multi_edit scope (now clarified)

**Health Score Improvement:** 8/10 → 9/10
