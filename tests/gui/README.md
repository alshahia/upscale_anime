# GUI Tests

**Status: Not Applicable**

This directory is reserved for GUI tests. The GUI component
(`apps/anime_upscaler_gui/) has its own test directory and requires
a display environment to run.

GUI tests are not included in the main test suite because:
1. They require a display/GUI environment
2. They are typically run separately from the core test suite
3. The GUI component has its own test infrastructure

See `apps/anime_upscaler_gui/tests/` for GUI-specific tests.
