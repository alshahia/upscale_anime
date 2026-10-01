# API Tests

**Status: No API tests currently**

This directory is reserved for API integration tests. Currently, there
are no API tests in the test suite.

When API tests are added, they should:
- Test the FastAPI endpoints in `anime_sr.api.inference_api`
- Use pytest fixtures for test client setup
- Mark tests with `@pytest.mark.integration`
- Test both success and error cases

See `src/anime_sr/api/inference_api.py` for the API implementation.
