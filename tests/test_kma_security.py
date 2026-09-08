"""실제 API 호출 없이 인증 누락과 오류 로그의 비밀 노출을 검사한다."""

import contextlib
import importlib
import io
import os
import unittest
from unittest.mock import patch

from weather import kma
import requests


class KmaSecurityTests(unittest.TestCase):
    def test_missing_environment_has_no_builtin_key(self):
        with patch.dict(os.environ, {}, clear=True):
            module = importlib.reload(kma)
            self.assertFalse(bool(module.API_KEY))

    def test_missing_key_stops_before_request(self):
        with patch.object(kma.requests, "get") as request:
            with self.assertRaisesRegex(ValueError, "KMA_API_KEY"):
                kma.request_data("202402010000", "")
            request.assert_not_called()

    def test_request_error_log_does_not_include_key(self):
        dummy = "test-credential-do-not-publish-real-keys"
        error = requests.RequestException(f"request failed ?authKey={dummy}")
        output = io.StringIO()
        with patch.object(kma, "request_data", side_effect=error):
            with contextlib.redirect_stdout(output):
                result = kma.request_data_with_retry("202402010000", dummy, retries=0)
        self.assertIsNone(result)
        self.assertNotIn(dummy, output.getvalue())
        self.assertIn("RequestException", output.getvalue())


if __name__ == "__main__":
    unittest.main()
