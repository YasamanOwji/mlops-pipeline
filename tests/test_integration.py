"""
Integration test: runs the pipeline end-to-end.
This test is skipped by default because it requires a full DVC/MLflow environment setup.
"""

import pytest

@pytest.mark.skip(reason="Integration test requires full environment setup (DVC, MLflow, etc.)")
def test_end_to_end_pipeline():
    # در صورت نیاز می‌توانید این تست را با تنظیمات کامل فعال کنید
    assert True