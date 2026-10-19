from argus.format import fmt_ago, fmt_duration, fmt_size


def test_fmt_size():
    assert fmt_size(None) == "–"
    assert fmt_size(0) == "0 B" and fmt_size(1023) == "1023 B"
    assert fmt_size(1024) == "1.0 KB" and fmt_size(1536) == "1.5 KB"
    assert fmt_size(5 * 1024**2) == "5.0 MB"
    assert fmt_size(int(4.9 * 1024**3)) == "4.9 GB"
    assert fmt_size(200 * 1024**3) == "200.0 GB"      # no unit above GB: stays readable


def test_fmt_ago():
    assert fmt_ago(0) == "just now" and fmt_ago(59) == "just now"
    assert fmt_ago(60) == "1 minute ago" and fmt_ago(150) == "2 minutes ago"
    assert fmt_ago(3600) == "1 hour ago" and fmt_ago(7300) == "2 hours ago"
    assert fmt_ago(86400) == "1 day ago" and fmt_ago(3 * 86400) == "3 days ago"
    assert fmt_ago(-500) == "just now"


def test_fmt_duration_still_behaves():
    assert fmt_duration(None) == "–" and fmt_duration(9_100_000_000) == "9.1 s"
