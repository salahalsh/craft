"""Port the pipeline files from the Django platform to a standalone package.

Applied once on top of the pristine copy (git commit "Pristine copy ...").
Every edit is asserted; defaults reproduce the platform's behaviour exactly.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "craft"


def edit(rel, pairs):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{rel}: expected 1 occurrence, found {n}: {old[:70]!r}"
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("edited", rel)


TTL_OLD = "self.cache_ttl = cache_ttl_hours * 3600"
TTL_NEW = ("from craft import config\n"
           "        self.cache_ttl = (config.cache_ttl_hours() if cache_ttl_hours is None\n"
           "                          else cache_ttl_hours) * 3600")


def django_client(name):
    return [
        ("cache_ttl_hours: int = 168):", "cache_ttl_hours: Optional[int] = None):"),
        (f"""            from django.conf import settings
            cache_dir = Path(settings.BASE_DIR) / 'craft' / 'data' / 'cache' / '{name}'""",
         f"""            from craft import config
            cache_dir = config.cache_dir('{name}')"""),
        (TTL_OLD, TTL_NEW),
    ]


edit("services/chembl_client.py", django_client("chembl"))
edit("services/uniprot_client.py", django_client("uniprot"))
edit("services/iuphar_client.py", django_client("iuphar"))

edit("services/pdbe_client.py", [
    ("                 cache_ttl_hours: int = 168):",
     "                 cache_ttl_hours: Optional[int] = None):"),
    ("            base = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'pdbe'",
     "            from craft import config\n            base = config.cache_dir('pdbe')"),
    (TTL_OLD, TTL_NEW),
])

edit("training/bioactivity_fetcher.py", [
    ("                 cache_ttl_hours: int = 168):",
     "                 cache_ttl_hours: Optional[int] = None):"),
    ("""            try:
                from django.conf import settings
                cache_dir = (Path(settings.BASE_DIR) / 'craft' / 'data'
                             / 'cache' / 'bioactivity')
            except Exception:
                cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'bioactivity'""",
     """            from craft import config
            cache_dir = config.cache_dir('bioactivity')"""),
    (TTL_OLD, TTL_NEW),
])

edit("training/baseline_featurizers.py", [
    ("            base = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'esm2'",
     "            from craft import config\n            base = config.cache_dir('esm2')"),
])

edit("services/craft_similarity.py", [
    ("""            from core_utils.user_errors import report_and_get_user_message, TASK_FAILED
            message, _code = report_and_get_user_message(TASK_FAILED, exc=e, user=None)
            # Generic message + code, safe to render - raw exception
            # preserved in ApplicationError.
            result.error = message""",
     """            result.error = f"Similarity search failed: {e}\""""),
])

# no Django or platform import may remain
for p in ROOT.rglob("*.py"):
    s = p.read_text(encoding="utf-8")
    for bad in ("django", "core_utils", "settings.BASE_DIR"):
        assert bad not in s, f"{p.relative_to(ROOT)} still contains {bad!r}"
print("no django/core_utils references remain")
