import sys
from types import ModuleType, SimpleNamespace


def _install_optional_dependency_stubs() -> None:
    if "pylatexenc" not in sys.modules:
        latex2text = ModuleType("pylatexenc.latex2text")

        class _LatexNodes2Text:
            def latex_to_text(self, expr):
                return expr

        latex2text.LatexNodes2Text = _LatexNodes2Text
        pylatexenc = ModuleType("pylatexenc")
        pylatexenc.latex2text = latex2text
        sys.modules["pylatexenc"] = pylatexenc
        sys.modules["pylatexenc.latex2text"] = latex2text

    if "httpx" not in sys.modules:
        httpx = ModuleType("httpx")

        class _DummyResponse:
            is_success = True
            status_code = 200
            text = ""

            def json(self):
                return {}

        class _DummyClient:
            def get(self, *args, **kwargs):
                return _DummyResponse()

            def post(self, *args, **kwargs):
                return _DummyResponse()

            def close(self):
                return None

        httpx.Client = _DummyClient
        httpx.AsyncClient = _DummyClient
        sys.modules["httpx"] = httpx

    if "firecrawl" not in sys.modules:
        firecrawl = ModuleType("firecrawl")
        firecrawl.FirecrawlApp = SimpleNamespace
        sys.modules["firecrawl"] = firecrawl

    if "transformers" not in sys.modules:
        transformers = ModuleType("transformers")

        class _PreTrainedTokenizerBase:
            pass

        transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = ModuleType("PIL")
        image = ModuleType("PIL.Image")
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image

    if "ray" not in sys.modules:
        ray = ModuleType("ray")

        def _remote(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(obj):
                return obj

            return decorator

        ray.remote = _remote
        ray.init = lambda *args, **kwargs: None
        ray.is_initialized = lambda: False
        ray.get = lambda value: value
        ray.timeline = lambda *args, **kwargs: None
        sys.modules["ray"] = ray


_install_optional_dependency_stubs()
