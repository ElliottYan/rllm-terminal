from importlib.machinery import ModuleSpec
import sys
from types import ModuleType, SimpleNamespace


def _make_stub_module(name: str, *, is_package: bool = False) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []
        module.__spec__.submodule_search_locations = []
    return module


def _install_optional_dependency_stubs() -> None:
    if "pylatexenc" not in sys.modules:
        latex2text = _make_stub_module("pylatexenc.latex2text")

        class _LatexNodes2Text:
            def latex_to_text(self, expr):
                return expr

        latex2text.LatexNodes2Text = _LatexNodes2Text
        pylatexenc = _make_stub_module("pylatexenc", is_package=True)
        pylatexenc.latex2text = latex2text
        sys.modules["pylatexenc"] = pylatexenc
        sys.modules["pylatexenc.latex2text"] = latex2text

    if "httpx" not in sys.modules:
        httpx = _make_stub_module("httpx")

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
        firecrawl = _make_stub_module("firecrawl")
        firecrawl.FirecrawlApp = SimpleNamespace
        sys.modules["firecrawl"] = firecrawl

    if "transformers" not in sys.modules:
        transformers = _make_stub_module("transformers")

        class _PreTrainedTokenizerBase:
            pass

        transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        sys.modules["transformers"] = transformers

    if "PIL" not in sys.modules:
        pil = _make_stub_module("PIL", is_package=True)
        image = _make_stub_module("PIL.Image")
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image

    if "ray" not in sys.modules:
        ray = _make_stub_module("ray")

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
