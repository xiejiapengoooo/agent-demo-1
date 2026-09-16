from .parsers import ParserRegistry


def run():
    parser_registry = ParserRegistry()
    print(parser_registry.parsers())


__all__ = ["run"]
