from .parsers import ParserRegistry, docx_parser


def run():
    parser_registry = ParserRegistry()

    print(parser_registry.parsers())


__all__ = ["run"]
