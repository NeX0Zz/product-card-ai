"""Command line: run the web service or generate cards from JSON files."""

import argparse
import asyncio
import ipaddress
import json
import logging
import sys
from pathlib import Path

from .config import LOOPBACK_HOSTS, get_settings
from .llm import LLMError, OpenRouterClient
from .models import InputError, load_json, parse_product
from .pipeline import generate_card
from .render import to_markdown

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_NEEDS_REVIEW = 2


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    if not _is_loopback(args.host):
        print(
            f"Внимание: сервис без авторизации и с вашим ключом LLM слушает {args.host}. "
            "Он рассчитан только на локальный запуск.",
            file=sys.stderr,
        )
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", server_header=False)
    return _EXIT_OK


def _is_loopback(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _generate_all(paths: list[Path], out_dir: Path | None) -> int:
    settings = get_settings()
    if settings.openrouter_api_key is None:
        print("Не задан OPENROUTER_API_KEY: добавьте ключ в файл .env.", file=sys.stderr)
        return _EXIT_FAILED

    client = OpenRouterClient(settings)
    exit_code = _EXIT_OK
    try:
        for path in paths:
            try:
                product, ignored = parse_product(load_json(path.read_text("utf-8")))
                card = await generate_card(product, client, settings, ignored)
            except (OSError, UnicodeDecodeError) as exc:
                print(f"{path}: не удалось прочитать файл ({type(exc).__name__}).", file=sys.stderr)
                exit_code = _EXIT_FAILED
                continue
            except InputError as exc:
                details = "; ".join(f"{d.field}: {d.message}" for d in exc.details)
                print(f"{path}: {exc.message} {details}".rstrip(), file=sys.stderr)
                exit_code = _EXIT_FAILED
                continue
            except LLMError as exc:
                print(f"{path}: {exc.message}", file=sys.stderr)
                exit_code = _EXIT_FAILED
                continue

            payload = json.dumps(card.model_dump(), ensure_ascii=False, indent=2)
            if out_dir is None:
                print(payload)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{path.stem}.json").write_text(payload + "\n", "utf-8")
                (out_dir / f"{path.stem}.md").write_text(to_markdown(card), "utf-8")
            print(f"{path.name}: {card.status}", file=sys.stderr)
            if card.status in {"needs_review", "insufficient_data"} and exit_code == _EXIT_OK:
                exit_code = _EXIT_NEEDS_REVIEW
    finally:
        await client.aclose()
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="product-card", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="запустить веб-интерфейс")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    generate = commands.add_parser("generate", help="сгенерировать карточки из JSON-файлов")
    generate.add_argument("files", nargs="+", type=Path)
    generate.add_argument("-o", "--out", type=Path, default=None, help="папка для .json и .md")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs request URLs at INFO. Secrets aren't there, but this is noise for the CLI.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _build_parser().parse_args()
    if args.command == "serve":
        sys.exit(_serve(args))
    sys.exit(asyncio.run(_generate_all(args.files, args.out)))


if __name__ == "__main__":
    main()
