# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List

from vllm.benchmarks.benchmark_serving import add_options, main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkServingSubcommand(CLISubcommand):
    """ The `benchmark-serving` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "benchmark-serving"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "benchmark-serving",
            help="Benchmark the online serving throughput.",
            usage="vllm benchmark-serving [options]")
        add_options(parser)
        return parser


def cmd_init() -> List[CLISubcommand]:
    return [BenchmarkServingSubcommand()]
