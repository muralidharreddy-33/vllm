# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List

from vllm.benchmarks.benchmark_throughput import (add_options, main,
                                                  validate_parsed_args)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkThroughputSubcommand(CLISubcommand):
    """ The `benchmark-throughput` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "benchmark-throughput"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "benchmark-throughput",
            help="Benchmark the throughput.",
            usage="vllm benchmark-throughput [options]")
        add_options(parser)
        return parser


def cmd_init() -> List[CLISubcommand]:
    return [BenchmarkThroughputSubcommand()]
