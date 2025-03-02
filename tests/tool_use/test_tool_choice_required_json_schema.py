# SPDX-License-Identifier: Apache-2.0
import json
import re
from copy import deepcopy
from typing import List
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionToolsParam)

EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                },
                "required": ["city"],
                "additionalProperties": False
            },
        },
        "strict": True
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the weather forecast for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to get the forecast for, e.g. 'New York'",
                    },
                    "days": {
                        "type":
                        "integer",
                        "description":
                        "Number of days to get the forecast for (1-7)",
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False
            },
        },
        "strict": True
    },
]


def _compile_and_check(tools: list[ChatCompletionToolsParam], sample_output,
                       should_match: bool):
    self = MagicMock(tool_choice="required", tools=tools)
    schema = ChatCompletionRequest._get_guided_json_from_tool(self)
    assert isinstance(schema, dict)

    # use build_regex_from_schema used in JSONLogitsProcessor to create Guide
    from outlines_core.fsm.json_schema import build_regex_from_schema
    regex = build_regex_from_schema(json.dumps(schema))
    compiled = re.compile(regex)
    matches = compiled.fullmatch(json.dumps(sample_output)) is not None

    assert matches == should_match


VALID_TOOL_OUTPUTS = [
    ([{
        "name": "get_current_weather",
        "parameters": {
            "city": "Vienna"
        }
    }], True),
    ([{
        "name": "get_current_weather",
        "parameters": {
            "city": "Vienna"
        }
    }, {
        "name": "get_current_weather",
        "parameters": {
            "city": "Berlin"
        }
    }], True),
    ([{
        "name": "get_forecast",
        "parameters": {
            "city": "Vienna",
            "days": 7
        }
    }], True),
    ([{
        "name": "get_forecast",
        "parameters": {
            "city": "Vienna",
            "days": 7
        }
    }, {
        "name": "get_current_weather",
        "parameters": {
            "city": "Vienna"
        }
    }], True),
]


@pytest.mark.parametrize(
    "sample_output, should_match",
    VALID_TOOL_OUTPUTS + [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [{  # function without required parameters cannot be generated
                "name": "get_current_weather"
            }],
            False),
        (
            [{  # function without required parameters cannot be generated
                "name": "get_current_weather",
                "parameters": {}
            }],
            False),
        (
            [{  # function without required parameters cannot be generated
                "name": "get_current_weather",
                "parameters": None
            }],
            False),
        (
            {  # tool call without lists cannot be generated
                "name": "get_current_weather",
                "parameters": {
                    "city": "Vienna"
                }
            },
            False),
        (
            [{  # tool call with extra parameters cannot be generated
                "name": "get_current_weather",
                "parameters": {
                    "city": "Vienna",
                    "extra": "value"
                }
            }],
            False),
        (
            [{  # tool call without all required parameters cannot be generated
                "name": "get_forecast",
                "parameters": {
                    "city": "Vienna"
                }
            }],
            False),
        (  # tool call with incorrect name/parameters cannot be generated
            [{
                "name": "get_weather",
                "parameters": {
                    "city": "Vienna",
                    "days": 7
                }
            }], False),
        (  #  tool call with both valid and empty function cannot be generated
            [{
                "name": "get_current_weather",
                "parameters": {
                    "city": "Vienna"
                }
            }, {}], False),
    ])
def test_guided_json(sample_output, should_match):
    _compile_and_check(tools=TypeAdapter(
        List[ChatCompletionToolsParam]).validate_python(EXAMPLE_TOOLS),
                       sample_output=sample_output,
                       should_match=should_match)


def update_parameters_none(
        tool: ChatCompletionToolsParam) -> ChatCompletionToolsParam:
    tool.function.parameters = None
    return tool


def update_parameters_empty_dict(
        tool: ChatCompletionToolsParam) -> ChatCompletionToolsParam:
    tool.function.parameters = {}
    return tool


@pytest.mark.parametrize(
    "sample_output, should_match",
    [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [{  # function without required parameters cannot be generated
                "name": "get_current_weather"
            }],
            False),
        (
            [{  # function without required parameters cannot be generated
                "name": "get_current_weather",
                "parameters": None
            }],
            False),
        (
            [{  # function with extra parameters cannot be generated
                "name": "get_current_weather",
                "parameters": {
                    "extra": "value"
                }
            }],
            False),
        (
            [{  # only function with empty parameters object is valid
                "name": "get_current_weather",
                "parameters": {}
            }],
            True),
    ])
@pytest.mark.parametrize(
    "update_parameters",
    [update_parameters_none, update_parameters_empty_dict])
def test_guided_json_without_parameters(sample_output, should_match,
                                        update_parameters):
    updated_tools = [deepcopy(EXAMPLE_TOOLS[0])]
    tools = TypeAdapter(
        List[ChatCompletionToolsParam]).validate_python(updated_tools)
    tools = list(map(update_parameters, tools))
    assert all([
        tool.function.parameters is None or tool.function.parameters == {}
        for tool in tools
    ])
    _compile_and_check(tools=tools,
                       sample_output=sample_output,
                       should_match=should_match)
