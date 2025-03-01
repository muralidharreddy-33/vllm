# Logging Configuration

vLLM leverages Python's `logging.config.dictConfig` functionality to enable
robust and flexible configuration of the various loggers used by vLLM.

vLLM offers two environment variables that can be used to accommodate a range
of logging configurations that range from simple-and-inflexible to
more-complex-and-more-flexible.

- No vLLM logging (simple and inflexible)
  - Set `VLLM_CONFIGURE_LOGGING=0` (leaving `VLLM_LOGGING_CONFIG_PATH` unset)
- vLLM's default logging configuration (simple and inflexible)
  - Leave `VLLM_CONFIGURE_LOGGING` unset or set `VLLM_CONFIGURE_LOGGING=1`
- Fine-grained custom logging configuration (more complex, more flexible)
  - Leave `VLLM_CONFIGURE_LOGGING` unset or set `VLLM_CONFIGURE_LOGGING=1` and
    set `VLLM_LOGGING_CONFIG_PATH=<path-to-logging-config.json>`

## Logging Configuration Environment Variables

### `VLLM_CONFIGURE_LOGGING`

`VLLM_CONFIGURE_LOGGING` controls whether or not vLLM takes any action to
configure the loggers used by vLLM. This functionality is enabled by default,
but can be disabled by setting `VLLM_CONFIGURE_LOGGING=0` when running vLLM.

If `VLLM_CONFIGURE_LOGGING` is enabled and no value is given for
`VLLM_LOGGING_CONFIG_PATH`, vLLM will use built-in default configuration to
configure the root vLLM logger. By default, no other vLLM loggers are
configured and, as such, all vLLM loggers defer to the root vLLM logger to make
all logging decisions.

If `VLLM_CONFIGURE_LOGGING` is disabled and a value is given for
`VLLM_LOGGING_CONFIG_PATH`, an error will occur while starting vLLM.

### `VLLM_LOGGING_CONFIG_PATH`

`VLLM_LOGGING_CONFIG_PATH` allows users to specify a path to a JSON file of
alternative, custom logging configuration that will be used instead of vLLM's
built-in default logging configuration. The logging configuration should be
provided in JSON format following the schema specified by Python's [logging
configuration dictionary
schema](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

If `VLLM_LOGGING_CONFIG_PATH` is specified, but `VLLM_CONFIGURE_LOGGING` is
disabled, an error will occur while starting vLLM.

## Examples

### Example 1: Customize vLLM root logger

For this example, we will customize the vLLM root logger to use
[`python-json-logger`](https://github.com/nhairs/python-json-logger)
(which is part of the container image) to log to
STDOUT of the console in JSON format with a log level of `INFO`.

To begin, first, create an appropriate JSON logging configuration file:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "console": {
      "class" : "logging.StreamHandler",
      "formatter": "json",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["console"],
      "level": "INFO",
      "propagate": false
    }
  },
  "version": 1
}
```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 2: Silence a particular vLLM logger

To silence a particular vLLM logger, it is necessary to provide custom logging
configuration for the target logger that configures the logger so that it won't
propagate its log messages to the root vLLM logger.

When custom configuration is provided for any logger, it is also necessary to
provide configuration for the root vLLM logger since any custom logger
configuration overrides the built-in default logging configuration used by vLLM.

First, create an appropriate JSON logging configuration file that includes
configuration for the root vLLM logger and for the logger you wish to silence:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "vllm": {
      "class": "vllm.logging_utils.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "vllm": {
      "class" : "logging.StreamHandler",
      "formatter": "vllm",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["vllm"],
      "level": "DEBUG",
      "propagage": false
    },
    "vllm.example_noisy_logger": {
      "propagate": false
    }
  },
  "version": 1
}
```

Finally, run vLLM with the `VLLM_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 3: Disable vLLM default logging configuration

To disable vLLM's default logging configuration and silence all vLLM loggers,
simple set `VLLM_CONFIGURE_LOGGING=0` when running vLLM. This will prevent vLLM
for configuring the root vLLM logger, which in turn, silences all other vLLM
loggers.

```bash
VLLM_CONFIGURE_LOGGING=0 \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### Example 4: Structured JSON Logging with Loguru

vLLM also supports structured JSON logging using the [`loguru`](https://github.com/Delgan/loguru) library. This approach provides more features for log handling and a more ergonomic format for log aggregation and analysis.

:::{note}
Note that to use this feature you will need to install `loguru` by running `pip install loguru`.
:::

Then, create a configuration file with a different format:

**/path/to/logging_config.json:**

```json
{
  "logger": {
    "level": "info",
    "rotation": "20 MB",
    "retention": "1 week",
    "format": "<level>{level: <8}</level> <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
  }
}
```

For file logging (optional), you can add file paths:

```json
{
  "logger": {
    "structured_log_file_path": "/path/to/logs/vllm_structured.log",
    "unstructured_log_file_path": "/path/to/logs/vllm.log",
    "level": "info",
    "rotation": "20 MB",
    "retention": "1 week",
    "format": "<level>{level: <8}</level> <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
  }
}
```

The loguru configuration supports:
- `level`: Log level (debug, info, warning, error, critical)
- `rotation`: When to rotate logs (size or time-based)
- `retention`: How long to keep logs
- `format`: Log message format with rich text formatting
- `structured_log_file_path`: Path for JSON structured logs (optional)
- `unstructured_log_file_path`: Path for human-readable logs (optional)

Run vLLM with this configuration:

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

## Additional resources

- [`logging.config` Dictionary Schema Details](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
