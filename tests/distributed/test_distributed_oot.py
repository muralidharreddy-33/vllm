from ..plugins.test_model_plugin import run_and_test_dummy_opt_api_server


def test_distributed_oot(dummy_opt_path: str):
    run_and_test_dummy_opt_api_server(dummy_opt_path, tp=2)
