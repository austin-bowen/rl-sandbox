from pathlib import Path

import mlflow

import rlsandbox


def log_code() -> None:
    code_path = Path(rlsandbox.__file__).parent
    mlflow.log_artifacts(str(code_path), artifact_path=f'code/{code_path.name}')


def log_metric(key: str, value, step: int = None, to_mlflow: bool = True) -> None:
    if step is None:
        print(f'{key}={value}')
    else:
        print(f'[step {step}] {key}={value}')

    if to_mlflow:
        mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: dict[str, ...], step: int = None, to_mlflow: bool = True) -> None:
    for key, value in metrics.items():
        log_metric(key, value, step=step, to_mlflow=False)

    if to_mlflow:
        mlflow.log_metrics(metrics, step=step)
