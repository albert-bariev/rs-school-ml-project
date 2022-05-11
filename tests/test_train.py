from src.rs_ml_project.train import train


def test_manual_model_fit(capsys) -> None:
    train([
        "-t", "True",
        "-d", "././tests/test_data.csv",
        "--target", "Cover_Type",
        "-m", "knn",
        "-n", "10",
        "-sc", "std"
    ])
    captured = capsys.readouterr()
    assert captured.out == "{'accuracy': 0.624, 'precision': 0.6278909959102664, 'recall': 0.624, 'f1_score': 0.6176700515938143}\n"
