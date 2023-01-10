def test_func1(data_path, split):
    from shutil import move, rmtree

    rmtree(data_path / split), move(data_path / "sample", data_path / split)


def test_func2(api, data_path):
    api.competition_download_file(
        "diabetic-retinopathy-detection", "sample.zip", path=data_path
    )
    api.competition_download_file(
        "diabetic-retinopathy-detection", "trainLabels.csv.zip", path=data_path
    )
