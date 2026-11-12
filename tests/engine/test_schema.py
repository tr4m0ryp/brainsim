import numpy as np
import pytest

from brainsim.schema.tables import CANONICAL_COLUMNS, HAVE_PYARROW, ConnectivityTable, validate_columns


def _tiny_table() -> ConnectivityTable:
    n = 4
    return ConnectivityTable(
        pre_id=np.array([0, 0, 1, 2]),
        post_id=np.array([1, 2, 2, 0]),
        weight=np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32),
        sign=np.array([1, -1, 1, 1], dtype=np.int8),
        cell_type=np.array(["a", "b", "a", "c"], dtype=object),
        dataset=np.full(n, "unit-test", dtype=object),
        version=np.full(n, "v0", dtype=object),
    )


def test_validate_columns_fails_loudly_on_missing():
    with pytest.raises(ValueError):
        validate_columns({"pre_id", "post_id", "weight"})
    validate_columns(set(CANONICAL_COLUMNS) | {"extra_column"})


def test_from_dict_missing_column_raises():
    data = _tiny_table().to_dict()
    del data["sign"]
    with pytest.raises(ValueError):
        ConnectivityTable.from_dict(data)


def test_bad_sign_raises():
    with pytest.raises(ValueError):
        ConnectivityTable(
            pre_id=np.array([0]),
            post_id=np.array([1]),
            weight=np.array([1.0]),
            sign=np.array([0]),  # invalid: must be -1 or +1
            cell_type=np.array(["a"], dtype=object),
            dataset=np.array(["d"], dtype=object),
            version=np.array(["v"], dtype=object),
        )


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        ConnectivityTable(
            pre_id=np.array([0]),
            post_id=np.array([1]),
            weight=np.array([-1.0]),
            sign=np.array([1]),
            cell_type=np.array(["a"], dtype=object),
            dataset=np.array(["d"], dtype=object),
            version=np.array(["v"], dtype=object),
        )


@pytest.mark.skipif(not HAVE_PYARROW, reason="pyarrow not installed")
def test_parquet_roundtrip(tmp_path):
    table = _tiny_table()
    path = tmp_path / "connectivity.parquet"
    table.write_parquet(path)
    loaded = ConnectivityTable.read_parquet(path)
    assert loaded.n_synapses == table.n_synapses
    assert np.array_equal(loaded.pre_id, table.pre_id)
    assert np.array_equal(loaded.post_id, table.post_id)
    assert np.allclose(loaded.weight, table.weight)
    assert np.array_equal(loaded.sign, table.sign)


@pytest.mark.skipif(not HAVE_PYARROW, reason="pyarrow not installed")
def test_read_parquet_missing_columns_fails_loudly(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"pre_id": [0], "post_id": [1]}), str(path))
    with pytest.raises(ValueError):
        ConnectivityTable.read_parquet(path)


@pytest.mark.skipif(HAVE_PYARROW, reason="only exercised when pyarrow is absent")
def test_write_parquet_without_pyarrow_raises(tmp_path):
    table = _tiny_table()
    with pytest.raises(RuntimeError):
        table.write_parquet(tmp_path / "x.parquet")
