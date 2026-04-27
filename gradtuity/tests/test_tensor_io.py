"""
Tests for tensor_io.py - SafeTensors-compatible save/load for GPU Tensor.

These tests require a CUDA-enabled GPU to run.
"""

import json
import struct

import pytest

from gradtuity.tensor import Tensor
from gradtuity.tensor_io import load_safetensors, save_safetensors

pytestmark = pytest.mark.requires_cuda


class TestRoundTripSingleTensor:
    """Round-trip save and load a single tensor."""

    def test_round_trip_1d(self, tmp_path):
        t = Tensor([1.0, 2.0, 3.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        state = load_safetensors(str(path))
        assert list(state.keys()) == ["x"]
        assert state["x"].shape == (3,)
        assert state["x"].to_list() == [1.0, 2.0, 3.0]

    def test_round_trip_2d(self, tmp_path):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"M": t})
        state = load_safetensors(str(path))
        assert state["M"].shape == (2, 2)
        assert state["M"].to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_round_trip_4d(self, tmp_path):
        t = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"conv": t})
        state = load_safetensors(str(path))
        assert state["conv"].shape == (1, 1, 2, 2)
        assert state["conv"].to_list() == [[[[1.0, 2.0], [3.0, 4.0]]]]

    def test_round_trip_3d(self, tmp_path):
        t = Tensor([[[1.0], [2.0]]])  # (1, 2, 1)
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        state = load_safetensors(str(path))
        assert state["x"].shape == (1, 2, 1)
        assert state["x"].to_list() == [[[1.0], [2.0]]]


class TestRoundTripStateDict:
    """Round-trip save and load multiple tensors."""

    def test_round_trip_many_tensors(self, tmp_path):
        W = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([0.5, -0.5])
        x = Tensor([1.0, 0.0])
        path = tmp_path / "state.safetensors"
        save_safetensors(str(path), {"W": W, "b": b, "x": x})
        state = load_safetensors(str(path))
        assert set(state.keys()) == {"W", "b", "x"}
        assert state["W"].to_list() == W.to_list()
        assert state["b"].to_list() == b.to_list()
        assert state["x"].to_list() == x.to_list()

    def test_requires_grad_on_load(self, tmp_path):
        t = Tensor([1.0, 2.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"t": t})
        loaded = load_safetensors(str(path), requires_grad=True)
        assert loaded["t"].requires_grad is True
        loaded_no_grad = load_safetensors(str(path), requires_grad=False)
        assert loaded_no_grad["t"].requires_grad is False


class TestDeterminism:
    """Same dict saved twice produces identical files."""

    def test_deterministic_output(self, tmp_path):
        tensors = {
            "b": Tensor([1.0, 2.0]),
            "a": Tensor([[1.0], [2.0]]),
        }
        path1 = tmp_path / "1.safetensors"
        path2 = tmp_path / "2.safetensors"
        save_safetensors(str(path1), tensors)
        save_safetensors(str(path2), tensors)
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            assert f1.read() == f2.read()


class TestCorruption:
    """Corrupt or invalid files raise with clear messages."""

    def test_corrupt_header_byte_fails(self, tmp_path):
        t = Tensor([1.0, 2.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = bytearray(f.read())
        # Flip a byte in the header (after the 8-byte length)
        if len(data) > 20:
            data[15] ^= 0xFF
        with open(path, "wb") as f:
            f.write(data)
        with pytest.raises(
            (ValueError, json.JSONDecodeError, UnicodeDecodeError)
        ) as exc_info:
            load_safetensors(str(path))
        msg = str(exc_info.value).lower()
        assert "corrupt" in msg or "json" in msg or "utf-8" in msg or "decode" in msg

    def test_offset_beyond_data_fails(self, tmp_path):
        # File has 8 bytes of data; corrupt header to claim [0, 16] and shape [4] (16 bytes)
        t = Tensor([1.0, 2.0])  # 8 bytes
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["data_offsets"] = [0, 16]
        header["x"]["shape"] = [4]  # 4 * 4 = 16 bytes claimed
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])  # only 8 bytes of data, so end (16) > data_size (8)
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "offset" in str(exc_info.value).lower()
            or "buffer" in str(exc_info.value).lower()
        )

    def test_invalid_dtype_fails(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["dtype"] = "F64"
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "F32" in str(exc_info.value) or "supported" in str(exc_info.value).lower()
        )

    def test_invalid_rank_fails(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["shape"] = [1, 2, 3, 4, 5]  # rank 5
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert "rank" in str(exc_info.value).lower() or "1..4" in str(exc_info.value)

    def test_offsets_hole_fails(self, tmp_path):
        # Two tensors with a gap: [0, 8] then [16, 24] (hole between 8 and 16).
        # Data region must be 24 bytes so both ranges are within bounds.
        t1 = Tensor([1.0, 2.0])  # 8 bytes
        t2 = Tensor([3.0, 4.0])  # 8 bytes
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"a": t1, "b": t2})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["a"]["data_offsets"] = [0, 8]
        header["b"]["data_offsets"] = [16, 24]  # hole
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        data_region = data[8 + n :]  # 16 bytes; pad to 24 so [16,24] is in bounds
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data_region + b"\x00" * 8)
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert "holes" in str(exc_info.value).lower()

    def test_offsets_overlap_fails(self, tmp_path):
        # Two tensors overlapping: [0, 8] and [4, 12]
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([3.0, 4.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"a": t1, "b": t2})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["a"]["data_offsets"] = [0, 8]
        header["b"]["data_offsets"] = [4, 12]
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert "overlap" in str(exc_info.value).lower()

    def test_file_too_small_for_header_length(self, tmp_path):
        path = tmp_path / "t.safetensors"
        path.write_bytes(struct.pack("<Q", 100)[:4])  # only 4 bytes
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "truncated" in str(exc_info.value).lower()
            or "header" in str(exc_info.value).lower()
        )

    def test_truncated_header(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        # Write correct length but only half the header bytes
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))
            f.write(data[8 : 8 + n // 2])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "truncated" in str(exc_info.value).lower()
            or "header" in str(exc_info.value).lower()
        )

    def test_truncated_data(self, tmp_path):
        t = Tensor([1.0, 2.0, 3.0, 4.0])  # 16 bytes
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        # Remove last 4 bytes of data
        with open(path, "wb") as f:
            f.write(data[:-4])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "truncated" in str(exc_info.value).lower()
            or "data" in str(exc_info.value).lower()
        )

    def test_header_leading_whitespace_loads(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header_raw = data[8 : 8 + n]
        # Prepend whitespace and update length
        new_header = b"  \n\t" + header_raw
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(new_header)))
            f.write(new_header)
            f.write(data[8 + n :])
        state = load_safetensors(str(path))
        assert state["x"].to_list() == [1.0]

    def test_header_no_brace_fails(self, tmp_path):
        path = tmp_path / "t.safetensors"
        # 8-byte length = 2, then 2 bytes that are not {
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", 2))
            f.write(b"[]")
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert "start" in str(exc_info.value).lower() or "{" in str(exc_info.value)


class TestConvenienceAPI:
    """Tensor.save and Tensor.load."""

    def test_save_and_load(self, tmp_path):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "t.safetensors"
        t.save(str(path), name="M")
        t2 = Tensor.load(str(path), name="M")
        assert t2.to_list() == t.to_list()
        assert t2.shape == t.shape

    def test_load_missing_key_raises(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with pytest.raises(KeyError) as exc_info:
            Tensor.load(str(path), name="nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestIncludeGrads:
    """Optional include_grads=True saves .grad tensors."""

    def test_include_grads_round_trip(self, tmp_path):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad = Tensor([0.1, 0.2])  # simulate having a grad
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"t": t}, include_grads=True)
        state = load_safetensors(str(path))
        assert "t" in state
        assert "t.grad" in state
        assert state["t"].shape == (2,)
        assert state["t.grad"].shape == (2,)
        assert state["t"].to_list() == [1.0, 2.0]
        assert state["t.grad"].to_list() == pytest.approx([0.1, 0.2])

    def test_include_grads_skips_when_no_grad(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"t": t}, include_grads=True)
        state = load_safetensors(str(path))
        assert list(state.keys()) == ["t"]


class TestMetadata:
    """Custom metadata is stored and valid."""

    def test_custom_metadata(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(
            str(path), {"x": t}, metadata={"creator": "test", "version": "1"}
        )
        state = load_safetensors(str(path))
        assert state["x"].to_list() == [1.0]

    def test_metadata_values_must_be_strings(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        with pytest.raises(ValueError) as exc_info:
            save_safetensors(str(path), {"x": t}, metadata={"creator": 123})
        assert (
            "metadata" in str(exc_info.value).lower()
            and "string" in str(exc_info.value).lower()
        )

    def test_metadata_merges_into_header(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t}, metadata={"custom_key": "custom_value"})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        assert "__metadata__" in header
        assert header["__metadata__"].get("custom_key") == "custom_value"


class TestValidation:
    """Input validation on save; device validation on load."""

    def test_empty_tensors_raises(self, tmp_path):
        path = tmp_path / "t.safetensors"
        with pytest.raises(ValueError) as exc_info:
            save_safetensors(str(path), {})
        assert (
            "empty" in str(exc_info.value).lower()
            or "not" in str(exc_info.value).lower()
        )

    def test_device_not_cuda_raises(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path), device="cpu")
        assert "cuda" in str(exc_info.value).lower()

    def test_non_string_key_raises(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        with pytest.raises(ValueError) as exc_info:
            save_safetensors(str(path), {123: t})  # type: ignore[dict-item]
        assert (
            "key" in str(exc_info.value).lower()
            and "string" in str(exc_info.value).lower()
        )

    def test_include_grads_key_collision_raises(self, tmp_path):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad = Tensor([0.1, 0.2])
        other = Tensor([9.0, 9.0])
        path = tmp_path / "t.safetensors"
        with pytest.raises(ValueError) as exc_info:
            save_safetensors(
                str(path),
                {"t": t, "t.grad": other},
                include_grads=True,
            )
        msg = str(exc_info.value).lower()
        assert "collide" in msg or "t.grad" in msg


class TestLoaderHeaderEdgeCases:
    """Loader validation: shape/dtype/offsets edge cases."""

    def test_negative_dim_fails(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["shape"] = [-1]
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "rank" in str(exc_info.value).lower()
            or "dim" in str(exc_info.value).lower()
        )

    def test_non_int_dim_fails(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["shape"] = [1.5]
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "rank" in str(exc_info.value).lower()
            or "dim" in str(exc_info.value).lower()
        )

    def test_data_offsets_not_length_two_fails(self, tmp_path):
        t = Tensor([1.0])
        path = tmp_path / "t.safetensors"
        save_safetensors(str(path), {"x": t})
        with open(path, "rb") as f:
            data = f.read()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
        header["x"]["data_offsets"] = [0]
        bad_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
            f.write(data[8 + n :])
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "data_offsets" in str(exc_info.value).lower()
            or "offset" in str(exc_info.value).lower()
        )

    def test_missing_required_fields_fails(self, tmp_path):
        path = tmp_path / "t.safetensors"
        # Minimal valid-looking header but missing dtype
        header = {"x": {"shape": [1], "data_offsets": [0, 4]}}
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            f.write(b"\x00" * 4)  # 4 bytes of data
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(str(path))
        assert (
            "dtype" in str(exc_info.value).lower()
            or "missing" in str(exc_info.value).lower()
        )


class TestSaveTimeValidation:
    """Save-time validation: rank and shape."""

    def test_save_rejects_rank_5(self, tmp_path):
        # Tensor._wrap rejects rank 5; use a normal tensor and override _shape
        # so save_safetensors' own validation is triggered.
        t = Tensor([1.0])
        t._shape = (1, 1, 1, 1, 1)
        path = tmp_path / "t.safetensors"
        with pytest.raises(ValueError) as exc_info:
            save_safetensors(str(path), {"x": t})
        assert "rank" in str(exc_info.value).lower() or "1..4" in str(exc_info.value)
