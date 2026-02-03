"""
ctypes wrapper for NCCL (libnccl.so).

Provides ncclUniqueId, ncclComm_t, and the minimal API for single-node
data parallel: ncclGetUniqueId, ncclCommInitRank, ncclAllReduce (in-place),
ncclCommDestroy. Uses default CUDA stream (NULL) for v1.
"""

from __future__ import annotations

import ctypes
import os

# NCCL_UNIQUE_ID_BYTES from nccl.h
NCCL_UNIQUE_ID_BYTES = 128

# ncclDataType_t (nccl.h): ncclFloat32 = 7
ncclFloat32 = 7

# ncclRedOp_t: ncclSum = 0
ncclSum = 0

# ncclResult_t: ncclSuccess = 0
ncclSuccess = 0


def _load_libnccl() -> ctypes.CDLL:
    """Load NCCL library. GRADTUITY_NCCL_LIBRARY must be set."""
    lib_name = os.environ.get("GRADTUITY_NCCL_LIBRARY")
    if not lib_name:
        raise RuntimeError(
            "GRADTUITY_NCCL_LIBRARY is not set. Set it to the NCCL library name or path (e.g. libnccl.so.2)."
        )
    return ctypes.CDLL(lib_name)


_libnccl = _load_libnccl()


# ncclUniqueId: struct with char internal[NCCL_UNIQUE_ID_BYTES]
class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


# ncclComm_t is an opaque pointer
ncclComm_t = ctypes.c_void_p

# cudaStream_t for default stream
cudaStream_t = ctypes.c_void_p
CUDA_STREAM_DEFAULT = None  # NULL


def _check_nccl(result: int) -> None:
    if result != ncclSuccess:
        try:
            err_str = _libnccl.ncclGetErrorString(ctypes.c_int(result))
            if err_str:
                msg = ctypes.string_at(err_str).decode("utf-8", errors="replace")
            else:
                msg = f"nccl error {result}"
        except Exception:
            msg = f"nccl error {result}"
        raise RuntimeError(msg)


# ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
_libnccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
_libnccl.ncclGetUniqueId.restype = ctypes.c_int


def nccl_get_unique_id() -> bytes:
    """Generate a unique ID for communicator creation. Returns bytes (128)."""
    # Use a raw buffer so we always get NCCL_UNIQUE_ID_BYTES regardless of struct layout/ABI
    buf = ctypes.create_string_buffer(NCCL_UNIQUE_ID_BYTES)
    result = _libnccl.ncclGetUniqueId(
        ctypes.cast(ctypes.byref(buf), ctypes.POINTER(_NcclUniqueId))
    )
    _check_nccl(result)
    return bytes(buf.raw)


# ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
_libnccl.ncclCommInitRank.argtypes = [
    ctypes.POINTER(ncclComm_t),
    ctypes.c_int,
    _NcclUniqueId,
    ctypes.c_int,
]
_libnccl.ncclCommInitRank.restype = ctypes.c_int


def nccl_comm_init_rank(nranks: int, comm_id: bytes, rank: int) -> int:
    """
    Create NCCL communicator for this rank.

    Args:
        nranks: Total number of ranks.
        comm_id: Bytes from nccl_get_unique_id() (length NCCL_UNIQUE_ID_BYTES).
        rank: This process's rank (0 .. nranks - 1).

    Returns:
        Communicator handle (opaque pointer value as int).
    """
    if len(comm_id) != NCCL_UNIQUE_ID_BYTES:
        raise ValueError(
            f"comm_id must be {NCCL_UNIQUE_ID_BYTES} bytes, got {len(comm_id)}"
        )
    uid = _NcclUniqueId.from_buffer_copy(comm_id)
    comm = ncclComm_t()
    result = _libnccl.ncclCommInitRank(ctypes.byref(comm), nranks, uid, rank)
    _check_nccl(result)
    return comm.value


# ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
#                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
#                           cudaStream_t stream);
_libnccl.ncclAllReduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ncclComm_t,
    cudaStream_t,
]
_libnccl.ncclAllReduce.restype = ctypes.c_int


def _nccl_all_reduce_inplace(
    comm: int, ptr: int, count: int, stream: int | None = None
) -> None:
    """In-place AllReduce; comm and stream passed by caller (comm.py)."""
    send_recv = ctypes.c_void_p(ptr)
    stream_ptr = ctypes.c_void_p(stream if stream is not None else 0)
    result = _libnccl.ncclAllReduce(
        send_recv,
        send_recv,
        count,
        ncclFloat32,
        ncclSum,
        ncclComm_t(comm),
        stream_ptr,
    )
    _check_nccl(result)


# ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
#                            ncclDataType_t datatype, int root, ncclComm_t comm,
#                            cudaStream_t stream);
_libnccl.ncclBroadcast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ncclComm_t,
    cudaStream_t,
]
_libnccl.ncclBroadcast.restype = ctypes.c_int


def _nccl_broadcast_inplace(
    comm: int, ptr: int, count: int, root: int, stream: int | None = None
) -> None:
    """In-place Broadcast from root; all ranks use ptr (root sends, others receive)."""
    send_recv = ctypes.c_void_p(ptr)
    stream_ptr = ctypes.c_void_p(stream if stream is not None else 0)
    result = _libnccl.ncclBroadcast(
        send_recv,
        send_recv,
        count,
        ncclFloat32,
        ctypes.c_int(root),
        ncclComm_t(comm),
        stream_ptr,
    )
    _check_nccl(result)


# ncclResult_t ncclCommDestroy(ncclComm_t comm);
_libnccl.ncclCommDestroy.argtypes = [ncclComm_t]
_libnccl.ncclCommDestroy.restype = ctypes.c_int


def nccl_comm_destroy(comm: int) -> None:
    """Destroy communicator."""
    result = _libnccl.ncclCommDestroy(ncclComm_t(comm))
    _check_nccl(result)


# Optional: ncclGetErrorString for better error messages
try:
    _libnccl.ncclGetErrorString.argtypes = [ctypes.c_int]
    _libnccl.ncclGetErrorString.restype = ctypes.c_char_p
except AttributeError:
    pass
