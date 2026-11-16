// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
static void nccl_check(ncclResult_t result) {
  if (result != ncclSuccess) throw std::runtime_error(ncclGetErrorString(result));
}

__global__ void send_chunk(ncclDevComm comm, ncclWindow_t window, size_t offset, size_t bytes,
                           unsigned slot, unsigned slots, uint64_t sequence, uint64_t timeout,
                           int *error) {
  ncclGin gin(comm, 0);
  auto status = gin.waitSignal(ncclCoopThread{}, slots + slot, sequence - 1, 64,
                               cuda::memory_order_acquire, timeout);
  if (status != ncclSuccess) {
    *error = int(status);
    return;
  }
  gin.put(ncclTeamWorld(comm), 1, window, offset, window, offset, bytes,
          ncclGin_StrongSignalInc{slot}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
          cuda::thread_scope_system, cuda::thread_scope_system);
  status = gin.flush(ncclCoopThread{}, cuda::memory_order_acquire, ncclGin_None{}, timeout);
  if (status != ncclSuccess) *error = int(status);
}

__global__ void wait_chunk(ncclDevComm comm, unsigned slot, uint64_t sequence, uint64_t timeout,
                           int *error) {
  ncclGin gin(comm, 0);
  auto status =
      gin.waitSignal(ncclCoopThread{}, slot, sequence, 64, cuda::memory_order_acquire, timeout);
  if (status != ncclSuccess) *error = int(status);
}

__global__ void drain_sender(ncclDevComm comm, unsigned slot, unsigned slots, uint64_t sequence,
                             uint64_t timeout, int *error) {
  ncclGin gin(comm, 0);
  auto status = gin.waitSignal(ncclCoopThread{}, slots + slot, sequence, 64,
                               cuda::memory_order_acquire, timeout);
  if (status != ncclSuccess) *error = int(status);
}

__global__ void ack_chunk(ncclDevComm comm, unsigned slot, unsigned slots, uint64_t timeout,
                          int *error) {
  if (*error) return;
  ncclGin gin(comm, 0);
  gin.signal(ncclTeamWorld(comm), 0, ncclGin_StrongSignalInc{slots + slot}, ncclCoopThread{},
             ncclGin_None{}, cuda::thread_scope_system, cuda::thread_scope_system);
  auto status = gin.flush(ncclCoopThread{}, cuda::memory_order_acquire, ncclGin_None{}, timeout);
  if (status != ncclSuccess) *error = int(status);
}

class Endpoint {
  ncclComm_t comm_ = nullptr;
  ncclDevComm dev_{};
  ncclWindow_t window_ = nullptr;
  void *buffer_ = nullptr;
  int *error_ = nullptr;
  bool dev_created_ = false;
  bool owns_buffer_ = true;
  int rank_, device_;
  size_t slot_bytes_;
  unsigned slots_;
  uint64_t timeout_;
  std::vector<uint64_t> sequence_;
  cudaStream_t stream_ = nullptr;

  void wait_ready(double timeout_seconds = 30.0) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(int(timeout_seconds * 1000));
    ncclResult_t status;
    do {
      nccl_check(ncclCommGetAsyncError(comm_, &status));
      if (status != ncclInProgress) {
        nccl_check(status);
        return;
      }
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("Sampling-mask GIN host operation timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (true);
  }

 public:
  Endpoint(const std::string &id, int rank, int device, size_t slot_bytes, unsigned slots,
           double timeout_seconds, uintptr_t reserved_buffer)
      : rank_(rank), device_(device), slot_bytes_(slot_bytes), slots_(slots), sequence_(slots, 0) {
    if (id.size() != sizeof(ncclUniqueId) || (rank != 0 && rank != 1) || slot_bytes == 0
        || slots == 0 || timeout_seconds <= 0)
      throw std::invalid_argument("Invalid sampling-mask GIN endpoint parameters");
    cuda_check(cudaSetDevice(device));
    cudaDeviceProp device_props;
    cuda_check(cudaGetDeviceProperties(&device_props, device));
    timeout_ = uint64_t(timeout_seconds * device_props.clockRate * 1000.0);
    ncclUniqueId unique_id;
    std::memcpy(&unique_id, id.data(), sizeof(unique_id));
    const char *stage = "communicator initialization";
    try {
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      ncclResult_t initializing = ncclCommInitRankConfig(&comm_, 2, unique_id, rank, &config);
      if (initializing != ncclSuccess && initializing != ncclInProgress) nccl_check(initializing);
      nccl_check(ncclCommGetAsyncError(comm_, &initializing));
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(int(timeout_seconds * 1000));
      while (initializing == ncclInProgress) {
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("Sampling-mask GIN communicator initialization timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        nccl_check(ncclCommGetAsyncError(comm_, &initializing));
      }
      nccl_check(initializing);
      stage = "capability query";
      ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
      nccl_check(ncclCommQueryProperties(comm_, &properties));
      if (!properties.deviceApiSupport)
        throw std::runtime_error(
            "NCCL device API unavailable: set NCCL_CUMEM_ENABLE=1 before process startup");
      if (properties.ginType != NCCL_GIN_TYPE_GDAKI)
        throw std::runtime_error("NCCL GIN GDAKI is unavailable; mask transport cannot fall back");
      owns_buffer_ = reserved_buffer == 0;
      stage = "window allocation";
      if (owns_buffer_)
        nccl_check(ncclMemAlloc(&buffer_, slot_bytes * slots));
      else
        buffer_ = reinterpret_cast<void *>(reserved_buffer);
      stage = "symmetric window registration";
      nccl_check(ncclCommWindowRegister(comm_, buffer_, slot_bytes * slots, &window_,
                                        NCCL_WIN_STRICT_ORDERING));
      wait_ready(timeout_seconds);
      if (!window_) throw std::runtime_error("NCCL returned an unregistered symmetric window");
      ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      requirements.ginType = NCCL_GIN_TYPE_GDAKI;
      requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      requirements.ginContextCount = 1;
      requirements.ginSignalCount = 2 * slots;
      requirements.ginExclusiveContexts = true;
      requirements.ginQueueDepth = 32;
      stage = "device communicator creation";
      nccl_check(ncclDevCommCreate(comm_, &requirements, &dev_));
      wait_ready(timeout_seconds);
      dev_created_ = true;
      cuda_check(cudaMalloc(&error_, sizeof(int)));
      cuda_check(cudaMemset(error_, 0, sizeof(int)));
      cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    } catch (const std::exception &error) {
      std::string message = std::string("Sampling-mask GIN ") + stage + ": " + error.what();
      abort();
      throw std::runtime_error(message);
    }
  }

  // Each endpoint owns one stream/context. Calls are serialized by its Python
  // owner. Source/destination tensors must remain live through finish().
  void transfer(uintptr_t pointer, size_t bytes, unsigned slot, uint64_t sequence) {
    if (!comm_ || slot >= slots_ || bytes > slot_bytes_ || bytes == 0
        || sequence != sequence_[slot] + 1)
      throw std::invalid_argument("Invalid GIN slot, size, or sequence number");
    cuda_check(cudaSetDevice(device_));
    const size_t offset = slot * slot_bytes_;
    if (rank_ == 0) {
      cuda_check(cudaMemcpyAsync(static_cast<char *>(buffer_) + offset,
                                 reinterpret_cast<void *>(pointer), bytes, cudaMemcpyDeviceToDevice,
                                 stream_));
      send_chunk<<<1, 1, 0, stream_>>>(dev_, window_, offset, bytes, slot, slots_, sequence,
                                       timeout_, error_);
    } else {
      wait_chunk<<<1, 1, 0, stream_>>>(dev_, slot, sequence, timeout_, error_);
      cuda_check(cudaMemcpyAsync(reinterpret_cast<void *>(pointer),
                                 static_cast<char *>(buffer_) + offset, bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      ack_chunk<<<1, 1, 0, stream_>>>(dev_, slot, slots_, timeout_, error_);
    }
    cuda_check(cudaGetLastError());
    sequence_[slot] = sequence;
  }

  void finish() {
    if (!comm_) throw std::runtime_error("Sampling-mask GIN endpoint is closed");
    cuda_check(cudaSetDevice(device_));
    cuda_check(cudaStreamSynchronize(stream_));
    int error = 0;
    cuda_check(cudaMemcpy(&error, error_, sizeof(error), cudaMemcpyDeviceToHost));
    ncclResult_t async_error;
    nccl_check(ncclCommGetAsyncError(comm_, &async_error));
    nccl_check(async_error);
    if (error)
      throw std::runtime_error("Sampling-mask GIN transfer timed out or failed: "
                               + std::to_string(error));
  }

  void close() {
    if (!comm_) return;
    if (rank_ == 0) {
      for (unsigned slot = 0; slot < slots_; ++slot)
        drain_sender<<<1, 1, 0, stream_>>>(dev_, slot, slots_, sequence_[slot], timeout_, error_);
    }
    finish();
    nccl_check(ncclDevCommDestroy(comm_, &dev_));
    dev_created_ = false;
    nccl_check(ncclCommWindowDeregister(comm_, window_));
    window_ = nullptr;
    if (owns_buffer_) nccl_check(ncclMemFree(buffer_));
    buffer_ = nullptr;
    cuda_check(cudaFree(error_));
    error_ = nullptr;
    cuda_check(cudaStreamDestroy(stream_));
    stream_ = nullptr;
    nccl_check(ncclCommDestroy(comm_));
    comm_ = nullptr;
  }

  void abort() noexcept {
    cudaSetDevice(device_);
    // Fatal setup/transport errors invalidate the whole two-rank session.
    if (stream_) cudaStreamSynchronize(stream_);
    if (comm_) ncclCommAbort(comm_);
    comm_ = nullptr;
    if (stream_) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    if (error_) cudaFree(error_);
    error_ = nullptr;
    if (buffer_ && owns_buffer_) ncclMemFree(buffer_);
    buffer_ = nullptr;
    window_ = nullptr;
  }
  ~Endpoint() { abort(); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("reserve", [](size_t bytes, int device) {
    cuda_check(cudaSetDevice(device));
    void *buffer = nullptr;
    nccl_check(ncclMemAlloc(&buffer, bytes));
    return reinterpret_cast<uintptr_t>(buffer);
  });
  module.def("free_reserved", [](uintptr_t pointer, int device) {
    cuda_check(cudaSetDevice(device));
    nccl_check(ncclMemFree(reinterpret_cast<void *>(pointer)));
  });
  module.def("bootstrap_id", []() {
    ncclUniqueId id;
    nccl_check(ncclGetUniqueId(&id));
    return py::bytes(reinterpret_cast<const char *>(&id), sizeof(id));
  });
  py::class_<Endpoint>(module, "Endpoint")
      .def(py::init<const std::string &, int, int, size_t, unsigned, double, uintptr_t>(),
           py::call_guard<py::gil_scoped_release>())
      .def("transfer", &Endpoint::transfer, py::call_guard<py::gil_scoped_release>())
      .def("finish", &Endpoint::finish, py::call_guard<py::gil_scoped_release>())
      .def("close", &Endpoint::close, py::call_guard<py::gil_scoped_release>())
      .def("abort", &Endpoint::abort, py::call_guard<py::gil_scoped_release>());
}
