/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#ifdef _WIN64
#define NOMINMAX
#endif /* _WIN64 */

#include "helper_multiprocess.hh"

#include <hip_test_common.hh>
#include <utils.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipMemPoolExportToShareableHandle hipMemPoolExportToShareableHandle
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
 * hipMemAllocationHandleType handle_type, unsigned int flags)` - Exports a memory pool to the
 * requested handle type.
 */

#ifdef __linux__

static const char shm_name[] = "mempool_test_shm";
static const char ipc_name[] = "mempool_test_pipe";

static constexpr int kMaxDevices = 8;

typedef struct shmStruct_st {
  Process processes[kMaxDevices];
  hipMemPoolPtrExportData exportPtrData[kMaxDevices];
} shmStruct;

typedef struct ipcBarrier {
  int count;
  bool sense;
  bool allExit;
} ipcBarrier_t;

typedef struct ipcDevices {
  int count;
  int ordinals[kMaxDevices];
} ipcDevices_t;

static ipcBarrier_t* g_Barrier{};
static bool g_procSense;
static int g_processCnt;

/*
  Get device with P2P access to device 0.
*/
static void get_devices(ipcDevices_t* devices) {
  pid_t pid = fork();

  if (!pid) {
    // HIP APIs are called in child process,
    // to avoid HIP initialization in main process.
    int i, device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    int mem_pool_support = 0;
    HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
    if (!mem_pool_support) {
      devices->count = 0;
      exit(EXIT_SUCCESS);
    }

    // Device 0
    devices->ordinals[0] = 0;
    devices->count = 1;

    if (device_count < 2) {
      exit(EXIT_SUCCESS);
    }

    int can_peer_access_0i, can_peer_access_i0;
    for (i = 1; i < device_count; i++) {
      HIP_CHECK(hipDeviceCanAccessPeer(&can_peer_access_0i, 0, i));
      HIP_CHECK(hipDeviceCanAccessPeer(&can_peer_access_i0, i, 0));
      HIP_CHECK(
          hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, i));

      if (can_peer_access_0i * can_peer_access_i0 * mem_pool_support) {
        devices->ordinals[i] = i;
        INFO("Two-way peer access is available between GPU" << devices->ordinals[0] << " and GPU"
                                                            << devices->ordinals[i]);
        devices->count += 1;
        if (devices->count >= kMaxDevices) break;
      } else {
        break;
      }
    }

    exit(EXIT_SUCCESS);
  } else {
    int status;
    waitpid(pid, &status, 0);
    HIP_ASSERT(!status);
  }
}

/*
 Calling process waits for other processes to signal/complete.
*/
static void process_barrier() {
  int newCount = __sync_add_and_fetch(&g_Barrier->count, 1);

  if (newCount == g_processCnt) {
    g_Barrier->count = 0;
    g_Barrier->sense = !g_procSense;

  } else {
    while (g_Barrier->sense == g_procSense) {
      if (!g_Barrier->allExit) {
        sched_yield();
      } else {
        exit(EXIT_FAILURE);
      }
    }
  }

  g_procSense = !g_procSense;
}

/* Child process(es) import shared memory pool and check if allocated memory can be accessed and
 * used*/
static void child_process(int id) {
  volatile shmStruct* shm = NULL;
  hipStream_t stream;
  sharedMemoryInfo info;
  void* ptr;

  LinearAllocGuard<int> host_ptr(LinearAllocs::hipHostMalloc, kPageSize);

  ipcHandle* ipc_child_handle = NULL;
  checkIpcErrors(ipcOpenSocket(ipc_child_handle));

  // wait for parent process to create shared memory
  process_barrier();

  if (sharedMemoryOpen(shm_name, sizeof(shmStruct), &info) != 0) {
    INFO("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = reinterpret_cast<volatile shmStruct*>(info.addr);
  shm->processes[id] = getpid();

  // wait for parent process to send shareable handle
  process_barrier();

  // Receive allocation handle shared by parent.
  std::vector<ShareableHandle> sh_handle(1);
  checkIpcErrors(ipcRecvShareableHandles(ipc_child_handle, sh_handle));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  hipMemPool_t pool;

  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;

  // Import mem pool from all the devices created in the master
  // process using shareable handles received via socket
  // and import the pointer to the allocated buffer using
  // exportData filled in shared memory by the master process.
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&pool, reinterpret_cast<void*>(sh_handle[0]),
                                                handle_type, 0));

  hipMemAccessFlags access_flags;
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = 0;
  HIP_CHECK(hipMemPoolGetAccess(&access_flags, pool, &location));
  if (access_flags != hipMemAccessFlagsProtReadWrite) {
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(pool, &desc, 1));
  }

  // Import the allocation from memory pool using the opaque export data retrieved through
  // the shared memory
  HIP_CHECK(hipMemPoolImportPointer(&ptr, pool,
                                    const_cast<hipMemPoolPtrExportData*>(&shm->exportPtrData[id])));

  // Since we have imported allocations shared by the parent with us, we can
  // close this ShareableHandle.
  checkIpcErrors(ipcCloseShareableHandle(sh_handle[0]));

  // Since we have imported allocations shared by the parent with us, we can
  // close the socket
  checkIpcErrors(ipcCloseSocket(ipc_child_handle));

  // Child processed accesses imported buffer
  const auto element_count = kPageSize / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  int expected_value = 12 + id;
  VectorSet<<<block_count, thread_count, 0, stream>>>((int*)ptr, expected_value, element_count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy the buffer locally
  HIP_CHECK(hipMemcpyAsync(host_ptr.host_ptr(), ptr, kPageSize, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  INFO("Process " << id << "verifying...\n");

  // Check if the content is as expected
  ArrayFindIfNot(host_ptr.host_ptr(), expected_value, element_count);

  // Free the memory before the exporter process frees it
  HIP_CHECK(hipFreeAsync(ptr, stream));

  // And wait for all the queued up work to complete
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

static void parent_process(int dev_count) {
  sharedMemoryInfo info;
  int i;
  volatile shmStruct* shm = NULL;
  std::vector<void*> ptrs;
  std::vector<Process> child_processes;

  if (sharedMemoryCreate(shm_name, sizeof(*shm), &info) != 0) {
    INFO("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct*)info.addr;
  memset((void*)shm, 0, sizeof(*shm));

  // wait for child processes to insert their pids into shared memory
  process_barrier();

  std::vector<ShareableHandle> shareable_handles(dev_count);
  std::vector<hipStream_t> streams(dev_count);
  std::vector<hipMemPool_t> pools(dev_count);

  // Now allocate memory for each process and fill the shared
  // memory buffer with the export data and get mempool handles to communicate
  for (i = 0; i < dev_count; i++) {
    void* ptr;
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
    // Allocate an explicit pool with IPC capabilities
    hipMemPoolProps pool_props;
    memset(&pool_props, 0, sizeof(hipMemPoolProps));
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;

    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.location.id = i;

    HIP_CHECK(hipMemPoolCreate(&pools[i], &pool_props));

    // Query the shareable handle for the pool
    hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
    // Allocate memory in a stream from the pool just created
    HIP_CHECK(hipMallocFromPoolAsync(&ptr, kPageSize, pools[i], streams[i]));

    HIP_CHECK(hipMemPoolExportToShareableHandle(&shareable_handles[i], pools[i], handle_type, 0));

    // Memset handle to 0 to make sure call to hipMemPoolImportPointer in
    // child process will fail if the following call to hipMemPoolExportPointer fails.
    memset((void*)&shm->exportPtrData[i], 0, sizeof(hipMemPoolPtrExportData));
    HIP_CHECK(
        hipMemPoolExportPointer(const_cast<hipMemPoolPtrExportData*>(&shm->exportPtrData[i]), ptr));
    ptrs.push_back(ptr);
    child_processes.push_back(static_cast<Process>(shm->processes[i]));
  }

  ipcHandle* ipc_parent_handle;
  checkIpcErrors(ipcCreateSocket(ipc_parent_handle, ipc_name, child_processes));

  for (i = 0; i < dev_count; i++) {
    std::vector<ShareableHandle> current_handle(1, shareable_handles[i]);
    std::vector<ShareableHandle> current_process(1, child_processes[i]);
    checkIpcErrors(ipcSendShareableHandles(ipc_parent_handle, current_handle, current_process));
  }

  // Close the shareable handles as they are not needed anymore.
  for (int i = 0; i < dev_count; i++) {
    checkIpcErrors(ipcCloseShareableHandle(shareable_handles[i]));
  }

  checkIpcErrors(ipcCloseSocket(ipc_parent_handle));

  process_barrier();

  // And wait for them to finish
  for (i = 0; i < child_processes.size(); i++) {
    if (waitProcess(&child_processes[i]) != EXIT_SUCCESS) {
      INFO("Process " << i << " failed!\n");
      exit(EXIT_FAILURE);
    }
  }

  // Clean up!
  for (i = 0; i < dev_count; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipFreeAsync(ptrs[i], streams[i]));
    HIP_CHECK(hipStreamSynchronize(streams[i]));
    HIP_CHECK(hipMemPoolDestroy(pools[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  sharedMemoryClose(&info);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify exporting/importing a shareable handle on a single device between parent and
 * child process using IPC mechanisms - shared memory and sockets.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolExportImportIPC.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolExportImport_IPC_Functional") {
  ipcDevices_t* shm_devices;
  shm_devices = reinterpret_cast<ipcDevices_t*>(
      mmap(NULL, sizeof(*shm_devices), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  REQUIRE(MAP_FAILED != shm_devices);
  // Barrier is used to synchronize created processes
  g_Barrier = reinterpret_cast<ipcBarrier_t*>(
      mmap(NULL, sizeof(*g_Barrier), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  memset(g_Barrier, 0, sizeof(*g_Barrier));

  // set local barrier sense flag
  g_procSense = 0;

  get_devices(shm_devices);
  if (!shm_devices->count) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  // Set device count to 1
  shm_devices->count = 1;
  g_processCnt = shm_devices->count + 1;
  int index = 0;

  Process process = fork();
  if (process != 0) {
    parent_process(shm_devices->count);
  } else {
    child_process(index);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify exporting/importing a shareable handle on multiple devices between parent and
 * child processes using IPC mechanisms - shared memory and sockets.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolExportImportIPC.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolExportImport_MultipleDevices_IPC_Functional") {
  ipcDevices_t* shm_devices;
  shm_devices = reinterpret_cast<ipcDevices_t*>(
      mmap(NULL, sizeof(*shm_devices), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  REQUIRE(MAP_FAILED != shm_devices);
  // Barrier is used to synchronize processes created.
  g_Barrier = reinterpret_cast<ipcBarrier_t*>(
      mmap(NULL, sizeof(*g_Barrier), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0));
  memset(g_Barrier, 0, sizeof(*g_Barrier));

  // set local barrier sense flag
  g_procSense = 0;

  get_devices(shm_devices);
  if (!shm_devices->count) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  g_processCnt = shm_devices->count + 1;

  int index = 0;

  for (int i = 1; i < g_processCnt; i++) {
    Process process = fork();
    if (!process) {
      index = i;
      break;
    }
  }

  if (index == 0) {
    parent_process(shm_devices->count);
  } else {
    child_process(index - 1);
  }
}
#endif
