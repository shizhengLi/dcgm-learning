# DCGM API设计：跨语言绑定的艺术

## 引言

在现代软件开发中，API设计是系统成功的关键因素。DCGM作为企业级GPU监控管理系统，需要支持多种编程语言的接入，这使得其API设计面临着独特的挑战。本文将深入剖析DCGM API的设计哲学、实现机制和跨语言绑定技术，揭示其如何构建优雅且高效的API体系。

## API设计哲学

### 核心设计原则

DCGM API设计遵循一系列核心原则：

```cpp
// dcgmlib/dcgm_api_design.h
// API设计原则
namespace DcgmApiDesign
{
    // 1. 一致性原则：所有API保持一致的命名和参数风格
    // 2. 最小化原则：API数量最少化，功能最大化
    // 3. 向后兼容：新版本保持向后兼容
    // 4. 错误处理：统一的错误处理机制
    // 5. 线程安全：API支持多线程环境
    // 6. 性能优化：零拷贝和缓存优化
    // 7. 可扩展性：支持未来功能扩展
}
```

### API架构设计

DCGM采用分层API架构：

```cpp
// dcgmlib/dcgm_api_architecture.h
// API层次结构
class DcgmApiArchitecture
{
public:
    // 底层C API - 稳定的基础接口
    class CoreCApi
    {
    public:
        dcgmReturn_t dcgmInit(dcgmHandle_t *handle);
        dcgmReturn_t dcgmShutdown(dcgmHandle_t handle);
        dcgmReturn_t dcgmGetAllGpuIds(dcgmHandle_t handle, unsigned int *gpuIds, unsigned int *count);
        // ... 更多底层API
    };

    // 中间层C++ API - 面向对象的封装
    class DcgmCppApi
    {
    public:
        DcgmCppApi();
        ~DcgmCppApi();
        dcgmReturn_t Initialize();
        dcgmReturn_t GetGpuIds(std::vector<unsigned int>& gpuIds);
        // ... 更多C++ API
    };

    // 高级语言绑定 - Python/Go/Rust等
    class DcgmLanguageBindings
    {
    public:
        // Python绑定
        std::unique_ptr<DcgmPythonBinding> CreatePythonBinding();
        // Go绑定
        std::unique_ptr<DcgmGoBinding> CreateGoBinding();
        // Rust绑定
        std::unique_ptr<DcgmRustBinding> CreateRustBinding();
    };
};
```

## 核心C API设计

### API接口定义

DCGM的C API设计简洁而强大：

```cpp
// dcgmlib/dcgm_core_api.h
/***************************************************************************************************/
/** @defgroup dcgmCoreAPI Core C API Functions
 *  Core DCGM C API functions that provide basic functionality
 *  @{
 */
/***************************************************************************************************/

/**
 * Initialize DCGM and return a handle for subsequent operations
 */
dcgmReturn_t dcgmInit(dcgmHandle_t *handle);

/**
 * Shutdown DCGM and release all resources
 */
dcgmReturn_t dcgmShutdown(dcgmHandle_t handle);

/**
 * Get all GPU IDs in the system
 */
dcgmReturn_t dcgmGetAllGpuIds(dcgmHandle_t handle, unsigned int *gpuIds, unsigned int *count);

/**
 * Get GPU information for a specific GPU
 */
dcgmReturn_t dcgmGetGpuInfo(dcgmHandle_t handle, unsigned int gpuId, dcgmGpuInfo_t *info);

/**
 * Watch a set of fields for updates
 */
dcgmReturn_t dcgmWatchFields(dcgmHandle_t handle,
                             dcgmGpuGrp_t groupId,
                             dcgmFieldGrp_t fieldGroupId,
                             double updateInterval,
                             double maxKeepAge);

/**
 * Get the latest values for watched fields
 */
dcgmReturn_t dcgmGetLatestValues(dcgmHandle_t handle,
                                dcgmGpuGrp_t groupId,
                                dcgmFieldGrp_t fieldGroupId,
                                dcgmFieldValue_v2_t *values,
                                int *count);

/**
 * Start health monitoring for GPUs
 */
dcgmReturn_t dcgmStartHealthMonitoring(dcgmHandle_t handle,
                                       dcgmGpuGrp_t groupId,
                                       dcgmHealthMonitorConfig_t *config);

/**
 * Get health status for GPUs
 */
dcgmReturn_t dcgmGetHealthStatus(dcgmHandle_t handle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmHealthResponse_t *response);

/** @} */
```

### 数据结构设计

DCGM使用精心设计的数据结构：

```cpp
// dcgmlib/dcgm_structs.h
// GPU信息结构
typedef struct dcgmGpuInfo_v1
{
    unsigned int gpuId;                //!< GPU ID
    char gpuName[DCGM_MAX_STR_LEN];    //!< GPU name
    char serial[DCGM_MAX_STR_LEN];     //!< GPU serial number
    unsigned int pciBusId;             //!< PCI bus ID
    unsigned int pciDeviceId;           //!< PCI device ID
    unsigned int pciSubSystemId;       //!< PCI subsystem ID
    uuid_t uuid;                       //!< GPU UUID
    unsigned int memorySize;           //!< Memory size in bytes
    unsigned int maxPowerLimit;        //!< Maximum power limit in milliwatts
    unsigned int clocks[DCGM_CLOCK_COUNT]; //!< Clock frequencies
    dcgmGpuState_t state;              //!< GPU state
    dcgmReturn_t status;               //!< Status of this structure
} dcgmGpuInfo_v1_t;

// 字段值结构
typedef struct dcgmFieldValue_v2
{
    unsigned int fieldId;              //!< Field ID
    unsigned int entityType;           //!< Entity type (GPU, VGPU, etc.)
    unsigned int entityId;             //!< Entity ID
    dcgmFieldType_t fieldType;        //!< Field type
    int64_t ts;                        //!< Timestamp when this value was recorded
    dcgmReturn_t status;               //!< Status of this field value

    union {
        int64_t i64;                   //!< For DCGM_FT_INT64
        double dbl;                    //!< For DCGM_FT_DOUBLE
        char str[DCGM_MAX_STR_LEN];   //!< For DCGM_FT_STRING
        struct {
            void *ptr;                 //!< Pointer to binary data
            size_t size;               //!< Size of binary data
        } blob;
    } value;
} dcgmFieldValue_v2_t;

// 健康状态结构
typedef struct dcgmHealthResponse_v1
{
    unsigned int gpuId;                //!< GPU ID
    dcgmHealthState_t overallHealth;  //!< Overall health state
    dcgmHealthComponent_t components[DCGM_HEALTH_COMPONENT_COUNT]; //!< Component health
    unsigned int issueCount;           //!< Number of health issues
    dcgmHealthIssue_t issues[DCGM_MAX_HEALTH_ISSUES]; //!< Health issues
    dcgmReturn_t status;               //!< Status of this structure
} dcgmHealthResponse_v1_t;
```

### 错误处理机制

DCGM实现了统一的错误处理机制：

```cpp
// dcgmlib/dcgm_errors.h
// 错误码定义
typedef enum
{
    DCGM_ST_OK = 0,                    //!< Success
    DCGM_ST_ERROR = 1,                  //!< Generic error
    DCGM_ST_NOT_INITIALIZED = 2,       //!< DCGM not initialized
    DCGM_ST_ALREADY_INITIALIZED = 3,   //!< DCGM already initialized
    DCGM_ST_INVALID_HANDLE = 4,        //!< Invalid handle
    DCGM_ST_INVALID_PARAMETER = 5,     //!< Invalid parameter
    DCGM_ST_TIMEOUT = 6,               //!< Operation timeout
    DCGM_ST_MEMORY = 7,                //!< Memory allocation error
    DCGM_ST_CONNECTION_ERROR = 8,     //!< Connection error
    DCGM_ST_PERMISSION_DENIED = 9,     //!< Permission denied
    DCGM_ST_GPU_NOT_FOUND = 10,        //!< GPU not found
    DCGM_ST_GROUP_NOT_FOUND = 11,      //!< Group not found
    DCGM_ST_FIELD_NOT_FOUND = 12,      //!< Field not found
    DCGM_ST_MODULE_NOT_FOUND = 13,     //!< Module not found
    DCGM_ST_POLICY_NOT_FOUND = 14,     //!< Policy not found
    DCGM_ST_HEALTH_ERROR = 15,         //!< Health monitoring error
    DCGM_ST_DIAGNOSTIC_ERROR = 16,     //!< Diagnostic error
    // ... 更多错误码
} dcgmReturn_t;

// 错误信息获取
const char* dcgmErrorString(dcgmReturn_t result);

// 错误详细信息
typedef struct dcgmErrorDetail
{
    dcgmReturn_t code;                 //!< Error code
    char message[DCGM_MAX_ERROR_MSG_LEN]; //!< Error message
    char function[DCGM_MAX_FUNCTION_LEN]; //!< Function name
    char file[DCGM_MAX_FILE_LEN];     //!< File name
    int line;                         //!< Line number
    dcgmSeverity_t severity;          //!< Error severity
    char suggestion[DCGM_MAX_SUGGESTION_LEN]; //!< Suggested action
} dcgmErrorDetail_t;

dcgmReturn_t dcgmGetLastError(dcgmHandle_t handle, dcgmErrorDetail_t *detail);
```

## C++ API封装

### 面向对象封装

DCGM提供了C++的面向对象封装：

```cpp
// dcgmlib/dcgm_cpp_api.h
class DcgmApi
{
public:
    // 构造和析构
    DcgmApi();
    ~DcgmApi();

    // 初始化和关闭
    dcgmReturn_t Initialize();
    dcgmReturn_t Shutdown();

    // GPU管理
    dcgmReturn_t GetGpuIds(std::vector<unsigned int>& gpuIds);
    dcgmReturn_t GetGpuInfo(unsigned int gpuId, DcgmGpuInfo& info);

    // 字段监控
    dcgmReturn_t WatchFields(unsigned int gpuId, const std::vector<unsigned int>& fieldIds,
                            std::chrono::milliseconds updateInterval,
                            std::chrono::seconds maxKeepAge);
    dcgmReturn_t GetLatestValues(unsigned int gpuId, const std::vector<unsigned int>& fieldIds,
                                 std::vector<DcgmFieldValue>& values);

    // 健康监控
    dcgmReturn_t StartHealthMonitoring(const std::vector<unsigned int>& gpuIds,
                                        const DcgmHealthConfig& config);
    dcgmReturn_t GetHealthStatus(unsigned int gpuId, DcgmHealthStatus& status);

    // 异步操作
    dcgmReturn_t WatchFieldsAsync(unsigned int gpuId, const std::vector<unsigned int>& fieldIds,
                                  std::chrono::milliseconds updateInterval,
                                  std::function<void(const std::vector<DcgmFieldValue>&)> callback);
    dcgmReturn_t CancelWatchAsync(unsigned int watchId);

    // 错误处理
    std::string GetLastErrorString() const;
    dcgmReturn_t GetLastError() const;

private:
    dcgmHandle_t m_handle;
    std::unique_ptr<DcgmConnection> m_connection;
    std::unique_ptr<DcgmThreadPool> m_threadPool;
    std::unique_ptr<DcgmCallbackManager> m_callbackManager;
    dcgmReturn_t m_lastError;

    // 内部工具方法
    dcgmReturn_t CheckInitialized() const;
    dcgmReturn_t ConvertToCError(const std::exception& e);
    void HandleCallback(unsigned int watchId, const std::vector<DcgmFieldValue>& values);
};

// RAII封装
class DcgmHandle
{
public:
    DcgmHandle() : m_api(std::make_unique<DcgmApi>()) {}
    ~DcgmHandle() { if (m_api) m_api->Shutdown(); }

    dcgmReturn_t Initialize() { return m_api->Initialize(); }
    dcgmReturn_t GetGpuIds(std::vector<unsigned int>& gpuIds) { return m_api->GetGpuIds(gpuIds); }
    // ... 更多方法

private:
    std::unique_ptr<DcgmApi> m_api;
};
```

### 智能指针和RAII

DCGM充分利用现代C++特性：

```cpp
// dcgmlib/dcgm_smart_ptr.h
// GPU信息智能指针
class DcgmGpuInfoPtr
{
public:
    explicit DcgmGpuInfoPtr(unsigned int gpuId);
    ~DcgmGpuInfoPtr();

    // 拷贝和移动
    DcgmGpuInfoPtr(const DcgmGpuInfoPtr& other);
    DcgmGpuInfoPtr(DcgmGpuInfoPtr&& other) noexcept;
    DcgmGpuInfoPtr& operator=(const DcgmGpuInfoPtr& other);
    DcgmGpuInfoPtr& operator=(DcgmGpuInfoPtr&& other) noexcept;

    // 访问
    const DcgmGpuInfo* operator->() const { return m_info; }
    const DcgmGpuInfo& operator*() const { return *m_info; }

private:
    DcgmGpuInfo* m_info;
    static std::mutex s_cacheMutex;
    static std::map<unsigned int, DcgmGpuInfo*> s_cache;
};

// 字段监控RAII包装
class DcgmFieldWatcher
{
public:
    DcgmFieldWatcher(DcgmApi& api, unsigned int gpuId, const std::vector<unsigned int>& fieldIds,
                    std::chrono::milliseconds updateInterval);
    ~DcgmFieldWatcher();

    // 禁用拷贝
    DcgmFieldWatcher(const DcgmFieldWatcher&) = delete;
    DcgmFieldWatcher& operator=(const DcgmFieldWatcher&) = delete;

    // 启用移动
    DcgmFieldWatcher(DcgmFieldWatcher&& other) noexcept;
    DcgmFieldWatcher& operator=(DcgmFieldWatcher&& other) noexcept;

    // 操作
    dcgmReturn_t GetLatestValues(std::vector<DcgmFieldValue>& values);
    dcgmReturn_t SetCallback(std::function<void(const std::vector<DcgmFieldValue>&)> callback);

private:
    DcgmApi* m_api;
    unsigned int m_gpuId;
    std::vector<unsigned int> m_fieldIds;
    unsigned int m_watchId;
    bool m_isActive;
};
```

## Python绑定实现

### 绑定框架

DCGM使用PyBind11实现Python绑定：

```cpp
// bindings/python/dcgm_python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <dcgmlib/dcgm_cpp_api.h>

namespace py = pybind11;

PYBIND11_MODULE(dcgm, m) {
    // 版本信息
    m.doc() = "DCGM Python Bindings";
    m.attr("__version__") = "3.0.0";

    // 错误码
    py::enum_<dcgmReturn_t>(m, "ReturnCode")
        .value("OK", DCGM_ST_OK)
        .value("ERROR", DCGM_ST_ERROR)
        .value("NOT_INITIALIZED", DCGM_ST_NOT_INITIALIZED)
        .value("INVALID_HANDLE", DCGM_ST_INVALID_HANDLE)
        // ... 更多错误码
        .export_values();

    // GPU信息类
    py::class_<DcgmGpuInfo>(m, "GpuInfo")
        .def(py::init<>())
        .def_readonly("gpu_id", &DcgmGpuInfo::gpuId)
        .def_readonly("gpu_name", &DcgmGpuInfo::gpuName)
        .def_readonly("memory_size", &DcgmGpuInfo::memorySize)
        .def_readonly("max_power_limit", &DcgmGpuInfo::maxPowerLimit)
        .def("__repr__", [](const DcgmGpuInfo& info) {
            return "<GpuInfo id=" + std::to_string(info.gpuId) +
                   " name='" + std::string(info.gpuName) + "'>";
        });

    // 字段值类
    py::class_<DcgmFieldValue>(m, "FieldValue")
        .def(py::init<>())
        .def_readonly("field_id", &DcgmFieldValue::fieldId)
        .def_readonly("field_type", &DcgmFieldValue::fieldType)
        .def_property_readonly("value", [](const DcgmFieldValue& fv) {
            switch (fv.fieldType) {
                case DCGM_FT_INT64:
                    return py::cast(fv.value.i64);
                case DCGM_FT_DOUBLE:
                    return py::cast(fv.value.dbl);
                case DCGM_FT_STRING:
                    return py::cast(std::string(fv.value.str));
                default:
                    return py::none();
            }
        });

    // 主要API类
    py::class_<DcgmApi>(m, "DcgmApi")
        .def(py::init<>())
        .def("initialize", &DcgmApi::Initialize)
        .def("shutdown", &DcgmApi::Shutdown)
        .def("get_gpu_ids", &DcgmApi::GetGpuIds)
        .def("get_gpu_info", &DcgmApi::GetGpuInfo)
        .def("watch_fields", &DcgmApi::WatchFields,
             py::call_guard<py::gil_scoped_release>())
        .def("get_latest_values", &DcgmApi::GetLatestValues)
        .def("start_health_monitoring", &DcgmApi::StartHealthMonitoring)
        .def("get_health_status", &DcgmApi::GetHealthStatus)
        .def("get_last_error_string", &DcgmApi::GetLastErrorString)
        .def("watch_fields_async", &DcgmApi::WatchFieldsAsync);

    // 异步回调支持
    m.def("set_async_callback", [](py::function callback) {
        return DcgmApi::SetAsyncCallback([callback](const std::vector<DcgmFieldValue>& values) {
            py::gil_scoped_acquire acquire;
            try {
                callback(values);
            } catch (const py::error_already_set& e) {
                PyErr_Print();
            }
        });
    });
}
```

### Python高级封装

DCGM提供了更Pythonic的高级接口：

```python
# bindings/python/dcgm/api.py
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time

@dataclass
class GpuInfo:
    gpu_id: int
    name: str
    memory_size: int
    max_power_limit: int
    temperature: float
    utilization: float

@dataclass
class FieldValue:
    field_id: int
    value: Any
    timestamp: int

class DcgmSession:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self._api = None
        self._lock = threading.Lock()
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def initialize(self):
        with self._lock:
            if self._initialized:
                return

            from dcgm import DcgmApi
            self._api = DcgmApi()
            result = self._api.initialize()
            if result != ReturnCode.OK:
                raise RuntimeError(f"Failed to initialize DCGM: {result}")
            self._initialized = True

    def shutdown(self):
        with self._lock:
            if self._api and self._initialized:
                self._api.shutdown()
                self._initialized = False

    def get_gpus(self) -> List[GpuInfo]:
        if not self._initialized:
            raise RuntimeError("DCGM not initialized")

        gpu_ids = self._api.get_gpu_ids()
        gpus = []

        for gpu_id in gpu_ids:
            info = self._api.get_gpu_info(gpu_id)
            gpu = GpuInfo(
                gpu_id=info.gpu_id,
                name=info.gpu_name.decode(),
                memory_size=info.memory_size,
                max_power_limit=info.max_power_limit,
                temperature=0.0,  # Will be filled by monitoring
                utilization=0.0    # Will be filled by monitoring
            )
            gpus.append(gpu)

        return gpus

    def monitor_gpu(self, gpu_id: int, fields: List[int],
                   callback: Optional[Callable[[List[FieldValue]], None]] = None):
        """Monitor GPU fields with optional callback"""
        if not self._initialized:
            raise RuntimeError("DCGM not initialized")

        return self._api.watch_fields_async(gpu_id, fields, callback)

    def get_gpu_metrics(self, gpu_id: int) -> Dict[str, float]:
        """Get current GPU metrics"""
        if not self._initialized:
            raise RuntimeError("DCGM not initialized")

        fields = [
            DCGM_FI_DEV_GPU_UTILIZATION,
            DCGM_FI_DEV_TEMP,
            DCGM_FI_DEV_POWER_USAGE,
            DCGM_FI_DEV_USED_MEMORY
        ]

        values = self._api.get_latest_values(gpu_id, fields)

        metrics = {}
        for value in values:
            if value.field_id == DCGM_FI_DEV_GPU_UTILIZATION:
                metrics['utilization'] = value.value
            elif value.field_id == DCGM_FI_DEV_TEMP:
                metrics['temperature'] = value.value
            elif value.field_id == DCGM_FI_DEV_POWER_USAGE:
                metrics['power'] = value.value
            elif value.field_id == DCGM_FI_DEV_USED_MEMORY:
                metrics['used_memory'] = value.value

        return metrics

# 高级监控装饰器
def monitor_gpu_metrics(gpu_id: int, interval: float = 1.0):
    """Decorator for monitoring GPU metrics during function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with DcgmSession() as session:
                metrics = []

                def callback(values):
                    timestamp = time.time()
                    metric_dict = {}
                    for value in values:
                        if value.field_id == DCGM_FI_DEV_GPU_UTILIZATION:
                            metric_dict['utilization'] = value.value
                        elif value.field_id == DCGM_FI_DEV_TEMP:
                            metric_dict['temperature'] = value.value

                    metrics.append((timestamp, metric_dict))

                # Start monitoring
                session.monitor_gpu(gpu_id,
                                 [DCGM_FI_DEV_GPU_UTILIZATION, DCGM_FI_DEV_TEMP],
                                 callback)

                try:
                    result = func(*args, **kwargs)
                finally:
                    time.sleep(interval)  # Wait for final metrics

                return result, metrics

        return wrapper
    return decorator
```

## Go绑定实现

### Go绑定框架

DCGM使用cgo实现Go绑定：

```go
// bindings/go/dcgm/dcgm.go
package dcgm

/*
#cgo pkg-config: dcgm
#include <dcgmlib/dcgm_core_api.h>
#include <dcgmlib/dcgm_structs.h>
*/
import "C"

import (
    "fmt"
    "time"
    "unsafe"
)

// 类型定义
type Handle struct {
    ptr C.dcgmHandle_t
}

type ReturnCode int

const (
    ReturnCodeOK ReturnCode = iota
    ReturnCodeError
    ReturnCodeNotInitialized
    ReturnCodeInvalidHandle
    ReturnCodeInvalidParameter
)

type GpuInfo struct {
    GpuId         uint32
    GpuName       string
    Serial        string
    MemorySize    uint64
    MaxPowerLimit uint32
}

type FieldValue struct {
    FieldId   uint32
    FieldType FieldType
    Value     interface{}
    Timestamp int64
}

// API包装
func Init() (*Handle, error) {
    var handle C.dcgmHandle_t
    result := C.dcgmInit(&handle)

    if result != C.DCGM_ST_OK {
        return nil, fmt.Errorf("failed to initialize DCGM: %d", result)
    }

    return &Handle{ptr: handle}, nil
}

func (h *Handle) Shutdown() error {
    if h.ptr == nil {
        return fmt.Errorf("handle is nil")
    }

    result := C.dcgmShutdown(h.ptr)
    if result != C.DCGM_ST_OK {
        return fmt.Errorf("failed to shutdown DCGM: %d", result)
    }

    h.ptr = nil
    return nil
}

func (h *Handle) GetGpuIds() ([]uint32, error) {
    if h.ptr == nil {
        return nil, fmt.Errorf("handle is nil")
    }

    var count C.uint
    result := C.dcgmGetAllGpuIds(h.ptr, nil, &count)

    if result != C.DCGM_ST_OK {
        return nil, fmt.Errorf("failed to get GPU count: %d", result)
    }

    if count == 0 {
        return []uint32{}, nil
    }

    gpuIds := make([]C.uint, count)
    result = C.dcgmGetAllGpuIds(h.ptr, &gpuIds[0], &count)

    if result != C.DCGM_ST_OK {
        return nil, fmt.Errorf("failed to get GPU IDs: %d", result)
    }

    ids := make([]uint32, count)
    for i := 0; i < int(count); i++ {
        ids[i] = uint32(gpuIds[i])
    }

    return ids, nil
}

func (h *Handle) GetGpuInfo(gpuId uint32) (*GpuInfo, error) {
    if h.ptr == nil {
        return nil, fmt.Errorf("handle is nil")
    }

    var info C.dcgmGpuInfo_v1_t
    result := C.dcgmGetGpuInfo(h.ptr, C.uint(gpuId), &info)

    if result != C.DCGM_ST_OK {
        return nil, fmt.Errorf("failed to get GPU info: %d", result)
    }

    return &GpuInfo{
        GpuId:         uint32(info.gpuId),
        GpuName:       C.GoString(&info.gpuName[0]),
        Serial:        C.GoString(&info.serial[0]),
        MemorySize:    uint64(info.memorySize),
        MaxPowerLimit: uint32(info.maxPowerLimit),
    }, nil
}

// 异步监控
type FieldWatcher struct {
    handle   *Handle
    gpuId    uint32
    fieldIds []uint32
    stopChan chan struct{}
    callback func([]FieldValue)
}

func (h *Handle) WatchFields(gpuId uint32, fieldIds []uint32,
                            interval time.Duration, callback func([]FieldValue)) (*FieldWatcher, error) {
    if h.ptr == nil {
        return nil, fmt.Errorf("handle is nil")
    }

    watcher := &FieldWatcher{
        handle:   h,
        gpuId:    gpuId,
        fieldIds: fieldIds,
        stopChan: make(chan struct{}),
        callback: callback,
    }

    go watcher.monitor(interval)

    return watcher, nil
}

func (w *FieldWatcher) monitor(interval time.Duration) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            values, err := w.handle.GetLatestValues(w.gpuId, w.fieldIds)
            if err != nil {
                continue
            }

            if w.callback != nil {
                w.callback(values)
            }

        case <-w.stopChan:
            return
        }
    }
}

func (w *FieldWatcher) Stop() {
    close(w.stopChan)
}
```

### Go高级API

DCGM提供了更Go风格的高级接口：

```go
// bindings/go/dcgm/advanced.go
package dcgm

import (
    "context"
    "sync"
    "time"
)

type GpuMonitor struct {
    handle    *Handle
    gpus      map[uint32]*GpuInfo
    watchers  map[uint32]*FieldWatcher
    mutex     sync.RWMutex
    callbacks []func(uint32, map[string]interface{})
}

func NewGpuMonitor() (*GpuMonitor, error) {
    handle, err := Init()
    if err != nil {
        return nil, err
    }

    monitor := &GpuMonitor{
        handle:   handle,
        gpus:     make(map[uint32]*GpuInfo),
        watchers: make(map[uint32]*FieldWatcher),
    }

    // Initialize GPU information
    err = monitor.refreshGpuInfo()
    if err != nil {
        handle.Shutdown()
        return nil, err
    }

    return monitor, nil
}

func (m *GpuMonitor) Close() error {
    m.mutex.Lock()
    defer m.mutex.Unlock()

    // Stop all watchers
    for _, watcher := range m.watchers {
        watcher.Stop()
    }
    m.watchers = make(map[uint32]*FieldWatcher)

    return m.handle.Shutdown()
}

func (m *GpuMonitor) refreshGpuInfo() error {
    gpuIds, err := m.handle.GetGpuIds()
    if err != nil {
        return err
    }

    for _, gpuId := range gpuIds {
        info, err := m.handle.GetGpuInfo(gpuId)
        if err != nil {
            return err
        }

        m.gpus[gpuId] = info
    }

    return nil
}

func (m *GpuMonitor) GetGpus() []*GpuInfo {
    m.mutex.RLock()
    defer m.mutex.RUnlock()

    gpus := make([]*GpuInfo, 0, len(m.gpus))
    for _, gpu := range m.gpus {
        gpus = append(gpus, gpu)
    }

    return gpus
}

func (m *GpuMonitor) MonitorGpu(ctx context.Context, gpuId uint32,
                               interval time.Duration) (<-chan map[string]interface{}, error) {
    m.mutex.Lock()
    defer m.mutex.Unlock()

    if _, exists := m.gpus[gpuId]; !exists {
        return nil, fmt.Errorf("GPU %d not found", gpuId)
    }

    if _, exists := m.watchers[gpuId]; exists {
        return nil, fmt.Errorf("GPU %d is already being monitored", gpuId)
    }

    metricChan := make(chan map[string]interface{}, 100)

    fieldIds := []uint32{
        DCGM_FI_DEV_GPU_UTILIZATION,
        DCGM_FI_DEV_TEMP,
        DCGM_FI_DEV_POWER_USAGE,
        DCGM_FI_DEV_USED_MEMORY,
    }

    watcher, err := m.handle.WatchFields(gpuId, fieldIds, interval, func(values []FieldValue) {
        metrics := make(map[string]interface{})

        for _, value := range values {
            switch value.FieldId {
            case DCGM_FI_DEV_GPU_UTILIZATION:
                metrics["utilization"] = value.Value
            case DCGM_FI_DEV_TEMP:
                metrics["temperature"] = value.Value
            case DCGM_FI_DEV_POWER_USAGE:
                metrics["power"] = value.Value
            case DCGM_FI_DEV_USED_MEMORY:
                metrics["used_memory"] = value.Value
            }
        }

        select {
        case metricChan <- metrics:
        default:
            // Channel full, drop metrics
        }
    })

    if err != nil {
        return nil, err
    }

    m.watchers[gpuId] = watcher

    // Handle context cancellation
    go func() {
        <-ctx.Done()
        watcher.Stop()

        m.mutex.Lock()
        delete(m.watchers, gpuId)
        m.mutex.Unlock()

        close(metricChan)
    }()

    return metricChan, nil
}

// 监控所有GPU
func (m *GpuMonitor) MonitorAllGpus(ctx context.Context,
                                    interval time.Duration) (<-chan map[uint32]map[string]interface{}, error) {
    m.mutex.RLock()
    gpuIds := make([]uint32, 0, len(m.gpus))
    for gpuId := range m.gpus {
        gpuIds = append(gpuIds, gpuId)
    }
    m.mutex.RUnlock()

    allMetricsChan := make(chan map[uint32]map[string]interface{}, 100)

    var wg sync.WaitGroup
    for _, gpuId := range gpuIds {
        wg.Add(1)

        go func(id uint32) {
            defer wg.Done()

            gpuChan, err := m.MonitorGpu(ctx, id, interval)
            if err != nil {
                return
            }

            for metrics := range gpuChan {
                allMetricsChan <- map[uint32]map[string]interface{}{id: metrics}
            }
        }(gpuId)
    }

    go func() {
        wg.Wait()
        close(allMetricsChan)
    }()

    return allMetricsChan, nil
}
```

## Rust绑定实现

### Rust绑定框架

DCGM使用bindgen实现Rust绑定：

```rust
// bindings/rust/dcgm-sys/build.rs
fn main() {
    // 生成绑定
    let bindings = bindgen::Builder::default()
        .header("dcgmlib/dcgm_core_api.h")
        .header("dcgmlib/dcgm_structs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // 写入绑定文件
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

// bindings/rust/dcgm-sys/src/lib.rs
// 生成的绑定代码
pub type dcgmHandle_t = *mut dcgmHandle;
pub type dcgmReturn_t = u32;

pub const DCGM_ST_OK: dcgmReturn_t = 0;
pub const DCGM_ST_ERROR: dcgmReturn_t = 1;
pub const DCGM_ST_NOT_INITIALIZED: dcgmReturn_t = 2;

#[repr(C)]
pub struct dcgmGpuInfo_v1 {
    pub gpuId: u32,
    pub gpuName: [std::os::raw::c_char; 64],
    pub serial: [std::os::raw::c_char; 64],
    pub memorySize: u32,
    pub maxPowerLimit: u32,
}

#[link(name = "dcgm")]
extern "C" {
    pub fn dcgmInit(handle: *mut dcgmHandle_t) -> dcgmReturn_t;
    pub fn dcgmShutdown(handle: dcgmHandle_t) -> dcgmReturn_t;
    pub fn dcgmGetAllGpuIds(handle: dcgmHandle_t, gpuIds: *mut u32, count: *mut u32) -> dcgmReturn_t;
    pub fn dcgmGetGpuInfo(handle: dcgmHandle_t, gpuId: u32, info: *mut dcgmGpuInfo_v1) -> dcgmReturn_t;
}
```

### Rust高级API

DCGM提供了安全的Rust API：

```rust
// bindings/rust/dcgm/src/lib.rs
use std::ptr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use dcgm_sys::*;

pub struct DcgmHandle {
    handle: dcgmHandle_t,
}

impl DcgmHandle {
    pub fn new() -> Result<Self, DcgmError> {
        let mut handle = ptr::null_mut();
        let result = unsafe { dcgmInit(&mut handle) };

        if result != DCGM_ST_OK {
            return Err(DcgmError::from_code(result));
        }

        Ok(DcgmHandle { handle })
    }

    pub fn get_gpu_ids(&self) -> Result<Vec<u32>, DcgmError> {
        let mut count = 0u32;
        let result = unsafe { dcgmGetAllGpuIds(self.handle, ptr::null_mut(), &mut count) };

        if result != DCGM_ST_OK {
            return Err(DcgmError::from_code(result));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut gpu_ids = vec![0u32; count as usize];
        let result = unsafe {
            dcgmGetAllGpuIds(self.handle, gpu_ids.as_mut_ptr(), &mut count)
        };

        if result != DCGM_ST_OK {
            return Err(DcgmError::from_code(result));
        }

        gpu_ids.truncate(count as usize);
        Ok(gpu_ids)
    }

    pub fn get_gpu_info(&self, gpu_id: u32) -> Result<GpuInfo, DcgmError> {
        let mut info = dcgmGpuInfo_v1 {
            gpuId: 0,
            gpuName: [0; 64],
            serial: [0; 64],
            memorySize: 0,
            maxPowerLimit: 0,
        };

        let result = unsafe { dcgmGetGpuInfo(self.handle, gpu_id, &mut info) };

        if result != DCGM_ST_OK {
            return Err(DcgmError::from_code(result));
        }

        Ok(GpuInfo::from_sys(info))
    }
}

impl Drop for DcgmHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { dcgmShutdown(self.handle) };
        }
    }
}

#[derive(Debug)]
pub enum DcgmError {
    NotInitialized,
    InvalidHandle,
    InvalidParameter,
    Timeout,
    Memory,
    ConnectionError,
    Unknown(u32),
}

impl DcgmError {
    fn from_code(code: u32) -> Self {
        match code {
            DCGM_ST_NOT_INITIALIZED => DcgmError::NotInitialized,
            DCGM_ST_INVALID_HANDLE => DcgmError::InvalidHandle,
            DCGM_ST_INVALID_PARAMETER => DcgmError::InvalidParameter,
            DCGM_ST_TIMEOUT => DcgmError::Timeout,
            DCGM_ST_MEMORY => DcgmError::Memory,
            DCGM_ST_CONNECTION_ERROR => DcgmError::ConnectionError,
            _ => DcgmError::Unknown(code),
        }
    }
}

// GPU信息结构
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub gpu_id: u32,
    pub gpu_name: String,
    pub serial: String,
    pub memory_size: u32,
    pub max_power_limit: u32,
}

impl GpuInfo {
    fn from_sys(info: dcgmGpuInfo_v1) -> Self {
        let gpu_name = unsafe {
            std::ffi::CStr::from_ptr(info.gpuName.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let serial = unsafe {
            std::ffi::CStr::from_ptr(info.serial.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        GpuInfo {
            gpu_id: info.gpuId,
            gpu_name,
            serial,
            memory_size: info.memorySize,
            max_power_limit: info.maxPowerLimit,
        }
    }
}

// 异步监控器
pub struct GpuMonitor {
    handle: Arc<DcgmHandle>,
    running: Arc<Mutex<bool>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl GpuMonitor {
    pub fn new(handle: Arc<DcgmHandle>) -> Self {
        GpuMonitor {
            handle,
            running: Arc::new(Mutex::new(true)),
            threads: Vec::new(),
        }
    }

    pub fn monitor_gpu<F>(&mut self, gpu_id: u32, interval: Duration, callback: F)
    where
        F: Fn(Vec<FieldValue>) + Send + 'static,
    {
        let handle = Arc::clone(&self.handle);
        let running = Arc::clone(&self.running);

        let thread = thread::spawn(move || {
            while *running.lock().unwrap() {
                match handle.get_gpu_metrics(gpu_id) {
                    Ok(metrics) => callback(metrics),
                    Err(e) => eprintln!("Error monitoring GPU {}: {:?}", gpu_id, e),
                }

                thread::sleep(interval);
            }
        });

        self.threads.push(thread);
    }

    pub fn stop(&mut self) {
        *self.running.lock().unwrap() = false;

        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }
}

impl Drop for GpuMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}
```

## 跨语言绑定最佳实践

### API版本兼容性

DCGM实现了完善的版本兼容机制：

```cpp
// dcgmlib/dcgm_versioning.h
// 版本信息
typedef struct dcgmVersionInfo
{
    unsigned int major;
    unsigned int minor;
    unsigned int patch;
    const char *build;
} dcgmVersionInfo_t;

// 版本查询
dcgmReturn_t dcgmGetVersion(dcgmVersionInfo_t *version);

// 兼容性检查
dcgmReturn_t dcgmCheckCompatibility(unsigned int requiredMajor,
                                      unsigned int requiredMinor,
                                      dcgmCompatibility_t *compatibility);

// 特性查询
dcgmReturn_t dcgmIsFeatureSupported(const char *featureName, bool *supported);

// API版本标记
#define DCGM_API_VERSION_1_0 0x01000000
#define DCGM_API_VERSION_2_0 0x02000000
#define DCGM_API_VERSION_3_0 0x03000000

// 版本化结构
typedef struct dcgmFieldValue_v3
{
    // v2 字段
    unsigned int fieldId;
    unsigned int entityType;
    unsigned int entityId;
    dcgmFieldType_t fieldType;
    int64_t ts;
    dcgmReturn_t status;

    union {
        int64_t i64;
        double dbl;
        char str[DCGM_MAX_STR_LEN];
        struct {
            void *ptr;
            size_t size;
        } blob;
    } value;

    // v3 新增字段
    unsigned int version;      // 结构版本
    unsigned int flags;        // 标志位
    int64_t sequenceId;       // 序列ID
    char source[64];           // 数据源
} dcgmFieldValue_v3_t;
```

### 内存管理

跨语言绑定的内存管理是关键：

```cpp
// dcgmlib/dcgm_memory_management.h
// 内存分配器
class DcgmMemoryAllocator
{
public:
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
    static void* Reallocate(void* ptr, size_t size);

private:
    // 内存池
    static std::map<size_t, std::stack<void*>> s_memoryPools;
    static std::mutex s_poolMutex;

    // 统计信息
    static std::atomic<size_t> s_totalAllocated;
    static std::atomic<size_t> s_totalFreed;
    static std::atomic<size_t> s_currentUsage;
};

// 字符串管理
class DcgmStringManager
{
public:
    static char* CreateString(const std::string& str);
    static void FreeString(char* str);
    static char* DuplicateString(const char* str);

private:
    // 字符串缓存
    static std::unordered_set<std::string> s_stringCache;
    static std::mutex s_cacheMutex;
};

// 引用计数
template<typename T>
class DcgmRefCounted
{
public:
    DcgmRefCounted(T* ptr) : m_ptr(ptr), m_refCount(1) {}
    ~DcgmRefCounted() { delete m_ptr; }

    void AddRef() { m_refCount++; }
    void Release() { if (--m_refCount == 0) delete this; }

    T* Get() const { return m_ptr; }

private:
    T* m_ptr;
    std::atomic<int> m_refCount;
};
```

## 实战案例分析

### 案例1：Python监控应用

```python
# examples/python_gpu_monitor.py
import dcgm
import time
import matplotlib.pyplot as plt
from datetime import datetime

class GpuMonitorApp:
    def __init__(self):
        self.session = dcgm.DcgmSession()
        self.session.initialize()
        self.metrics_history = {}

    def monitor_gpus(self, duration_minutes=60, interval_seconds=5):
        """Monitor all GPUs for specified duration"""
        gpus = self.session.get_gpus()
        print(f"Found {len(gpus)} GPUs")

        # Initialize history for each GPU
        for gpu in gpus:
            self.metrics_history[gpu.gpu_id] = {
                'timestamps': [],
                'utilization': [],
                'temperature': [],
                'power': [],
                'memory_usage': []
            }

        # Setup monitoring for all GPUs
        for gpu in gpus:
            self.session.monitor_gpu(
                gpu.gpu_id,
                [dcgm.DCGM_FI_DEV_GPU_UTILIZATION,
                 dcgm.DCGM_FI_DEV_TEMP,
                 dcgm.DCGM_FI_DEV_POWER_USAGE,
                 dcgm.DCGM_FI_DEV_USED_MEMORY],
                interval_seconds,
                self._create_callback(gpu.gpu_id)
            )

        # Monitor for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        try:
            while time.time() < end_time:
                time.sleep(interval_seconds)
                self._print_current_metrics()
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

    def _create_callback(self, gpu_id):
        """Create callback for GPU monitoring"""
        def callback(values):
            timestamp = datetime.now()

            for value in values:
                if value.field_id == dcgm.DCGM_FI_DEV_GPU_UTILIZATION:
                    self.metrics_history[gpu_id]['utilization'].append(value.value)
                elif value.field_id == dcgm.DCGM_FI_DEV_TEMP:
                    self.metrics_history[gpu_id]['temperature'].append(value.value)
                elif value.field_id == dcgm.DCGM_FI_DEV_POWER_USAGE:
                    self.metrics_history[gpu_id]['power'].append(value.value)
                elif value.field_id == dcgm.DCGM_FI_DEV_USED_MEMORY:
                    self.metrics_history[gpu_id]['memory_usage'].append(value.value)

            self.metrics_history[gpu_id]['timestamps'].append(timestamp)

        return callback

    def _print_current_metrics(self):
        """Print current metrics for all GPUs"""
        print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        for gpu_id, history in self.metrics_history.items():
            if history['utilization']:
                latest_util = history['utilization'][-1]
                latest_temp = history['temperature'][-1]
                latest_power = history['power'][-1]
                latest_mem = history['memory_usage'][-1]

                print(f"GPU {gpu_id}: "
                      f"Util={latest_util:.1f}%, "
                      f"Temp={latest_temp:.1f}°C, "
                      f"Power={latest_power:.1f}W, "
                      f"Mem={latest_mem/1024/1024/1024:.1f}GB")

    def generate_report(self):
        """Generate monitoring report"""
        print("\n=== GPU Monitoring Report ===")

        for gpu_id, history in self.metrics_history.items():
            print(f"\nGPU {gpu_id}:")

            if history['utilization']:
                avg_util = sum(history['utilization']) / len(history['utilization'])
                max_util = max(history['utilization'])
                min_util = min(history['utilization'])

                print(f"  Utilization: Avg={avg_util:.1f}%, Max={max_util:.1f}%, Min={min_util:.1f}%")

            if history['temperature']:
                avg_temp = sum(history['temperature']) / len(history['temperature'])
                max_temp = max(history['temperature'])

                print(f"  Temperature: Avg={avg_temp:.1f}°C, Max={max_temp:.1f}°C")

            if history['power']:
                avg_power = sum(history['power']) / len(history['power'])
                max_power = max(history['power'])

                print(f"  Power: Avg={avg_power:.1f}W, Max={max_power:.1f}W")

    def plot_metrics(self, gpu_id):
        """Plot metrics for specific GPU"""
        if gpu_id not in self.metrics_history:
            print(f"No data available for GPU {gpu_id}")
            return

        history = self.metrics_history[gpu_id]

        if not history['timestamps']:
            print(f"No timestamp data available for GPU {gpu_id}")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'GPU {gpu_id} Metrics')

        timestamps = history['timestamps']

        # Utilization
        if history['utilization']:
            ax1.plot(timestamps, history['utilization'], 'b-')
            ax1.set_title('GPU Utilization')
            ax1.set_ylabel('Utilization (%)')
            ax1.grid(True)

        # Temperature
        if history['temperature']:
            ax2.plot(timestamps, history['temperature'], 'r-')
            ax2.set_title('Temperature')
            ax2.set_ylabel('Temperature (°C)')
            ax2.grid(True)

        # Power
        if history['power']:
            ax3.plot(timestamps, history['power'], 'g-')
            ax3.set_title('Power Usage')
            ax3.set_ylabel('Power (W)')
            ax3.set_xlabel('Time')
            ax3.grid(True)

        # Memory Usage
        if history['memory_usage']:
            memory_gb = [mem / 1024 / 1024 / 1024 for mem in history['memory_usage']]
            ax4.plot(timestamps, memory_gb, 'm-')
            ax4.set_title('Memory Usage')
            ax4.set_ylabel('Memory (GB)')
            ax4.set_xlabel('Time')
            ax4.grid(True)

        plt.tight_layout()
        plt.show()

def main():
    app = GpuMonitorApp()

    try:
        # Monitor for 30 minutes with 5-second intervals
        app.monitor_gpus(duration_minutes=30, interval_seconds=5)

        # Generate report
        app.generate_report()

        # Plot metrics for first GPU
        if app.metrics_history:
            first_gpu_id = next(iter(app.metrics_history.keys()))
            app.plot_metrics(first_gpu_id)

    finally:
        app.session.shutdown()

if __name__ == "__main__":
    main()
```

### 案例2：Go服务集成

```go
// examples/go_gpu_service/main.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/gorilla/mux"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

type GpuService struct {
    monitor *dcgm.GpuMonitor
    server  *http.Server
    metrics struct {
        gpuUtilization   *prometheus.GaugeVec
        gpuTemperature  *prometheus.GaugeVec
        gpuPowerUsage    *prometheus.GaugeVec
        gpuMemoryUsage  *prometheus.GaugeVec
    }
}

func NewGpuService() (*GpuService, error) {
    monitor, err := dcgm.NewGpuMonitor()
    if err != nil {
        return nil, fmt.Errorf("failed to create GPU monitor: %v", err)
    }

    service := &GpuService{
        monitor: monitor,
    }

    // Initialize Prometheus metrics
    service.metrics.gpuUtilization = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dcgm_gpu_utilization_percent",
            Help: "GPU utilization percentage",
        },
        []string{"gpu_id", "gpu_name"},
    )

    service.metrics.gpuTemperature = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dcgm_gpu_temperature_celsius",
            Help: "GPU temperature in Celsius",
        },
        []string{"gpu_id", "gpu_name"},
    )

    service.metrics.gpuPowerUsage = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dcgm_gpu_power_usage_watts",
            Help: "GPU power usage in watts",
        },
        []string{"gpu_id", "gpu_name"},
    )

    service.metrics.gpuMemoryUsage = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dcgm_gpu_memory_usage_bytes",
            Help: "GPU memory usage in bytes",
        },
        []string{"gpu_id", "gpu_name"},
    )

    // Register metrics
    prometheus.MustRegister(service.metrics.gpuUtilization)
    prometheus.MustRegister(service.metrics.gpuTemperature)
    prometheus.MustRegister(service.metrics.gpuPowerUsage)
    prometheus.MustRegister(service.metrics.gpuMemoryUsage)

    return service, nil
}

func (s *GpuService) Start(ctx context.Context) error {
    // Setup HTTP server
    router := mux.NewRouter()

    // Health check endpoint
    router.HandleFunc("/health", s.healthCheck).Methods("GET")

    // GPU info endpoint
    router.HandleFunc("/api/v1/gpus", s.getGpus).Methods("GET")
    router.HandleFunc("/api/v1/gpus/{gpuId}", s.getGpu).Methods("GET")

    // Metrics endpoint
    router.Handle("/metrics", promhttp.Handler()).Methods("GET")

    // WebSocket for real-time updates
    router.HandleFunc("/ws", s.handleWebSocket).Methods("GET")

    s.server = &http.Server{
        Addr:    ":8080",
        Handler: router,
    }

    // Start monitoring all GPUs
    go s.startMonitoring(ctx)

    // Start HTTP server
    go func() {
        log.Printf("Starting GPU service on :8080")
        if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Printf("HTTP server error: %v", err)
        }
    }()

    return nil
}

func (s *GpuService) startMonitoring(ctx context.Context) {
    // Monitor all GPUs every 5 seconds
    allMetricsChan, err := s.monitor.MonitorAllGpus(ctx, 5*time.Second)
    if err != nil {
        log.Printf("Failed to start monitoring: %v", err)
        return
    }

    for metrics := range allMetricsChan {
        for gpuId, gpuMetrics := range metrics {
            // Update Prometheus metrics
            if gpuInfo, err := s.monitor.GetGpuInfo(gpuId); err == nil {
                gpuIdStr := fmt.Sprintf("%d", gpuId)
                gpuName := gpuInfo.Name

                if utilization, ok := gpuMetrics["utilization"].(float64); ok {
                    s.metrics.gpuUtilization.WithLabelValues(gpuIdStr, gpuName).Set(utilization)
                }

                if temperature, ok := gpuMetrics["temperature"].(float64); ok {
                    s.metrics.gpuTemperature.WithLabelValues(gpuIdStr, gpuName).Set(temperature)
                }

                if power, ok := gpuMetrics["power"].(float64); ok {
                    s.metrics.gpuPowerUsage.WithLabelValues(gpuIdStr, gpuName).Set(power)
                }

                if memory, ok := gpuMetrics["used_memory"].(float64); ok {
                    s.metrics.gpuMemoryUsage.WithLabelValues(gpuIdStr, gpuName).Set(memory)
                }
            }
        }
    }
}

func (s *GpuService) healthCheck(w http.ResponseWriter, r *http.Request) {
    gpus := s.monitor.GetGpus()
    response := map[string]interface{}{
        "status":  "healthy",
        "gpus":    len(gpus),
        "version": "1.0.0",
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (s *GpuService) getGpus(w http.ResponseWriter, r *http.Request) {
    gpus := s.monitor.GetGpus()

    var gpuList []map[string]interface{}
    for _, gpu := range gpus {
        gpuInfo := map[string]interface{}{
            "gpu_id":         gpu.GpuId,
            "gpu_name":       gpu.GpuName,
            "serial":         gpu.Serial,
            "memory_size":    gpu.MemorySize,
            "max_power_limit": gpu.MaxPowerLimit,
        }
        gpuList = append(gpuList, gpuInfo)
    }

    response := map[string]interface{}{
        "gpus": gpuList,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (s *GpuService) getGpu(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    gpuId := vars["gpuId"]

    var gpuIdInt uint32
    _, err := fmt.Sscanf(gpuId, "%d", &gpuIdInt)
    if err != nil {
        http.Error(w, "Invalid GPU ID", http.StatusBadRequest)
        return
    }

    gpu, err := s.monitor.GetGpuInfo(gpuIdInt)
    if err != nil {
        http.Error(w, "GPU not found", http.StatusNotFound)
        return
    }

    // Get current metrics
    metrics, err := s.monitor.GetGpuMetrics(gpuIdInt)
    if err != nil {
        log.Printf("Failed to get GPU metrics: %v", err)
        metrics = map[string]float64{}
    }

    response := map[string]interface{}{
        "gpu_id":         gpu.GpuId,
        "gpu_name":       gpu.GpuName,
        "serial":         gpu.Serial,
        "memory_size":    gpu.MemorySize,
        "max_power_limit": gpu.MaxPowerLimit,
        "current_metrics": metrics,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (s *GpuService) handleWebSocket(w http.ResponseWriter, r *http.Request) {
    // WebSocket implementation for real-time updates
    // This would use a WebSocket library like gorilla/websocket
    w.Write([]byte("WebSocket endpoint - to be implemented"))
}

func (s *GpuService) Shutdown(ctx context.Context) error {
    if s.server != nil {
        shutdownCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
        defer cancel()

        if err := s.server.Shutdown(shutdownCtx); err != nil {
            return fmt.Errorf("failed to shutdown server: %v", err)
        }
    }

    return s.monitor.Close()
}

func main() {
    service, err := NewGpuService()
    if err != nil {
        log.Fatalf("Failed to create GPU service: %v", err)
    }

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if err := service.Start(ctx); err != nil {
        log.Fatalf("Failed to start GPU service: %v", err)
    }

    // Handle graceful shutdown
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    <-sigChan
    log.Println("Shutting down GPU service...")

    if err := service.Shutdown(ctx); err != nil {
        log.Fatalf("Failed to shutdown GPU service: %v", err)
    }

    log.Println("GPU service shutdown complete")
}
```

## 总结

DCGM API设计是一个优秀的跨语言绑定案例，其核心特点包括：

1. **分层架构**：从C API到高级语言绑定的完整层次
2. **一致性设计**：所有语言绑定保持一致的API风格
3. **类型安全**：充分利用各语言的类型系统
4. **内存管理**：安全的跨语言内存管理机制
5. **性能优化**：零拷贝和缓存优化技术
6. **版本兼容**：完善的版本兼容性保证
7. **错误处理**：统一的错误处理机制

通过深入理解DCGM API的设计，我们可以学到构建跨语言系统的关键技术：

- **API设计**：如何设计稳定且易用的API
- **跨语言绑定**：如何实现高效的跨语言调用
- **类型映射**：如何在类型系统间进行映射
- **内存管理**：如何管理跨语言内存
- **性能优化**：如何优化跨语言调用性能

DCGM API的设计体现了现代跨语言系统的最佳实践，为我们提供了构建多语言生态系统的优秀范例。

---

*下一篇文章我们将深入探讨DCGM性能优化，解析十万级GPU监控的实战经验。*