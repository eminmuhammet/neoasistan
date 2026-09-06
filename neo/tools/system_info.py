from __future__ import annotations

import asyncio
import logging
import platform
import socket
import winreg

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)


def _cpu_name() -> str:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except OSError:
        return platform.processor() or "bilinmiyor"


_own_process = None


def get_own_usage() -> dict[str, float] | None:
    """NEO's *own* CPU/RAM footprint, as opposed to the machine-wide numbers
    the gauges show. Worth reporting separately: a full-screen gauge reading
    "66%" sitting inside NEO's window reads as "NEO is using 66%", when in
    the measured case NEO held 577 MB of 15.8 GB and the rest belonged to
    everything else running.

    CPU is scaled to whole-machine percent (psutil reports per-core, so a
    busy single thread would otherwise read as 100%). Returns None if the
    reading fails rather than guessing a number.
    """
    global _own_process
    try:
        import psutil

        # cpu_percent() measures usage since the previous call *on the same
        # Process object*, so this instance is kept around -- building a new
        # one each refresh would report 0.0 forever.
        if _own_process is None:
            _own_process = psutil.Process()
            _own_process.cpu_percent()

        cores = psutil.cpu_count() or 1
        return {
            "cpu_percent": _own_process.cpu_percent() / cores,
            "memory_mb": _own_process.memory_info().rss / (1024 * 1024),
        }
    except Exception:
        logger.exception("NEO'nun kendi kaynak kullanımı okunamadı")
        return None


class GetSystemInfoTool(Tool):
    name = "get_system_info"
    description = (
        "Bilgisayarın genel özelliklerini döndürür: işletim sistemi, CPU modeli, "
        "çekirdek sayısı, toplam RAM ve bilgisayar adı."
    )
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return await asyncio.to_thread(self._read)

    def _read(self) -> ToolResult:
        import psutil

        try:
            data = {
                "computer_name": platform.node(),
                "os": f"{platform.system()} {platform.release()}",
                "os_version": platform.version(),
                "cpu": _cpu_name(),
                "cpu_cores_physical": psutil.cpu_count(logical=False),
                "cpu_cores_logical": psutil.cpu_count(logical=True),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            }
        except Exception:
            logger.exception("Sistem bilgisi okunamadı")
            return ToolResult(success=False, error="Sistem bilgisi okunurken bir hata oluştu.")
        return ToolResult(success=True, data=data)


class GetCpuUsageTool(Tool):
    name = "get_cpu_usage"
    description = "İşlemcinin (CPU) anlık kullanım yüzdesini döndürür."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        import psutil

        percent = await asyncio.to_thread(psutil.cpu_percent, 0.5)
        return ToolResult(success=True, data={"cpu_percent": percent})


class GetRamUsageTool(Tool):
    name = "get_ram_usage"
    description = "RAM kullanım bilgisini (toplam, kullanılan, yüzde) döndürür."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        import psutil

        vm = psutil.virtual_memory()
        return ToolResult(
            success=True,
            data={
                "total_gb": round(vm.total / (1024**3), 1),
                "used_gb": round(vm.used / (1024**3), 1),
                "percent": vm.percent,
            },
        )


class GetDiskUsageTool(Tool):
    name = "get_disk_usage"
    description = "Diskteki toplam ve boş alan bilgisini döndürür (tüm sabit sürücüler)."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return await asyncio.to_thread(self._read)

    def _read(self) -> ToolResult:
        import psutil

        drives = []
        for part in psutil.disk_partitions(all=False):
            if not part.fstype:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            drives.append(
                {
                    "drive": part.device,
                    "total_gb": round(usage.total / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "percent_used": usage.percent,
                }
            )
        if not drives:
            return ToolResult(success=False, error="Disk bilgisine erişilemedi.")
        return ToolResult(success=True, data={"drives": drives})


class GetGpuStatusTool(Tool):
    name = "get_gpu_status"
    description = (
        "NVIDIA ekran kartının anlık kullanım yüzdesini, VRAM kullanımını ve "
        "sıcaklığını döndürür. Desteklenen bir NVIDIA GPU yoksa veya sürücü "
        "verisine erişilemiyorsa bunu açıkça belirtir, veri uydurmaz."
    )
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return await asyncio.to_thread(self._read)

    def _read(self) -> ToolResult:
        try:
            import pynvml
        except ImportError:
            return ToolResult(success=False, error="pynvml kurulu değil, GPU verisine erişemiyorum.")

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return ToolResult(
                success=False,
                error="Bu sistemde NVIDIA GPU verisine erişemiyorum (sürücü/NVML hatası).",
            )

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            data = {
                "name": name,
                "gpu_percent": util.gpu,
                "vram_used_mb": round(mem.used / (1024**2)),
                "vram_total_mb": round(mem.total / (1024**2)),
            }
            try:
                data["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                pass
            return ToolResult(success=True, data=data)
        except pynvml.NVMLError:
            logger.exception("GPU verisi okunamadı")
            return ToolResult(success=False, error="GPU verisi okunurken bir hata oluştu.")
        finally:
            pynvml.nvmlShutdown()


class GetNetworkStatusTool(Tool):
    name = "get_network_status"
    description = "İnternet bağlantısının aktif olup olmadığını kontrol eder."
    risk = RiskLevel.LOW
    input_schema = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return await asyncio.to_thread(self._check)

    def _check(self) -> ToolResult:
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=2).close()
            return ToolResult(success=True, data={"connected": True})
        except OSError:
            return ToolResult(success=True, data={"connected": False})
