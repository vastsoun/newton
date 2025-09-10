###########################################################################
# KAMINO: Utilities: Profiling
###########################################################################


from warp.context import Devicelike


def get_device_info(device: Devicelike) -> str:
    dinfo = "[device]:\n"
    dinfo += f"                name: {device.name}\n"
    dinfo += f"               alias: {device.alias}\n"
    dinfo += f"                arch: {device.arch}\n"
    dinfo += f"                uuid: {device.uuid}\n"
    dinfo += f"             ordinal: {device.ordinal}\n"
    dinfo += f"          pci_bus_id: {device.pci_bus_id}\n"
    dinfo += f"              is_uva: {device.is_uva}\n"
    dinfo += f"          is_primary: {device.is_primary}\n"
    dinfo += f"  is_cubin_supported: {device.is_cubin_supported}\n"
    dinfo += f"is_mempool_supported: {device.is_mempool_supported}\n"
    dinfo += f"  is_mempool_enabled: {device.is_mempool_enabled}\n"
    dinfo += f"    is_ipc_supported: {device.is_ipc_supported}\n"
    dinfo += f"              is_cpu: {device.is_cpu}\n"
    dinfo += f"             is_cuda: {device.is_cuda}\n"
    dinfo += f"        is_capturing: {device.is_capturing}\n"
    dinfo += f"         has_context: {device.has_context}\n"
    dinfo += f"             context: {device.context}\n"
    dinfo += f"          has_stream: {device.has_stream}\n"
    dinfo += f"            sm_count: {device.sm_count}\n"
    dinfo += f"total_memory (bytes): {device.total_memory}\n"
    dinfo += f" free_memory (bytes): {device.free_memory}\n"
    return dinfo


def get_device_memory_allocation_info(device: Devicelike) -> str:
    return ""
