from .logger import logger_instance as log
from .safe_utils import safe_div


def profiling(profiling_data, image_total):
    e2e_cost_time_per_image = 0
    for module_name in profiling_data:
        data = profiling_data[module_name]
        total_time = data[0]
        process_time = data[0] - data[1]
        send_time = data[1]
        process_avg = safe_div(process_time * 1000, image_total)
        e2e_cost_time_per_image += process_avg
        log.info(
            f"{module_name} cost total {total_time:.2f} s, process avg cost {process_avg:.2f} ms, "
            f"send waiting time avg cost {safe_div(send_time * 1000, image_total):.2f} ms"
        )
        log.info("----------------------------------------------------")
    log.info(f"e2e cost time per image {e2e_cost_time_per_image}ms")
