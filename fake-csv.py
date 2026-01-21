import csv
import math
import time
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate fake DCGM-like CSV data using sin/cos"
    )
    parser.add_argument(
        "-n", "--seconds",
        type=int,
        default=100,
        help="Number of seconds of data to generate"
    )
    parser.add_argument(
        "-g", "--gpus",
        type=int,
        default=4,
        help="Number of fake GPUs (gpuId = 0..gpus-1)"
    )
    parser.add_argument(
        "-o", "--output",
        default="dcgm_fake.csv",
        help="Output CSV filename"
    )

    args = parser.parse_args()

    columns = [
        "timestamp","gpuId",
        "DCGM_FI_DEV_GPU_UTIL_min","DCGM_FI_DEV_GPU_UTIL_avg","DCGM_FI_DEV_GPU_UTIL_max",
        "DCGM_FI_DEV_FB_FREE_min","DCGM_FI_DEV_FB_FREE_avg","DCGM_FI_DEV_FB_FREE_max",
        "DCGM_FI_DEV_FB_USED_min","DCGM_FI_DEV_FB_USED_avg","DCGM_FI_DEV_FB_USED_max",
        "DCGM_FI_DEV_FB_RESERVED_min","DCGM_FI_DEV_FB_RESERVED_avg","DCGM_FI_DEV_FB_RESERVED_max",
        "DCGM_FI_PROF_SM_ACTIVE_min","DCGM_FI_PROF_SM_ACTIVE_avg","DCGM_FI_PROF_SM_ACTIVE_max",
        "DCGM_FI_PROF_SM_OCCUPANCY_min","DCGM_FI_PROF_SM_OCCUPANCY_avg","DCGM_FI_PROF_SM_OCCUPANCY_max",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_min","DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_avg","DCGM_FI_PROF_PIPE_TENSOR_ACTIVE_max",
        "DCGM_FI_PROF_PIPE_FP64_ACTIVE_min","DCGM_FI_PROF_PIPE_FP64_ACTIVE_avg","DCGM_FI_PROF_PIPE_FP64_ACTIVE_max",
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE_min","DCGM_FI_PROF_PIPE_FP32_ACTIVE_avg","DCGM_FI_PROF_PIPE_FP32_ACTIVE_max",
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE_min","DCGM_FI_PROF_PIPE_FP16_ACTIVE_avg","DCGM_FI_PROF_PIPE_FP16_ACTIVE_max",
        "DCGM_FI_PROF_DRAM_ACTIVE_min","DCGM_FI_PROF_DRAM_ACTIVE_avg","DCGM_FI_PROF_DRAM_ACTIVE_max",
        "DCGM_FI_PROF_PCIE_TX_BYTES_min","DCGM_FI_PROF_PCIE_TX_BYTES_avg","DCGM_FI_PROF_PCIE_TX_BYTES_max",
        "DCGM_FI_PROF_PCIE_RX_BYTES_min","DCGM_FI_PROF_PCIE_RX_BYTES_avg","DCGM_FI_PROF_PCIE_RX_BYTES_max",
        "DCGM_FI_PROF_NVLINK_RX_BYTES_min","DCGM_FI_PROF_NVLINK_RX_BYTES_avg","DCGM_FI_PROF_NVLINK_RX_BYTES_max",
        "DCGM_FI_PROF_NVLINK_TX_BYTES_min","DCGM_FI_PROF_NVLINK_TX_BYTES_avg","DCGM_FI_PROF_NVLINK_TX_BYTES_max",
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_min","DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_avg","DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_max",
        "DCGM_FI_DEV_POWER_USAGE_min","DCGM_FI_DEV_POWER_USAGE_avg","DCGM_FI_DEV_POWER_USAGE_max",
    ]

    start_ts = int(time.time())

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i in range(args.seconds):

            for gpu in range(args.gpus):

                base = (math.sin(i / 10.0 + gpu) + 1.) / 2
                base2 = (math.cos(i / 15.0 + gpu) + 1.) / 2

                row = [i, gpu]

                # number of (min, avg, max) triplets
                n_triplets = (len(columns) - 2) // 3

                for j in range(n_triplets):
                    avg = base #.50 * (base + 0.1 * j)
                    spread = .05 * abs(math.sin(i + j))
                    row.extend([
                        avg - spread,
                        avg,
                        avg + spread
                    ])

                writer.writerow(row)

    print(f"Wrote {args.seconds} seconds to {args.output}")

if __name__ == "__main__":
    main()
