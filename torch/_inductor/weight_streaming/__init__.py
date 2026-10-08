"""Three-tier weight streaming for captured Inductor graphs: weights homed in
pinned DRAM or on an NVMe SSD are copied into a fixed VRAM arena just ahead of
use, gated by in-graph stream memops; the SSD reads are issued by the GPU
through the standalone gpu-co library."""
