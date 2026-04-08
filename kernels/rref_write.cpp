// rref_writer.cpp
// DataMovement kernel (RISCV_1, NOC 1)
//
// Runtime args (set by host per-core):
//   0: dram_addr – base byte address of this matrix's tile buffer in DRAM
//   1: nt_r      – number of tile-rows  (ceil(rows / 32))
//   2: nt_c      – number of tile-cols  (ceil(cols / 32))
//
// What this kernel does
// ─────────────────────
// Drains CB::c_out0 one tile-row (nt_c tiles) at a time and writes each tile
// back to its original location in the DRAM buffer, overwriting the input.
//
// The flat tile index matches the reader:
//   tile(tr, tc)  →  index  tr * nt_c + tc
//
// The double-buffer depth of 2*nt_c in c_out0 means the writer can be
// issuing writes for one row while the compute kernel fills the next,
// keeping both pipelines busy without stalls.

#include "dataflow_api.h"

void kernel_main() {
    // ── Runtime args ────────────────────────────────────────────────────────
    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t nt_r      = get_arg_val<uint32_t>(1);
    uint32_t nt_c      = get_arg_val<uint32_t>(2);

    // ── DRAM address generator ───────────────────────────────────────────────
    // InterleavedAddrGenFast is the current canonical interleaved-DRAM helper.
    // page_size and data_format are resolved at compile time so address
    // calculation costs nothing at runtime.
    constexpr uint32_t cb_out = CB::c_out0;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    const InterleavedAddrGenFast</*DRAM=*/true> dst = {
        .bank_base_address = dram_addr,
        .page_size         = tile_bytes,
        .data_format       = DataFormat::Float16_b,
    };

    // ── Drain c_out0 one tile-row at a time ─────────────────────────────────
    for (uint32_t tr = 0; tr < nt_r; tr++) {
        // Block until the compute kernel has produced the next nt_c tiles.
        cb_wait_front(cb_out, nt_c);

        uint32_t l1_read_ptr = get_read_ptr(cb_out);

        for (uint32_t tc = 0; tc < nt_c; tc++) {
            // noc_async_write_tile resolves the interleaved DRAM bank address
            // from the flat tile index and issues a non-blocking NoC write.
            noc_async_write_tile(tr * nt_c + tc, dst, l1_read_ptr);
            l1_read_ptr += tile_bytes;
        }

        // Flush all in-flight writes before releasing the CB pages so L1 is
        // not reclaimed while the NoC transaction is still in progress.
        noc_async_write_barrier();

        cb_pop_front(cb_out, nt_c);
    }
}