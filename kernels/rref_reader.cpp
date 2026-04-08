// rref_reader.cpp
// DataMovement kernel (RISCV_0, NOC 0)
//
// Runtime args (set by host per-core):
//   0: dram_addr       – base byte address of this matrix's tile buffer in DRAM
//   1: nt_r            – number of tile-rows  (ceil(rows / 32))
//   2: nt_c            – number of tile-cols  (ceil(cols / 32))
//   3: pivot_row       – element-row index of the pivot row for this step
//   4: pivot_tile_col  – tile-col index that contains the pivot column
//                        (pivot_col / TILE_W, passed pre-divided by host)
//
// What this kernel does
// ─────────────────────
// Phase 1 – load pivot row tiles into CB::c_in1  (once, single-buffered)
//   Read the nt_c tiles that make up the pivot row (tile-row = pivot_row/32).
//
// Phase 2 – stream every row into CB::c_in0  (double-buffered)
//   For each of the nt_r tile-rows, push nt_c tiles into c_in0.
//   The compute kernel consumes each row of tiles before we push the next,
//   so the double-buffer depth of 2*nt_c is sufficient.
//
// Tile layout in DRAM (row-major over tile grid):
//   tile(tr, tc)  starts at  dram_addr + (tr * nt_c + tc) * TILE_BYTES
//   where TILE_BYTES = 32*32*2 = 2048 bytes  (bfloat16).

#include "dataflow_api.h"

static constexpr uint32_t TILE_BYTES = 32 * 32 * sizeof(uint16_t); // 2048

void kernel_main() {
    // ── Runtime args ────────────────────────────────────────────────────────
    uint32_t dram_addr      = get_arg_val<uint32_t>(0);
    uint32_t nt_r           = get_arg_val<uint32_t>(1);
    uint32_t nt_c           = get_arg_val<uint32_t>(2);
    uint32_t pivot_row      = get_arg_val<uint32_t>(3);
    uint32_t pivot_tile_row = pivot_row / 32; // tile-row that holds the pivot

    // ── DRAM accessor ───────────────────────────────────────────────────────
    // All tiles live in a single flat DRAM buffer; page size == one tile.
    constexpr uint32_t DRAM_CHANNEL = 0;
    const InterleavedAddrGen</*DRAM=*/true> s = {
        .bank_base_address = dram_addr,
        .page_size         = TILE_BYTES,
    };

    // ── Phase 1: load pivot row → CB c_in1 ─────────────────────────────────
    // c_in1 is single-buffered with capacity nt_c; we push it exactly once.
    cb_reserve_back(CB::c_in1, nt_c);
    uint32_t l1_write_ptr = get_write_ptr(CB::c_in1);

    for (uint32_t tc = 0; tc < nt_c; tc++) {
        uint32_t tile_idx = pivot_tile_row * nt_c + tc;
        uint64_t src_noc  = get_noc_addr(tile_idx, s);
        noc_async_read(src_noc, l1_write_ptr, TILE_BYTES);
        l1_write_ptr += TILE_BYTES;
    }
    noc_async_read_barrier(); // wait for all pivot-row reads to land
    cb_push_back(CB::c_in1, nt_c);

    // ── Phase 2: stream every tile-row → CB c_in0  (double-buffered) ───────
    // We iterate tile-rows in order; the compute kernel consumes each set of
    // nt_c tiles before we write the next, enforced via cb_reserve_back /
    // cb_push_back handshake.
    for (uint32_t tr = 0; tr < nt_r; tr++) {
        cb_reserve_back(CB::c_in0, nt_c);
        l1_write_ptr = get_write_ptr(CB::c_in0);

        for (uint32_t tc = 0; tc < nt_c; tc++) {
            uint32_t tile_idx = tr * nt_c + tc;
            uint64_t src_noc  = get_noc_addr(tile_idx, s);
            noc_async_read(src_noc, l1_write_ptr, TILE_BYTES);
            l1_write_ptr += TILE_BYTES;
        }
        noc_async_read_barrier(); // ensure all tiles for this row have arrived
        cb_push_back(CB::c_in0, nt_c);
    }
}