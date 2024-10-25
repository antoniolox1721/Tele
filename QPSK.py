#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import pandas as pd
from PyQt5 import Qt
from gnuradio import gr
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
import time
import os

FILE_2_OFFSET = 49 * 0.02
DATA_SIZE = 216
OUTPUT_FILENAME = "QPSKFINAL.xlsx"

def calc(file1, file2):
    """Compare binary files accounting for offset"""
    try:
        # Check if files exist and have content
        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"Files missing: {file1} exists: {os.path.exists(file1)}, {file2} exists: {os.path.exists(file2)}")
            return -1

        # Get file sizes
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)

        if size1 == 0 or size2 == 0:
            print(f"Empty files: {file1} size: {size1}, {file2} size: {size2}")
            return -1

        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            # Read entire contents
            content1 = f1.read()
            content2 = f2.read()

            # Verify we have enough data
            required_bytes1 = math.ceil(DATA_SIZE / 8)
            required_bytes2 = math.ceil((DATA_SIZE + FILE_2_OFFSET) / 8)

            if len(content1) < required_bytes1 or len(content2) < required_bytes2:
                print(f"Insufficient data: need {required_bytes1}/{required_bytes2} bytes, got {len(content1)}/{len(content2)}")
                return -1

            differences = 0
            for i in range(DATA_SIZE):
                # Get bit from first file
                byte_idx1 = i // 8
                bit_idx1 = 7 - (i % 8)
                bit1 = bool(content1[byte_idx1] & (1 << bit_idx1))

                # Get bit from second file (with offset)
                adjusted_bit_pos = i + FILE_2_OFFSET
                byte_idx2 = adjusted_bit_pos // 8
                bit_idx2 = 7 - (adjusted_bit_pos % 8)
                bit2 = bool(content2[byte_idx2] & (1 << bit_idx2))

                if bit1 != bit2:
                    differences += 1

            return differences

    except Exception as e:
        print(f"Error comparing files: {str(e)}")
        return -1

class top_block(gr.top_block):
    def __init__(self, noise_voltage, loop_bandwidth):
        gr.top_block.__init__(self, "QPSK Analysis")
        
        ##################################################
        # Variables with optimized parameters
        ##################################################
        self.sps = sps = 8  # Back to 8 samples per symbol for better accuracy
        self.samp_rate = samp_rate = 32000
        self.rolloff = rolloff = 0.75  # Back to original rolloff
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(
            1.0,           # Gain
            samp_rate,     # Sampling rate
            samp_rate/sps, # Symbol rate
            rolloff,       # Roll-off factor
            11*sps        # Number of taps
        )
        
        # QPSK constellation with precise mapping
        constellation = digital.constellation_calcdist([-1-1j, -1+1j, 1+1j, 1-1j], [0, 1, 3, 2],
        4, 1, digital.constellation.AMPLITUDE_NORMALIZATION).base()
        constellation.set_npwr(1.0)
        self.QPSK = constellation

        ##################################################
        # Blocks
        ##################################################
        input_data = [240,240,240,15,15,15,240,240,240] + \
                    [10, 38, 33, 10, 74, 72, 11, 6, 34] + \
                    [15,15,15,240,240,240,15,15,15,0,0,0,0,0,0,0,0]

        self.blocks_vector_source_x_0 = blocks.vector_source_b(input_data, False, 1, [])

        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=self.QPSK,
            differential=False,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=rolloff,
            verbose=False,
            log=False)

        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=42,
            block_tags=False)

        self.fir_filter_xxx_0 = filter.fir_filter_ccc(1, rrc_taps)
        self.fir_filter_xxx_0.declare_sample_delay(0)

        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_SIGNAL_TIMES_SLOPE_ML,  # Back to original timing error detector
            sps,
            loop_bandwidth,
            1.0,    # Damping factor
            4.0,    # Loop bandwidth
            1.5,    # Ted gain
            1,      # Maximum deviation
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP,
            128,    # Back to original buffer size
            []
        )

        # Demodulation chain
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_imag_0 = blocks.complex_to_imag(1)
        self.digital_binary_slicer_fb_0 = digital.binary_slicer_fb()
        self.digital_binary_slicer_fb_1 = digital.binary_slicer_fb()
        self.blocks_char_to_float_0 = blocks.char_to_float(1, 1)
        self.blocks_char_to_float_1 = blocks.char_to_float(1, 1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(2)
        self.blocks_add_xx_0 = blocks.add_ff(1)
        self.blocks_float_to_char_0 = blocks.float_to_char(1, 1)
        self.blocks_pack_k_bits_bb_0 = blocks.pack_k_bits_bb(8)

        # File sinks
        self.blocks_file_sink_sent = blocks.file_sink(gr.sizeof_char*1, 'qpsk_sent.dat', False)
        self.blocks_file_sink_rec = blocks.file_sink(gr.sizeof_char*1, 'qpsk_rec.dat', False)
        self.blocks_file_sink_sent.set_unbuffered(True)  # Back to unbuffered for immediate writing
        self.blocks_file_sink_rec.set_unbuffered(True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_file_sink_sent, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.blocks_complex_to_real_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.blocks_complex_to_imag_0, 0))
        self.connect((self.blocks_complex_to_real_0, 0), (self.digital_binary_slicer_fb_0, 0))
        self.connect((self.blocks_complex_to_imag_0, 0), (self.digital_binary_slicer_fb_1, 0))
        self.connect((self.digital_binary_slicer_fb_0, 0), (self.blocks_char_to_float_0, 0))
        self.connect((self.digital_binary_slicer_fb_1, 0), (self.blocks_char_to_float_1, 0))
        self.connect((self.blocks_char_to_float_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_char_to_float_1, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_float_to_char_0, 0))
        self.connect((self.blocks_float_to_char_0, 0), (self.blocks_pack_k_bits_bb_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0, 0), (self.blocks_file_sink_rec, 0))

def main():
    app = Qt.QApplication(sys.argv)

    noise_voltages = [x/2 for x in range(9)]
    loop_bandwidths = np.arange(0, 0.25, 0.0025)

    columns = ['Loop BW'] + [f'{n:.1f}' for n in noise_voltages]
    df = pd.DataFrame(columns=columns)

    try:
        for bw in loop_bandwidths:
            print(f"\nTesting Loop BW = {bw:.4f}")
            row_data = {'Loop BW': f"{bw:.4f}"}
            
            for noise in noise_voltages:
                print(f"  Noise voltage = {noise:.1f}", end='', flush=True)
                
                # Clean up old files
                for filename in ['qpsk_sent.dat', 'qpsk_rec.dat']:
                    if os.path.exists(filename):
                        os.remove(filename)
                
                # Run flowgraph
                tb = top_block(noise, bw)
                tb.start()
                time.sleep(2.0)  # Increased processing time
                tb.stop()
                tb.wait()
                time.sleep(1.0)  # Increased file writing time
                
                differences = calc("qpsk_sent.dat", "qpsk_rec.dat")
                row_data[f'{noise:.1f}'] = differences
                print(f" -> {differences} errors")
                
                del tb
                time.sleep(0.5)
            
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
            df.to_excel(OUTPUT_FILENAME, index=False)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        if not df.empty:
            df.to_excel(OUTPUT_FILENAME, index=False)

if __name__ == '__main__':
    main()