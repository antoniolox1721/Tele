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

FILE_2_OFFSET = 49
DATA_SIZE = 216
OUTPUT_FILENAME = "BPSKFINAL.xlsx"

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

            # Skip offset in second file
            offset_bytes = FILE_2_OFFSET // 8
            offset_bits = FILE_2_OFFSET % 8

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
        gr.top_block.__init__(self, "BPSK Analysis")
        
        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 8
        self.samp_rate = samp_rate = 32000
        self.rolloff = rolloff = 0.75
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0, samp_rate, samp_rate/sps, rolloff, 11*sps)
        self.noise_voltage = noise_voltage
        self.loop_bandwidth = loop_bandwidth
        self.BPSK = BPSK = digital.constellation_calcdist([-1 + 0j, 1 + 0j], [0, 1],
        2, 1).base()

        ##################################################
        # Blocks
        ##################################################
        # Input data - Fixed sequence as per requirements
        input_data = [240,240,240,15,15,15,240,240,240] + \
                    [10, 38, 33, 10, 74, 72, 11, 6, 34] + \
                    [15,15,15,240,240,240,15,15,15,0,0,0,0,0,0,0,0]

        self.blocks_vector_source_x_0 = blocks.vector_source_b(input_data, False, 1, [])

        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=BPSK,
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
            noise_seed=0,
            block_tags=False)

        self.fir_filter_xxx_0 = filter.fir_filter_ccc(1, rrc_taps)
        self.fir_filter_xxx_0.declare_sample_delay(0)

        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_SIGNAL_TIMES_SLOPE_ML,
            sps,
            loop_bandwidth,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_bpsk().base(),
            digital.IR_MMSE_8TAP,
            128,
            [])

        # Processing blocks
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.digital_binary_slicer_fb_0 = digital.binary_slicer_fb()
        self.blocks_pack_k_bits_bb_0 = blocks.pack_k_bits_bb(8)
        
        # File sinks
        self.blocks_file_sink_sent = blocks.file_sink(gr.sizeof_char*1, 'bpsk_sent.dat', False)
        self.blocks_file_sink_sent.set_unbuffered(True)
        self.blocks_file_sink_rec = blocks.file_sink(gr.sizeof_char*1, 'bpsk_rec.dat', False)
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
        self.connect((self.blocks_complex_to_real_0, 0), (self.digital_binary_slicer_fb_0, 0))
        self.connect((self.digital_binary_slicer_fb_0, 0), (self.blocks_pack_k_bits_bb_0, 0))
        self.connect((self.blocks_pack_k_bits_bb_0, 0), (self.blocks_file_sink_rec, 0))

def main():
    app = Qt.QApplication(sys.argv)

    # Test parameters
    noise_voltages = [x/2 for x in range(9)]  # 0 to 4 in steps of 0.5
    loop_bandwidths = np.arange(0, 0.25, 0.0025)  # 0 to 0.25 in steps of 0.0025

    # Create DataFrame
    columns = ['Loop BW'] + [f'{n:.1f}' for n in noise_voltages]
    df = pd.DataFrame(columns=columns)

    try:
        for bw in loop_bandwidths:
            print(f"\nTesting Loop BW = {bw:.4f}")
            row_data = {'Loop BW': f"{bw:.4f}"}
            
            for noise in noise_voltages:
                print(f"  Noise voltage = {noise:.1f}", end='', flush=True)
                
                # Run flowgraph
                tb = top_block(noise, bw)
                tb.start()
                time.sleep(0.2)  # Give more time for data to flow
                tb.stop()
                tb.wait()
                
                # Calculate errors
                differences = calc("bpsk_sent.dat", "bpsk_rec.dat")
                row_data[f'{noise:.1f}'] = differences
                print(f" -> {differences} errors")
                
                # Cleanup
                del tb
                time.sleep(0.1)  # Add delay between tests
            
            # Add row to DataFrame
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
            
            # Save progress
            df.to_excel(OUTPUT_FILENAME, index=False)
            df.to_csv('results.csv', index=False, sep='\t')
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    finally:
        # Final save
        if not df.empty:
            df.to_excel(OUTPUT_FILENAME, index=False)
            df.to_csv('results.csv', index=False, sep='\t')

if __name__ == '__main__':
    main()