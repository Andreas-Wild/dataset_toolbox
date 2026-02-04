"""This module is used to convert label studio masks into numpy arrays saved as PNG images. The module includes a DatasetConverter class that can be imported for use elsewhere.
The module may also be run as a script, where the annotations.json, image source directory and, optionally, output directory are provided.
If the output directory is not specified the dataset will be created at the current working directory."""

import json
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import argparse

class DatasetConverter():
    def __init__(self, annotation_input_json:Path|str, image_source_path:Path|str, output_directory: Path|str = Path.cwd()) -> None:
        self.annotation_json: Path = Path(annotation_input_json)
        self.image_source: Path = Path(image_source_path)
        self.output_directory:Path = Path(output_directory)
        
    def decode_rle(self, rle, height, width):
        def access_bit(data, num):
            base = int(num // 8)
            shift = 7 - int(num % 8)
            return (data[base] & (1 << shift)) >> shift
        
        def bytes2bit(data):
            return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])
        
        class InputStream:
            def __init__(self, data):
                self.data, self.i = data, 0
            def read(self, size):
                out = self.data[self.i:self.i + size]
                self.i += size
                return int(out, 2)
        
        bits = bytes2bit(bytes(rle))
        stream = InputStream(bits)
        num = stream.read(32)
        word_size = stream.read(5) + 1
        rle_sizes = [stream.read(4) + 1 for _ in range(4)]
        
        out = np.zeros(num, dtype=np.uint8)
        i = 0
        while i < num:
            x = stream.read(1)
            j = i + 1 + stream.read(rle_sizes[stream.read(2)])
            if x:
                val = stream.read(word_size)
                out[i:j] = val
                i = j
            else:
                while i < j:
                    out[i] = stream.read(word_size)
                    i += 1
        return np.reshape(out, [height, width, 4])[:, :, 3]

    def prepare_dataset(self):
        img_dir = os.path.join(self.output_directory, 'images')
        mask_dir = os.path.join(self.output_directory, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        with open(self.annotation_json, 'r') as f:
            tasks = json.load(f)

        print(f"Total tasks found in JSON: {len(tasks)}")
        processed_count = 0
        skipped_count = 0

        for task in tasks:
            if 'mask' not in task or not task['mask']:
                skipped_count += 1
                continue

            file_name = task['img'].split('/')[-1].split('?')[0]
            src_path = os.path.join(self.image_source, file_name)

            if not os.path.exists(src_path):
                print(f"Skipping {file_name}: File not found in {self.image_source}")
                skipped_count += 1
                continue

            try:
                h = task['mask'][0]['original_height']
                w = task['mask'][0]['original_width']
            except KeyError:
                print(f"Skipping {file_name}: Missing dimension metadata.")
                skipped_count += 1
                continue

            # Create Instance Mask (background=0, objects=1,2,3...)
            instance_mask = np.zeros((h, w), dtype=np.uint8)

            for i, mask_data in enumerate(task['mask']):
                try:
                    binary_mask = self.decode_rle(mask_data['rle'], h, w)
                    instance_id = i + 1
                    instance_mask[binary_mask > 0] = instance_id
                except Exception as e:
                    print(f"Error decoding mask for {file_name}: {e}")
                    continue

            # Save Image and Mask as PNG
            shutil.copy(src_path, os.path.join(img_dir, file_name))
            mask_filename = os.path.splitext(file_name)[0] + '.png'
            cv2.imwrite(os.path.join(mask_dir, mask_filename), instance_mask)
            
            processed_count += 1

        print("\nProcessing complete!")
        print(f"Images with masks created: {processed_count}")
        print(f"Tasks skipped (no masks/missing files): {skipped_count}")
        print(f"Dataset saved to: {self.output_directory}")

if __name__ == "__main__":
    # The module can also be run from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json", type=str, help="The annotation.json file returned by Label Studio.")
    parser.add_argument("image_source_directory", type=str, help="The directory where raw image files are saved. (Typically, label-studio/files/)")
    parser.add_argument("-o", "--output-dir", type=str, help="The output location where the dataset will be saved.")
    args = parser.parse_args()

    converter = DatasetConverter(args.annotation_json, args.image_source_directory, args.output_dir)
    converter.prepare_dataset()
        