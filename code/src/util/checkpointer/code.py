import os
import shutil


def dump_code(in_dir_path: os.path, out_file_path: os.path):
    shutil.make_archive(out_file_path, "zip", in_dir_path)
