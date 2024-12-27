#!/usr/bin/env python

# Created by TG on 2024-12-25
# 
# This script is for post build tasks
# back story: I tried to make static build with libtorch and failed
#     Then I tried to make cmake create installation package with dynamic libs and remap the paths
#   there is a cmake package to it, but.. it didnt work for me. Wasted a lot of time
#  So I decided to make a python script to do it manually 


import argparse
from pathlib import Path
import subprocess
import shutil
import logging as log
from typing import List

log.basicConfig(level=log.INFO)


def get_dependencies(bin_path:Path) -> List[str]:
    assert bin_path.exists()
    ldd_output = subprocess.check_output(['ldd', str(bin_path)]).decode()
    deps = []
    for line in ldd_output.splitlines():
        if '=>' in line:
            p = line.split('=>')[1].strip().split()[0]
            deps.append(p)
    return deps

def main():
    args = parse_args()
    build_dir = args.build
    # we pick the executable from the build directory to discover deps
    #   but we patchelf the executable in the install directory
    # this is because when cmake installs the executable, it loses the rpath
    executable = build_dir / 'tahoma'
    install_dir = args.install
    install_libs_dir = install_dir / 'lib'
    assert executable.exists(), f'{executable} does not exist'
    deps = get_dependencies(executable)
    log.info(f'Found {len(deps)} dependencies')
    install_libs_dir.mkdir(parents=True, exist_ok=True)
    for src in deps:
        dest = install_libs_dir / Path(src).name
        log.info(f'Copy {src} --> {dest}')
        shutil.copy2(src, dest, follow_symlinks=True)
    if shutil.which('patchelf'):
        exec_final = install_dir / 'bin' / executable.name
        cmd = f"patchelf --set-rpath '$ORIGIN/../lib' {exec_final}"
        log.info(f'Run:\n\t{cmd}')
        subprocess.run(cmd, shell=True)
    else:
        log.warning('patchelf not found. Please install it to fix the rpath. or use LD_LIBRARY_PATH at runtime')
    if args.archive:
        basename = f'tahoma-{args.version}'
        archive_file: Path = install_dir.parent / f'{basename}.tar.gz'
        # TODO: make the top level dir in the archive to be tahoma-<version>
        cmd = f'cd {install_dir} && tar -czvf {archive_file} *'
        log.info(f'Archiving to {archive_file}:\n\t{cmd}')
        subprocess.run(cmd, shell=True, check=True)
        log.info(f'Archive created: {archive_file}')
    log.info('Done')

def parse_args():
    parser = argparse.ArgumentParser(description='Post build tasks')
    parser.add_argument('-b', '--build', type=Path, required=True,
                        help='Build directory')
    parser.add_argument('-i', '--install', type=Path,
                        help='Install directory. Default is <build>/install')
    parser.add_argument('-a', '--archive', action='store_true',
                        help='Archive the installation directory')
    parser.add_argument('-v', '--version', type=str, required=True,
                        help='Version string')
    args = parser.parse_args()
    if not args.install:
        args.install = args.build / 'install'
    args.install = args.install.resolve()
    return args


if __name__ == '__main__':
    main()
