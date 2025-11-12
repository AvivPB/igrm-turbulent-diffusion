import os
import argparse

import yt
import caesar

## For dougpython: module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

parser = argparse.ArgumentParser(prog='make_caesar_files.py', description='Create caesar files for snapshots')
parser.add_argument('--snap_dir', action='store', type=str, required=True, 
                    help='directory containing snapshots')
parser.add_argument('--snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers')
parser.add_argument('--snap_base', action='store', type=str, default='snapshot_',
                    help='base name for snapshots, e.g. snapshot_')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory to write caesar files')
parser.add_argument('--haloid', action='store', type=str, default='snap', choices=['fof', 'snap'],
                    help='caesar haloid option')
parser.add_argument('--blackholes', action=argparse.BooleanOptionalAction, default=True,
                    help='caesar blackholes option')
parser.add_argument('--aperture', action='store', type=float, default=30,
                    help='Aperture size in ckpc for galaxy masses')
parser.add_argument('--half_stellar_radius_property', action=argparse.BooleanOptionalAction, default=True,
                    help='caesar half_stellar_radius_property option')
parser.add_argument('--lowres', action='store', nargs='*', type=int, default=[],
                    help='caesar lowres option')
parser.add_argument('--nproc', action='store', type=int, default=1,
                    help='caesar nproc option')
args = parser.parse_args()

# snap_dir = args.snap_dir #'/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/halo_3224_v3/'
# snap_list = args.snap_nums
# snap_list = list(range(152))
print(f'snap_list={args.snap_nums}')
# snap_base = args.snap_base #'snapshot_'

# haloid = args.haloid #'fof'
# blackholes = args.blackholes #True
# lowres = args.lowres #[2]
# nproc = args.nproc #32

print(f'lowres={args.lowres}')
print()

# caesar_dir = os.path.join(snap_dir, f'caesar_{haloid}')
if not os.path.exists(args.caesar_dir):
    print('Making caesar directory...')
    os.makedirs(args.caesar_dir)
    print()


for snap_num in args.snap_nums:
    snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{snap_num:03}.hdf5')
    if not os.path.exists(snap_file):
        print(f'{snap_file} does not exist')
        print()
        continue

    print(snap_file)
    try:
        snap = yt.load(snap_file)
    except Exception as error:
        print(f'Error occurred loading snapshot: {error}')
        print()
        continue

    try:
        obj = caesar.CAESAR(snap)
    except Exception as error:
        print(f'Error occurred creating caesar object: {error}')
        print()
        continue

    try:
        obj.member_search(haloid=args.haloid, blackholes=args.blackholes,
                          aperture=args.aperture, half_stellar_radius_property=args.half_stellar_radius_property,
                          nproc=args.nproc, lowres=args.lowres)
    except Exception as error:
        print(f'Error occurred running caesar member_search: {error}')
        print()
        continue
        

    # caesar_file = os.path.join(caesar_dir, f'caesar_{snap_num:03}_haloid-{haloid}_lowres-{lowres}.hdf5')
    caesar_file = os.path.join(args.caesar_dir, f'caesar_{snap_num:03}.hdf5')
    print(caesar_file)
    obj.save(caesar_file)

    print()
    print()
