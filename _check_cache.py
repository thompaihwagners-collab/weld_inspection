import os, glob
cache = os.path.expanduser(r'~\AppData\Local\pip\cache\http')
if os.path.exists(cache):
    files = glob.glob(cache + '/**/*torch*', recursive=True)
    sizes = [os.path.getsize(f) for f in files]
    print(f'Found {len(files)} torch cache files, total {sum(sizes)/1e6:.0f}MB')
    for f, s in sorted(zip(files, sizes), key=lambda x: -x[1])[:5]:
        print(f'  {s/1e6:.0f}MB - {os.path.basename(f)}')
else:
    print('No pip cache found')
