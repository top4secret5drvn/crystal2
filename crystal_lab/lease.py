"""Cross-process workspace lease for the supported single-process launcher."""
from pathlib import Path
import os

class WorkspaceLease:
    def __init__(self,workspace):
        root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
        path=root/'.crystal-lab.lock'
        if path.is_symlink():raise ValueError('Lock path must not be a symlink')
        self.handle=path.open('a+b')
        try:
            if os.name=='nt':
                import msvcrt
                self.handle.seek(0,2)
                if self.handle.tell()==0:self.handle.write(b'0');self.handle.flush()
                self.handle.seek(0);msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.handle.close();raise RuntimeError('Another Crystal Lab process owns this workspace')
    def close(self):self.handle.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
