import os
os.environ["OMP_NUM_THREADS"] = "1" # limit numy to a single thread
MULTITHREAD = False

from profile import *

if __name__ == "__main__":
    main()
