from os import listdir

import numpy as np
from absl import app
from absl.logging import info
from os.path import join, isfile, isdir
import event_frame_pb2 as epb


def compress(it, pb):
    event_pb = epb.EventFrame()
    events = np.load(it)
    h, w = events.shape[:2]
    event_pb.height = h
    event_pb.width = w
    for i in range(h):
        for j in range(w):
            if events[i, j] != 0:
                event = event_pb.events.add()
                event.x = i
                event.y = j
                event.p = events[i, j]
    with open(pb, "wb") as f:
        f.write(event_pb.SerializeToString())
    info(f"Compressed \n  {it} to \n  {pb}")


def walk_root(root):
    items = sorted(listdir(root))
    for i, it in enumerate(items):
        it = join(root, it)
        if isfile(it):
            if it.endswith("_events.npy"):
                pb = it.replace("_events.npy", "_events.pb")
                pb = join(root, pb)
                compress(it, pb)
        elif isdir(it):
            walk_root(it)
        # info
        info(f"{i}/{len(items)}: {it}")


def main(args):
    del args
    dataset_root = "/mnt/dev-my-book-yunfan-8T/backup-202211028/yunfanlu/dataset/01-EG-VSR-1023-2022/01-2022-10-24-vsr/"
    walk_root(dataset_root)


if __name__ == "__main__":
    app.run(main)
