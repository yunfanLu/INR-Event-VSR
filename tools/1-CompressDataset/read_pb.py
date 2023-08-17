from os.path import isfile
from absl import app
import event_frame_pb2 as epb
from absl.logging import info


def main(args):
    del args
    pb_path = A_PB_FILE_PATH
    assert isfile(pb_path), f"pb file not found: {pb_path}"

    ef = epb.EventFrame()
    with open(pb_path, "rb") as f:
        ef.ParseFromString(f.read())

    info(f"height: {ef.height}")
    info(f"width: {ef.width}")
    info(f"events: {len(ef.events)}")
    for e in ef.events:
        info(f"  x,y,p: {e.x, e.y, e.p}")
        break


if __name__ == "__main__":
    app.run(main)
