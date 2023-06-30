from absl import app
import event_frame_pb2 as epb
from absl.logging import info


def main(args):
    del args
    pb_path = "/mnt/dev-my-book-yunfan-8T/backup-202211028/yunfanlu/dataset/01-EG-VSR-1023-2022/01-2022-10-24-vsr/20221022152113743/141694506_1632_1224_8_20221022152113743_0_events.pb"

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
