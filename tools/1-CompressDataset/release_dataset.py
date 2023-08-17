from os import listdir
import os
from absl import app, flags, logging
from os.path import join, isfile, isdir
from absl.logging import info
from tqdm import tqdm
from shutil import move, copy
from os import makedirs

def move_pb_and_copy_png_to_release(vsr_folder, release_folder):
    info(f"Processing {vsr_folder} -> {release_folder}")
    makedirs(release_folder, mode=0o777, exist_ok=True)

    # Some comments can not be executed, use the following code instead
    # comments = f"mv {vsr_folder}/*.pb {release_folder}/"
    # print(comments)
    # os.system(comments)
    # comments = f"cp {vsr_folder}/*.png {release_folder}/"
    # print(comments)
    # os.system(comments)

    for it in tqdm(listdir(vsr_folder)):
        if it.endswith("pb"):
            pb = join(vsr_folder, it)
            pb_release = join(release_folder, it)
            if isfile(pb_release):
                continue
            move(pb, pb_release)
            # info(f"Moved {pb} -> {pb_release}")
        elif it.endswith("png"):
            png = join(vsr_folder, it)
            png_release = join(release_folder, it)
            if isfile(png_release):
                continue
            copy(png, png_release)
            # info(f"Copied {png} -> {png_release}")

def remove_all_event_vis_pngs(vsr_folder):
    for video in listdir(vsr_folder):
        video_folder = join(vsr_folder, video)
        if isdir(video_folder):
            comments = f"rm {video_folder}/*eventsvis.png"
            print(comments)
            os.system(comments)

        # if the comments is not working, use the following code
        # for f in listdir(video_folder):
        #     if f.endswith("eventsvis.png"):
        #         path = join(video_folder, f)
        #         info(f"Removing {path}")
        #         os.remove(path)

def main(argv):
    import pudb

    pudb.set_trace()

    ROOT = ""
    VSR_ROOT = join(ROOT, "01-2022-10-24-vsr")
    RELEASE_ROOT = join(ROOT, "01-2022-10-24-vsr-release")

    for folder in listdir(VSR_ROOT):
        if isdir(join(VSR_ROOT, folder)):
            move_pb_and_copy_png_to_release(join(VSR_ROOT, folder), join(RELEASE_ROOT, folder))
        else:
            copy(join(VSR_ROOT, folder), join(RELEASE_ROOT, folder))

    # remove_all_event_vis_pngs(RELEASE_ROOT)

if __name__ == "__main__":
    app.run(main)