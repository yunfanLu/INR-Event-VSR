import pudb
import os
from os.path import join, isdir, isfile
from os import listdir


def main(args=None):
    RELEASE_ROOT = ""
    video_folders = listdir(RELEASE_ROOT)

    for video_folder in video_folders:
        video_folder_path = join(RELEASE_ROOT, video_folder)
        zip_file = join(RELEASE_ROOT, f"{video_folder}.zip")
        if isdir(video_folder_path):
            zip_comments = f'zip -r "{zip_file}" "{video_folder_path}"'
            print(zip_comments)
            os.system(zip_comments)

# pudb.set_trace()
main()