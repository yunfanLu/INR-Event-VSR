from logging import info, warning
from os import listdir
from os.path import isdir, join

from absl.testing import absltest
from absl.testing.absltest import TestCase

from config import global_path as gp


class GlobalPathTest(TestCase):
    def test_color_event_dataset(self):
        info(f"Color Event Path: {gp.color_event_dataset}")
        self.assertTrue(isdir(gp.color_event_dataset))
        folder_count = 0
        for folder in listdir(gp.color_event_dataset):
            folder_path = join(gp.color_event_dataset, folder)
            if not isdir(folder_path):
                warning(f"{folder_path} is not a folder.")
            else:
                folder_count += 1
        self.assertEqual(folder_count, 84)


if __name__ == "__main__":
    absltest.main()
